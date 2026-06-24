import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * Structure-Based GP engine for symbolic regression of electricity load.
 *
 * <p>Extends canonical tree-based GP (Assignment 1) by incorporating structural
 * information directly into the fitness function via <em>fitness sharing</em>.
 * The structural mechanism is derived from the lecture slides on SBGP:
 * "counting the number of nodes that are the same from the root."
 *
 * <h2>How Structure Is Incorporated</h2>
 * <p>After standard MSE-based fitness evaluation each generation, an additional
 * structural penalty is applied to individuals that are too similar (share too
 * much common prefix structure) to many of their peers. The adjusted fitness is:
 * <pre>
 *   adjustedFitness[i] = rawFitness[i] × (1 + α × nicheCount[i])
 * </pre>
 * where {@code nicheCount[i]} is the number of sampled peers whose structural
 * similarity to individual {@code i} meets or exceeds the threshold σ
 * ({@link #similarityThreshold}).  Similarity is normalised prefix match count
 * divided by the size of the larger tree ({@link Tree#structuralSimilarity}).
 *
 * <p>This mechanism discourages structural convergence, maintains population
 * diversity, and helps avoid premature convergence to local optima — addressing
 * GP's known tendency to converge to structurally homogeneous populations.
 *
 * <p>Selection, tournament, and elitism all use the <em>adjusted</em> fitness.
 * The best individual is tracked and reported by <em>raw</em> fitness, since
 * raw fitness (MSE + parsimony) reflects actual prediction quality.
 *
 * <h2>Base GP Components (unchanged from Assignment 1)</h2>
 * <ul>
 *   <li>Grow initialisation (root always a function node)</li>
 *   <li>MSE + parsimony fitness (minimisation)</li>
 *   <li>Tournament selection (k=2)</li>
 *   <li>Subtree crossover (one offspring per operation)</li>
 *   <li>Subtree mutation (replace a random subtree with a freshly grown one)</li>
 *   <li>Depth-bounded bloat control via {@link Tree#trimToDepth}</li>
 *   <li>Exact operator rates via Fisher-Yates slot assignment</li>
 *   <li>2% elitism: top individuals carried forward each generation</li>
 * </ul>
 *
 * <p>COS 710 Assignment 2 – Structure-Based Genetic Programming for Regression.
 */
public class GeneticProgramming {

    // ─── GP Language ──────────────────────────────────────────────────────────

    /** Binary arithmetic operators forming the function set. */
    private static final String[] FUNCTION_SET = {"+", "-", "*", "/"};

    /**
     * Variable terminals representing previous load values.
     * x1 = Load(t−1), x2 = Load(t−2), …, x7 = Load(t−7).
     */
    private static final String[] VARIABLE_TERMINALS = {"x1","x2","x3","x4","x5","x6","x7"};

    /**
     * Parsimony pressure coefficient. A small penalty proportional to tree size
     * is added to the MSE fitness to discourage bloated trees.
     * Formula: rawFitness = MSE * (1 + PARSIMONY_COEFF * treeSize).
     */
    private static final double PARSIMONY_COEFF = 0.001;

    /**
     * Absolute-error threshold for counting a prediction as a "hit".
     * A training case is a hit if |predicted − actual| < HITS_THRESHOLD.
     */
    private static final double HITS_THRESHOLD = 0.01;

    // ─── Configurable Base GP Parameters ──────────────────────────────────────

    private final int    populationSize;
    private final int    maxDepth;
    private final int    generations;
    private final int    tournamentSize;
    private final double crossoverRate;
    private final double mutationRate;

    // ─── SBGP Parameters ──────────────────────────────────────────────────────

    /**
     * Similarity threshold (σ): two individuals are considered structurally
     * similar if their normalised prefix match similarity is ≥ this value.
     * Individuals accumulate a niche count for each peer that meets the threshold.
     */
    private final double similarityThreshold;

    /**
     * Sharing coefficient (α): scales the structural penalty applied to crowded
     * individuals. A small value ensures the structural component complements the
     * behavioural fitness signal without overwhelming it.
     * Formula: adjustedFitness = rawFitness × (1 + α × nicheCount).
     */
    private final double sharingAlpha;

    /**
     * Number of random peers sampled when computing each individual's niche count.
     * Avoids the O(n²) cost of all-pairs comparison; 50 samples provide a
     * representative estimate at a fraction of the full comparison cost.
     */
    private final int structuralSampleSize;

    // ─── Dataset ──────────────────────────────────────────────────────────────

    private final double[][] trainInputs;
    private final double[]   trainTargets;
    private final double[][] testInputs;
    private final double[]   testTargets;

    // ─── Reproducibility ──────────────────────────────────────────────────────

    private final Random rng;

    // ─── Result Container ─────────────────────────────────────────────────────

    /**
     * Holds the results of a single completed SBGP run.
     */
    public static class RunResult {

        /** Best raw training MSE achieved during this run. */
        public final double trainFitness;

        /** Test MSE of the best training individual evaluated on held-out data. */
        public final double testFitness;

        /** The best evolved expression tree found during this run. */
        public final Tree bestTree;

        /** Wall-clock duration of this run in milliseconds. */
        public final long runtimeMs;

        /** Per-generation average raw training MSE across the population. */
        public final double[] avgFitnessPerGen;

        /** Per-generation average tree size (node count) across the population. */
        public final double[] avgSizePerGen;

        /** Per-generation hit count for the best individual in that generation. */
        public final int[] hitsPerGen;

        /** Per-generation population variety: count of unique infix strings. */
        public final int[] varietyPerGen;

        /**
         * Per-generation average pairwise structural similarity across sampled
         * individual pairs. Tracks how structurally homogeneous the population
         * becomes over time; higher values indicate structural convergence.
         */
        public final double[] avgStructuralSimilarityPerGen;

        RunResult(double trainFitness, double testFitness, Tree bestTree, long runtimeMs,
                  double[] avgFitnessPerGen, double[] avgSizePerGen,
                  int[] hitsPerGen, int[] varietyPerGen,
                  double[] avgStructuralSimilarityPerGen) {
            this.trainFitness                  = trainFitness;
            this.testFitness                   = testFitness;
            this.bestTree                      = bestTree;
            this.runtimeMs                     = runtimeMs;
            this.avgFitnessPerGen              = avgFitnessPerGen;
            this.avgSizePerGen                 = avgSizePerGen;
            this.hitsPerGen                    = hitsPerGen;
            this.varietyPerGen                 = varietyPerGen;
            this.avgStructuralSimilarityPerGen = avgStructuralSimilarityPerGen;
        }
    }

    // ─── Constructor ──────────────────────────────────────────────────────────

    /**
     * Creates a new SBGP engine with the given base GP parameters, SBGP
     * structural parameters, dataset, and random seed.
     *
     * @param populationSize      number of individuals maintained per generation
     * @param maxDepth            maximum allowed expression tree depth
     * @param generations         number of evolutionary generations to run
     * @param tournamentSize      number of competitors drawn for each tournament
     * @param crossoverRate       fraction of non-elite slots produced by crossover
     * @param mutationRate        fraction of non-elite slots produced by mutation
     * @param similarityThreshold structural similarity threshold σ for niche counting
     * @param sharingAlpha        sharing coefficient α scaling the structural penalty
     * @param structuralSampleSize number of random peers sampled for fitness sharing
     * @param seed                random seed for full reproducibility
     * @param trainInputs         training input matrix [trainSize][windowSize]
     * @param trainTargets        training target vector [trainSize]
     * @param testInputs          test input matrix [testSize][windowSize]
     * @param testTargets         test target vector [testSize]
     */
    public GeneticProgramming(int populationSize, int maxDepth, int generations,
                               int tournamentSize, double crossoverRate, double mutationRate,
                               double similarityThreshold, double sharingAlpha,
                               int structuralSampleSize,
                               long seed,
                               double[][] trainInputs, double[] trainTargets,
                               double[][] testInputs,  double[] testTargets) {
        this.populationSize       = populationSize;
        this.maxDepth             = maxDepth;
        this.generations          = generations;
        this.tournamentSize       = tournamentSize;
        this.crossoverRate        = crossoverRate;
        this.mutationRate         = mutationRate;
        this.similarityThreshold  = similarityThreshold;
        this.sharingAlpha         = sharingAlpha;
        this.structuralSampleSize = structuralSampleSize;
        this.rng                  = new Random(seed);
        this.trainInputs          = trainInputs;
        this.trainTargets         = trainTargets;
        this.testInputs           = testInputs;
        this.testTargets          = testTargets;
    }

    // ─── Population Initialisation ────────────────────────────────────────────

    /**
     * Initialises the population using the Grow method.
     * The root of every tree is forced to be a function node. Up to three
     * generation attempts are made per slot to promote initial diversity.
     *
     * @return array of {@code populationSize} freshly generated individuals
     */
    public Tree[] initializePopulation() {
        Tree[] population = new Tree[populationSize];
        Set<String> seen  = new HashSet<>();

        for (int i = 0; i < populationSize; i++) {
            Tree tree = null;
            for (int attempt = 0; attempt < 3; attempt++) {
                Tree candidate = growTree(maxDepth, 0, true);
                String repr = candidate.toString();
                if (!seen.contains(repr)) {
                    seen.add(repr);
                    tree = candidate;
                    break;
                }
            }
            if (tree == null) {
                tree = growTree(maxDepth, 0, true);
            }
            population[i] = tree;
        }
        return population;
    }

    private Tree growTree(int maxDepth, int currentDepth, boolean forceFunction) {
        Node root = growNode(maxDepth, currentDepth, forceFunction);
        Node.setParents(root, null);
        return new Tree(root);
    }

    private Node growNode(int maxDepth, int currentDepth, boolean forceFunction) {
        if (!forceFunction && currentDepth >= maxDepth) {
            return Tree.makeTerminalNode(VARIABLE_TERMINALS, rng);
        }
        boolean chooseFunction = forceFunction || rng.nextBoolean();
        if (chooseFunction) {
            String op   = FUNCTION_SET[rng.nextInt(FUNCTION_SET.length)];
            Node   node = new Node(op, 2);
            node.children.add(growNode(maxDepth, currentDepth + 1, false));
            node.children.add(growNode(maxDepth, currentDepth + 1, false));
            return node;
        } else {
            return Tree.makeTerminalNode(VARIABLE_TERMINALS, rng);
        }
    }

    // ─── Fitness Evaluation ───────────────────────────────────────────────────

    /**
     * Computes the raw fitness of {@code tree} on the training dataset.
     * Raw fitness = MSE + parsimony pressure (proportional to tree size).
     * Returns {@link Double#MAX_VALUE} on any numerical error.
     *
     * @param tree the expression tree to evaluate
     * @return raw fitness (lower is better)
     */
    public double evaluateFitness(Tree tree) {
        double sumSquaredError = 0.0;
        int n = trainInputs.length;

        for (int i = 0; i < n; i++) {
            double predicted = tree.evaluate(trainInputs[i]);
            if (Double.isNaN(predicted) || Double.isInfinite(predicted)) {
                return Double.MAX_VALUE;
            }
            double error = predicted - trainTargets[i];
            sumSquaredError += error * error;
        }

        double mse = sumSquaredError / n;
        return mse * (1.0 + PARSIMONY_COEFF * tree.size());
    }

    /**
     * Computes the MSE of {@code tree} on the held-out test dataset.
     *
     * @param tree the expression tree to evaluate
     * @return test MSE; {@link Double#MAX_VALUE} on numerical error
     */
    public double evaluateTestFitness(Tree tree) {
        double sumSquaredError = 0.0;
        int n = testInputs.length;

        for (int i = 0; i < n; i++) {
            double predicted = tree.evaluate(testInputs[i]);
            if (Double.isNaN(predicted) || Double.isInfinite(predicted)) {
                return Double.MAX_VALUE;
            }
            double error = predicted - testTargets[i];
            sumSquaredError += error * error;
        }
        return sumSquaredError / n;
    }

    /**
     * Evaluates the raw fitness of every individual in the population.
     *
     * @param population the array of trees to evaluate
     * @return array of raw fitness values aligned with {@code population}
     */
    public double[] evaluatePopulation(Tree[] population) {
        double[] fitnesses = new double[population.length];
        for (int i = 0; i < population.length; i++) {
            fitnesses[i] = evaluateFitness(population[i]);
        }
        return fitnesses;
    }

    // ─── Structural Fitness Sharing (SBGP) ───────────────────────────────────

    /**
     * Applies structural fitness sharing to produce an adjusted fitness array
     * used for selection and elitism.
     *
     * <p>For each individual {@code i}, {@link #structuralSampleSize} peers are
     * sampled uniformly at random. The niche count is the number of sampled peers
     * whose structural similarity to {@code i} meets or exceeds
     * {@link #similarityThreshold}. The adjusted fitness is then:
     * <pre>
     *   adjustedFitness[i] = rawFitness[i] × (1 + α × nicheCount[i])
     * </pre>
     *
     * <p>Individuals with {@link Double#MAX_VALUE} raw fitness (numerical failures)
     * are left at {@code MAX_VALUE} and excluded from niche count contributions.
     * This ensures failed individuals cannot cause over-penalisation of their peers.
     *
     * @param population the current population of trees
     * @param rawFitness raw (unadjusted) fitness values from {@link #evaluatePopulation}
     * @return adjusted fitness array for use in selection; same length as population
     */
    private double[] applyFitnessSharing(Tree[] population, double[] rawFitness) {
        int n = population.length;
        double[] adjusted = new double[n];

        for (int i = 0; i < n; i++) {
            if (rawFitness[i] == Double.MAX_VALUE) {
                adjusted[i] = Double.MAX_VALUE;
                continue;
            }

            int nicheCount = 0;
            for (int s = 0; s < structuralSampleSize; s++) {
                int j = rng.nextInt(n);
                if (j == i || rawFitness[j] == Double.MAX_VALUE) continue;
                double sim = population[i].structuralSimilarity(population[j]);
                if (sim >= similarityThreshold) nicheCount++;
            }

            adjusted[i] = rawFitness[i] * (1.0 + sharingAlpha * nicheCount);
        }
        return adjusted;
    }

    // ─── Selection ────────────────────────────────────────────────────────────

    /**
     * Tournament selection: draws {@link #tournamentSize} candidates uniformly
     * at random and returns the index of the one with the lowest fitness.
     *
     * @param fitnesses fitness values for the current population (adjusted)
     * @return index of the selected individual
     */
    public int tournamentSelection(double[] fitnesses) {
        int best = rng.nextInt(fitnesses.length);
        for (int i = 1; i < tournamentSize; i++) {
            int candidate = rng.nextInt(fitnesses.length);
            if (fitnesses[candidate] < fitnesses[best]) {
                best = candidate;
            }
        }
        return best;
    }

    // ─── Genetic Operators ────────────────────────────────────────────────────

    /**
     * Subtree crossover: produces one offspring by replacing a randomly chosen
     * subtree in a copy of {@code parent1} with a randomly chosen subtree from
     * {@code parent2}. Bloat control is applied immediately after the swap.
     *
     * @param parent1 the first parent (supplies the structural base)
     * @param parent2 the second parent (donor of the inserted subtree)
     * @return one offspring tree
     */
    public Tree crossover(Tree parent1, Tree parent2) {
        Tree offspring = parent1.copy();
        Tree donor     = parent2.copy();

        Node offspringPoint = offspring.getRandomNode(rng);
        Node donorSubtree   = donor.getRandomNode(rng).copy();

        replaceNode(offspring, offspringPoint, donorSubtree);

        Tree.trimToDepth(offspring.root, 0, maxDepth, VARIABLE_TERMINALS, rng);
        Node.setParents(offspring.root, null);

        return offspring;
    }

    /**
     * Subtree mutation: replaces a randomly chosen subtree in a copy of
     * {@code individual} with a freshly grown subtree.
     *
     * @param individual the tree to mutate
     * @return a mutated copy (the original is not modified)
     */
    public Tree mutate(Tree individual) {
        Tree mutant = individual.copy();

        Node mutationPoint = mutant.getRandomNode(rng);
        int  mutationDepth = getNodeDepth(mutationPoint);
        int  allowedDepth  = Math.max(0, maxDepth - mutationDepth);

        Node newSubtree = growNode(allowedDepth, 0, false);

        replaceNode(mutant, mutationPoint, newSubtree);

        Tree.trimToDepth(mutant.root, 0, maxDepth, VARIABLE_TERMINALS, rng);
        Node.setParents(mutant.root, null);

        return mutant;
    }

    private void replaceNode(Tree tree, Node target, Node replacement) {
        if (target.parent == null) {
            tree.root = replacement;
            replacement.parent = null;
        } else {
            Node parent = target.parent;
            for (int i = 0; i < parent.children.size(); i++) {
                if (parent.children.get(i) == target) {
                    replacement.parent = parent;
                    parent.children.set(i, replacement);
                    break;
                }
            }
        }
        Node.setParents(replacement, replacement.parent);
    }

    private int getNodeDepth(Node node) {
        int depth = 0;
        Node current = node;
        while (current.parent != null) {
            depth++;
            current = current.parent;
        }
        return depth;
    }

    // ─── Per-Generation Statistics Helpers ───────────────────────────────────

    /** Average raw fitness across viable individuals (raw fitness < 1.0). */
    private double computeAvgFitness(double[] rawFitness) {
        double sum = 0.0;
        int count = 0;
        for (double f : rawFitness) {
            if (f < 1.0) { sum += f; count++; }
        }
        return count == 0 ? Double.MAX_VALUE : sum / count;
    }

    /** Average number of nodes per tree across the population. */
    private double computeAvgSize(Tree[] population) {
        long total = 0;
        for (Tree t : population) total += t.size();
        return (double) total / population.length;
    }

    /**
     * Counts training cases where {@code individual}'s prediction is within
     * {@link #HITS_THRESHOLD} of the true value.
     */
    public int computeHits(Tree individual) {
        int hits = 0;
        for (int i = 0; i < trainInputs.length; i++) {
            double predicted = individual.evaluate(trainInputs[i]);
            if (!Double.isNaN(predicted) && !Double.isInfinite(predicted)
                    && Math.abs(predicted - trainTargets[i]) < HITS_THRESHOLD) {
                hits++;
            }
        }
        return hits;
    }

    /** Number of structurally distinct individuals (unique infix strings). */
    private int computeVariety(Tree[] population) {
        Set<String> unique = new HashSet<>();
        for (Tree t : population) unique.add(t.toString());
        return unique.size();
    }

    /**
     * Estimates the average pairwise structural similarity in the population by
     * sampling 100 random pairs. Used to monitor structural diversity over time.
     *
     * @param population the current population
     * @return estimated mean pairwise structural similarity in [0.0, 1.0]
     */
    private double computeAvgStructuralSimilarity(Tree[] population) {
        int n = population.length;
        if (n < 2) return 0.0;
        double totalSim = 0.0;
        int numSamples  = 100;
        for (int s = 0; s < numSamples; s++) {
            int i = rng.nextInt(n);
            int j = rng.nextInt(n);
            if (j == i) j = (j + 1) % n;
            totalSim += population[i].structuralSimilarity(population[j]);
        }
        return totalSim / numSamples;
    }

    // ─── Main Evolution Loop ──────────────────────────────────────────────────

    /**
     * Executes a complete SBGP run: initialise, evolve for {@link #generations}
     * generations with structural fitness sharing, and return the best individual.
     *
     * <p>Each generation:
     * <ol>
     *   <li>Evaluate raw fitness (MSE + parsimony) for the full population.</li>
     *   <li>Apply structural fitness sharing → adjusted fitness array.</li>
     *   <li>Carry the best {@code eliteCount} individuals (by adjusted fitness)
     *       to the next generation unchanged.</li>
     *   <li>Fill remaining slots using tournament selection on adjusted fitness,
     *       applying crossover ({@code crossoverRate}) or mutation ({@code mutationRate})
     *       via exact Fisher-Yates slot assignment.</li>
     *   <li>Track global best by raw fitness for reporting accuracy.</li>
     * </ol>
     *
     * @return {@link RunResult} containing best fitness, test fitness, best tree,
     *         runtime, and per-generation statistics
     */
    public RunResult run() {
        long startTime = System.currentTimeMillis();

        // ── Initialise and evaluate the starting population ───────────────────
        Tree[]   population  = initializePopulation();
        double[] rawFitness  = evaluatePopulation(population);
        double[] adjFitness  = applyFitnessSharing(population, rawFitness);

        // Best is tracked by raw fitness (true prediction quality)
        int    bestIdx     = getBestIndex(rawFitness);
        double bestFitness = rawFitness[bestIdx];
        Tree   bestTree    = population[bestIdx].copy();

        // Per-generation tracking arrays
        double[] avgFitnessPerGen             = new double[generations];
        double[] avgSizePerGen                = new double[generations];
        int[]    hitsPerGen                   = new int[generations];
        int[]    varietyPerGen                = new int[generations];
        double[] avgStructuralSimilarityPerGen = new double[generations];

        // ── Evolution loop ────────────────────────────────────────────────────
        for (int gen = 0; gen < generations; gen++) {
            Tree[] newPopulation = new Tree[populationSize];

            // ── Elitism: carry top ~2% of individuals forward by adjusted fitness ──
            int eliteCount = Math.max(1, (int) Math.round(populationSize * 0.02));
            int[] eliteIdx = getTopIndices(adjFitness, eliteCount);
            for (int e = 0; e < eliteCount; e++) {
                newPopulation[e] = population[eliteIdx[e]].copy();
            }

            // ── Exact operator slot assignment via Fisher-Yates shuffle ───────
            int nonEliteSlots  = populationSize - eliteCount;
            int crossoverCount = (int) Math.round(nonEliteSlots * crossoverRate);
            int mutationCount  = (int) Math.round(nonEliteSlots * mutationRate);
            // Reconcile any floating-point rounding so all slots are filled
            if (crossoverCount + mutationCount != nonEliteSlots) {
                crossoverCount = nonEliteSlots - mutationCount;
            }

            int[] slotType = new int[nonEliteSlots]; // 0=crossover, 1=mutation
            for (int i = crossoverCount; i < crossoverCount + mutationCount; i++) slotType[i] = 1;
            for (int i = nonEliteSlots - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = slotType[i]; slotType[i] = slotType[j]; slotType[j] = tmp;
            }

            // ── Selection and operator application (uses adjusted fitness) ────
            for (int i = eliteCount; i < populationSize; i++) {
                Tree offspring;
                if (slotType[i - eliteCount] == 0) {
                    int p1 = tournamentSelection(adjFitness);
                    int p2 = tournamentSelection(adjFitness);
                    offspring = crossover(population[p1], population[p2]);
                } else {
                    int p = tournamentSelection(adjFitness);
                    offspring = mutate(population[p]);
                }
                newPopulation[i] = offspring;
            }

            // ── Evaluate new population ───────────────────────────────────────
            population = newPopulation;
            rawFitness = evaluatePopulation(population);
            adjFitness = applyFitnessSharing(population, rawFitness);

            // ── Update global best by raw fitness ─────────────────────────────
            int genBest = getBestIndex(rawFitness);
            if (rawFitness[genBest] < bestFitness) {
                bestFitness = rawFitness[genBest];
                bestTree    = population[genBest].copy();
            }

            // ── Record per-generation statistics ──────────────────────────────
            avgFitnessPerGen[gen]              = computeAvgFitness(rawFitness);
            avgSizePerGen[gen]                 = computeAvgSize(population);
            hitsPerGen[gen]                    = computeHits(population[genBest]);
            varietyPerGen[gen]                 = computeVariety(population);
            avgStructuralSimilarityPerGen[gen] = computeAvgStructuralSimilarity(population);
        }

        // ── Evaluate best individual on test data ─────────────────────────────
        long   runtimeMs   = System.currentTimeMillis() - startTime;
        double testFitness = evaluateTestFitness(bestTree);

        return new RunResult(bestFitness, testFitness, bestTree, runtimeMs,
                avgFitnessPerGen, avgSizePerGen, hitsPerGen, varietyPerGen,
                avgStructuralSimilarityPerGen);
    }

    // ─── Utility Helpers ──────────────────────────────────────────────────────

    private int getBestIndex(double[] fitnesses) {
        int best = 0;
        for (int i = 1; i < fitnesses.length; i++) {
            if (fitnesses[i] < fitnesses[best]) best = i;
        }
        return best;
    }

    private int[] getTopIndices(double[] fitnesses, int count) {
        int[]     indices = new int[count];
        boolean[] used    = new boolean[fitnesses.length];
        for (int i = 0; i < count; i++) {
            int best = -1;
            for (int j = 0; j < fitnesses.length; j++) {
                if (!used[j] && (best < 0 || fitnesses[j] < fitnesses[best])) {
                    best = j;
                }
            }
            indices[i] = best;
            used[best]  = true;
        }
        return indices;
    }
}
