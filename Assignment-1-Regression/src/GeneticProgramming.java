import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * Core Genetic Programming engine for symbolic regression of electricity load.
 *
 * <p>Implements canonical tree-based GP with:
 * <ul>
 *   <li>Grow initialisation (root always a function node)</li>
 *   <li>Mean Squared Error (MSE) fitness (minimisation)</li>
 *   <li>Tournament selection (k=2 to reduce premature convergence)</li>
 *   <li>Subtree crossover (one offspring per operation)</li>
 *   <li>Subtree mutation (replace a random subtree with a freshly grown one)</li>
 *   <li>Depth-bounded bloat control via {@link Tree#trimToDepth}</li>
 *   <li>2% elitism: top individuals carried forward each generation</li>
 * </ul>
 *
 * <p>The function set is {"+", "−", "*", "/"} (all binary; division is protected).
 * The terminal set is {"x1"…"x7"} plus ERCs sampled in [−5, 5].
 *
 * <p>COS 710 Assignment 1 – Genetic Programming for Symbolic Regression.
 */
public class GeneticProgramming {

    // ─── GP Language ──────────────────────────────────────────────────────────

    /** Binary arithmetic operators forming the function set. */
    private static final String[] FUNCTION_SET = {"+", "-", "*", "/"};

    /**
     * Variable terminals representing previous load values.
     * x1 = Load(t−1), x2 = Load(t−2), …, x7 = Load(t−7).
     * Must match the window size used in {@link Main}.
     */
    private static final String[] VARIABLE_TERMINALS = {"x1", "x2", "x3", "x4", "x5", "x6", "x7"};

    /**
     * Parsimony pressure coefficient. A small penalty proportional to tree size
     * is added to the MSE fitness to discourage bloated trees and improve
     * generalisation. Formula: fitness = MSE * (1 + PARSIMONY_COEFF * treeSize).
     */
    private static final double PARSIMONY_COEFF = 0.001;

    /**
     * Absolute-error threshold for counting a prediction as a "hit".
     * A training case is a hit if |predicted − actual| < HITS_THRESHOLD.
     * With load values typically in [0.04, 0.15], a threshold of 0.01
     * corresponds to roughly a 10% tolerance on a typical reading.
     */
    private static final double HITS_THRESHOLD = 0.01;

    // ─── Configurable Parameters ──────────────────────────────────────────────

    private final int    populationSize;
    private final int    maxDepth;
    private final int    generations;
    private final int    tournamentSize;
    private final double crossoverRate;
    private final double mutationRate;

    // ─── Dataset ──────────────────────────────────────────────────────────────

    private final double[][] trainInputs;
    private final double[]   trainTargets;
    private final double[][] testInputs;
    private final double[]   testTargets;

    // ─── Reproducibility ──────────────────────────────────────────────────────

    /** Seeded RNG – set once per run to guarantee reproducibility. */
    private final Random rng;

    // ─── Result Container ─────────────────────────────────────────────────────

    /**
     * Holds the results of a single completed GP run.
     */
    public static class RunResult {

        /** Best training MSE achieved during this run. */
        public final double trainFitness;

        /** Test MSE of the best training individual evaluated on held-out data. */
        public final double testFitness;

        /** The best evolved expression tree found during this run. */
        public final Tree bestTree;

        /** Wall-clock duration of this run in milliseconds. */
        public final long runtimeMs;

        /**
         * Per-generation average training MSE across the whole population.
         * Index 0 = generation 1, index {@code generations-1} = final generation.
         */
        public final double[] avgFitnessPerGen;

        /**
         * Per-generation average tree size (number of nodes) across the population.
         */
        public final double[] avgSizePerGen;

        /**
         * Per-generation hit count for the best individual in that generation.
         * A hit is a training case where |predicted − actual| < {@link #HITS_THRESHOLD}.
         */
        public final int[] hitsPerGen;

        /**
         * Per-generation population variety: number of structurally distinct
         * individuals (unique infix expression strings) in the population.
         */
        public final int[] varietyPerGen;

        RunResult(double trainFitness, double testFitness, Tree bestTree, long runtimeMs,
                  double[] avgFitnessPerGen, double[] avgSizePerGen,
                  int[] hitsPerGen, int[] varietyPerGen) {
            this.trainFitness      = trainFitness;
            this.testFitness       = testFitness;
            this.bestTree          = bestTree;
            this.runtimeMs         = runtimeMs;
            this.avgFitnessPerGen  = avgFitnessPerGen;
            this.avgSizePerGen     = avgSizePerGen;
            this.hitsPerGen        = hitsPerGen;
            this.varietyPerGen     = varietyPerGen;
        }
    }

    // ─── Constructor ──────────────────────────────────────────────────────────

    /**
     * Creates a new GP engine with the given parameters, dataset, and random seed.
     *
     * @param populationSize  number of individuals maintained per generation
     * @param maxDepth        maximum allowed expression tree depth
     * @param generations     number of evolutionary generations to run
     * @param tournamentSize  number of competitors drawn for each tournament
     * @param crossoverRate   probability of applying subtree crossover
     * @param mutationRate    probability of applying subtree mutation
     * @param seed            random seed that determines all stochastic decisions
     *                        (set once per run for reproducibility)
     * @param trainInputs     training input matrix [trainSize][windowSize]
     * @param trainTargets    training target vector [trainSize]
     * @param testInputs      test input matrix [testSize][windowSize]
     * @param testTargets     test target vector [testSize]
     */
    public GeneticProgramming(int populationSize, int maxDepth, int generations,
                               int tournamentSize, double crossoverRate, double mutationRate,
                               long seed,
                               double[][] trainInputs, double[] trainTargets,
                               double[][] testInputs,  double[] testTargets) {
        this.populationSize = populationSize;
        this.maxDepth       = maxDepth;
        this.generations    = generations;
        this.tournamentSize = tournamentSize;
        this.crossoverRate  = crossoverRate;
        this.mutationRate   = mutationRate;
        this.rng            = new Random(seed);
        this.trainInputs    = trainInputs;
        this.trainTargets   = trainTargets;
        this.testInputs     = testInputs;
        this.testTargets    = testTargets;
    }

    // ─── Population Initialisation ────────────────────────────────────────────

    /**
     * Initialises the population using the Grow method.
     *
     * <p>The root of every tree is forced to be a function node so that no
     * trivial single-terminal individual is created at startup. Up to three
     * generation attempts are made per slot to promote diversity by avoiding
     * duplicate tree expressions.
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
            // Fall back to the last generated tree if all attempts were duplicates
            if (tree == null) {
                tree = growTree(maxDepth, 0, true);
            }
            population[i] = tree;
        }
        return population;
    }

    /**
     * Grows a full expression tree using the Grow method and wraps it in a
     * {@link Tree} object with parent references properly established.
     *
     * @param maxDepth      maximum depth permitted for the grown tree
     * @param currentDepth  depth of the node currently being generated (root = 0)
     * @param forceFunction if {@code true}, the current node must be a function;
     *                      used to force the root to be a non-terminal
     * @return a new expression tree
     */
    private Tree growTree(int maxDepth, int currentDepth, boolean forceFunction) {
        Node root = growNode(maxDepth, currentDepth, forceFunction);
        Node.setParents(root, null);
        return new Tree(root);
    }

    /**
     * Recursively constructs a tree node using the Grow algorithm.
     *
     * <p>Rules:
     * <ul>
     *   <li>If {@code currentDepth >= maxDepth}: always create a terminal.</li>
     *   <li>If {@code forceFunction}: always create a function node.</li>
     *   <li>Otherwise: 50 % chance of function, 50 % chance of terminal.</li>
     * </ul>
     *
     * @param maxDepth      maximum allowed depth
     * @param currentDepth  depth of the current node from the root
     * @param forceFunction if {@code true}, force a function node
     * @return the created node (with subtree fully populated)
     */
    private Node growNode(int maxDepth, int currentDepth, boolean forceFunction) {
        // At or beyond max depth: must use a terminal
        if (!forceFunction && currentDepth >= maxDepth) {
            return Tree.makeTerminalNode(VARIABLE_TERMINALS, rng);
        }

        // Decide whether to place a function or a terminal at this position
        boolean chooseFunction = forceFunction || rng.nextBoolean();

        if (chooseFunction) {
            String op   = FUNCTION_SET[rng.nextInt(FUNCTION_SET.length)];
            Node   node = new Node(op, 2);
            // Recursively grow both children (neither is forced to be a function)
            node.children.add(growNode(maxDepth, currentDepth + 1, false));
            node.children.add(growNode(maxDepth, currentDepth + 1, false));
            return node;
        } else {
            return Tree.makeTerminalNode(VARIABLE_TERMINALS, rng);
        }
    }

    // ─── Fitness Evaluation ───────────────────────────────────────────────────

    /**
     * Computes the Mean Squared Error (MSE) of {code tree} on the training dataset.
     *
     * <p>If the tree produces a {@code NaN} or {@code Infinity} for any training
     * case, {@link Double#MAX_VALUE} is returned so the individual is effectively
     * penalised out of the gene pool.
     *
     * @param tree the expression tree to evaluate
     * @return MSE fitness (lower is better); {@link Double#MAX_VALUE} on numerical error
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
        // Parsimony pressure: penalise larger trees to encourage simpler,
        // more generalisable expressions.
        return mse * (1.0 + PARSIMONY_COEFF * tree.size());
    }

    /**
     * Computes the MSE of {@code tree} on the held-out test dataset.
     *
     * @param tree the expression tree to evaluate
     * @return test MSE (lower is better); {@link Double#MAX_VALUE} on numerical error
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
     * Evaluates the fitness of every individual in the population in order.
     *
     * @param population the array of trees to evaluate
     * @return array of MSE fitness values aligned with {@code population}
     */
    public double[] evaluatePopulation(Tree[] population) {
        double[] fitnesses = new double[population.length];
        for (int i = 0; i < population.length; i++) {
            fitnesses[i] = evaluateFitness(population[i]);
        }
        return fitnesses;
    }

    // ─── Selection ────────────────────────────────────────────────────────────

    /**
     * Tournament selection: draws {@link #tournamentSize} candidates uniformly
     * at random (with replacement) and returns the index of the one with the
     * lowest MSE fitness.
     *
     * @param fitnesses array of fitness values for the current population
     * @return index of the selected individual in the population array
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
     * {@code parent2}.
     *
     * <p>Bloat control: {@link Tree#trimToDepth} is applied to the offspring
     * immediately after the swap, followed by a full parent-reference refresh.
     *
     * @param parent1 the first parent (supplies the structural base)
     * @param parent2 the second parent (donor of the inserted subtree)
     * @return one offspring tree
     */
    public Tree crossover(Tree parent1, Tree parent2) {
        Tree offspring = parent1.copy();
        Tree donor     = parent2.copy();

        // Select a random node in the offspring to be replaced
        Node offspringPoint = offspring.getRandomNode(rng);

        // Select a random node from the donor and take a deep copy of its subtree
        Node donorSubtree = donor.getRandomNode(rng).copy();

        // Graft the donor subtree onto the offspring at the chosen point
        replaceNode(offspring, offspringPoint, donorSubtree);

        // Enforce depth limit and repair parent references
        Tree.trimToDepth(offspring.root, 0, maxDepth, VARIABLE_TERMINALS, rng);
        Node.setParents(offspring.root, null);

        return offspring;
    }

    /**
     * Subtree mutation: replaces a randomly chosen subtree in a copy of
     * {@code individual} with a freshly grown subtree.
     *
     * <p>The depth allowed for the replacement subtree is bounded so the total
     * tree depth cannot increase beyond {@link #maxDepth}. Bloat control is
     * applied afterwards as a safety net.
     *
     * @param individual the tree to mutate
     * @return a mutated copy (the original is not modified)
     */
    public Tree mutate(Tree individual) {
        Tree mutant = individual.copy();

        // Choose a random mutation point
        Node mutationPoint = mutant.getRandomNode(rng);

        // Depth remaining from the mutation point down to the limit
        int mutationDepth = getNodeDepth(mutationPoint);
        int allowedDepth  = Math.max(0, maxDepth - mutationDepth);

        // Grow a new subtree to insert at the mutation point
        Node newSubtree = growNode(allowedDepth, 0, false);

        // Replace the mutation point with the new subtree
        replaceNode(mutant, mutationPoint, newSubtree);

        // Safety: enforce depth limit and repair parent references
        Tree.trimToDepth(mutant.root, 0, maxDepth, VARIABLE_TERMINALS, rng);
        Node.setParents(mutant.root, null);

        return mutant;
    }

    /**
     * Replaces {@code target} in {@code tree} with {@code replacement}.
     *
     * <p>If {@code target} is the root, the tree's root reference is updated
     * directly. Otherwise, the replacement is spliced into the parent's children
     * list using reference equality to locate the correct slot.
     * Parent references for the replacement subtree are set after the splice.
     *
     * @param tree        the tree being structurally modified (in-place)
     * @param target      the node to be removed and replaced
     * @param replacement the new subtree to insert at {@code target}'s position
     */
    private void replaceNode(Tree tree, Node target, Node replacement) {
        if (target.parent == null) {
            // Target is the root – replace the whole tree
            tree.root = replacement;
            replacement.parent = null;
        } else {
            // Splice into parent's children list
            Node parent = target.parent;
            for (int i = 0; i < parent.children.size(); i++) {
                if (parent.children.get(i) == target) { // identity comparison
                    replacement.parent = parent;
                    parent.children.set(i, replacement);
                    break;
                }
            }
        }
        // Repair parent references throughout the inserted subtree
        Node.setParents(replacement, replacement.parent);
    }

    /**
     * Returns the depth of {@code node} from the root of its tree by walking
     * the parent chain upward.
     *
     * @param node the node whose root-relative depth is required
     * @return depth from root (root = 0)
     */
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

    /** Returns the average fitness across viable individuals (fitness < 1.0),
     *  excluding both Double.MAX_VALUE sentinels and trees with exploding
     *  (finite but astronomically large) predictions. */
    private double computeAvgFitness(double[] fitnesses) {
        double sum = 0.0;
        int count = 0;
        for (double f : fitnesses) {
            if (f < 1.0) { sum += f; count++; }
        }
        return count == 0 ? Double.MAX_VALUE : sum / count;
    }

    /** Returns the average number of nodes per tree across the population. */
    private double computeAvgSize(Tree[] population) {
        long total = 0;
        for (Tree t : population) total += t.size();
        return (double) total / population.length;
    }

    /**
     * Counts the number of training cases where {@code individual}'s prediction
     * is within {@link #HITS_THRESHOLD} of the true value (absolute error).
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

    /**
     * Returns the number of structurally distinct individuals in the population,
     * measured by unique infix expression strings.
     */
    private int computeVariety(Tree[] population) {
        java.util.HashSet<String> unique = new java.util.HashSet<>();
        for (Tree t : population) unique.add(t.toString());
        return unique.size();
    }

    // ─── Main Evolution Loop ──────────────────────────────────────────────────

    /**
     * Executes a complete GP run: initialise, evolve for {@link #generations}
     * generations, and return the best individual found.
     *
     * <p>Each generation follows the standard generational model:
     * <ol>
     *   <li>Evaluate all individuals (MSE on training data).</li>
     *   <li>Build a new population by repeatedly applying tournament selection
     *       followed by crossover, mutation, or reproduction.</li>
     *   <li>Replace the old population entirely.</li>
     * </ol>
     * The global best individual (lowest training MSE seen across all generations)
     * is tracked and returned.
     *
     * @return a {@link RunResult} containing best fitness, test fitness, best tree,
     *         and wall-clock runtime
     */
    public RunResult run() {
        long startTime = System.currentTimeMillis();

        // ── Initialise and evaluate the starting population ───────────────────
        Tree[]   population = initializePopulation();
        double[] fitnesses  = evaluatePopulation(population);

        int    bestIdx     = getBestIndex(fitnesses);
        double bestFitness = fitnesses[bestIdx];
        Tree   bestTree    = population[bestIdx].copy();

        // Per-generation tracking arrays
        double[] avgFitnessPerGen = new double[generations];
        double[] avgSizePerGen    = new double[generations];
        int[]    hitsPerGen       = new int[generations];
        int[]    varietyPerGen    = new int[generations];

        // ── Evolution loop ────────────────────────────────────────────────────
        for (int gen = 0; gen < generations; gen++) {
            Tree[] newPopulation = new Tree[populationSize];

            // ── Elitism: carry the best ~1% of individuals forward unchanged ──
            // This prevents regression between generations and ensures the best
            // solution found so far is never lost to genetic drift.
            int eliteCount = Math.max(1, (int) Math.round(populationSize * 0.02));
            int[] eliteIdx = getTopIndices(fitnesses, eliteCount);
            for (int e = 0; e < eliteCount; e++) {
                newPopulation[e] = population[eliteIdx[e]].copy();
            }

            // ── Determine exact operator counts from rates ────────────────────
            // Rates must sum to 1. Each rate describes exactly what fraction of
            // the non-elite slots is created by that operator, guaranteeing the
            // intended distribution regardless of random variation.
            int nonEliteSlots  = populationSize - eliteCount;
            int crossoverCount = (int) Math.round(nonEliteSlots * crossoverRate);
            int mutationCount  = nonEliteSlots - crossoverCount; // remainder = mutation

            // Build a slot-type array and shuffle it so operator assignments
            // are random across positions (not all crossover first, then mutation).
            // 0 = crossover, 1 = mutation
            int[] slotType = new int[nonEliteSlots];
            for (int i = crossoverCount; i < nonEliteSlots; i++) slotType[i] = 1;
            // Fisher-Yates shuffle
            for (int i = nonEliteSlots - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = slotType[i]; slotType[i] = slotType[j]; slotType[j] = tmp;
            }

            for (int i = eliteCount; i < populationSize; i++) {
                Tree offspring;
                int slotIdx = i - eliteCount;

                if (slotType[slotIdx] == 0) {
                    // ── Crossover ─────────────────────────────────────────────
                    int p1 = tournamentSelection(fitnesses);
                    int p2 = tournamentSelection(fitnesses);
                    offspring = crossover(population[p1], population[p2]);
                } else {
                    // ── Mutation ──────────────────────────────────────────────
                    int p = tournamentSelection(fitnesses);
                    offspring = mutate(population[p]);
                }

                newPopulation[i] = offspring;
            }

            // Replace old population and evaluate the new one
            population = newPopulation;
            fitnesses  = evaluatePopulation(population);

            // Update global best
            int genBest = getBestIndex(fitnesses);
            if (fitnesses[genBest] < bestFitness) {
                bestFitness = fitnesses[genBest];
                bestTree    = population[genBest].copy();
            }

            // ── Record per-generation statistics ──────────────────────────────
            avgFitnessPerGen[gen] = computeAvgFitness(fitnesses);
            avgSizePerGen[gen]    = computeAvgSize(population);
            hitsPerGen[gen]       = computeHits(population[genBest]);
            varietyPerGen[gen]    = computeVariety(population);
        }

        // ── Evaluate best individual on test data ─────────────────────────────
        long   runtimeMs   = System.currentTimeMillis() - startTime;
        double testFitness = evaluateTestFitness(bestTree);

        return new RunResult(bestFitness, testFitness, bestTree, runtimeMs,
                avgFitnessPerGen, avgSizePerGen, hitsPerGen, varietyPerGen);
    }

    /**
     * Returns the index of the individual with the lowest (best) fitness value.
     *
     * @param fitnesses array of fitness values
     * @return index of the best individual in the array
     */
    private int getBestIndex(double[] fitnesses) {
        int best = 0;
        for (int i = 1; i < fitnesses.length; i++) {
            if (fitnesses[i] < fitnesses[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Returns the indices of the {@code count} individuals with the lowest
     * (best) fitness values, used for elitist carry-over between generations.
     *
     * @param fitnesses array of fitness values
     * @param count     number of elite individuals to identify
     * @return indices of the {@code count} best individuals
     */
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
