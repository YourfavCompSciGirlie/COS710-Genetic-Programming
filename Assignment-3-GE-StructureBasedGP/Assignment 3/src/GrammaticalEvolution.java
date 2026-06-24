import java.util.Random;

/**
 * Structure-Based Grammatical Evolution (SBGE) engine for symbolic regression.
 *
 * <h2>Algorithm overview</h2>
 * Canonical Grammatical Evolution (O'Neill &amp; Ryan, 2001) maintains a
 * population of variable-length integer codon genomes. Each genome is mapped
 * via a BNF grammar (see {@link Grammar}) to a phenotype program. Structure is
 * incorporated into this canonical scheme by augmenting fitness with a
 * <em>structural sharing penalty</em> that is computed over the derivation
 * signature of each individual:
 * <pre>
 *   adjustedFitness[i] = rawFitness[i] × (1 + α × nicheCount[i])
 * </pre>
 * where {@code nicheCount[i]} counts how many sampled peers have a
 * derivation-signature prefix match ratio above {@link #similarityThreshold}.
 * Crowded structural niches are therefore penalised, encouraging the search
 * to explore a wider region of program-structure space.
 *
 * <h2>Elitism</h2>
 * Elitism uses <em>raw</em> training fitness, not the adjusted (shared)
 * fitness. The purpose of elitism is to preserve the best behavioural
 * solution found so far between generations; structural crowding should
 * not influence whether an individual survives unchanged.
 *
 * <h2>Operators</h2>
 * Children for the next generation are produced by an exact-rate operator
 * split (not per-child Bernoulli probabilities): given a population of size
 * {@code N}, exactly 1 individual is reproduced by elitism, exactly
 * {@code round(crossoverRate × (N − 1))} are produced by crossover, and the
 * remaining slots are produced by mutation. {@code crossoverRate} and
 * {@code mutationRate} must sum to 1.0.
 * <ul>
 *   <li><strong>Tournament selection</strong> over adjusted fitness — drives
 *       parent selection towards both high-quality and structurally rare
 *       individuals.</li>
 *   <li><strong>One-point crossover</strong> on the integer codon array —
 *       the standard GE recombination operator; cleanly mixes leading
 *       (structural) and trailing (refinement) codon material between parents.</li>
 *   <li><strong>Integer mutation</strong> (per-codon) — replaces a codon with
 *       a fresh random integer at probability {@code codonMutationProb}.</li>
 * </ul>
 *
 * <p>COS 710 Assignment 3 – Structure-Based Grammatical Evolution.
 */
public final class GrammaticalEvolution {

    // ─── GA hyper-parameters ──────────────────────────────────────────────────

    private final int    populationSize;
    private final int    generations;
    private final int    tournamentSize;
    private final double crossoverRate;
    private final double mutationRate;
    private final double codonMutationProb;
    private final int    genomeLength;
    private final int    codonMax;

    // ─── Structural sharing parameters ────────────────────────────────────────

    /**
     * Normalised prefix-match threshold above which two derivation signatures
     * are considered structurally similar (and so share a niche).
     */
    private final double similarityThreshold;

    /** Sharing coefficient α scaling the structural penalty. */
    private final double sharingAlpha;

    /** Number of random peers sampled per individual for niche counting. */
    private final int    structuralSampleSize;

    // ─── Parsimony coefficient for raw fitness ───────────────────────────────

    /** Small parsimony pressure to discourage bloated phenotypes (same as A1/A2). */
    private static final double PARSIMONY_COEFF = 0.001;

    // ─── Runtime state ────────────────────────────────────────────────────────

    private final long       seed;
    private final Random     rng;
    private final double[][] trainInputs;
    private final double[]   trainTargets;
    private final double[][] testInputs;
    private final double[]   testTargets;

    /**
     * Aggregated results of a single GE run, returned by {@link #run()}.
     */
    public static final class RunResult {
        public final double      trainFitness;
        public final double      testFitness;
        public final Expression  bestExpression;
        public final int[]       bestGenome;
        public final long        runtimeMs;
        public final double[]    avgFitnessPerGen;
        public final double[]    avgSizePerGen;
        public final int[]       hitsPerGen;
        public final int[]       varietyPerGen;
        public final double[]    avgStructuralSimilarityPerGen;
        public final int[]       invalidPerGen;

        RunResult(double trainFitness, double testFitness,
                  Expression bestExpression, int[] bestGenome,
                  long runtimeMs,
                  double[] avgFitnessPerGen, double[] avgSizePerGen,
                  int[] hitsPerGen, int[] varietyPerGen,
                  double[] avgStructuralSimilarityPerGen,
                  int[] invalidPerGen) {
            this.trainFitness        = trainFitness;
            this.testFitness         = testFitness;
            this.bestExpression      = bestExpression;
            this.bestGenome          = bestGenome;
            this.runtimeMs           = runtimeMs;
            this.avgFitnessPerGen    = avgFitnessPerGen;
            this.avgSizePerGen       = avgSizePerGen;
            this.hitsPerGen          = hitsPerGen;
            this.varietyPerGen       = varietyPerGen;
            this.avgStructuralSimilarityPerGen = avgStructuralSimilarityPerGen;
            this.invalidPerGen       = invalidPerGen;
        }
    }

    /**
     * Constructs an SBGE engine.
     *
     * @param populationSize       population size
     * @param generations          number of generations
     * @param tournamentSize       tournament size for parent selection
     * @param crossoverRate        fraction of non-elite children produced by crossover
     * @param mutationRate         fraction of non-elite children produced by mutation
     *                             ({@code crossoverRate + mutationRate} must equal 1.0)
     * @param codonMutationProb    per-codon probability inside the mutation operator
     * @param genomeLength         length of the integer codon genome
     * @param codonMax             upper bound (exclusive) on random codon values
     * @param similarityThreshold  prefix-match ratio threshold σ
     * @param sharingAlpha         sharing coefficient α
     * @param structuralSampleSize sampled peers per individual for niche count
     * @param seed                 RNG seed for this run
     * @param trainInputs          training input matrix
     * @param trainTargets         training target vector
     * @param testInputs           test input matrix
     * @param testTargets          test target vector
     */
    public GrammaticalEvolution(int populationSize, int generations,
                            int tournamentSize, double crossoverRate,
                            double mutationRate, double codonMutationProb,
                            int genomeLength, int codonMax,
                            double similarityThreshold, double sharingAlpha,
                            int structuralSampleSize, long seed,
                            double[][] trainInputs, double[] trainTargets,
                            double[][] testInputs,  double[] testTargets) {
        this.populationSize       = populationSize;
        this.generations          = generations;
        this.tournamentSize       = tournamentSize;
        this.crossoverRate        = crossoverRate;
        this.mutationRate         = mutationRate;
        this.codonMutationProb    = codonMutationProb;
        this.genomeLength         = genomeLength;
        this.codonMax             = codonMax;
        this.similarityThreshold  = similarityThreshold;
        this.sharingAlpha         = sharingAlpha;
        this.structuralSampleSize = structuralSampleSize;
        this.seed                 = seed;
        this.rng                  = new Random(seed);
        this.trainInputs          = trainInputs;
        this.trainTargets         = trainTargets;
        this.testInputs           = testInputs;
        this.testTargets          = testTargets;
    }

    // ─── Main evolutionary loop ───────────────────────────────────────────────

    /**
     * Runs the evolutionary loop for {@link #generations} generations and
     * returns aggregated statistics.
     */
    public RunResult run() {
        long startTime = System.currentTimeMillis();

        Individual[] population = initialisePopulation();
        double[]     rawFitness = evaluate(population);

        // Track best-ever (raw fitness, A2 feedback: elitism uses RAW fitness).
        int bestIdx = argMin(rawFitness);
        Individual bestEver       = population[bestIdx];
        double     bestEverRawFit = rawFitness[bestIdx];

        // Per-generation telemetry.
        double[] avgFitnessPerGen = new double[generations];
        double[] avgSizePerGen    = new double[generations];
        int[]    hitsPerGen       = new int[generations];
        int[]    varietyPerGen    = new int[generations];
        double[] avgStructSimGen  = new double[generations];
        int[]    invalidPerGen    = new int[generations];

        for (int gen = 0; gen < generations; gen++) {

            double[] adjustedFitness = applyFitnessSharing(population, rawFitness);

            // Telemetry on the current (parent) generation.
            populateTelemetry(gen, population, rawFitness,
                              avgFitnessPerGen, avgSizePerGen,
                              hitsPerGen, varietyPerGen,
                              avgStructSimGen, invalidPerGen);

            // Build next generation using EXACT operator rates (not per-child
            // Bernoulli draws). Given a population of size N:
            //   ‣ 1 slot      → elitism (raw fitness)
            //   ‣ Nc          → crossover  (Nc = round(crossoverRate × (N − 1)))
            //   ‣ N − 1 − Nc  → mutation
            // crossoverRate and mutationRate must sum to 1.0.
            Individual[] next = new Individual[populationSize];

            int eliteIdx = argMin(rawFitness);
            next[0] = population[eliteIdx];

            int nonElite       = populationSize - 1;
            int crossoverSlots = (int) Math.round(crossoverRate * nonElite);
            int mutationSlots  = nonElite - crossoverSlots;

            int idx = 1;
            for (int k = 0; k < crossoverSlots; k++) {
                Individual p1 = tournament(population, adjustedFitness);
                Individual p2 = tournament(population, adjustedFitness);
                int[] child = onePointCrossover(p1.genome, p2.genome);
                next[idx++] = new Individual(child);
            }
            for (int k = 0; k < mutationSlots; k++) {
                Individual p = tournament(population, adjustedFitness);
                int[] child = mutate(p.genome);
                next[idx++] = new Individual(child);
            }

            population = next;
            rawFitness = evaluate(population);

            int curBest = argMin(rawFitness);
            if (rawFitness[curBest] < bestEverRawFit) {
                bestEverRawFit = rawFitness[curBest];
                bestEver       = population[curBest];
            }
        }

        // Final test-set evaluation of best-ever individual.
        double testFit = mse(bestEver.phenotype, testInputs, testTargets);
        long runtimeMs = System.currentTimeMillis() - startTime;

        return new RunResult(bestEverRawFit, testFit,
                             bestEver.phenotype, bestEver.genome,
                             runtimeMs,
                             avgFitnessPerGen, avgSizePerGen,
                             hitsPerGen, varietyPerGen,
                             avgStructSimGen, invalidPerGen);
    }

    // ─── Initial population ───────────────────────────────────────────────────

    /**
     * Random codon initialisation. Each individual is a fresh int[] of
     * {@link #genomeLength} uniformly sampled codons in {@code [0, codonMax)}.
     *
     * <p>This is the canonical GE initialisation (O'Neill &amp; Ryan, 2001).
     * Unlike GP's Grow / Full / Ramped Half-and-Half, GE does not directly
     * control derivation-tree shape at initialisation: it samples in the
     * <em>genotype</em> space and lets the grammar mapping induce phenotype
     * structure. Pure random codon initialisation is preferred over
     * "sensible initialisation" (Ryan &amp; Azad, 2003) here because:
     * (i) it preserves the full structural diversity of the codon space, which
     *     is essential for the structural-sharing penalty to be meaningful;
     * (ii) the depth-bounded grammar fallback already guarantees terminating
     *      derivations, removing the main motivation for sensible init.
     */
    private Individual[] initialisePopulation() {
        Individual[] pop = new Individual[populationSize];
        for (int i = 0; i < populationSize; i++) {
            pop[i] = new Individual(randomGenome());
        }
        return pop;
    }

    private int[] randomGenome() {
        int[] g = new int[genomeLength];
        for (int i = 0; i < genomeLength; i++) g[i] = rng.nextInt(codonMax);
        return g;
    }

    // ─── Fitness evaluation ───────────────────────────────────────────────────

    /**
     * Raw fitness = MSE × (1 + PARSIMONY_COEFF × phenotypeSize).
     * Invalid individuals (mapping failed) receive {@code Double.MAX_VALUE}.
     */
    private double[] evaluate(Individual[] pop) {
        double[] f = new double[pop.length];
        for (int i = 0; i < pop.length; i++) {
            Individual ind = pop[i];
            if (!ind.valid) { f[i] = Double.MAX_VALUE; continue; }
            double m = mse(ind.phenotype, trainInputs, trainTargets);
            if (Double.isNaN(m) || Double.isInfinite(m)) {
                f[i] = Double.MAX_VALUE;
            } else {
                f[i] = m * (1.0 + PARSIMONY_COEFF * ind.phenotype.size());
            }
        }
        return f;
    }

    private static double mse(Expression e, double[][] inputs, double[] targets) {
        double sum = 0.0;
        int n = inputs.length;
        for (int i = 0; i < n; i++) {
            double y = e.evaluate(inputs[i]);
            if (Double.isNaN(y) || Double.isInfinite(y)) return Double.MAX_VALUE;
            double d = y - targets[i];
            sum += d * d;
        }
        return sum / n;
    }

    // ─── Structural fitness sharing ───────────────────────────────────────────

    /**
     * Computes adjusted fitness from raw fitness via prefix-match based
     * structural sharing over derivation signatures. For each individual we
     * sample {@link #structuralSampleSize} random peers (without replacement
     * conceptually; with-replacement here for speed) and count peers whose
     * normalised prefix-match ratio is at least {@link #similarityThreshold}.
     */
    private double[] applyFitnessSharing(Individual[] pop, double[] raw) {
        double[] adj = new double[pop.length];
        for (int i = 0; i < pop.length; i++) {
            if (raw[i] == Double.MAX_VALUE) { adj[i] = Double.MAX_VALUE; continue; }
            int[] sigI = pop[i].signature;
            int   nicheCount = 0;
            int   trials = Math.min(structuralSampleSize, pop.length - 1);
            for (int k = 0; k < trials; k++) {
                int j = rng.nextInt(pop.length);
                if (j == i || raw[j] == Double.MAX_VALUE) continue;
                int[] sigJ = pop[j].signature;
                int   maxLen = Math.max(sigI.length, sigJ.length);
                if (maxLen == 0) continue;
                double sim = Grammar.prefixMatchCount(sigI, sigJ) / (double) maxLen;
                if (sim >= similarityThreshold) nicheCount++;
            }
            adj[i] = raw[i] * (1.0 + sharingAlpha * nicheCount);
        }
        return adj;
    }

    // ─── Selection ────────────────────────────────────────────────────────────

    /**
     * Tournament selection over adjusted fitness. Returns the individual with
     * the lowest adjusted fitness among {@link #tournamentSize} random draws.
     */
    private Individual tournament(Individual[] pop, double[] fit) {
        int bestIdx = rng.nextInt(pop.length);
        double bestFit = fit[bestIdx];
        for (int t = 1; t < tournamentSize; t++) {
            int idx = rng.nextInt(pop.length);
            if (fit[idx] < bestFit) {
                bestFit = fit[idx];
                bestIdx = idx;
            }
        }
        return pop[bestIdx];
    }

    // ─── Crossover (one-point on codons) ──────────────────────────────────────

    /**
     * Standard GE one-point crossover. A single cut point is selected
     * uniformly along the codon array; the child inherits codons before the
     * cut from {@code p1} and codons after the cut from {@code p2}.
     */
    private int[] onePointCrossover(int[] p1, int[] p2) {
        int len = p1.length;
        int cut = 1 + rng.nextInt(len - 1);
        int[] c = new int[len];
        System.arraycopy(p1, 0, c, 0, cut);
        System.arraycopy(p2, cut, c, cut, len - cut);
        return c;
    }

    // ─── Mutation (per-codon integer mutation) ────────────────────────────────

    /**
     * Integer mutation: each codon is independently replaced with probability
     * {@link #codonMutationProb} by a fresh uniform draw in {@code [0, codonMax)}.
     * To guarantee at least one codon change per application, if no codon was
     * mutated by the per-codon roll a single uniformly chosen codon is forced.
     */
    private int[] mutate(int[] genome) {
        int[] c = genome.clone();
        boolean any = false;
        for (int i = 0; i < c.length; i++) {
            if (rng.nextDouble() < codonMutationProb) {
                c[i] = rng.nextInt(codonMax);
                any = true;
            }
        }
        if (!any) {
            int j = rng.nextInt(c.length);
            c[j] = rng.nextInt(codonMax);
        }
        return c;
    }

    // ─── Telemetry ────────────────────────────────────────────────────────────

    private void populateTelemetry(int gen, Individual[] pop, double[] raw,
                                   double[] avgFitness, double[] avgSize,
                                   int[] hits, int[] variety,
                                   double[] avgStructSim, int[] invalidCount) {
        double sumFit = 0.0, sumSize = 0.0;
        int    invalid = 0, validCount = 0;
        java.util.HashSet<String> uniques = new java.util.HashSet<>();
        for (int i = 0; i < pop.length; i++) {
            if (raw[i] == Double.MAX_VALUE) { invalid++; continue; }
            validCount++;
            sumFit  += raw[i];
            sumSize += pop[i].phenotype.size();
            uniques.add(pop[i].phenotype.toString());
        }
        avgFitness[gen] = validCount == 0 ? Double.NaN : sumFit  / validCount;
        avgSize[gen]    = validCount == 0 ? Double.NaN : sumSize / validCount;
        variety[gen]    = uniques.size();
        invalidCount[gen] = invalid;

        // Hits (best individual): training cases with |pred - actual| < 0.01.
        int bestIdx = argMin(raw);
        Expression bestE = pop[bestIdx].phenotype;
        int h = 0;
        for (int i = 0; i < trainInputs.length; i++) {
            double y = bestE.evaluate(trainInputs[i]);
            if (!Double.isNaN(y) && !Double.isInfinite(y) &&
                Math.abs(y - trainTargets[i]) < 0.01) h++;
        }
        hits[gen] = h;

        // Average pairwise structural similarity over a random sample.
        int samples = Math.min(50, pop.length);
        double simSum = 0.0;
        int    simCount = 0;
        for (int k = 0; k < samples; k++) {
            int a = rng.nextInt(pop.length);
            int b = rng.nextInt(pop.length);
            if (a == b) continue;
            int[] sa = pop[a].signature, sb = pop[b].signature;
            int   maxLen = Math.max(sa.length, sb.length);
            if (maxLen == 0) continue;
            simSum += Grammar.prefixMatchCount(sa, sb) / (double) maxLen;
            simCount++;
        }
        avgStructSim[gen] = simCount == 0 ? 0.0 : simSum / simCount;
    }

    // ─── Utility ──────────────────────────────────────────────────────────────

    private static int argMin(double[] arr) {
        int idx = 0;
        double best = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] < best) { best = arr[i]; idx = i; }
        }
        return idx;
    }
}
