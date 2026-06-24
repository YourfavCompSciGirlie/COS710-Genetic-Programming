import java.io.IOException;
import java.util.Locale;

/**
 * Entry point for COS 710 Assignment 3: Structure-Based Grammatical Evolution
 * for Symbolic Regression.
 *
 * <h2>Problem</h2>
 * Evolve an arithmetic expression that predicts residential electricity load
 * at time t using the seven most recent observed load values
 * (Load(t−1) … Load(t−7)). Structure is incorporated into canonical GE via a
 * structural fitness-sharing penalty applied to the derivation signature of
 * each codon genome (see {@link GrammaticalEvolution}).
 *
 * <h2>Usage</h2>
 * From the {@code Assignment 3} project root:
 * <pre>
 *   javac -d out src/*.java && java -cp out Main
 * </pre>
 *
 * <h2>Output</h2>
 * Banner, configuration, per-run table (train MSE / test MSE / time),
 * summary statistics across 10 independent runs, per-generation statistics
 * of the best run, hits / success-predicate report, and a final comparison
 * table against the Assignment 1 (canonical GP) and Assignment 2 (SBGP)
 * results.
 */
public class Main {

    // ─── Dataset ──────────────────────────────────────────────────────────────

    private static final String DATASET_PATH =
            "documentation/Residential_Energy_Dataset_UK- 2014-2020.csv";

    // ─── GE parameters ────────────────────────────────────────────────────────

    /** Population size (matches Assignments 1 &amp; 2 for fair comparison). */
    private static final int    POPULATION_SIZE = 1000;

    /** Number of generations. */
    private static final int    GENERATIONS     = 100;

    /** Tournament size for parent selection. */
    private static final int    TOURNAMENT_SIZE = 2;

    /** Fraction of non-elite children produced by one-point crossover. */
    private static final double CROSSOVER_RATE  = 0.70;

    /** Fraction of non-elite children produced by integer mutation.
     *  Crossover + mutation must equal 1.0 (exact-rate operator split). */
    private static final double MUTATION_RATE   = 0.30;

    /** Per-codon probability inside the integer mutation operator. */
    private static final double CODON_MUTATION_PROB = 0.10;

    /** Length of the integer codon genome (initial). */
    private static final int    GENOME_LENGTH   = 64;

    /** Upper bound (exclusive) on uniformly sampled codon values. */
    private static final int    CODON_MAX       = 256;

    /** Number of previous load values used as input features. */
    private static final int    WINDOW_SIZE     = 7;

    /** Fraction of windowed examples assigned to training. */
    private static final double TRAIN_RATIO     = 0.8;

    // ─── Structural sharing parameters ────────────────────────────────────────

    /**
     * σ – structural similarity threshold. Derivation signatures whose
     * normalised prefix-match ratio is ≥ σ are deemed similar (in the same
     * niche). 0.6 mirrors the value used for prefix-match in A2.
     */
    private static final double SIMILARITY_THRESHOLD   = 0.6;

    /**
     * α – sharing coefficient. {@code adj = raw × (1 + α × nicheCount)}.
     * Small (0.01) so the structural penalty complements rather than
     * dominates the behavioural fitness signal.
     */
    private static final double SHARING_ALPHA          = 0.01;

    /** Sample size for structural niche estimation per individual. */
    private static final int    STRUCTURAL_SAMPLE_SIZE = 50;

    // ─── Seeds (identical to A1 and A2) ───────────────────────────────────────

    private static final long[] SEEDS = {101, 202, 303, 404, 505, 606, 707, 808, 909, 1010};

    // ─── A1 / A2 reference results (for comparison) ───────────────────────────

    private static final double A1_AVG_MSE   = 0.00014812;
    private static final double A1_STD_MSE   = 0.00000658;
    private static final double A1_BEST_MSE  = 0.00014257;
    private static final double A1_AVG_TEST  = 0.00017889;

    private static final double A2_AVG_MSE   = 0.00014611;
    private static final double A2_STD_MSE   = 0.00000342;
    private static final double A2_BEST_MSE  = 0.00014120;
    private static final double A2_AVG_TEST  = 0.00017394;

    // ─── Main ─────────────────────────────────────────────────────────────────

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);

        printBanner();

        // Load the time series.
        double[] series;
        try {
            series = DatasetLoader.loadElectricityLoad(DATASET_PATH);
        } catch (IOException e) {
            System.err.println("ERROR: Could not read dataset: " + DATASET_PATH);
            System.err.println("Run the program from the 'Assignment 3' directory.");
            System.err.println("Details: " + e.getMessage());
            System.exit(1);
            return;
        }

        String capLabel = DatasetLoader.DATA_SAMPLE_SIZE < 0
                ? "full dataset"
                : DatasetLoader.DATA_SAMPLE_SIZE + " rows (sample cap)";
        System.out.printf("Loaded %d electricity load values (%s)%n%n",
                series.length, capLabel);

        DatasetLoader.TrainingData data =
                DatasetLoader.createAndSplitData(series, WINDOW_SIZE, TRAIN_RATIO);

        System.out.printf("Training examples : %d%n", data.trainInputs.length);
        System.out.printf("Test examples     : %d%n%n", data.testInputs.length);

        // Print configuration.
        System.out.println("SBGE Configuration");
        System.out.println("────────────────────────────────────────────────────────");
        System.out.printf("  Population size       : %d%n",   POPULATION_SIZE);
        System.out.printf("  Generations           : %d%n",   GENERATIONS);
        System.out.printf("  Tournament size       : %d%n",   TOURNAMENT_SIZE);
        System.out.printf("  Crossover rate        : %.2f%n", CROSSOVER_RATE);
        System.out.printf("  Mutation rate         : %.2f%n", MUTATION_RATE);
        System.out.printf("  Codon mutation prob   : %.2f%n", CODON_MUTATION_PROB);
        System.out.printf("  Genome length         : %d%n",   GENOME_LENGTH);
        System.out.printf("  Codon range           : [0, %d)%n", CODON_MAX);
        System.out.printf("  Max derivation depth  : %d%n",   Grammar.MAX_DEPTH);
        System.out.printf("  Max wraps             : %d%n",   Grammar.MAX_WRAPS);
        System.out.printf("  Window size           : %d%n",   WINDOW_SIZE);
        System.out.printf("  Similarity threshold  : %.2f%n", SIMILARITY_THRESHOLD);
        System.out.printf("  Sharing alpha (α)     : %.4f%n", SHARING_ALPHA);
        System.out.printf("  Structural sample sz  : %d%n",   STRUCTURAL_SAMPLE_SIZE);
        System.out.printf("  Independent runs      : %d%n%n", SEEDS.length);

        // Header.
        System.out.println("────────────────────────────────────────────────────────");
        System.out.printf("%-5s  %-20s  %-20s  %-10s%n",
                "Run", "Train MSE", "Test MSE", "Time (s)");
        System.out.println("────────────────────────────────────────────────────────");

        double[] trainFitnesses = new double[SEEDS.length];
        double[] testFitnesses  = new double[SEEDS.length];
        long[]   runtimesMs     = new long[SEEDS.length];
        double   bestSoFar      = Double.MAX_VALUE;
        Expression bestExpr     = null;
        int        bestRunNum   = -1;
        GrammaticalEvolution.RunResult bestRunResult = null;

        for (int i = 0; i < SEEDS.length; i++) {
            GrammaticalEvolution ge = new GrammaticalEvolution(
                    POPULATION_SIZE, GENERATIONS, TOURNAMENT_SIZE,
                    CROSSOVER_RATE, MUTATION_RATE, CODON_MUTATION_PROB,
                    GENOME_LENGTH, CODON_MAX,
                    SIMILARITY_THRESHOLD, SHARING_ALPHA, STRUCTURAL_SAMPLE_SIZE,
                    SEEDS[i],
                    data.trainInputs, data.trainTargets,
                    data.testInputs,  data.testTargets);

            GrammaticalEvolution.RunResult r = ge.run();
            trainFitnesses[i] = r.trainFitness;
            testFitnesses[i]  = r.testFitness;
            runtimesMs[i]     = r.runtimeMs;

            System.out.printf("%-5d  %-20s  %-20s  %.2f%n",
                    (i + 1),
                    formatFitness(r.trainFitness),
                    formatFitness(r.testFitness),
                    r.runtimeMs / 1000.0);

            if (r.trainFitness < bestSoFar) {
                bestSoFar     = r.trainFitness;
                bestExpr      = r.bestExpression;
                bestRunNum    = i + 1;
                bestRunResult = r;
            }
        }

        // Summary statistics.
        double mean     = computeMean(trainFitnesses);
        double stdDev   = computeStdDev(trainFitnesses, mean);
        double meanTest = computeMean(testFitnesses);
        double avgRuntimeSec = 0.0;
        for (long t : runtimesMs) avgRuntimeSec += t / 1000.0;
        avgRuntimeSec /= SEEDS.length;

        System.out.println("════════════════════════════════════════════════════════");
        System.out.println("Summary (Train MSE across all runs)");
        System.out.println("════════════════════════════════════════════════════════");
        System.out.printf("  Average train MSE  : %s%n", formatFitness(mean));
        System.out.printf("  Standard deviation : %s%n", formatFitness(stdDev));
        System.out.printf("  Best train MSE     : %s  (Run %d)%n",
                formatFitness(bestSoFar), bestRunNum);
        System.out.printf("  Average test MSE   : %s%n", formatFitness(meanTest));
        System.out.printf("  Average runtime    : %.2f s%n", avgRuntimeSec);
        System.out.println();
        System.out.println("Best evolved expression:");
        System.out.println("  " + (bestExpr != null ? bestExpr : "N/A"));
        System.out.println("════════════════════════════════════════════════════════");

        // Hits + success predicate.
        final double SUCCESS_THRESHOLD = 0.00020;
        int successful = 0;
        for (double f : trainFitnesses) if (f < SUCCESS_THRESHOLD) successful++;

        System.out.println();
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println("Hits & Success Predicate");
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println("  Hits threshold     : |predicted - actual| < 0.01");
        if (bestRunResult != null) {
            int finalHits = bestRunResult.hitsPerGen[bestRunResult.hitsPerGen.length - 1];
            System.out.printf("  Hits (best run)    : %d / %d training cases (%.1f%%)%n",
                    finalHits, data.trainInputs.length,
                    100.0 * finalHits / data.trainInputs.length);
        }
        System.out.printf("  Success threshold  : Train MSE < %.5f%n", SUCCESS_THRESHOLD);
        System.out.printf("  Successful runs    : %d / %d%n", successful, SEEDS.length);
        System.out.println("════════════════════════════════════════════════════════");

        // Per-generation statistics for the best run.
        if (bestRunResult != null) {
            System.out.println();
            System.out.println("════════════════════════════════════════════════════════");
            System.out.printf("Per-Generation Statistics (Best Run %d, Seed %d)%n",
                    bestRunNum, SEEDS[bestRunNum - 1]);
            System.out.println("════════════════════════════════════════════════════════");
            System.out.printf("%-6s  %-14s  %-12s  %-8s  %-8s  %-12s  %-8s%n",
                    "Gen", "Avg Train MSE", "Avg Size", "Hits", "Variety", "Struct Sim", "Invalid");
            System.out.println("────────────────────────────────────────────────────────");
            int[] printAt = buildPrintGenerations(GENERATIONS);
            for (int g : printAt) {
                System.out.printf("%-6d  %-14s  %-12.2f  %-8d  %-8d  %-12.4f  %-8d%n",
                        g + 1,
                        formatFitness(bestRunResult.avgFitnessPerGen[g]),
                        bestRunResult.avgSizePerGen[g],
                        bestRunResult.hitsPerGen[g],
                        bestRunResult.varietyPerGen[g],
                        bestRunResult.avgStructuralSimilarityPerGen[g],
                        bestRunResult.invalidPerGen[g]);
            }
            System.out.println("════════════════════════════════════════════════════════");
        }

        // Comparison.
        System.out.println();
        System.out.println("════════════════════════════════════════════════════════════════════");
        System.out.println("Comparison: A1 (Canonical GP)  vs  A2 (SBGP)  vs  A3 (SBGE)");
        System.out.println("════════════════════════════════════════════════════════════════════");
        System.out.printf("%-16s  %-14s  %-14s  %-14s  %-14s%n",
                "Assignment", "Avg Train MSE", "StdDev", "Best Train MSE", "Avg Test MSE");
        System.out.println("────────────────────────────────────────────────────────────────────");
        System.out.printf("%-16s  %-14s  %-14s  %-14s  %-14s%n",
                "A1 (Canonical)", formatFitness(A1_AVG_MSE), formatFitness(A1_STD_MSE),
                formatFitness(A1_BEST_MSE), formatFitness(A1_AVG_TEST));
        System.out.printf("%-16s  %-14s  %-14s  %-14s  %-14s%n",
                "A2 (SBGP)", formatFitness(A2_AVG_MSE), formatFitness(A2_STD_MSE),
                formatFitness(A2_BEST_MSE), formatFitness(A2_AVG_TEST));
        System.out.printf("%-16s  %-14s  %-14s  %-14s  %-14s%n",
                "A3 (SBGE)", formatFitness(mean), formatFitness(stdDev),
                formatFitness(bestSoFar), formatFitness(meanTest));
        System.out.println("════════════════════════════════════════════════════════════════════");
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    private static void printBanner() {
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println("  COS 710 Assignment 3");
        System.out.println("  Structure-Based Grammatical Evolution for Regression");
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println();
    }

    private static int[] buildPrintGenerations(int totalGenerations) {
        java.util.List<Integer> list = new java.util.ArrayList<>();
        for (int g = 9; g < totalGenerations; g += 10) list.add(g);
        if (list.isEmpty() || list.get(list.size() - 1) != totalGenerations - 1) {
            list.add(totalGenerations - 1);
        }
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    private static String formatFitness(double f) {
        return f == Double.MAX_VALUE ? "N/A" : String.format(Locale.US, "%.8f", f);
    }

    private static double computeMean(double[] v) {
        double s = 0.0; int c = 0;
        for (double x : v) if (x != Double.MAX_VALUE) { s += x; c++; }
        return c == 0 ? Double.NaN : s / c;
    }

    private static double computeStdDev(double[] v, double mean) {
        double s = 0.0; int c = 0;
        for (double x : v) if (x != Double.MAX_VALUE) { s += (x - mean) * (x - mean); c++; }
        return c == 0 ? Double.NaN : Math.sqrt(s / c);
    }
}
