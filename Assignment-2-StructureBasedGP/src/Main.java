import java.io.IOException;
import java.util.Locale;

/**
 * Entry point for COS 710 Assignment 2: Structure-Based Genetic Programming
 * for Symbolic Regression.
 *
 * <h2>Problem</h2>
 * Evolve an expression tree that predicts the residential electricity load at
 * time t using the seven most recent observed load values (Load(t−1) … Load(t−7)),
 * incorporating structural information into the GP fitness function via fitness
 * sharing based on prefix node matching.
 *
 * <h2>Usage</h2>
 * Compile and run from the <em>Assignment 2</em> directory:
 * <pre>
 *   javac -d out src/Node.java src/Tree.java src/DatasetLoader.java \
 *              src/GeneticProgramming.java src/Main.java
 *   java -cp out Main
 * </pre>
 * Or in one step:
 * <pre>
 *   javac -d out src/*.java && java -cp out Main
 * </pre>
 *
 * <h2>Output</h2>
 * <ul>
 *   <li>Per-run: run number, best raw training MSE, test MSE, elapsed time.</li>
 *   <li>Summary: mean, standard deviation, and best MSE over all runs.</li>
 *   <li>Per-generation statistics for the best run (avg MSE, avg tree size,
 *       hits, variety, avg structural similarity).</li>
 *   <li>Comparison table against Assignment 1 canonical GP results.</li>
 * </ul>
 *
 * <p>COS 710 Assignment 2 – Structure-Based Genetic Programming for Regression.
 */
public class Main {

    // ─── Dataset ──────────────────────────────────────────────────────────────

    /**
     * Path to the CSV dataset file, relative to the working directory.
     * Run the program from the <em>Assignment 2</em> project root.
     */
    private static final String DATASET_PATH =
            "documentation/Residential_Energy_Dataset_UK- 2014-2020.csv";

    // ─── Base GP Parameters (identical to Assignment 1 for fair comparison) ──

    /** Number of individuals maintained in the population each generation. */
    private static final int    POPULATION_SIZE = 1000;

    /** Maximum allowed depth of any evolved expression tree. */
    private static final int    MAX_DEPTH       = 6;

    /** Number of generations to run the evolutionary process. */
    private static final int    GENERATIONS     = 100;

    /** Number of competitors drawn in each tournament for parent selection. */
    private static final int    TOURNAMENT_SIZE = 2;

    /** Fraction of non-elite slots produced by subtree crossover. */
    private static final double CROSSOVER_RATE  = 0.9;

    /** Fraction of non-elite slots produced by subtree mutation. */
    private static final double MUTATION_RATE   = 0.1;

    /** Number of previous load values used as input features. */
    private static final int    WINDOW_SIZE     = 7;

    /** Fraction of windowed examples used for training. */
    private static final double TRAIN_RATIO     = 0.8;

    // ─── SBGP Parameters ──────────────────────────────────────────────────────

    /**
     * Structural similarity threshold σ.
     * Two individuals are considered structurally similar (and counted in each
     * other's niche) if their normalised prefix match score ≥ SIMILARITY_THRESHOLD.
     * A value of 0.6 means trees sharing at least 60% of their top-down
     * node structure are treated as neighbours in the structural niche.
     */
    private static final double SIMILARITY_THRESHOLD   = 0.6;

    /**
     * Sharing coefficient α.
     * Scales the structural penalty: adjustedFitness = rawFitness × (1 + α × nicheCount).
     * Set small (0.01) so the structural component complements rather than
     * dominates the behavioural fitness signal.
     */
    private static final double SHARING_ALPHA          = 0.01;

    /**
     * Number of random peers sampled per individual when computing niche counts.
     * Avoids the O(n²) cost of full pairwise comparison while still providing
     * a representative estimate of structural crowding.
     */
    private static final int    STRUCTURAL_SAMPLE_SIZE = 50;

    // ─── Experimental Seeds ───────────────────────────────────────────────────

    /** One unique random seed per independent run (same as Assignment 1). */
    private static final long[] SEEDS = {101, 202, 303, 404, 505, 606, 707, 808, 909, 1010};

    // ─── Assignment 1 Reference Results (for comparison) ────────────────────

    /** Assignment 1 average train MSE across 10 runs. */
    private static final double A1_AVG_MSE   = 0.00014812;
    /** Assignment 1 standard deviation of train MSE across 10 runs. */
    private static final double A1_STD_MSE   = 0.00000658;
    /** Assignment 1 best train MSE (Run 8, seed 808). */
    private static final double A1_BEST_MSE  = 0.00014257;
    /** Assignment 1 average test MSE across 10 runs. */
    private static final double A1_AVG_TEST  = 0.00017889;

    // ─── Main Method ──────────────────────────────────────────────────────────

    /**
     * Loads the dataset, configures the SBGP engine, executes all independent
     * runs, and prints per-run, summary, per-generation, and comparison statistics.
     *
     * @param args command-line arguments (not used)
     */
    public static void main(String[] args) {
        Locale.setDefault(Locale.US);

        printBanner();

        // ── Load and preprocess the dataset ───────────────────────────────────
        double[] series;
        try {
            series = DatasetLoader.loadElectricityLoad(DATASET_PATH);
        } catch (IOException e) {
            System.err.println("ERROR: Could not read dataset: " + DATASET_PATH);
            System.err.println("Run the program from the 'Assignment 2' directory.");
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

        // ── Print GP configuration ─────────────────────────────────────────────
        System.out.println("SBGP Configuration");
        System.out.println("────────────────────────────────────────────────────────");
        System.out.printf("  Population size       : %d%n",    POPULATION_SIZE);
        System.out.printf("  Max tree depth        : %d%n",    MAX_DEPTH);
        System.out.printf("  Generations           : %d%n",    GENERATIONS);
        System.out.printf("  Tournament size       : %d%n",    TOURNAMENT_SIZE);
        System.out.printf("  Crossover rate        : %.2f%n",  CROSSOVER_RATE);
        System.out.printf("  Mutation rate         : %.2f%n",  MUTATION_RATE);
        System.out.printf("  Window size           : %d%n",    WINDOW_SIZE);
        System.out.printf("  Similarity threshold  : %.2f%n",  SIMILARITY_THRESHOLD);
        System.out.printf("  Sharing alpha (α)     : %.4f%n",  SHARING_ALPHA);
        System.out.printf("  Structural sample size: %d%n",    STRUCTURAL_SAMPLE_SIZE);
        System.out.printf("  Independent runs      : %d%n%n",  SEEDS.length);

        // ── Column headers ─────────────────────────────────────────────────────
        System.out.println("────────────────────────────────────────────────────────");
        System.out.printf("%-5s  %-20s  %-20s  %-10s%n",
                "Run", "Train MSE", "Test MSE", "Time (s)");
        System.out.println("────────────────────────────────────────────────────────");

        // ── Run SBGP for each seed ─────────────────────────────────────────────
        double[] trainFitnesses = new double[SEEDS.length];
        double[] testFitnesses  = new double[SEEDS.length];
        double   overallBestFitness = Double.MAX_VALUE;
        Tree     overallBestTree    = null;
        int      overallBestRun     = -1;
        GeneticProgramming.RunResult bestRunResult = null;

        for (int i = 0; i < SEEDS.length; i++) {
            GeneticProgramming gp = new GeneticProgramming(
                    POPULATION_SIZE, MAX_DEPTH, GENERATIONS, TOURNAMENT_SIZE,
                    CROSSOVER_RATE, MUTATION_RATE,
                    SIMILARITY_THRESHOLD, SHARING_ALPHA, STRUCTURAL_SAMPLE_SIZE,
                    SEEDS[i],
                    data.trainInputs, data.trainTargets,
                    data.testInputs,  data.testTargets);

            GeneticProgramming.RunResult result = gp.run();
            trainFitnesses[i] = result.trainFitness;
            testFitnesses[i]  = result.testFitness;

            System.out.printf("%-5d  %-20s  %-20s  %.2f%n",
                    (i + 1),
                    formatFitness(result.trainFitness),
                    formatFitness(result.testFitness),
                    result.runtimeMs / 1000.0);

            if (result.trainFitness < overallBestFitness) {
                overallBestFitness = result.trainFitness;
                overallBestTree    = result.bestTree;
                overallBestRun     = i + 1;
                bestRunResult      = result;
            }
        }

        // ── Summary statistics ─────────────────────────────────────────────────
        double mean     = computeMean(trainFitnesses);
        double stdDev   = computeStdDev(trainFitnesses, mean);
        double meanTest = computeMean(testFitnesses);

        System.out.println("════════════════════════════════════════════════════════");
        System.out.println("Summary (Train MSE across all runs)");
        System.out.println("════════════════════════════════════════════════════════");
        System.out.printf("  Average train MSE  : %s%n", formatFitness(mean));
        System.out.printf("  Standard deviation : %s%n", formatFitness(stdDev));
        System.out.printf("  Best train MSE     : %s  (Run %d)%n",
                formatFitness(overallBestFitness), overallBestRun);
        System.out.printf("  Average test MSE   : %s%n", formatFitness(meanTest));
        System.out.println();
        System.out.println("Best evolved expression:");
        System.out.println("  " + (overallBestTree != null ? overallBestTree : "N/A"));
        System.out.println("════════════════════════════════════════════════════════");

        // ── Hits and success predicate ─────────────────────────────────────────
        final double SUCCESS_THRESHOLD = 0.00020;
        int successfulRuns = 0;
        for (double f : trainFitnesses) if (f < SUCCESS_THRESHOLD) successfulRuns++;

        System.out.println();
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println("Hits & Success Predicate");
        System.out.println("════════════════════════════════════════════════════════");
        System.out.printf("  Hits threshold     : |predicted - actual| < 0.01%n");
        if (bestRunResult != null) {
            int finalHits = bestRunResult.hitsPerGen[bestRunResult.hitsPerGen.length - 1];
            System.out.printf("  Hits (best run)    : %d / %d training cases (%.1f%%)%n",
                    finalHits, data.trainInputs.length,
                    100.0 * finalHits / data.trainInputs.length);
        }
        System.out.printf("  Success threshold  : Train MSE < %.5f%n", SUCCESS_THRESHOLD);
        System.out.printf("  Successful runs    : %d / %d%n", successfulRuns, SEEDS.length);
        System.out.println("════════════════════════════════════════════════════════");

        // ── Per-generation statistics for the best run ─────────────────────────
        if (bestRunResult != null) {
            System.out.println();
            System.out.println("════════════════════════════════════════════════════════");
            System.out.printf("Per-Generation Statistics (Best Run %d, Seed %d)%n",
                    overallBestRun, SEEDS[overallBestRun - 1]);
            System.out.println("════════════════════════════════════════════════════════");
            System.out.printf("%-6s  %-14s  %-12s  %-8s  %-8s  %-12s%n",
                    "Gen", "Avg Train MSE", "Avg TreeSize", "Hits", "Variety", "Avg Struct Sim");
            System.out.println("────────────────────────────────────────────────────────");
            int[] printAt = buildPrintGenerations(GENERATIONS);
            for (int g : printAt) {
                System.out.printf("%-6d  %-14s  %-12.2f  %-8d  %-8d  %-12.4f%n",
                        g + 1,
                        formatFitness(bestRunResult.avgFitnessPerGen[g]),
                        bestRunResult.avgSizePerGen[g],
                        bestRunResult.hitsPerGen[g],
                        bestRunResult.varietyPerGen[g],
                        bestRunResult.avgStructuralSimilarityPerGen[g]);
            }
            System.out.println("════════════════════════════════════════════════════════");
        }

        // ── Comparison with Assignment 1 ───────────────────────────────────────
        System.out.println();
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println("Comparison: Assignment 1 (Canonical GP) vs Assignment 2 (SBGP)");
        System.out.println("════════════════════════════════════════════════════════");
        System.out.printf("%-14s  %-14s  %-14s  %-14s  %-14s%n",
                "Assignment", "Avg Train MSE", "StdDev", "Best Train MSE", "Avg Test MSE");
        System.out.println("──────────────────────────────────────────────────────────────────────");
        System.out.printf("%-14s  %-14s  %-14s  %-14s  %-14s%n",
                "A1 (Canonical)",
                formatFitness(A1_AVG_MSE),
                formatFitness(A1_STD_MSE),
                formatFitness(A1_BEST_MSE),
                formatFitness(A1_AVG_TEST));
        System.out.printf("%-14s  %-14s  %-14s  %-14s  %-14s%n",
                "A2 (SBGP)",
                formatFitness(mean),
                formatFitness(stdDev),
                formatFitness(overallBestFitness),
                formatFitness(meanTest));
        System.out.println("════════════════════════════════════════════════════════");
    }

    // ─── Utility Helpers ──────────────────────────────────────────────────────

    /** Returns indices of generations to print: every 10th plus the final. */
    private static int[] buildPrintGenerations(int totalGenerations) {
        java.util.List<Integer> list = new java.util.ArrayList<>();
        for (int g = 9; g < totalGenerations; g += 10) list.add(g);
        if (list.isEmpty() || list.get(list.size() - 1) != totalGenerations - 1) {
            list.add(totalGenerations - 1);
        }
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    private static String formatFitness(double fitness) {
        return fitness == Double.MAX_VALUE ? "N/A" : String.format(Locale.US, "%.8f", fitness);
    }

    private static double computeMean(double[] values) {
        double sum = 0.0;
        int    count = 0;
        for (double v : values) {
            if (v != Double.MAX_VALUE) { sum += v; count++; }
        }
        return count == 0 ? Double.NaN : sum / count;
    }

    private static double computeStdDev(double[] values, double mean) {
        double variance = 0.0;
        int    count    = 0;
        for (double v : values) {
            if (v != Double.MAX_VALUE) {
                double diff = v - mean;
                variance += diff * diff;
                count++;
            }
        }
        return count == 0 ? Double.NaN : Math.sqrt(variance / count);
    }

    private static void printBanner() {
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println("  COS 710 Artificial Intelligence – Assignment 2");
        System.out.println("  Structure-Based Genetic Programming for Regression");
        System.out.println("  Dataset: Residential Energy Usage UK (2014–2020)");
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println();
    }
}
