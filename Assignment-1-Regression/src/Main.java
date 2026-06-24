import java.io.IOException;
import java.util.Locale;

/**
 * Entry point for COS 710 Assignment 1: Genetic Programming for Symbolic Regression.
 *
 * <h2>Problem</h2>
 * Evolve an expression tree that predicts the residential electricity load at
 * time t using the five most recent observed load values (Load(t−1) … Load(t−5)).
 *
 * <h2>Usage</h2>
 * Compile and run from the <em>Assignment 1</em> directory:
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
 *   <li>Per-run: run number, best training MSE, test MSE, elapsed time.</li>
 *   <li>Summary: mean and standard deviation of training MSE over all runs,
 *       overall best fitness and its run number, and the best evolved expression.</li>
 * </ul>
 *
 * <p>COS 710 Assignment 1 – Genetic Programming for Symbolic Regression.
 */
public class Main {

    // ─── Dataset ──────────────────────────────────────────────────────────────

    /**
     * Path to the CSV dataset file, relative to the working directory.
     * Run the program from the <em>Assignment 1</em> project root.
     */
    private static final String DATASET_PATH =
            "documentation/Residential_Energy_Dataset_UK- 2014-2020.csv";

    // ─── GP Parameters ────────────────────────────────────────────────────────

    /** Number of individuals maintained in the population each generation. */
    private static final int    POPULATION_SIZE = 1000;

    /** Maximum allowed depth of any evolved expression tree. */
    private static final int    MAX_DEPTH       = 6;

    /** Number of generations to run the evolutionary process. */
    private static final int    GENERATIONS     = 100;

    /**
     * Number of competitors drawn in each tournament for parent selection.
     * Reduced to 2 to lower selection pressure and reduce premature convergence.
     */
    private static final int    TOURNAMENT_SIZE = 2;

    /** Probability of applying subtree crossover to produce each new individual. */
    private static final double CROSSOVER_RATE  = 0.9;

    /**
     * Probability of applying subtree mutation.
     * When crossover is also applied, this probability is checked independently
     * on the crossover offspring (i.e. mutation can be chained after crossover).
     */
    private static final double MUTATION_RATE   = 0.1;

    /**
     * Number of previous load values used as input features (window size).
     * Extended to 7 (1h 45m of 15-min-interval history) for more temporal context.
     */
    private static final int    WINDOW_SIZE     = 7;

    /** Fraction of windowed examples used for training; remainder goes to testing. */
    private static final double TRAIN_RATIO     = 0.8;

    // ─── Experimental Seeds ───────────────────────────────────────────────────

    /**
     * One unique random seed per independent run.
     * Using different seeds ensures the results explore the search space broadly
     * and allows statistical summary across runs.
     */
    private static final long[] SEEDS = {101, 202, 303, 404, 505, 606, 707, 808, 909, 1010};

    // ─── Main Method ──────────────────────────────────────────────────────────

    /**
     * Loads the dataset, configures the GP engine, executes all independent runs,
     * and prints per-run and summary statistics to standard output.
     *
     * @param args command-line arguments (not used)
     */
    public static void main(String[] args) {
        // Force US locale so that all numeric output uses dots as decimal separators,
        // regardless of the JVM system locale.
        Locale.setDefault(Locale.US);

        printBanner();

        // ── Load and preprocess the dataset ───────────────────────────────────
        double[] series;
        try {
            series = DatasetLoader.loadElectricityLoad(DATASET_PATH);
        } catch (IOException e) {
            System.err.println("ERROR: Could not read dataset: " + DATASET_PATH);
            System.err.println("Run the program from the 'Assignment 1' directory.");
            System.err.println("Details: " + e.getMessage());
            System.exit(1);
            return; // appease compiler
        }

        String capLabel = DatasetLoader.DATA_SAMPLE_SIZE < 0
                ? "full dataset"
                : DatasetLoader.DATA_SAMPLE_SIZE + " rows (sample cap)";
        System.out.printf("Loaded %d electricity load values (%s)%n%n",
                series.length, capLabel);

        // Build sliding-window examples and split temporally 80/20
        DatasetLoader.TrainingData data =
                DatasetLoader.createAndSplitData(series, WINDOW_SIZE, TRAIN_RATIO);

        System.out.printf("Training examples : %d%n", data.trainInputs.length);
        System.out.printf("Test examples     : %d%n%n",  data.testInputs.length);

        // ── Print GP configuration ─────────────────────────────────────────────
        System.out.println("GP Configuration");
        System.out.println("────────────────────────────────────────────────────────");
        System.out.printf("  Population size  : %d%n",    POPULATION_SIZE);
        System.out.printf("  Max tree depth   : %d%n",    MAX_DEPTH);
        System.out.printf("  Generations      : %d%n",    GENERATIONS);
        System.out.printf("  Tournament size  : %d%n",    TOURNAMENT_SIZE);
        System.out.printf("  Crossover rate   : %.2f%n",  CROSSOVER_RATE);
        System.out.printf("  Mutation rate    : %.2f%n",  MUTATION_RATE);
        System.out.printf("  Window size      : %d%n",    WINDOW_SIZE);
        System.out.printf("  Independent runs : %d%n%n",  SEEDS.length);

        // ── Column headers ─────────────────────────────────────────────────────
        System.out.println("────────────────────────────────────────────────────────");
        System.out.printf("%-5s  %-20s  %-20s  %-10s%n",
                "Run", "Train MSE", "Test MSE", "Time (s)");
        System.out.println("────────────────────────────────────────────────────────");

        // ── Run GP for each seed ───────────────────────────────────────────────
        double[] trainFitnesses = new double[SEEDS.length];
        double   overallBestFitness = Double.MAX_VALUE;
        Tree     overallBestTree    = null;
        int      overallBestRun     = -1;
        GeneticProgramming.RunResult bestRunResult = null;
        GeneticProgramming           bestRunGP     = null;

        for (int i = 0; i < SEEDS.length; i++) {
            GeneticProgramming gp = new GeneticProgramming(
                    POPULATION_SIZE, MAX_DEPTH, GENERATIONS, TOURNAMENT_SIZE,
                    CROSSOVER_RATE,  MUTATION_RATE,
                    SEEDS[i],
                    data.trainInputs, data.trainTargets,
                    data.testInputs,  data.testTargets);

            GeneticProgramming.RunResult result = gp.run();
            trainFitnesses[i] = result.trainFitness;

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
                bestRunGP          = gp;
            }
        }

        // ── Summary statistics ─────────────────────────────────────────────────
        double mean     = computeMean(trainFitnesses);
        double stdDev   = computeStdDev(trainFitnesses, mean);

        System.out.println("════════════════════════════════════════════════════════");
        System.out.println("Summary (Train MSE across all runs)");
        System.out.println("════════════════════════════════════════════════════════");
        System.out.printf("  Average fitness    : %s%n", formatFitness(mean));
        System.out.printf("  Standard deviation : %s%n", formatFitness(stdDev));
        System.out.printf("  Best fitness       : %s  (Run %d)%n",
                formatFitness(overallBestFitness), overallBestRun);
        System.out.println();
        System.out.println("Best evolved expression:");
        System.out.println("  " + (overallBestTree != null ? overallBestTree : "N/A"));
        System.out.println("════════════════════════════════════════════════════════");

        // ── Hits and success predicate ─────────────────────────────────────────
        // Success predicate: a run is successful if train MSE < 0.00020
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
            System.out.printf("%-6s  %-14s  %-12s  %-8s  %-8s%n",
                    "Gen", "Avg Train MSE", "Avg TreeSize", "Hits", "Variety");
            System.out.println("──────────────────────────────────────────────────────");
            // Print every 10th generation plus the last
            int[] printAt = buildPrintGenerations(GENERATIONS);
            for (int g : printAt) {
                System.out.printf("%-6d  %-14s  %-12.2f  %-8d  %-8d%n",
                        g + 1,
                        formatFitness(bestRunResult.avgFitnessPerGen[g]),
                        bestRunResult.avgSizePerGen[g],
                        bestRunResult.hitsPerGen[g],
                        bestRunResult.varietyPerGen[g]);
            }
            System.out.println("════════════════════════════════════════════════════════");
        }
    }

    /** Returns indices of generations to print: every 10th plus the final. */
    private static int[] buildPrintGenerations(int totalGenerations) {
        java.util.List<Integer> list = new java.util.ArrayList<>();
        for (int g = 9; g < totalGenerations; g += 10) list.add(g);
        if (list.isEmpty() || list.get(list.size() - 1) != totalGenerations - 1) {
            list.add(totalGenerations - 1);
        }
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    // ─── Utility Helpers ──────────────────────────────────────────────────────

    /**
     * Formats a fitness value for display using a dot decimal separator.
     * Returns "N/A" if the value is {@link Double#MAX_VALUE} (numerical failure).
     *
     * @param fitness the fitness value to format
     * @return formatted string
     */
    private static String formatFitness(double fitness) {
        return fitness == Double.MAX_VALUE ? "N/A" : String.format(Locale.US, "%.8f", fitness);
    }

    /**
     * Computes the arithmetic mean of an array of doubles.
     *
     * @param values array of values (ignores {@link Double#MAX_VALUE} entries)
     * @return mean of valid (non-MAX_VALUE) entries; NaN if all entries are invalid
     */
    private static double computeMean(double[] values) {
        double sum = 0.0;
        int    count = 0;
        for (double v : values) {
            if (v != Double.MAX_VALUE) {
                sum += v;
                count++;
            }
        }
        return count == 0 ? Double.NaN : sum / count;
    }

    /**
     * Computes the population standard deviation of an array of doubles.
     *
     * @param values array of values (ignores {@link Double#MAX_VALUE} entries)
     * @param mean   the pre-computed mean of the valid entries
     * @return population standard deviation of valid entries
     */
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

    /** Prints the program banner. */
    private static void printBanner() {
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println("  COS 710 Artificial Intelligence – Assignment 1");
        System.out.println("  Genetic Programming for Symbolic Regression");
        System.out.println("  Dataset: Residential Energy Usage UK (2014–2020)");
        System.out.println("════════════════════════════════════════════════════════");
        System.out.println();
    }
}
