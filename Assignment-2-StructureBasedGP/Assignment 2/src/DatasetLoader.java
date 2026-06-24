import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Handles all dataset loading and preprocessing for the GP regression task.
 *
 * <p>The dataset is a CSV file with the following columns:
 * <pre>
 *   utc_timestamp, Electricity_load, Residential_electricity_price,
 *   Residential_solar_generation, Residential_wind_generation,
 *   Temperature, Relative Humidity
 * </pre>
 * Only the {@code Electricity_load} column (index 1) is used; all other columns
 * are ignored. The time series is converted into a supervised learning dataset
 * using a sliding window of size {@code windowSize}.
 *
 * <p>For each time step t:
 * <pre>
 *   inputs[i][0] = x1 = Load(t−1)   (most recent lag)
 *   inputs[i][1] = x2 = Load(t−2)
 *   …
 *   inputs[i][windowSize−1] = Load(t−windowSize)   (oldest lag)
 *   targets[i] = Load(t)
 * </pre>
 *
 * <p>The split preserves temporal order (no shuffling) since shuffling would
 * introduce data leakage in a time-series regression task.
 *
 * <p>COS 710 Assignment 2 – Structure-Based Genetic Programming for Regression.
 */
public class DatasetLoader {

    /**
     * Maximum number of valid data rows to load from the CSV file.
     * {@code -1} loads the entire dataset (~201 604 rows).
     * 100 000 rows covers roughly half the full dataset (~80 000 training,
     * ~20 000 test examples) while keeping each run to a manageable duration.
     */
    public static final int DATA_SAMPLE_SIZE = 100000;

    // ─── Inner Classes ────────────────────────────────────────────────────────

    /**
     * Container for the split training and testing datasets produced by
     * {@link #createAndSplitData(double[], int, double)}.
     */
    public static class TrainingData {

        /** 2-D array of training inputs; shape [trainSize][windowSize]. */
        public final double[][] trainInputs;

        /** 1-D array of training targets; length trainSize. */
        public final double[] trainTargets;

        /** 2-D array of test inputs; shape [testSize][windowSize]. */
        public final double[][] testInputs;

        /** 1-D array of test targets; length testSize. */
        public final double[] testTargets;

        /**
         * Constructs a TrainingData container with the given split arrays.
         *
         * @param trainInputs  training input matrix
         * @param trainTargets training target vector
         * @param testInputs   test input matrix
         * @param testTargets  test target vector
         */
        public TrainingData(double[][] trainInputs, double[] trainTargets,
                            double[][] testInputs,  double[] testTargets) {
            this.trainInputs  = trainInputs;
            this.trainTargets = trainTargets;
            this.testInputs   = testInputs;
            this.testTargets  = testTargets;
        }
    }

    // ─── Data Loading ─────────────────────────────────────────────────────────

    /**
     * Loads the {@code Electricity_load} column from the CSV file.
     *
     * <p>The first line (header) is skipped. Rows with missing or non-numeric
     * load values are silently skipped. Loading stops once {@link #DATA_SAMPLE_SIZE}
     * valid rows have been collected (or at EOF if {@code DATA_SAMPLE_SIZE == -1}).
     *
     * @param csvPath path to the CSV file (relative to the working directory)
     * @return array of electricity load values in chronological order
     * @throws IOException if the file cannot be opened or read
     */
    public static double[] loadElectricityLoad(String csvPath) throws IOException {
        List<Double> values = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            br.readLine(); // skip header line

            String line;
            int rowsRead = 0;

            while ((line = br.readLine()) != null) {
                if (DATA_SAMPLE_SIZE >= 0 && rowsRead >= DATA_SAMPLE_SIZE) {
                    break;
                }

                String[] parts = line.split(",");
                if (parts.length < 2) continue;

                String cell = parts[1].trim();
                if (cell.isEmpty()) continue;

                try {
                    double val = Double.parseDouble(cell);
                    if (!Double.isNaN(val) && !Double.isInfinite(val)) {
                        values.add(val);
                        rowsRead++;
                    }
                } catch (NumberFormatException e) {
                    // Malformed cell – skip this row
                }
            }
        }

        double[] result = new double[values.size()];
        for (int i = 0; i < values.size(); i++) {
            result[i] = values.get(i);
        }
        return result;
    }

    // ─── Window Creation and Splitting ────────────────────────────────────────

    /**
     * Converts the raw time series into windowed training/test examples and
     * performs the temporal 80/20 split in a single pass.
     *
     * <p>For each valid target position {@code t} (where {@code t >= windowSize}):
     * <pre>
     *   inputs[i][j]  = series[t − 1 − j]   (j=0 → Load(t−1) = x1, etc.)
     *   targets[i]    = series[t]
     * </pre>
     *
     * <p>The first {@code trainRatio * N} examples form the training set;
     * the remainder form the test set. No shuffling is performed.
     *
     * @param series     the raw electricity load time series
     * @param windowSize number of previous time steps to use as input features
     * @param trainRatio fraction of examples to use for training (e.g. 0.8)
     * @return a {@link TrainingData} object containing the split datasets
     */
    public static TrainingData createAndSplitData(double[] series,
                                                   int windowSize,
                                                   double trainRatio) {
        int n = series.length - windowSize;

        double[][] inputs  = new double[n][windowSize];
        double[]   targets = new double[n];

        for (int i = 0; i < n; i++) {
            targets[i] = series[i + windowSize];
            for (int j = 0; j < windowSize; j++) {
                inputs[i][j] = series[i + windowSize - 1 - j];
            }
        }

        int trainSize = (int) (n * trainRatio);

        double[][] trainInputs  = Arrays.copyOfRange(inputs,  0,         trainSize);
        double[]   trainTargets = Arrays.copyOfRange(targets, 0,         trainSize);
        double[][] testInputs   = Arrays.copyOfRange(inputs,  trainSize, n);
        double[]   testTargets  = Arrays.copyOfRange(targets, trainSize, n);

        return new TrainingData(trainInputs, trainTargets, testInputs, testTargets);
    }
}
