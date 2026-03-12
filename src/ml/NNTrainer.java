package ml;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import org.bson.Document;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Trains the neural network using game data from MongoDB.
 *
 * Process:
 * 1. Load all games from MongoDB
 * 2. Extract (features, label) pairs from each turn
 *    - features = the 12 game features
 *    - label = 1.0 if currentPlayer won the game, 0.0 if lost
 * 3. Normalize features to [0,1] range
 * 4. Split into training (80%) and validation (20%) sets
 * 5. Train for multiple epochs with shuffling
 * 6. Save the trained network to a file
 *
 * Usage:
 *   java NNTrainer [mongoUri] [outputFile]
 *   java NNTrainer mongodb+srv://user:pass@cluster.net/ nn_weights.dat
 */
public class NNTrainer {

    private static final String DEFAULT_DB = "quoridor";
    private static final String DEFAULT_OUTPUT = "nn_weights.dat";
    private static final int EPOCHS = 30;
    private static final double TRAIN_SPLIT = 0.8;

    // Feature normalization ranges (approximate from game knowledge)
    // These transform raw features into roughly [0, 1] range
    private static final double[] FEATURE_MIN = {
        1, 1, 0, 0, 1, 1, 1, 1, -15, 0, 0, 2
    };
    private static final double[] FEATURE_MAX = {
        30, 30, 10, 10, 5, 5, 40, 40, 15, 200, 20, 16
    };

    public static void main(String[] args) {
        String mongoUri = args.length >= 1 ? args[0] : null;
        String outputFile = args.length >= 2 ? args[1] : DEFAULT_OUTPUT;

        if (mongoUri == null) {
            System.out.println("Usage: NNTrainer [mongoUri] [outputFile]");
            System.out.println("Example: NNTrainer mongodb+srv://user:pass@cluster.net/ nn_weights.dat");
            return;
        }

        System.out.println("=== Neural Network Trainer ===");
        System.out.println("Loading data from MongoDB...");

        // Step 1: Load training samples from MongoDB
        List<double[]> features = new ArrayList<>();
        List<Double> labels = new ArrayList<>();
        loadData(mongoUri, features, labels);

        if (features.isEmpty()) {
            System.out.println("No training data found!");
            return;
        }

        System.out.println("Loaded " + features.size() + " training samples from "
                + countGames(mongoUri) + " games");

        // Step 2: Normalize features
        normalize(features);

        // Step 3: Split into train/validation
        int splitIndex = (int) (features.size() * TRAIN_SPLIT);
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < features.size(); i++) indices.add(i);
        Collections.shuffle(indices, new Random(42));

        List<double[]> trainFeatures = new ArrayList<>();
        List<Double> trainLabels = new ArrayList<>();
        List<double[]> valFeatures = new ArrayList<>();
        List<Double> valLabels = new ArrayList<>();

        for (int i = 0; i < indices.size(); i++) {
            int idx = indices.get(i);
            if (i < splitIndex) {
                trainFeatures.add(features.get(idx));
                trainLabels.add(labels.get(idx));
            } else {
                valFeatures.add(features.get(idx));
                valLabels.add(labels.get(idx));
            }
        }

        System.out.println("Training samples: " + trainFeatures.size());
        System.out.println("Validation samples: " + valFeatures.size());
        System.out.println();

        // Step 4: Create and train network
        NeuralNetwork nn = new NeuralNetwork();
        System.out.println("Network: " + nn);
        System.out.println("Epochs: " + EPOCHS);
        System.out.println("Learning rate: " + nn.getLearningRate());
        System.out.println();

        Random shuffleRng = new Random(123);

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            // Shuffle training data each epoch
            List<Integer> trainIdx = new ArrayList<>();
            for (int i = 0; i < trainFeatures.size(); i++) trainIdx.add(i);
            Collections.shuffle(trainIdx, shuffleRng);

            // Train
            double trainLoss = 0;
            for (int idx : trainIdx) {
                trainLoss += nn.train(trainFeatures.get(idx), trainLabels.get(idx));
            }
            trainLoss /= trainFeatures.size();

            // Validate
            double valLoss = 0;
            int correct = 0;
            for (int i = 0; i < valFeatures.size(); i++) {
                double pred = nn.predict(valFeatures.get(i));
                double label = valLabels.get(i);
                valLoss += (pred - label) * (pred - label);

                // Accuracy: prediction > 0.5 matches label > 0.5
                if ((pred > 0.5) == (label > 0.5)) correct++;
            }
            valLoss /= valFeatures.size();
            double accuracy = 100.0 * correct / valFeatures.size();

            System.out.printf("Epoch %2d/%d | Train Loss: %.4f | Val Loss: %.4f | Accuracy: %.1f%%%n",
                    epoch, EPOCHS, trainLoss, valLoss, accuracy);

            // Learning rate decay every 10 epochs
            if (epoch % 10 == 0) {
                nn.setLearningRate(nn.getLearningRate() * 0.5);
                System.out.println("  → Learning rate reduced to " + nn.getLearningRate());
            }
        }

        // Step 5: Save
        try {
            nn.save(outputFile);
            System.out.println();
            System.out.println("=== Training Complete ===");
            System.out.println("Weights saved to: " + outputFile);
        } catch (Exception e) {
            System.err.println("Failed to save: " + e.getMessage());
        }
    }

    /**
     * Loads training data from MongoDB.
     * For each turn in each game:
     *   - features = the 12 game features
     *   - label = 1.0 if the current player won, 0.0 if lost
     * Draws/timeouts are skipped.
     */
    private static void loadData(String mongoUri, List<double[]> features, List<Double> labels) {
        try (MongoClient client = MongoClients.create(mongoUri)) {
            MongoDatabase db = client.getDatabase(DEFAULT_DB);
            MongoCollection<Document> collection = db.getCollection("games");

            String[] featureNames = GameFeatures.featureNames();

            for (Document game : collection.find()) {
                int winner = game.getInteger("winner", -1);
                if (winner == -1) continue; // skip draws/timeouts

                List<Document> turns = game.getList("turns", Document.class);
                if (turns == null) continue;

                for (Document turn : turns) {
                    int currentPlayer = turn.getInteger("currentPlayer", 0);
                    Document featDoc = turn.get("features", Document.class);
                    if (featDoc == null) continue;

                    // Extract features
                    double[] feat = new double[featureNames.length];
                    for (int i = 0; i < featureNames.length; i++) {
                        Number val = featDoc.get(featureNames[i], Number.class);
                        feat[i] = val != null ? val.doubleValue() : 0.0;
                    }

                    // Label: 1.0 if this player won, 0.0 if lost
                    double label = (currentPlayer == winner) ? 1.0 : 0.0;

                    features.add(feat);
                    labels.add(label);
                }
            }
        }
    }

    /**
     * Normalizes features to [0, 1] range using predefined min/max values.
     */
    private static void normalize(List<double[]> features) {
        for (double[] feat : features) {
            for (int i = 0; i < feat.length; i++) {
                double range = FEATURE_MAX[i] - FEATURE_MIN[i];
                if (range > 0) {
                    feat[i] = (feat[i] - FEATURE_MIN[i]) / range;
                    // Clamp to [0, 1]
                    feat[i] = Math.max(0, Math.min(1, feat[i]));
                }
            }
        }
    }

    /**
     * Normalizes a single feature vector (for use by NNBot at runtime).
     */
    public static double[] normalizeSingle(double[] raw) {
        double[] normalized = new double[raw.length];
        for (int i = 0; i < raw.length; i++) {
            double range = FEATURE_MAX[i] - FEATURE_MIN[i];
            if (range > 0) {
                normalized[i] = (raw[i] - FEATURE_MIN[i]) / range;
                normalized[i] = Math.max(0, Math.min(1, normalized[i]));
            }
        }
        return normalized;
    }

    private static long countGames(String mongoUri) {
        try (MongoClient client = MongoClients.create(mongoUri)) {
            MongoDatabase db = client.getDatabase(DEFAULT_DB);
            return db.getCollection("games").countDocuments();
        }
    }
}
