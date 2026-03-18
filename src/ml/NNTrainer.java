package ml;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import org.bson.Document;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.structure.NetworkCODEC;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Trains the neural network using game data stored in MongoDB.
 *
 * Training pipeline:
 *   1. Load game documents from MongoDB
 *   2. Extract (features, label) pairs from each turn
 *   3. Expand to 18 features (12 raw + 6 derived interaction terms)
 *   4. Normalize features to [0,1] range
 *   5. Train multiple times with RPROP, keep the best model
 *   6. Validate on a held-out 20% of the data
 *   7. Save the trained network
 *
 * Usage:
 *   java ml.NNTrainer [mongoUri] [outputFile]
 */
public class NNTrainer {

    // Total input size: 12 raw features + 6 derived interaction features
    public static final int INPUT_SIZE = 18;

    // Feature normalization ranges (indices 0-11: raw features, 12-17: derived)
    private static final double[] FEATURE_MIN = {
        1, 1, 0, 0, 1, 1, 1, 1, -15, 0, 0, 2,  // raw features
        0, 0, 0, 0, 0, -150                        // derived features
    };
    private static final double[] FEATURE_MAX = {
        30, 30, 10, 10, 5, 5, 40, 40, 15, 200, 20, 16,  // raw features
        1, 1, 1, 300, 300, 150                             // derived features
    };

    private static final int EPOCHS = 500;
    private static final int PATIENCE = 50;
    private static final int MIN_TURN = 8;  // include most turns (was 40, losing too much data)
    private static final int NUM_RUNS = 20;  // train multiple times, keep best
    private static final int ENSEMBLE_SIZE = 5; // keep top-K models for ensemble

    /**
     * Normalizes a single feature value to [0,1] based on predefined ranges.
     */
    public static double normalizeSingle(double value, int featureIndex) {
        double min = FEATURE_MIN[featureIndex];
        double max = FEATURE_MAX[featureIndex];
        if (max == min) return 0.5;
        double normalized = (value - min) / (max - min);
        return Math.max(0.0, Math.min(1.0, normalized));
    }

    /**
     * Normalizes an entire feature vector.
     */
    public static double[] normalizeAll(double[] features) {
        double[] normalized = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            normalized[i] = normalizeSingle(features[i], i);
        }
        return normalized;
    }

    /**
     * Expands 12 raw features into 18 by adding 6 derived interaction features.
     *
     * Derived features:
     *  [12] distRatio       = myDist / (myDist + oppDist + 1)
     *  [13] flowRatio       = myMaxFlow / (myMaxFlow + oppMaxFlow + 1)
     *  [14] safetyRatio     = mySafety / (mySafety + oppSafety + 1)
     *  [15] myThreat        = myDist * oppWallsLeft
     *  [16] oppThreat       = oppDist * myWallsLeft
     *  [17] raceWallBalance = raceGap * (oppWallsLeft - myWallsLeft)
     */
    public static double[] expandFeatures(double[] raw) {
        double[] expanded = new double[INPUT_SIZE];
        System.arraycopy(raw, 0, expanded, 0, 12);

        double myDist = raw[0], oppDist = raw[1];
        double myWalls = raw[2], oppWalls = raw[3];
        double myFlow = raw[4], oppFlow = raw[5];
        double mySafety = raw[6], oppSafety = raw[7];
        double raceGap = raw[8];

        expanded[12] = myDist / (myDist + oppDist + 1);
        expanded[13] = myFlow / (myFlow + oppFlow + 1);
        expanded[14] = mySafety / (mySafety + oppSafety + 1);
        expanded[15] = myDist * oppWalls;
        expanded[16] = oppDist * myWalls;
        expanded[17] = raceGap * (oppWalls - myWalls);

        return expanded;
    }

    public static void main(String[] args) {
        String mongoUri = System.getenv("MONGODB_URI") != null ? System.getenv("MONGODB_URI") : "mongodb://localhost:27017";
        String outputFile = "quoridor_nn.eg";

        if (args.length >= 1) mongoUri = args[0];
        if (args.length >= 2) outputFile = args[1];

        System.out.println("=== Quoridor Neural Network Trainer ===");
        System.out.println("Connecting to MongoDB...");

        // Step 1: Load data from MongoDB
        List<double[]> allFeatures = new ArrayList<>();
        List<Double> allLabels = new ArrayList<>();
        loadDataFromMongo(mongoUri, allFeatures, allLabels);

        if (allFeatures.isEmpty()) {
            System.out.println("No training data found! Run TrainingDataGenerator first.");
            return;
        }

        System.out.println("Loaded " + allFeatures.size() + " training samples");

        // Step 2: Normalize features
        System.out.println("Normalizing features...");
        for (int i = 0; i < allFeatures.size(); i++) {
            allFeatures.set(i, normalizeAll(allFeatures.get(i)));
        }

        // Step 3: Shuffle and split into train/validation (80/20)
        long seed = 42;
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < allFeatures.size(); i++) indices.add(i);
        Collections.shuffle(indices, new java.util.Random(seed));

        int splitPoint = (int) (indices.size() * 0.8);
        MLDataSet trainSet = buildDataSet(allFeatures, allLabels, indices, 0, splitPoint);
        MLDataSet valSet = buildDataSet(allFeatures, allLabels, indices, splitPoint, indices.size());

        System.out.println("Training samples: " + trainSet.getRecordCount());
        System.out.println("Validation samples: " + valSet.getRecordCount());

        // Step 4: Multi-run training — RPROP is stochastic, so different
        // random initializations find different optima. Keep the best.
        System.out.println("\nArchitecture: 18 -> 32 (tanh) -> 16 (tanh) -> 1 (Sigmoid)");
        System.out.println("Algorithm: Resilient Propagation (RPROP) + early stopping");
        System.out.println("Runs: " + NUM_RUNS + " | Epochs: " + EPOCHS + " | Patience: " + PATIENCE);

        // Track top-K models for ensemble
        List<double[]> topWeights = new ArrayList<>();
        List<Double> topErrors = new ArrayList<>();

        for (int run = 1; run <= NUM_RUNS; run++) {
            System.out.println("\n--- Run " + run + "/" + NUM_RUNS + " ---");
            NeuralNetwork nn = new NeuralNetwork();
            ResilientPropagation trainer = new ResilientPropagation(nn.getNetwork(), trainSet);

            double bestValError = Double.MAX_VALUE;
            int bestEpoch = 0;
            double[] bestWeights = NetworkCODEC.networkToArray(nn.getNetwork());
            int epochsWithoutImprovement = 0;

            for (int epoch = 1; epoch <= EPOCHS; epoch++) {
                trainer.iteration();
                double trainError = trainer.getError();
                double valError = nn.getNetwork().calculateError(valSet);

                if (valError < bestValError) {
                    bestValError = valError;
                    bestEpoch = epoch;
                    bestWeights = NetworkCODEC.networkToArray(nn.getNetwork());
                    epochsWithoutImprovement = 0;
                } else {
                    epochsWithoutImprovement++;
                }

                if (epoch % 20 == 0 || epoch == 1) {
                    System.out.printf("  Epoch %3d | Train: %.6f | Val: %.6f | Best: %.6f (ep %d)%n",
                            epoch, trainError, valError, bestValError, bestEpoch);
                }

                if (epochsWithoutImprovement >= PATIENCE) {
                    System.out.printf("  Early stop at epoch %d%n", epoch);
                    break;
                }
            }
            trainer.finishTraining();

            System.out.printf("  Run %d best: val_error=%.6f at epoch %d%n", run, bestValError, bestEpoch);

            // Insert into top-K sorted list
            int insertIdx = topErrors.size();
            for (int k = 0; k < topErrors.size(); k++) {
                if (bestValError < topErrors.get(k)) {
                    insertIdx = k;
                    break;
                }
            }
            topWeights.add(insertIdx, bestWeights);
            topErrors.add(insertIdx, bestValError);
            if (topWeights.size() > ENSEMBLE_SIZE) {
                topWeights.remove(topWeights.size() - 1);
                topErrors.remove(topErrors.size() - 1);
            }
        }

        System.out.println("\n=== Ensemble: Top " + topWeights.size() + " models ===");
        for (int k = 0; k < topWeights.size(); k++) {
            System.out.printf("  Model %d: val_error=%.6f%n", k + 1, topErrors.get(k));
        }

        // Build ensemble networks
        List<NeuralNetwork> ensemble = new ArrayList<>();
        for (double[] weights : topWeights) {
            NeuralNetwork nn = new NeuralNetwork();
            NetworkCODEC.arrayToNetwork(weights, nn.getNetwork());
            ensemble.add(nn);
        }

        // Step 5: Final evaluation with ensemble averaging
        System.out.println();
        System.out.println("=== Training Complete ===");

        // Single-best accuracy (for comparison)
        NeuralNetwork bestNN = ensemble.get(0);
        System.out.printf("Best single model val error: %.6f%n", topErrors.get(0));

        // Ensemble accuracy
        int correctSingle = 0;
        int correctEnsemble = 0;
        int total = 0;
        for (int i = splitPoint; i < indices.size(); i++) {
            int idx = indices.get(i);
            double actual = allLabels.get(idx);

            // Single best
            double singlePred = bestNN.predict(allFeatures.get(idx));
            if ((singlePred >= 0.5) == (actual >= 0.5)) correctSingle++;

            // Ensemble average
            double ensembleSum = 0;
            for (NeuralNetwork nn : ensemble) {
                ensembleSum += nn.predict(allFeatures.get(idx));
            }
            double ensemblePred = ensembleSum / ensemble.size();
            if ((ensemblePred >= 0.5) == (actual >= 0.5)) correctEnsemble++;

            total++;
        }
        System.out.printf("Single Best Accuracy:  %.1f%% (%d/%d)%n",
                100.0 * correctSingle / total, correctSingle, total);
        System.out.printf("Ensemble Accuracy:     %.1f%% (%d/%d)%n",
                100.0 * correctEnsemble / total, correctEnsemble, total);

        // Step 7: Save all ensemble models
        for (int k = 0; k < ensemble.size(); k++) {
            String path = (k == 0) ? outputFile : outputFile.replace(".eg", "_" + (k + 1) + ".eg");
            ensemble.get(k).save(path);
        }
        System.out.println("Done! Ensemble of " + ensemble.size() + " models saved.");
    }

    /**
     * Mirrors a feature vector by swapping player perspectives.
     * Swaps: myDist <-> oppDist, myWalls <-> oppWalls, myFlow <-> oppFlow,
     *        mySafety <-> oppSafety, raceGap negated.
     * Then recomputes derived features from the swapped raw features.
     * Label is flipped: 1.0 -> 0.0, 0.0 -> 1.0.
     */
    public static double[] mirrorFeatures(double[] expanded) {
        double[] raw = new double[12];
        // Swap my/opp pairs
        raw[0] = expanded[1]; // myDist <- oppDist
        raw[1] = expanded[0]; // oppDist <- myDist
        raw[2] = expanded[3]; // myWalls <- oppWalls
        raw[3] = expanded[2]; // oppWalls <- myWalls
        raw[4] = expanded[5]; // myFlow <- oppFlow
        raw[5] = expanded[4]; // oppFlow <- myFlow
        raw[6] = expanded[7]; // mySafety <- oppSafety
        raw[7] = expanded[6]; // oppSafety <- mySafety
        raw[8] = -expanded[8]; // raceGap negated
        raw[9] = expanded[9]; // turnNumber unchanged
        raw[10] = expanded[10]; // totalWalls unchanged
        raw[11] = expanded[11]; // distToOpponent unchanged
        return expandFeatures(raw);
    }

    /**
     * Loads game data from MongoDB and extracts (features, label) pairs.
     *
     * TURN-WEIGHTED LABELS: Instead of binary 0/1, we use weighted labels that
     * give more certainty to late-game positions:
     *   - Early turns: label closer to 0.5 (uncertain - game outcome not clear yet)
     *   - Late turns: label closer to 0/1 (certain - outcome is more predictable)
     *
     * This reduces noise from early positions where the eventual winner might
     * still be in a losing position (or vice versa).
     *
     * Skips draws/timeouts and very early turns.
     * Each sample is also mirrored (player swap) for data augmentation.
     */
    private static void loadDataFromMongo(String mongoUri,
                                          List<double[]> features,
                                          List<Double> labels) {
        MongoClient client = MongoClients.create(mongoUri);
        MongoDatabase db = client.getDatabase("quoridor");
        MongoCollection<Document> collection = db.getCollection("games");

        String[] featureNames = GameFeatures.featureNames();
        int gameCount = 0;

        for (Document game : collection.find()) {
            int winner = game.getInteger("winner", -1);
            if (winner == -1) continue;

            List<Document> turns = game.getList("turns", Document.class);
            if (turns == null) continue;

            int totalTurns = turns.size();

            for (Document turn : turns) {
                int currentPlayer = turn.getInteger("currentPlayer", 0);
                Document featureDoc = turn.get("features", Document.class);
                if (featureDoc == null) continue;

                double[] rawFeatures = new double[GameFeatures.FEATURE_COUNT];
                for (int f = 0; f < featureNames.length; f++) {
                    Number val = featureDoc.get(featureNames[f], Number.class);
                    rawFeatures[f] = (val != null) ? val.doubleValue() : 0.0;
                }

                int turnNumber = (int) rawFeatures[9];
                if (turnNumber < MIN_TURN) continue; // skip early turns

                double[] expanded = expandFeatures(rawFeatures);

                // TURN-WEIGHTED LABEL:
                // Weight increases from 0.3 at MIN_TURN to 1.0 at game end
                // This gives early turns labels closer to 0.5 (uncertain)
                // and late turns labels closer to 0/1 (certain)
                double progress = (double) turnNumber / Math.max(totalTurns, turnNumber + 1);
                double turnWeight = 0.3 + 0.7 * progress;  // Range: [0.3, 1.0]

                double baseLabel = (currentPlayer == winner) ? 1.0 : 0.0;
                double weightedLabel = 0.5 + (baseLabel - 0.5) * turnWeight;

                features.add(expanded);
                labels.add(weightedLabel);

                // Data augmentation: mirror (swap player perspectives)
                double[] mirrored = mirrorFeatures(expanded);
                features.add(mirrored);
                labels.add(1.0 - weightedLabel);
            }

            gameCount++;
            if (gameCount % 1000 == 0) {
                System.out.println("Loaded " + gameCount + " games (" + features.size() + " samples)...");
            }
        }

        client.close();
        System.out.println("Loaded " + gameCount + " games total (" + features.size() + " samples)");
    }

    private static MLDataSet buildDataSet(List<double[]> features, List<Double> labels,
                                          List<Integer> indices, int from, int to) {
        BasicMLDataSet dataSet = new BasicMLDataSet();
        for (int i = from; i < to; i++) {
            int idx = indices.get(i);
            BasicMLData input = new BasicMLData(features.get(idx));
            BasicMLData ideal = new BasicMLData(new double[] { labels.get(idx) });
            dataSet.add(new BasicMLDataPair(input, ideal));
        }
        return dataSet;
    }
}
