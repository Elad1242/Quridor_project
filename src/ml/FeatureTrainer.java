package ml;

import java.util.Random;

/**
 * Trains the from-scratch NeuralNetwork on feature data.
 * Mini-batch SGD with momentum, MSE loss, early stopping.
 *
 * Usage: java ml.FeatureTrainer [dataFile] [modelOutputPath]
 */
public class FeatureTrainer {

    public static void main(String[] args) throws Exception {
        String dataFile = (args.length > 0) ? args[0] : "training_features.dat";
        String modelPath = (args.length > 1) ? args[1] : "feature_model.bin";

        System.out.println("=== Feature Network Trainer ===");
        System.out.println("Data: " + dataFile);

        FeatureDataGenerator.TrainingData data = FeatureDataGenerator.readData(dataFile);
        System.out.println("Samples: " + data.size());

        // Label distribution
        int wins = 0;
        for (double l : data.labels) if (l > 0.5) wins++;
        System.out.printf("Labels — win: %d (%.1f%%), loss: %d (%.1f%%)%n",
                wins, 100.0 * wins / data.size(),
                data.size() - wins, 100.0 * (data.size() - wins) / data.size());

        // Train
        NeuralNetwork nn = trainNetwork(data, modelPath);

        System.out.println("\nTraining complete. Model saved to " + modelPath);
        System.out.println("Parameters: " + nn.getParamCount());
    }

    public static NeuralNetwork trainNetwork(FeatureDataGenerator.TrainingData data, String modelPath) throws Exception {
        Random rng = new Random(42);

        // Split: 90% train, 10% val
        data.shuffle(rng);
        int trainSize = (int) (data.size() * 0.9);
        double[][] trainX = new double[trainSize][];
        double[] trainY = new double[trainSize];
        double[][] valX = new double[data.size() - trainSize][];
        double[] valY = new double[data.size() - trainSize];

        System.arraycopy(data.features, 0, trainX, 0, trainSize);
        System.arraycopy(data.labels, 0, trainY, 0, trainSize);
        System.arraycopy(data.features, trainSize, valX, 0, valX.length);
        System.arraycopy(data.labels, trainSize, valY, 0, valY.length);

        System.out.println("Train: " + trainSize + ", Val: " + valX.length);

        // Create network: 22 → 64 → 32 → 16 → 1
        NeuralNetwork nn = new NeuralNetwork(GameFeatures.TOTAL_FEATURES, 64, 32, 16, 1);
        System.out.println("Network: 22→64→32→16→1, params=" + nn.getParamCount());

        // Hyperparameters
        double lr = 0.001;
        double momentum = 0.9;
        double weightDecay = 1e-5;
        int batchSize = 64;
        int maxEpochs = 200;
        int patience = 50;
        int noImproveCount = 0;
        double bestValLoss = Double.MAX_VALUE;

        long startTime = System.currentTimeMillis();

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            // Shuffle training data
            for (int i = trainSize - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                double[] tmpX = trainX[i]; trainX[i] = trainX[j]; trainX[j] = tmpX;
                double tmpY = trainY[i]; trainY[i] = trainY[j]; trainY[j] = tmpY;
            }

            // Train one epoch
            double trainLoss = 0;
            int batches = 0;

            for (int i = 0; i < trainSize; i += batchSize) {
                int end = Math.min(i + batchSize, trainSize);
                int len = end - i;

                // Accumulate gradients
                double[][][] gradW = null;
                double[][] gradB = null;
                double batchLoss = 0;

                for (int j = i; j < end; j++) {
                    NeuralNetwork.Gradients g = nn.backprop(trainX[j], trainY[j]);
                    batchLoss += g.loss;

                    if (gradW == null) {
                        gradW = g.dW;
                        gradB = g.dB;
                    } else {
                        for (int L = 0; L < g.dW.length; L++) {
                            for (int r = 0; r < g.dW[L].length; r++) {
                                gradB[L][r] += g.dB[L][r];
                                for (int c = 0; c < g.dW[L][r].length; c++) {
                                    gradW[L][r][c] += g.dW[L][r][c];
                                }
                            }
                        }
                    }
                }

                nn.updateWeights(gradW, gradB, len, lr, momentum, weightDecay);
                trainLoss += batchLoss / len;
                batches++;
            }

            trainLoss /= batches;

            // Validation loss (every 5 epochs)
            if (epoch % 5 == 0 || epoch == maxEpochs - 1) {
                double valLoss = 0;
                int correct = 0;
                for (int i = 0; i < valX.length; i++) {
                    double pred = nn.predict(valX[i]);
                    double err = pred - valY[i];
                    valLoss += 0.5 * err * err;
                    if ((pred > 0.5) == (valY[i] > 0.5)) correct++;
                }
                valLoss /= valX.length;
                double valAcc = 100.0 * correct / valX.length;

                long elapsed = (System.currentTimeMillis() - startTime) / 1000;
                System.out.printf("Epoch %d/%d (%ds) — trainLoss=%.5f, valLoss=%.5f, valAcc=%.1f%%, lr=%.6f%n",
                        epoch + 1, maxEpochs, elapsed, trainLoss, valLoss, valAcc, lr);

                if (valLoss < bestValLoss) {
                    bestValLoss = valLoss;
                    noImproveCount = 0;
                    nn.save(modelPath);
                    System.out.printf("  >> Best model saved (valLoss=%.5f)%n", bestValLoss);
                } else {
                    noImproveCount += 5;
                    if (noImproveCount >= patience / 2 && lr > 1e-5) {
                        lr *= 0.5;
                        System.out.printf("  >> LR reduced to %.6f%n", lr);
                        noImproveCount = 0; // reset after LR reduction
                    }
                    if (noImproveCount >= patience) {
                        System.out.println("  >> Early stopping!");
                        break;
                    }
                }
            }
        }

        // Load best model
        return NeuralNetwork.load(modelPath);
    }
}
