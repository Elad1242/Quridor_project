package ml;

import java.util.Random;

// Trains the neural network on feature data. Mini-batch SGD with momentum and early stopping.
public class FeatureTrainer {

    public static void main(String[] args) throws Exception {
        String dataFile = (args.length > 0) ? args[0] : "training_features.dat";
        String modelPath = (args.length > 1) ? args[1] : "feature_model.bin";

        System.out.println("Feature Network Trainer");
        System.out.println("Data: " + dataFile);

        FeatureDataGenerator.TrainingData data = FeatureDataGenerator.readData(dataFile);
        System.out.println("Samples: " + data.size());

        // label distribution
        int wins = 0;
        for (double l : data.labels) if (l > 0.5) wins++;
        System.out.println("Labels - win: " + wins + " (" + (100.0 * wins / data.size()) + "%), loss: " + (data.size() - wins));

        NeuralNetwork nn = trainNetwork(data, modelPath);

        System.out.println("\nTraining complete. Model saved to " + modelPath);
        System.out.println("Parameters: " + nn.getParamCount());
    }

    public static NeuralNetwork trainNetwork(FeatureDataGenerator.TrainingData data, String modelPath) throws Exception {
        Random rng = new Random(42);

        // split 90/10 train/val
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

        // create network
        NeuralNetwork nn = new NeuralNetwork(GameFeatures.TOTAL_FEATURES, 64, 32, 16, 1);
        System.out.println("Network: 22->64->32->16->1, params=" + nn.getParamCount());

        // hyperparams
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
            // shuffle training data
            for (int i = trainSize - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                double[] tmpX = trainX[i]; trainX[i] = trainX[j]; trainX[j] = tmpX;
                double tmpY = trainY[i]; trainY[i] = trainY[j]; trainY[j] = tmpY;
            }

            // train one epoch
            double trainLoss = 0;
            int batches = 0;

            for (int i = 0; i < trainSize; i += batchSize) {
                int end = Math.min(i + batchSize, trainSize);
                int len = end - i;

                // accumulate gradients
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

            // validation every 5 epochs
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
                System.out.println("Epoch " + (epoch + 1) + "/" + maxEpochs + " (" + elapsed + "s) loss=" + trainLoss + " valLoss=" + valLoss + " acc=" + valAcc + "%");

                if (valLoss < bestValLoss) {
                    bestValLoss = valLoss;
                    noImproveCount = 0;
                    nn.save(modelPath);
                    System.out.println("  >> best model saved (valLoss=" + bestValLoss + ")");
                } else {
                    noImproveCount += 5;
                    if (noImproveCount >= patience / 2 && lr > 1e-5) {
                        lr *= 0.5;
                        System.out.println("  >> lr reduced to " + lr);
                        noImproveCount = 0; // reset after LR reduction
                    }
                    if (noImproveCount >= patience) {
                        System.out.println("  >> early stopping!");
                        break;
                    }
                }
            }
        }

        // load best model
        return NeuralNetwork.load(modelPath);
    }
}
