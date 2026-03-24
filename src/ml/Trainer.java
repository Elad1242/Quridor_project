package ml;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Training loop for the Quoridor Value Network.
 *
 * Usage: java ml.Trainer [dataDir] [modelDir]
 * Default: dataDir=training_data, modelDir=models
 *
 * Trains with:
 *   - Adam optimizer, LR=0.001 with cosine decay
 *   - Batch size 256
 *   - MSE loss
 *   - 20 epochs, 90/10 train/val split
 *   - Early stopping (patience=5 on val loss)
 */
public class MLTrainer {

    private static final int BATCH_SIZE = 256;
    private static final int EPOCHS = 20;
    private static final float INITIAL_LR = 0.001f;
    private static final float MIN_LR = 0.00001f;
    private static final double TRAIN_RATIO = 0.9;
    private static final int PATIENCE = 5;

    public static void main(String[] args) throws Exception {
        String dataDir = (args.length > 0) ? args[0] : "training_data";
        String modelDir = (args.length > 1) ? args[1] : "models";

        new File(modelDir).mkdirs();

        System.out.println("=== Quoridor Value Network Trainer ===");
        System.out.println("Data dir: " + dataDir);
        System.out.println("Model dir: " + modelDir);

        // Load all .qdat files from data directory
        File[] dataFiles = new File(dataDir).listFiles((dir, name) -> name.endsWith(".qdat"));
        if (dataFiles == null || dataFiles.length == 0) {
            System.err.println("No .qdat files found in " + dataDir);
            return;
        }

        String[] paths = new String[dataFiles.length];
        for (int i = 0; i < dataFiles.length; i++) {
            paths[i] = dataFiles[i].getAbsolutePath();
        }

        System.out.println("Loading " + paths.length + " data files...");
        TrainingDataWriter.TrainingData allData = TrainingDataWriter.readAll(paths);
        System.out.println("Total samples: " + allData.size());

        // Print label distribution
        printLabelStats(allData);

        // Split train/val
        Random rng = new Random(42);
        TrainingDataWriter.TrainingData[] split = allData.split(TRAIN_RATIO, rng);
        TrainingDataWriter.TrainingData trainData = split[0];
        TrainingDataWriter.TrainingData valData = split[1];
        System.out.println("Train: " + trainData.size() + ", Val: " + valData.size());

        // Create model
        Model model = Model.newInstance("quoridor-value-net");
        Block block = ValueNetwork.buildNetwork();
        model.setBlock(block);

        // Training config
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                .optOptimizer(Optimizer.adam()
                        .optLearningRateTracker(Tracker.fixed(INITIAL_LR))
                        .build())
                .addTrainingListeners(TrainingListener.Defaults.basic());

        try (Trainer trainer = model.newTrainer(config)) {
            // Initialize with input shape
            Shape inputShape = new Shape(BATCH_SIZE, BoardEncoder.CHANNELS,
                    BoardEncoder.BOARD_SIZE, BoardEncoder.BOARD_SIZE);
            trainer.initialize(inputShape);

            long totalParams = model.getBlock().getParameters().stream()
                    .mapToLong(p -> p.getValue().getArray().size())
                    .sum();
            System.out.println("Model parameters: " + totalParams);

            // Training loop
            float bestValLoss = Float.MAX_VALUE;
            int patienceCounter = 0;

            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                // Shuffle training data each epoch
                trainData.shuffle(rng);

                float epochTrainLoss = trainEpoch(trainer, trainData, epoch);
                float epochValLoss = evaluateValidation(trainer, valData);

                System.out.printf("Epoch %d/%d — Train Loss: %.6f, Val Loss: %.6f%n",
                        epoch + 1, EPOCHS, epochTrainLoss, epochValLoss);

                // Save checkpoint
                model.setProperty("Epoch", String.valueOf(epoch + 1));
                model.save(java.nio.file.Paths.get(modelDir), "checkpoint-epoch-" + (epoch + 1));

                // Early stopping check
                if (epochValLoss < bestValLoss) {
                    bestValLoss = epochValLoss;
                    patienceCounter = 0;
                    model.save(java.nio.file.Paths.get(modelDir), "best-model");
                    System.out.println("  >> New best model saved (val loss: " + bestValLoss + ")");
                } else {
                    patienceCounter++;
                    System.out.println("  >> No improvement (" + patienceCounter + "/" + PATIENCE + ")");
                    if (patienceCounter >= PATIENCE) {
                        System.out.println("Early stopping at epoch " + (epoch + 1));
                        break;
                    }
                }
            }

            System.out.println("\nTraining complete. Best val loss: " + bestValLoss);
            System.out.println("Best model saved to " + modelDir + "/best-model");
        }

        model.close();
    }

    private static float trainEpoch(Trainer trainer, TrainingDataWriter.TrainingData data,
                                     int epoch) throws TranslateException {
        NDManager manager = trainer.getManager();
        float totalLoss = 0;
        int batchCount = 0;

        for (int i = 0; i < data.size(); i += BATCH_SIZE) {
            int batchEnd = Math.min(i + BATCH_SIZE, data.size());
            int batchLen = batchEnd - i;

            try (NDManager subManager = manager.newSubManager()) {
                // Create batch tensors
                float[][][][] batchBoards = new float[batchLen][][][];
                float[][] batchLabels = new float[batchLen][1];
                for (int j = 0; j < batchLen; j++) {
                    batchBoards[j] = data.boards[i + j];
                    batchLabels[j][0] = data.labels[i + j];
                }

                NDArray input = subManager.create(batchBoards);
                NDArray label = subManager.create(batchLabels);

                try (ai.djl.training.GradientCollector gc = trainer.newGradientCollector()) {
                    NDArray pred = trainer.forward(new NDList(input)).singletonOrThrow();
                    NDArray loss = trainer.getLoss().evaluate(new NDList(label), new NDList(pred));
                    gc.backward(loss);
                    totalLoss += loss.getFloat();
                    batchCount++;
                }

                trainer.step();
            }

            // Progress report every 1000 batches
            if (batchCount % 1000 == 0) {
                System.out.printf("  Epoch %d, batch %d/%d, avg loss: %.6f%n",
                        epoch + 1, batchCount, data.size() / BATCH_SIZE, totalLoss / batchCount);
            }
        }

        return totalLoss / Math.max(1, batchCount);
    }

    private static float evaluateValidation(Trainer trainer, TrainingDataWriter.TrainingData data) {
        NDManager manager = trainer.getManager();
        float totalLoss = 0;
        int batchCount = 0;

        for (int i = 0; i < data.size(); i += BATCH_SIZE) {
            int batchEnd = Math.min(i + BATCH_SIZE, data.size());
            int batchLen = batchEnd - i;

            try (NDManager subManager = manager.newSubManager()) {
                float[][][][] batchBoards = new float[batchLen][][][];
                float[][] batchLabels = new float[batchLen][1];
                for (int j = 0; j < batchLen; j++) {
                    batchBoards[j] = data.boards[i + j];
                    batchLabels[j][0] = data.labels[i + j];
                }

                NDArray input = subManager.create(batchBoards);
                NDArray label = subManager.create(batchLabels);

                NDArray pred = trainer.evaluate(new NDList(input)).singletonOrThrow();
                NDArray loss = trainer.getLoss().evaluate(new NDList(label), new NDList(pred));
                totalLoss += loss.getFloat();
                batchCount++;
            }
        }

        return totalLoss / Math.max(1, batchCount);
    }

    private static void printLabelStats(TrainingDataWriter.TrainingData data) {
        float sum = 0, min = 1, max = 0;
        int[] histogram = new int[10]; // 0-0.1, 0.1-0.2, ..., 0.9-1.0

        for (float label : data.labels) {
            sum += label;
            min = Math.min(min, label);
            max = Math.max(max, label);
            int bin = Math.min(9, (int) (label * 10));
            histogram[bin]++;
        }

        System.out.printf("Labels — mean: %.3f, min: %.3f, max: %.3f%n",
                sum / data.size(), min, max);
        System.out.print("Distribution: ");
        for (int i = 0; i < 10; i++) {
            System.out.printf("[%.1f-%.1f]: %d  ", i * 0.1, (i + 1) * 0.1, histogram[i]);
        }
        System.out.println();
    }
}
