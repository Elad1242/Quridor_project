package ml.cnn;

import bot.BotBrain;
import bot.WallEvaluator;
import logic.MoveValidator;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.util.*;

/**
 * Cloud-Optimized Trainer for Quoridor CNN
 *
 * Designed for GPU cloud instances (RunPod, Vast.ai, Lambda Labs)
 * Uses IMITATION LEARNING + SELF-PLAY REINFORCEMENT
 *
 * Target: 75% win rate against BotBrain with PURE ML (no heuristics)
 */
public class CloudTrainer {

    // ============== CLOUD-SCALE PARAMETERS ==============
    // Adjust these based on your cloud instance

    private static final int IMITATION_GAMES = 20000;      // Learn from BotBrain (faster)
    private static final int SELF_PLAY_ROUNDS = 10;        // Self-improvement rounds (faster)
    private static final int GAMES_PER_ROUND = 3000;       // Games per self-play round
    private static final int EPOCHS_INITIAL = 30;          // Initial training epochs
    private static final int EPOCHS_PER_ROUND = 10;        // Epochs per round
    private static final int BATCH_SIZE = 512;             // Larger batch for GPU utilization
    private static final double LEARNING_RATE = 0.001;
    private static final double EXPLORATION_RATE = 0.15;
    private static final int MAX_WALLS_TO_EVAL = 50;       // More wall evaluation
    private static final int TEST_GAMES = 200;             // Games for win rate test

    // Larger network for more capacity
    private static final int CONV_FILTERS = 128;           // More filters
    private static final int DENSE_UNITS = 256;            // Larger dense layer

    private MultiLayerNetwork model;
    private Random random = new Random(42);

    public static void main(String[] args) {
        WallEvaluator.silent = true;

        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║     QUORIDOR CNN - CLOUD TRAINING (PURE ML)          ║");
        System.out.println("╚══════════════════════════════════════════════════════╝");

        new CloudTrainer().train();
    }

    public void train() {
        // Print system info
        printSystemInfo();

        // Phase 1: Imitation Learning
        System.out.println("\n▶ PHASE 1: IMITATION LEARNING");
        System.out.println("  Learning to copy BotBrain's moves...");

        List<INDArray> states = new ArrayList<>();
        List<INDArray> moveLabels = new ArrayList<>();
        generateImitationData(states, moveLabels, IMITATION_GAMES);

        // Create larger network
        model = createNetwork();
        System.out.println("  Network parameters: " + model.numParams());

        // Train on imitation data
        trainImitation(states, moveLabels, EPOCHS_INITIAL);

        // Test initial performance
        System.out.println("\n  ── Initial Performance ──");
        int initialWins = testAgainstBotBrain(TEST_GAMES);
        double initialRate = initialWins * 100.0 / TEST_GAMES;
        System.out.printf("  Win rate: %d/%d (%.1f%%)%n", initialWins, TEST_GAMES, initialRate);

        // Save initial model
        saveModel("quoridor_cnn_imitation.zip");

        // Phase 2: Self-Play Reinforcement
        System.out.println("\n▶ PHASE 2: SELF-PLAY REINFORCEMENT");

        List<INDArray> valueStates = new ArrayList<>();
        List<Double> valueLabels = new ArrayList<>();

        int bestWinRate = initialWins;

        for (int round = 1; round <= SELF_PLAY_ROUNDS; round++) {
            System.out.println("\n  ── Round " + round + "/" + SELF_PLAY_ROUNDS + " ──");

            // Generate self-play data
            List<INDArray> roundStates = new ArrayList<>();
            List<Double> roundLabels = new ArrayList<>();
            generateSelfPlayData(roundStates, roundLabels, GAMES_PER_ROUND);

            valueStates.addAll(roundStates);
            valueLabels.addAll(roundLabels);

            // Limit memory
            int maxSamples = 500000;
            if (valueStates.size() > maxSamples) {
                int toRemove = valueStates.size() - maxSamples;
                valueStates.subList(0, toRemove).clear();
                valueLabels.subList(0, toRemove).clear();
            }

            System.out.println("  Total samples: " + valueStates.size());

            // Train on value prediction
            trainValue(valueStates, valueLabels, EPOCHS_PER_ROUND);

            // Test performance
            int wins = testAgainstBotBrain(TEST_GAMES);
            double winRate = wins * 100.0 / TEST_GAMES;
            System.out.printf("  Win rate: %d/%d (%.1f%%)%n", wins, TEST_GAMES, winRate);

            // Save if improved
            if (wins > bestWinRate) {
                bestWinRate = wins;
                saveModel("quoridor_cnn_best.zip");
                System.out.println("  ★ New best model saved!");
            }

            // Save checkpoint every 5 rounds
            if (round % 5 == 0) {
                saveModel("quoridor_cnn_checkpoint_r" + round + ".zip");
            }

            // Early success
            if (winRate >= 75) {
                System.out.println("\n  ★★★ TARGET 75% ACHIEVED! ★★★");
                break;
            }
        }

        // Final evaluation
        System.out.println("\n▶ PHASE 3: FINAL EVALUATION");
        saveModel("quoridor_cnn_final.zip");

        int finalWins = testAgainstBotBrain(500);
        double finalRate = finalWins * 100.0 / 500;

        System.out.println("\n╔══════════════════════════════════════════════════════╗");
        System.out.printf("║  FINAL WIN RATE: %d/500 (%.1f%%)                      ║%n", finalWins, finalRate);
        System.out.println("╚══════════════════════════════════════════════════════╝");

        if (finalRate >= 75) {
            System.out.println("SUCCESS: Target achieved!");
        } else if (finalRate >= 50) {
            System.out.println("GOOD: Better than random. More training may help.");
        } else {
            System.out.println("NEEDS MORE: Consider increasing IMITATION_GAMES or SELF_PLAY_ROUNDS.");
        }
    }

    private void printSystemInfo() {
        System.out.println("\n  System Information:");
        System.out.println("  ─────────────────────────────────────");
        System.out.println("  Backend: " + Nd4j.getBackend().getClass().getSimpleName());
        System.out.println("  Device: " + Nd4j.getAffinityManager().getDeviceForCurrentThread());

        try {
            long[] memInfo = Nd4j.getMemoryManager().getCurrentWorkspace() != null ?
                new long[]{0, 0} : new long[]{
                    Runtime.getRuntime().maxMemory() / (1024*1024),
                    Runtime.getRuntime().totalMemory() / (1024*1024)
                };
            System.out.println("  Max Memory: " + Runtime.getRuntime().maxMemory() / (1024*1024) + " MB");
        } catch (Exception e) {
            // Ignore
        }

        System.out.println("  ─────────────────────────────────────");
        System.out.println("  Training Config:");
        System.out.println("    Imitation games: " + IMITATION_GAMES);
        System.out.println("    Self-play rounds: " + SELF_PLAY_ROUNDS);
        System.out.println("    Games per round: " + GAMES_PER_ROUND);
        System.out.println("    Batch size: " + BATCH_SIZE);
    }

    private MultiLayerNetwork createNetwork() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(42)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(LEARNING_RATE))
            .list()
            // Conv block 1
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nIn(8).nOut(CONV_FILTERS)
                .activation(Activation.RELU)
                .build())
            .layer(new BatchNormalization())
            // Conv block 2
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(CONV_FILTERS)
                .activation(Activation.RELU)
                .build())
            .layer(new BatchNormalization())
            // Conv block 3
            .layer(new ConvolutionLayer.Builder(3, 3)
                .nOut(CONV_FILTERS)
                .activation(Activation.RELU)
                .build())
            .layer(new BatchNormalization())
            // Global pooling
            .layer(new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
            // Dense layers
            .layer(new DenseLayer.Builder().nOut(DENSE_UNITS)
                .activation(Activation.RELU).build())
            .layer(new DenseLayer.Builder().nOut(DENSE_UNITS / 2)
                .activation(Activation.RELU).build())
            // Output
            .layer(new OutputLayer.Builder(LossFunction.XENT)
                .nOut(1).activation(Activation.SIGMOID).build())
            .setInputType(org.deeplearning4j.nn.conf.inputs.InputType.convolutional(9, 9, 8))
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1000));
        return net;
    }

    /**
     * Generate imitation learning data from BotBrain games - PARALLEL VERSION.
     * Uses all CPU cores for maximum speed.
     */
    private void generateImitationData(List<INDArray> states, List<INDArray> labels, int numGames) {
        int numThreads = Runtime.getRuntime().availableProcessors();
        System.out.println("    Using " + numThreads + " threads for parallel game generation");

        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(numThreads);
        java.util.concurrent.ConcurrentLinkedQueue<GameResult> results = new java.util.concurrent.ConcurrentLinkedQueue<>();
        java.util.concurrent.atomic.AtomicInteger completed = new java.util.concurrent.atomic.AtomicInteger(0);

        // Submit all games to thread pool
        for (int g = 0; g < numGames; g++) {
            executor.submit(() -> {
                GameResult result = playOneGame();
                results.add(result);
                int done = completed.incrementAndGet();
                if (done % 2500 == 0) {
                    System.out.println("    Generated " + done + "/" + numGames + " games");
                }
            });
        }

        // Wait for all to complete
        executor.shutdown();
        try {
            executor.awaitTermination(2, java.util.concurrent.TimeUnit.HOURS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // Collect results
        for (GameResult result : results) {
            states.addAll(result.states);
            labels.addAll(result.labels);
        }
        System.out.println("    Total samples: " + states.size());
    }

    private static class GameResult {
        List<INDArray> states = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();
    }

    private GameResult playOneGame() {
        GameResult result = new GameResult();
        List<INDArray> gameStates = new ArrayList<>();

        GameState state = new GameState();
        BotBrain bot1 = new BotBrain(0.05, 6);
        BotBrain bot2 = new BotBrain(0.05, 6);
        bot1.setSilent(true);
        bot2.setSilent(true);

        for (int turn = 0; turn < 200 && !state.isGameOver(); turn++) {
            Player me = state.getCurrentPlayer();
            int idx = state.getCurrentPlayerIndex();
            BotBrain bot = (idx == 0) ? bot1 : bot2;

            BotBrain.BotAction action = bot.computeBestAction(state);
            if (action == null) break;

            gameStates.add(BoardEncoder.encode(state));

            if (action.type == BotBrain.BotAction.Type.MOVE) {
                me.setPosition(action.moveTarget);
            } else {
                action.wallToPlace.setOwnerIndex(idx);
                state.addWall(action.wallToPlace);
                me.setWallsRemaining(me.getWallsRemaining() - 1);
            }

            state.checkWinCondition();
            if (!state.isGameOver()) state.nextTurn();
        }

        // Label based on outcome
        boolean p0Won = state.isGameOver() && state.getWinner() == state.getPlayers()[0];
        for (int i = 0; i < gameStates.size(); i++) {
            boolean wasP0Turn = (i % 2 == 0);
            double label = (wasP0Turn == p0Won) ? 0.9 : 0.1;
            result.states.add(gameStates.get(i));
            result.labels.add(Nd4j.create(new double[]{label}));
        }

        return result;
    }

    private void trainImitation(List<INDArray> states, List<INDArray> labels, int epochs) {
        System.out.println("  Training on " + states.size() + " samples for " + epochs + " epochs...");

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < states.size(); i++) indices.add(i);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            Collections.shuffle(indices, random);

            for (int batch = 0; batch < indices.size(); batch += BATCH_SIZE) {
                int end = Math.min(batch + BATCH_SIZE, indices.size());
                int batchSize = end - batch;

                INDArray[] batchStates = new INDArray[batchSize];
                INDArray[] batchLabels = new INDArray[batchSize];

                for (int i = 0; i < batchSize; i++) {
                    int idx = indices.get(batch + i);
                    batchStates[i] = states.get(idx);
                    batchLabels[i] = labels.get(idx);
                }

                INDArray input = Nd4j.stack(0, batchStates);
                INDArray output = Nd4j.vstack(batchLabels);

                model.fit(new DataSet(input, output));
            }

            if (epoch % 5 == 0 || epoch == epochs) {
                System.out.println("    Epoch " + epoch + "/" + epochs + " complete");
            }
        }
    }

    /**
     * Generate self-play data - PARALLEL VERSION.
     * Uses all CPU cores. Model.output() is thread-safe for inference.
     */
    private void generateSelfPlayData(List<INDArray> states, List<Double> labels, int numGames) {
        int numThreads = Runtime.getRuntime().availableProcessors();
        System.out.println("    Using " + numThreads + " threads for parallel self-play");

        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(numThreads);
        java.util.concurrent.ConcurrentLinkedQueue<SelfPlayResult> results = new java.util.concurrent.ConcurrentLinkedQueue<>();
        java.util.concurrent.atomic.AtomicInteger completed = new java.util.concurrent.atomic.AtomicInteger(0);

        for (int g = 0; g < numGames; g++) {
            executor.submit(() -> {
                SelfPlayResult result = playOneSelfPlayGame();
                results.add(result);
                int done = completed.incrementAndGet();
                if (done % 1000 == 0) {
                    System.out.println("    Self-play: " + done + "/" + numGames);
                }
            });
        }

        executor.shutdown();
        try {
            executor.awaitTermination(2, java.util.concurrent.TimeUnit.HOURS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // Collect results
        for (SelfPlayResult result : results) {
            states.addAll(result.states);
            labels.addAll(result.labels);
        }
    }

    private static class SelfPlayResult {
        List<INDArray> states = new ArrayList<>();
        List<Double> labels = new ArrayList<>();
    }

    private SelfPlayResult playOneSelfPlayGame() {
        SelfPlayResult result = new SelfPlayResult();
        java.util.concurrent.ThreadLocalRandom rnd = java.util.concurrent.ThreadLocalRandom.current();

        GameState state = new GameState();
        List<INDArray> p0States = new ArrayList<>();
        List<INDArray> p1States = new ArrayList<>();

        for (int turn = 0; turn < 200 && !state.isGameOver(); turn++) {
            Player me = state.getCurrentPlayer();
            int idx = state.getCurrentPlayerIndex();

            INDArray encoded = BoardEncoder.encode(state);
            if (idx == 0) p0States.add(encoded);
            else p1States.add(encoded);

            // CNN plays with exploration
            boolean explore = rnd.nextDouble() < EXPLORATION_RATE;
            Position bestMove = null;
            Wall bestWall = null;
            double bestScore = Double.NEGATIVE_INFINITY;
            boolean isMove = true;

            List<Position> validMoves = MoveValidator.getValidMoves(state, me);

            if (explore && !validMoves.isEmpty()) {
                bestMove = validMoves.get(rnd.nextInt(validMoves.size()));
            } else {
                for (Position move : validMoves) {
                    INDArray after = BoardEncoder.encodeAfterMove(state, move);
                    // After our move, it's opponent's turn - minimize their win prob
                    double score = 1.0 - predict(after);
                    if (score > bestScore) {
                        bestScore = score;
                        bestMove = move;
                        isMove = true;
                    }
                }

                if (me.getWallsRemaining() > 0) {
                    List<Wall> walls = getValidWalls(state);
                    if (walls.size() > MAX_WALLS_TO_EVAL) {
                        Collections.shuffle(walls, rnd);
                        walls = walls.subList(0, MAX_WALLS_TO_EVAL);
                    }
                    for (Wall wall : walls) {
                        INDArray after = BoardEncoder.encodeAfterWall(state, wall);
                        // After our wall, it's opponent's turn - minimize their win prob
                        double score = 1.0 - predict(after);
                        if (score > bestScore) {
                            bestScore = score;
                            bestWall = wall;
                            isMove = false;
                        }
                    }
                }
            }

            if (isMove && bestMove != null) {
                me.setPosition(bestMove);
            } else if (bestWall != null) {
                bestWall.setOwnerIndex(idx);
                state.addWall(bestWall);
                me.setWallsRemaining(me.getWallsRemaining() - 1);
            } else if (bestMove != null) {
                me.setPosition(bestMove);
            }

            state.checkWinCondition();
            if (!state.isGameOver()) state.nextTurn();
        }

        // Label based on outcome
        boolean p0Won = state.isGameOver() && state.getWinner() == state.getPlayers()[0];
        boolean draw = !state.isGameOver() || state.getWinner() == null;
        double discount = 0.97;

        int n0 = p0States.size();
        for (int i = 0; i < n0; i++) {
            double weight = Math.pow(discount, n0 - 1 - i);
            double baseLabel = draw ? 0.5 : (p0Won ? 0.8 : 0.2);
            double label = 0.5 + (baseLabel - 0.5) * weight;
            result.states.add(p0States.get(i));
            result.labels.add(Math.max(0.05, Math.min(0.95, label)));
        }

        int n1 = p1States.size();
        for (int i = 0; i < n1; i++) {
            double weight = Math.pow(discount, n1 - 1 - i);
            double baseLabel = draw ? 0.5 : (p0Won ? 0.2 : 0.8);
            double label = 0.5 + (baseLabel - 0.5) * weight;
            result.states.add(p1States.get(i));
            result.labels.add(Math.max(0.05, Math.min(0.95, label)));
        }

        return result;
    }

    private void trainValue(List<INDArray> states, List<Double> labels, int epochs) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < states.size(); i++) indices.add(i);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            Collections.shuffle(indices, random);

            for (int batch = 0; batch < indices.size(); batch += BATCH_SIZE) {
                int end = Math.min(batch + BATCH_SIZE, indices.size());
                int batchSize = end - batch;

                INDArray[] batchStates = new INDArray[batchSize];
                double[] batchLabels = new double[batchSize];

                for (int i = 0; i < batchSize; i++) {
                    int idx = indices.get(batch + i);
                    batchStates[i] = states.get(idx);
                    batchLabels[i] = labels.get(idx);
                }

                INDArray input = Nd4j.stack(0, batchStates);
                INDArray output = Nd4j.create(batchLabels).reshape(batchSize, 1);

                model.fit(new DataSet(input, output));
            }

            if (epoch % 2 == 0 || epoch == epochs) {
                System.out.println("    Epoch " + epoch + "/" + epochs);
            }
        }
    }

    private double predict(INDArray state) {
        INDArray input = state.reshape(1, 8, 9, 9);
        return model.output(input).getDouble(0);
    }

    private int testAgainstBotBrain(int games) {
        int wins = 0;

        for (int g = 0; g < games; g++) {
            GameState state = new GameState();
            BotBrain bot = new BotBrain(0.0, 3);
            bot.setSilent(true);

            int cnnIdx = g % 2;

            for (int turn = 0; turn < 200 && !state.isGameOver(); turn++) {
                Player me = state.getCurrentPlayer();
                int idx = state.getCurrentPlayerIndex();

                if (idx == cnnIdx) {
                    // CNN plays
                    Position bestMove = null;
                    Wall bestWall = null;
                    double bestScore = Double.NEGATIVE_INFINITY;
                    boolean isMove = true;

                    for (Position move : MoveValidator.getValidMoves(state, me)) {
                        INDArray after = BoardEncoder.encodeAfterMove(state, move);
                        // After our move, it's opponent's turn - minimize their win prob
                        double score = 1.0 - predict(after);
                        if (score > bestScore) {
                            bestScore = score;
                            bestMove = move;
                            isMove = true;
                        }
                    }

                    if (me.getWallsRemaining() > 0) {
                        List<Wall> walls = getValidWalls(state);
                        if (walls.size() > MAX_WALLS_TO_EVAL) {
                            Collections.shuffle(walls, random);
                            walls = walls.subList(0, MAX_WALLS_TO_EVAL);
                        }
                        for (Wall wall : walls) {
                            INDArray after = BoardEncoder.encodeAfterWall(state, wall);
                            // After our wall, it's opponent's turn - minimize their win prob
                            double score = 1.0 - predict(after);
                            if (score > bestScore) {
                                bestScore = score;
                                bestWall = wall;
                                isMove = false;
                            }
                        }
                    }

                    if (isMove && bestMove != null) {
                        me.setPosition(bestMove);
                    } else if (bestWall != null) {
                        bestWall.setOwnerIndex(cnnIdx);
                        state.addWall(bestWall);
                        me.setWallsRemaining(me.getWallsRemaining() - 1);
                    } else if (bestMove != null) {
                        me.setPosition(bestMove);
                    }
                } else {
                    // BotBrain plays
                    BotBrain.BotAction action = bot.computeBestAction(state);
                    if (action == null) break;

                    if (action.type == BotBrain.BotAction.Type.MOVE) {
                        me.setPosition(action.moveTarget);
                    } else {
                        action.wallToPlace.setOwnerIndex(idx);
                        state.addWall(action.wallToPlace);
                        me.setWallsRemaining(me.getWallsRemaining() - 1);
                    }
                }

                state.checkWinCondition();
                if (!state.isGameOver()) state.nextTurn();
            }

            if (state.isGameOver() && state.getWinner() == state.getPlayers()[cnnIdx]) {
                wins++;
            }

            if ((g + 1) % 50 == 0) {
                System.out.println("    Test: " + (g + 1) + "/" + games + " - Wins: " + wins);
            }
        }

        return wins;
    }

    private List<Wall> getValidWalls(GameState state) {
        List<Wall> walls = new ArrayList<>();
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                Wall h = new Wall(new Position(r, c), Wall.Orientation.HORIZONTAL);
                if (WallValidator.isValidWallPlacement(state, h)) walls.add(h);
                Wall v = new Wall(new Position(r, c), Wall.Orientation.VERTICAL);
                if (WallValidator.isValidWallPlacement(state, v)) walls.add(v);
            }
        }
        return walls;
    }

    private void saveModel(String filename) {
        try {
            File file = new File(filename);
            model.save(file);
            System.out.println("  Model saved: " + filename);
        } catch (Exception e) {
            System.err.println("  Failed to save model: " + e.getMessage());
        }
    }
}
