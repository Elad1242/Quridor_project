package ml.cnn;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.Batchifier;

import bot.BotBrain;
import bot.WallEvaluator;
import logic.MoveValidator;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * DJL (Deep Java Library) CNN Trainer for Quoridor
 *
 * Uses PyTorch backend - supports ALL GPUs including H100, A100, RTX 4090
 * Much faster than DeepLearning4J
 */
public class DJLTrainer {

    // ============== TRAINING PARAMETERS ==============
    private static final int IMITATION_GAMES = 50000;
    private static final int SELF_PLAY_ROUNDS = 20;
    private static final int GAMES_PER_ROUND = 5000;
    private static final int EPOCHS_INITIAL = 30;
    private static final int EPOCHS_PER_ROUND = 10;
    private static final int BATCH_SIZE = 256;
    private static final float LEARNING_RATE = 0.001f;
    private static final float EXPLORATION_RATE = 0.15f;
    private static final int MAX_WALLS_TO_EVAL = 50;
    private static final int TEST_GAMES = 200;

    // Network architecture
    private static final int CONV_FILTERS = 128;
    private static final int DENSE_UNITS = 256;

    private Model model;
    private NDManager manager;
    private Device device;
    private Random random = new Random(42);

    public static void main(String[] args) {
        WallEvaluator.silent = true;

        System.out.println("╔══════════════════════════════════════════════════════╗");
        System.out.println("║   QUORIDOR CNN - DJL PyTorch (ALL GPUs SUPPORTED)    ║");
        System.out.println("╚══════════════════════════════════════════════════════╝");

        new DJLTrainer().train();
    }

    public void train() {
        // Force GPU device
        if (Engine.getInstance().getGpuCount() > 0) {
            device = Device.gpu(0);
            System.out.println("  Using GPU: " + device);
        } else {
            device = Device.cpu();
            System.out.println("  WARNING: No GPU detected, using CPU (will be slow)");
        }

        // Create manager on specific device
        manager = NDManager.newBaseManager(device);

        printSystemInfo();

        // Phase 1: Imitation Learning
        System.out.println("\n▶ PHASE 1: IMITATION LEARNING");
        System.out.println("  Learning to copy BotBrain's moves...");

        List<float[]> states = new ArrayList<>();
        List<Float> labels = new ArrayList<>();
        generateImitationData(states, labels, IMITATION_GAMES);

        // Create network
        model = createModel();
        System.out.println("  Model created successfully");

        // Train on imitation data
        trainModel(states, labels, EPOCHS_INITIAL);

        // Test initial performance
        System.out.println("\n  ── Initial Performance ──");
        int initialWins = testAgainstBotBrain(TEST_GAMES);
        double initialRate = initialWins * 100.0 / TEST_GAMES;
        System.out.printf("  Win rate: %d/%d (%.1f%%)%n", initialWins, TEST_GAMES, initialRate);

        saveModel("quoridor_cnn_imitation");

        // Phase 2: Self-Play Reinforcement
        System.out.println("\n▶ PHASE 2: SELF-PLAY REINFORCEMENT");

        List<float[]> valueStates = new ArrayList<>();
        List<Float> valueLabels = new ArrayList<>();

        int bestWinRate = initialWins;

        for (int round = 1; round <= SELF_PLAY_ROUNDS; round++) {
            System.out.println("\n  ── Round " + round + "/" + SELF_PLAY_ROUNDS + " ──");

            List<float[]> roundStates = new ArrayList<>();
            List<Float> roundLabels = new ArrayList<>();
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

            trainModel(valueStates, valueLabels, EPOCHS_PER_ROUND);

            int wins = testAgainstBotBrain(TEST_GAMES);
            double winRate = wins * 100.0 / TEST_GAMES;
            System.out.printf("  Win rate: %d/%d (%.1f%%)%n", wins, TEST_GAMES, winRate);

            if (wins > bestWinRate) {
                bestWinRate = wins;
                saveModel("quoridor_cnn_best");
                System.out.println("  ★ New best model saved!");
            }

            if (round % 5 == 0) {
                saveModel("quoridor_cnn_checkpoint_r" + round);
            }

            if (winRate >= 75) {
                System.out.println("\n  ★★★ TARGET 75% ACHIEVED! ★★★");
                break;
            }
        }

        // Final evaluation
        System.out.println("\n▶ PHASE 3: FINAL EVALUATION");
        saveModel("quoridor_cnn_final");

        int finalWins = testAgainstBotBrain(500);
        double finalRate = finalWins * 100.0 / 500;

        System.out.println("\n╔══════════════════════════════════════════════════════╗");
        System.out.printf("║  FINAL WIN RATE: %d/500 (%.1f%%)                      ║%n", finalWins, finalRate);
        System.out.println("╚══════════════════════════════════════════════════════╝");

        manager.close();
    }

    private void printSystemInfo() {
        System.out.println("\n  System Information:");
        System.out.println("  ─────────────────────────────────────────");
        System.out.println("  Engine: " + Engine.getInstance().getEngineName());
        System.out.println("  Version: " + Engine.getInstance().getVersion());
        System.out.println("  GPU Count: " + Engine.getInstance().getGpuCount());
        if (Engine.getInstance().getGpuCount() > 0) {
            System.out.println("  GPU: Available ✓");
        } else {
            System.out.println("  GPU: Not available (using CPU)");
        }
        System.out.println("  ─────────────────────────────────────────");
        System.out.println("  Training Config:");
        System.out.println("    Imitation games: " + IMITATION_GAMES);
        System.out.println("    Self-play rounds: " + SELF_PLAY_ROUNDS);
        System.out.println("    Games per round: " + GAMES_PER_ROUND);
        System.out.println("    Batch size: " + BATCH_SIZE);
    }

    private Model createModel() {
        Model model = Model.newInstance("quoridor-cnn", device);

        SequentialBlock block = new SequentialBlock();

        // Conv block 1: 8 -> 128 filters
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .setFilters(CONV_FILTERS)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);

        // Conv block 2
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .setFilters(CONV_FILTERS)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);

        // Conv block 3
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .setFilters(CONV_FILTERS)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);

        // Global average pooling
        block.add(Pool.globalAvgPool2dBlock());

        // Flatten
        block.add(Blocks.batchFlattenBlock());

        // Dense layers
        block.add(Linear.builder().setUnits(DENSE_UNITS).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(DENSE_UNITS / 2).build());
        block.add(Activation::relu);

        // Output (sigmoid for binary classification)
        block.add(Linear.builder().setUnits(1).build());
        block.add(Activation::sigmoid);

        model.setBlock(block);
        return model;
    }

    private void generateImitationData(List<float[]> states, List<Float> labels, int numGames) {
        int progressInterval = numGames / 20;

        for (int g = 0; g < numGames; g++) {
            GameState state = new GameState();
            BotBrain bot1 = new BotBrain(0.05, 6);
            BotBrain bot2 = new BotBrain(0.05, 6);
            bot1.setSilent(true);
            bot2.setSilent(true);

            List<float[]> gameStates = new ArrayList<>();
            List<Integer> gamePlayers = new ArrayList<>();

            for (int turn = 0; turn < 200 && !state.isGameOver(); turn++) {
                Player me = state.getCurrentPlayer();
                int idx = state.getCurrentPlayerIndex();
                BotBrain bot = (idx == 0) ? bot1 : bot2;

                BotBrain.BotAction action = bot.computeBestAction(state);
                if (action == null) break;

                gameStates.add(encodeBoard(state));
                gamePlayers.add(idx);

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

            // Label based on outcome - clear 0.9/0.1 labels
            boolean p0Won = state.isGameOver() && state.getWinner() == state.getPlayers()[0];

            for (int i = 0; i < gameStates.size(); i++) {
                int player = gamePlayers.get(i);
                // Clear labels: winner's states = 0.9, loser's states = 0.1
                float label = (player == 0) == p0Won ? 0.9f : 0.1f;
                states.add(gameStates.get(i));
                labels.add(label);
            }

            if ((g + 1) % progressInterval == 0) {
                System.out.println("    Generated " + (g + 1) + "/" + numGames + " games (" + states.size() + " samples)");
            }
        }
    }

    private void trainModel(List<float[]> states, List<Float> labels, int epochs) {
        System.out.println("  Training on " + states.size() + " samples for " + epochs + " epochs...");

        try (NDManager trainManager = manager.newSubManager()) {
            // Convert to NDArrays
            int n = states.size();
            float[] flatStates = new float[n * 8 * 9 * 9];
            float[] flatLabels = new float[n];

            for (int i = 0; i < n; i++) {
                System.arraycopy(states.get(i), 0, flatStates, i * 8 * 9 * 9, 8 * 9 * 9);
                flatLabels[i] = labels.get(i);
            }

            NDArray xData = trainManager.create(flatStates, new Shape(n, 8, 9, 9));
            NDArray yData = trainManager.create(flatLabels, new Shape(n, 1));

            // Training config
            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.sigmoidBinaryCrossEntropyLoss())
                    .optOptimizer(Optimizer.adam()
                            .optLearningRateTracker(Tracker.fixed(LEARNING_RATE))
                            .build())
                    .addTrainingListeners(TrainingListener.Defaults.basic());

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(BATCH_SIZE, 8, 9, 9));

                for (int epoch = 1; epoch <= epochs; epoch++) {
                    // Shuffle indices
                    List<Integer> indices = new ArrayList<>();
                    for (int i = 0; i < n; i++) indices.add(i);
                    Collections.shuffle(indices, random);

                    float totalLoss = 0;
                    int batches = 0;

                    for (int batch = 0; batch < n; batch += BATCH_SIZE) {
                        int end = Math.min(batch + BATCH_SIZE, n);
                        int batchSize = end - batch;

                        float[] batchX = new float[batchSize * 8 * 9 * 9];
                        float[] batchY = new float[batchSize];

                        for (int i = 0; i < batchSize; i++) {
                            int idx = indices.get(batch + i);
                            System.arraycopy(states.get(idx), 0, batchX, i * 8 * 9 * 9, 8 * 9 * 9);
                            batchY[i] = labels.get(idx);
                        }

                        try (NDManager batchManager = trainManager.newSubManager()) {
                            NDArray x = batchManager.create(batchX, new Shape(batchSize, 8, 9, 9));
                            NDArray y = batchManager.create(batchY, new Shape(batchSize, 1));

                            try (ai.djl.training.GradientCollector gc = trainer.newGradientCollector()) {
                                NDArray pred = trainer.forward(new NDList(x)).singletonOrThrow();
                                NDArray loss = Loss.sigmoidBinaryCrossEntropyLoss().evaluate(new NDList(y), new NDList(pred));
                                gc.backward(loss);
                                totalLoss += loss.getFloat();
                                batches++;
                            }
                            trainer.step();
                        }
                    }

                    if (epoch % 5 == 0 || epoch == epochs) {
                        System.out.printf("    Epoch %d/%d - Loss: %.6f%n", epoch, epochs, totalLoss / batches);
                    }
                }
            }
        }
    }

    private void generateSelfPlayData(List<float[]> states, List<Float> labels, int numGames) {
        int progressInterval = Math.max(1, numGames / 10);

        for (int g = 0; g < numGames; g++) {
            GameState state = new GameState();

            List<float[]> p0States = new ArrayList<>();
            List<float[]> p1States = new ArrayList<>();

            for (int turn = 0; turn < 200 && !state.isGameOver(); turn++) {
                Player me = state.getCurrentPlayer();
                int idx = state.getCurrentPlayerIndex();

                float[] encoded = encodeBoard(state);
                if (idx == 0) p0States.add(encoded);
                else p1States.add(encoded);

                boolean explore = random.nextDouble() < EXPLORATION_RATE;
                Position bestMove = null;
                Wall bestWall = null;
                double bestScore = Double.NEGATIVE_INFINITY;
                boolean isMove = true;

                List<Position> validMoves = MoveValidator.getValidMoves(state, me);

                if (explore && !validMoves.isEmpty()) {
                    bestMove = validMoves.get(random.nextInt(validMoves.size()));
                } else {
                    for (Position move : validMoves) {
                        float[] after = encodeAfterMove(state, move);
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
                            float[] after = encodeAfterWall(state, wall);
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

            labelGameStates(states, labels, p0States, p1States, state);

            if ((g + 1) % progressInterval == 0) {
                System.out.println("    Self-play: " + (g + 1) + "/" + numGames);
            }
        }
    }

    private void labelGameStates(List<float[]> states, List<Float> labels,
                                  List<float[]> p0States, List<float[]> p1States,
                                  GameState finalState) {
        boolean p0Won = finalState.isGameOver() && finalState.getWinner() == finalState.getPlayers()[0];
        boolean draw = !finalState.isGameOver() || finalState.getWinner() == null;
        double discount = 0.97;

        for (int i = 0; i < p0States.size(); i++) {
            double weight = Math.pow(discount, p0States.size() - 1 - i);
            double baseLabel = draw ? 0.5 : (p0Won ? 0.8 : 0.2);
            double label = 0.5 + (baseLabel - 0.5) * weight;
            states.add(p0States.get(i));
            labels.add((float) Math.max(0.05, Math.min(0.95, label)));
        }

        for (int i = 0; i < p1States.size(); i++) {
            double weight = Math.pow(discount, p1States.size() - 1 - i);
            double baseLabel = draw ? 0.5 : (p0Won ? 0.2 : 0.8);
            double label = 0.5 + (baseLabel - 0.5) * weight;
            states.add(p1States.get(i));
            labels.add((float) Math.max(0.05, Math.min(0.95, label)));
        }
    }

    private double predict(float[] state) {
        try (NDManager predManager = manager.newSubManager()) {
            NDArray input = predManager.create(state, new Shape(1, 8, 9, 9));
            NDArray output = model.getBlock().forward(
                    new ai.djl.training.ParameterStore(predManager, false),
                    new NDList(input),
                    false
            ).singletonOrThrow();
            return output.getFloat(0);
        }
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
                    Position bestMove = null;
                    Wall bestWall = null;
                    double bestScore = Double.NEGATIVE_INFINITY;
                    boolean isMove = true;

                    for (Position move : MoveValidator.getValidMoves(state, me)) {
                        float[] after = encodeAfterMove(state, move);
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
                            float[] after = encodeAfterWall(state, wall);
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

    // Board encoding - 8 channels, 9x9 board
    private float[] encodeBoard(GameState state) {
        float[] encoded = new float[8 * 9 * 9];

        Player p0 = state.getPlayers()[0];
        Player p1 = state.getPlayers()[1];
        int current = state.getCurrentPlayerIndex();

        // Channel 0: Current player position
        Position myPos = (current == 0) ? p0.getPosition() : p1.getPosition();
        encoded[0 * 81 + myPos.getRow() * 9 + myPos.getCol()] = 1.0f;

        // Channel 1: Opponent position
        Position oppPos = (current == 0) ? p1.getPosition() : p0.getPosition();
        encoded[1 * 81 + oppPos.getRow() * 9 + oppPos.getCol()] = 1.0f;

        // Channel 2: My goal row
        int myGoal = (current == 0) ? 0 : 8;
        for (int c = 0; c < 9; c++) {
            encoded[2 * 81 + myGoal * 9 + c] = 1.0f;
        }

        // Channel 3: Opponent goal row
        int oppGoal = (current == 0) ? 8 : 0;
        for (int c = 0; c < 9; c++) {
            encoded[3 * 81 + oppGoal * 9 + c] = 1.0f;
        }

        // Channel 2: Horizontal walls
        // Channel 3: Vertical walls
        for (Wall wall : state.getWalls()) {
            Position pos = wall.getPosition();
            int r = pos.getRow();
            int c = pos.getCol();
            if (wall.isHorizontal()) {
                encoded[2 * 81 + r * 9 + c] = 1.0f;
                if (c + 1 < 9) {
                    encoded[2 * 81 + r * 9 + c + 1] = 1.0f;
                }
            } else {
                encoded[3 * 81 + r * 9 + c] = 1.0f;
                if (r + 1 < 9) {
                    encoded[3 * 81 + (r + 1) * 9 + c] = 1.0f;
                }
            }
        }

        // Channel 6: My walls remaining (normalized)
        int myWalls = (current == 0) ? p0.getWallsRemaining() : p1.getWallsRemaining();
        float myWallsNorm = myWalls / 10.0f;
        for (int i = 0; i < 81; i++) {
            encoded[6 * 81 + i] = myWallsNorm;
        }

        // Channel 7: Opponent walls remaining (normalized)
        int oppWalls = (current == 0) ? p1.getWallsRemaining() : p0.getWallsRemaining();
        float oppWallsNorm = oppWalls / 10.0f;
        for (int i = 0; i < 81; i++) {
            encoded[7 * 81 + i] = oppWallsNorm;
        }

        return encoded;
    }

    private float[] encodeAfterMove(GameState state, Position move) {
        float[] encoded = new float[8 * 9 * 9];

        Player current = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();

        // Channel 0: Current player's NEW position (after move)
        encoded[0 * 81 + move.getRow() * 9 + move.getCol()] = 1.0f;

        // Channel 1: Opponent stays same
        Position oppPos = opponent.getPosition();
        encoded[1 * 81 + oppPos.getRow() * 9 + oppPos.getCol()] = 1.0f;

        // Channel 2: Horizontal walls
        // Channel 3: Vertical walls
        for (Wall wall : state.getWalls()) {
            encodeWall(encoded, wall);
        }

        // Channel 4: Current player's goal row
        int myGoal = current.getGoalRow();
        for (int c = 0; c < 9; c++) {
            encoded[4 * 81 + myGoal * 9 + c] = 1.0f;
        }

        // Channel 5: Opponent's goal row
        int oppGoal = opponent.getGoalRow();
        for (int c = 0; c < 9; c++) {
            encoded[5 * 81 + oppGoal * 9 + c] = 1.0f;
        }

        // Channel 6: My walls remaining (unchanged for move)
        float myWallsNorm = current.getWallsRemaining() / 10.0f;
        for (int i = 0; i < 81; i++) {
            encoded[6 * 81 + i] = myWallsNorm;
        }

        // Channel 7: Opponent walls remaining
        float oppWallsNorm = opponent.getWallsRemaining() / 10.0f;
        for (int i = 0; i < 81; i++) {
            encoded[7 * 81 + i] = oppWallsNorm;
        }

        return encoded;
    }

    private float[] encodeAfterWall(GameState state, Wall wall) {
        float[] encoded = new float[8 * 9 * 9];

        Player current = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();

        // Channel 0: Current player position unchanged
        Position myPos = current.getPosition();
        encoded[0 * 81 + myPos.getRow() * 9 + myPos.getCol()] = 1.0f;

        // Channel 1: Opponent position unchanged
        Position oppPos = opponent.getPosition();
        encoded[1 * 81 + oppPos.getRow() * 9 + oppPos.getCol()] = 1.0f;

        // Channel 2 & 3: Existing walls
        for (Wall w : state.getWalls()) {
            encodeWall(encoded, w);
        }
        // Add the new wall
        encodeWall(encoded, wall);

        // Channel 4: Current player's goal row
        int myGoal = current.getGoalRow();
        for (int c = 0; c < 9; c++) {
            encoded[4 * 81 + myGoal * 9 + c] = 1.0f;
        }

        // Channel 5: Opponent's goal row
        int oppGoal = opponent.getGoalRow();
        for (int c = 0; c < 9; c++) {
            encoded[5 * 81 + oppGoal * 9 + c] = 1.0f;
        }

        // Channel 6: My walls remaining (decremented by 1)
        float myWallsNorm = Math.max(0, current.getWallsRemaining() - 1) / 10.0f;
        for (int i = 0; i < 81; i++) {
            encoded[6 * 81 + i] = myWallsNorm;
        }

        // Channel 7: Opponent walls remaining
        float oppWallsNorm = opponent.getWallsRemaining() / 10.0f;
        for (int i = 0; i < 81; i++) {
            encoded[7 * 81 + i] = oppWallsNorm;
        }

        return encoded;
    }

    private void encodeWall(float[] encoded, Wall wall) {
        Position pos = wall.getPosition();
        int r = pos.getRow();
        int c = pos.getCol();

        if (wall.isHorizontal()) {
            if (r < 9 && c < 9) encoded[2 * 81 + r * 9 + c] = 1.0f;
            if (r < 9 && c + 1 < 9) encoded[2 * 81 + r * 9 + c + 1] = 1.0f;
        } else {
            if (r < 9 && c < 9) encoded[3 * 81 + r * 9 + c] = 1.0f;
            if (r + 1 < 9 && c < 9) encoded[3 * 81 + (r + 1) * 9 + c] = 1.0f;
        }
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

    private void saveModel(String name) {
        try {
            Path modelDir = Paths.get(name);
            Files.createDirectories(modelDir);
            model.save(modelDir, name);
            System.out.println("  Model saved: " + name);
        } catch (IOException e) {
            System.err.println("  Failed to save model: " + e.getMessage());
        }
    }
}
