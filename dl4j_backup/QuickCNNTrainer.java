package ml.cnn;

import bot.BotBrain;
import bot.WallEvaluator;
import logic.MoveValidator;
import logic.PathFinder;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Quick CNN Trainer for Quoridor - Minimal version for fast training.
 *
 * PURE ML APPROACH: The CNN learns to evaluate board positions
 * without any heuristic assistance.
 */
public class QuickCNNTrainer {

    // Parameters for effective training
    private static final int SIMULATION_GAMES = 500;
    private static final int EPOCHS = 15;
    private static final int BATCH_SIZE = 64;
    private static final double TRAIN_RATIO = 0.85;

    public static void main(String[] args) {
        WallEvaluator.silent = true;
        String outputPath = "quoridor_cnn.zip";

        System.out.println("=== Quick CNN Trainer for Quoridor ===");
        System.out.println("=== PURE ML - No Heuristics ===");
        System.out.println("Games: " + SIMULATION_GAMES + ", Epochs: " + EPOCHS);

        // Step 1: Generate minimal training data
        System.out.println("\n[Step 1] Generating training data...");
        List<INDArray> boardStates = new ArrayList<>();
        List<Double> labels = new ArrayList<>();
        generateTrainingData(boardStates, labels, SIMULATION_GAMES);
        System.out.println("Total samples: " + boardStates.size());

        // Step 2: Create and train CNN
        System.out.println("\n[Step 2] Training CNN...");
        QuoridorCNN cnn = new QuoridorCNN();
        System.out.println("Parameters: " + cnn.getParameterCount());

        trainCNN(cnn, boardStates, labels);

        // Step 3: Save model
        System.out.println("\n[Step 3] Saving model...");
        cnn.save(outputPath);

        // Step 4: Quick test
        System.out.println("\n[Step 4] Testing (PURE ML)...");
        int wins = testAgainstBotBrain(cnn, 50);
        double winRate = wins * 100.0 / 50;
        System.out.printf("Win rate: %d/50 (%.1f%%)%n", wins, winRate);

        if (winRate >= 75) {
            System.out.println("SUCCESS: Target win rate achieved!");
        } else if (winRate >= 50) {
            System.out.println("DECENT: Better than random.");
        } else {
            System.out.println("NOTE: More training data may improve results.");
        }

        System.out.println("\n=== Training Complete ===");
    }

    private static void generateTrainingData(List<INDArray> boardStates, List<Double> labels, int numGames) {
        for (int game = 0; game < numGames; game++) {
            GameState state = new GameState();
            BotBrain bot1 = new BotBrain(0.1, 5);
            BotBrain bot2 = new BotBrain(0.1, 5);
            bot1.setSilent(true);
            bot2.setSilent(true);

            List<INDArray> p0States = new ArrayList<>();
            List<INDArray> p1States = new ArrayList<>();

            for (int turn = 0; turn < 200; turn++) {
                if (state.isGameOver()) break;

                Player me = state.getCurrentPlayer();
                int playerIdx = state.getCurrentPlayerIndex();

                INDArray boardState = BoardEncoder.encode(state);
                if (playerIdx == 0) p0States.add(boardState);
                else p1States.add(boardState);

                BotBrain bot = (playerIdx == 0) ? bot1 : bot2;
                BotBrain.BotAction action = bot.computeBestAction(state);
                if (action == null) break;

                if (action.type == BotBrain.BotAction.Type.MOVE) {
                    me.setPosition(action.moveTarget);
                } else {
                    action.wallToPlace.setOwnerIndex(playerIdx);
                    state.addWall(action.wallToPlace);
                }

                state.checkWinCondition();
                if (!state.isGameOver()) state.nextTurn();
            }

            // Label based on winner with temporal weighting
            boolean p0Won = state.isGameOver() && state.getWinner() == state.getPlayers()[0];
            int n0 = p0States.size();
            int n1 = p1States.size();

            // Temporal weighting: later states matter more
            for (int i = 0; i < n0; i++) {
                double temporalWeight = 0.5 + 0.5 * (i / (double) Math.max(1, n0 - 1));
                double label = p0Won ? (0.5 + 0.4 * temporalWeight) : (0.5 - 0.4 * temporalWeight);
                boardStates.add(p0States.get(i));
                labels.add(label);
            }
            for (int i = 0; i < n1; i++) {
                double temporalWeight = 0.5 + 0.5 * (i / (double) Math.max(1, n1 - 1));
                double label = p0Won ? (0.5 - 0.4 * temporalWeight) : (0.5 + 0.4 * temporalWeight);
                boardStates.add(p1States.get(i));
                labels.add(label);
            }

            if ((game + 1) % 50 == 0) {
                System.out.println("  Generated " + (game + 1) + "/" + numGames + " games");
            }
        }
    }

    private static void trainCNN(QuoridorCNN cnn, List<INDArray> boardStates, List<Double> labelList) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < boardStates.size(); i++) indices.add(i);
        Collections.shuffle(indices, new Random(42));

        int splitPoint = (int) (indices.size() * TRAIN_RATIO);
        List<Integer> trainIndices = new ArrayList<>(indices.subList(0, splitPoint));

        cnn.getNetwork().setListeners(new ScoreIterationListener(200));

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            Collections.shuffle(trainIndices);

            for (int batch = 0; batch < trainIndices.size(); batch += BATCH_SIZE) {
                int end = Math.min(batch + BATCH_SIZE, trainIndices.size());
                int batchSize = end - batch;

                INDArray[] batchStates = new INDArray[batchSize];
                double[] batchLabelArr = new double[batchSize];

                for (int i = 0; i < batchSize; i++) {
                    int idx = trainIndices.get(batch + i);
                    batchStates[i] = boardStates.get(idx);
                    batchLabelArr[i] = labelList.get(idx);
                }

                INDArray batchInput = Nd4j.stack(0, batchStates);
                INDArray batchLabels = Nd4j.create(batchLabelArr).reshape(batchSize, 1);

                DataSet ds = new DataSet(batchInput, batchLabels);
                cnn.getNetwork().fit(ds);
            }

            System.out.println("  Epoch " + epoch + "/" + EPOCHS + " complete");
        }
    }

    private static int testAgainstBotBrain(QuoridorCNN cnn, int games) {
        int wins = 0;

        for (int g = 0; g < games; g++) {
            GameState state = new GameState();
            BotBrain bot = new BotBrain(0.0, 3);
            bot.setSilent(true);

            int cnnIdx = g % 2;

            for (int turn = 0; turn < 200; turn++) {
                if (state.isGameOver()) break;

                Player me = state.getCurrentPlayer();
                boolean isCNNTurn = (state.getCurrentPlayerIndex() == cnnIdx);

                if (isCNNTurn) {
                    // CNN plays - PURE ML (evaluate moves and some walls)
                    Position bestMove = null;
                    Wall bestWall = null;
                    double bestScore = Double.NEGATIVE_INFINITY;
                    boolean bestIsMove = true;

                    // Evaluate moves
                    List<Position> validMoves = MoveValidator.getValidMoves(state, me);
                    for (Position move : validMoves) {
                        INDArray stateAfter = BoardEncoder.encodeAfterMove(state, move);
                        double score = cnn.predict(stateAfter);
                        if (score > bestScore) {
                            bestScore = score;
                            bestMove = move;
                            bestIsMove = true;
                        }
                    }

                    // Evaluate some walls if available
                    if (me.getWallsRemaining() > 0) {
                        List<Wall> walls = getValidWalls(state);
                        // Sample 20 walls for speed
                        if (walls.size() > 20) {
                            Collections.shuffle(walls);
                            walls = walls.subList(0, 20);
                        }
                        for (Wall wall : walls) {
                            INDArray stateAfter = BoardEncoder.encodeAfterWall(state, wall);
                            double score = cnn.predict(stateAfter);
                            if (score > bestScore) {
                                bestScore = score;
                                bestWall = wall;
                                bestIsMove = false;
                            }
                        }
                    }

                    // Apply best action
                    if (bestIsMove && bestMove != null) {
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
                        action.wallToPlace.setOwnerIndex(state.getCurrentPlayerIndex());
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

            if ((g + 1) % 10 == 0) {
                System.out.println("    Game " + (g + 1) + "/" + games + ": " + wins + " wins");
            }
        }

        return wins;
    }

    private static List<Wall> getValidWalls(GameState state) {
        List<Wall> walls = new ArrayList<>();
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                Wall hWall = new Wall(new Position(r, c), Wall.Orientation.HORIZONTAL);
                if (WallValidator.isValidWallPlacement(state, hWall)) {
                    walls.add(hWall);
                }
                Wall vWall = new Wall(new Position(r, c), Wall.Orientation.VERTICAL);
                if (WallValidator.isValidWallPlacement(state, vWall)) {
                    walls.add(vWall);
                }
            }
        }
        return walls;
    }
}
