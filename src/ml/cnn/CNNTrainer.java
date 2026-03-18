package ml.cnn;

import bot.BotBrain;
import bot.WallEvaluator;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import logic.MoveValidator;
import logic.PathFinder;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;
import org.bson.Document;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * CNN Trainer for Quoridor.
 *
 * Trains a convolutional neural network to evaluate board positions.
 * The CNN learns spatial patterns on the board (wall configurations,
 * player positions relative to goals, etc.) without hand-crafted features.
 *
 * PURE ML APPROACH: The CNN directly predicts position quality without
 * any heuristic scoring or feature engineering.
 */
public class CNNTrainer {

    // Training parameters
    private static final int SIMULATION_GAMES = 1000;   // BotBrain vs BotBrain games (minimal for memory)
    private static final int EPOCHS = 30;               // Fewer epochs
    private static final int PATIENCE = 8;              // Early stopping patience
    private static final int BATCH_SIZE = 32;           // Small batch for memory
    private static final double TRAIN_RATIO = 0.85;

    // RL-style parameters
    private static final double TEMPORAL_DISCOUNT = 0.95;
    private static final double MOVE_QUALITY_WEIGHT = 0.3;
    private static final double WIN_BONUS = 0.7;
    private static final double LOSE_PENALTY = 0.3;

    public static void main(String[] args) {
        // Suppress verbose output
        WallEvaluator.silent = true;

        String mongoUri = "mongodb+srv://EladMongo:Elad1234@eladprojectcluster.ebx4rpd.mongodb.net/?appName=EladProjectCluster";
        String outputPath = args.length > 0 ? args[0] : "quoridor_cnn.zip";

        System.out.println("=== CNN Trainer for Quoridor ===");
        System.out.println("=== PURE ML - No Heuristics ===");
        System.out.println("Output: " + outputPath);

        // Step 1: Generate training data from BotBrain simulations
        System.out.println("\n[Step 1] Generating training data from BotBrain simulations...");
        List<INDArray> boardStates = new ArrayList<>();
        List<Double> labels = new ArrayList<>();
        generateTrainingData(boardStates, labels, SIMULATION_GAMES);
        System.out.println("Simulation samples: " + boardStates.size());

        // Step 2: Skip MongoDB (not contributing samples, saves memory)
        System.out.println("\n[Step 2] Skipping MongoDB (memory optimization)...");
        // Force GC to free memory before training
        System.gc();

        System.out.println("\nTotal samples: " + boardStates.size());

        // Step 3: Create CNN and train
        System.out.println("\n[Step 3] Creating and training CNN...");
        QuoridorCNN cnn = new QuoridorCNN();
        cnn.printSummary();

        trainCNN(cnn, boardStates, labels);

        // Step 4: Save model
        System.out.println("\n[Step 4] Saving model...");
        cnn.save(outputPath);

        // Step 5: Evaluate against BotBrain
        System.out.println("\n[Step 5] Evaluating against BotBrain (PURE ML)...");
        int wins = testAgainstBotBrain(cnn, 100);
        System.out.printf("Win rate: %d/100 (%.1f%%)%n", wins, wins * 1.0);

        System.out.println("\n=== Training Complete ===");
    }

    /**
     * Generates training data from BotBrain vs BotBrain games.
     *
     * For each position in the game, we encode the board state and
     * label it based on the eventual outcome + move quality.
     */
    private static void generateTrainingData(List<INDArray> boardStates, List<Double> labels, int numGames) {
        Random rand = new Random(42);

        for (int game = 0; game < numGames; game++) {
            GameState state = new GameState();

            BotBrain bot1 = new BotBrain(0.1, 5);  // Slight randomness for variety
            BotBrain bot2 = new BotBrain(0.1, 5);
            bot1.setSilent(true);
            bot2.setSilent(true);

            // Record game states and move qualities
            List<StateRecord> player0Records = new ArrayList<>();
            List<StateRecord> player1Records = new ArrayList<>();

            for (int turn = 0; turn < 200; turn++) {
                if (state.isGameOver()) break;

                Player me = state.getCurrentPlayer();
                Player opp = state.getOtherPlayer();
                int playerIdx = state.getCurrentPlayerIndex();

                // Get race gap BEFORE move
                int myDistBefore = PathFinder.aStarShortestPath(state, me);
                int oppDistBefore = PathFinder.aStarShortestPath(state, opp);
                int raceGapBefore = oppDistBefore - myDistBefore;

                // Encode current state
                INDArray boardState = BoardEncoder.encode(state);

                // Get action from BotBrain
                BotBrain bot = (playerIdx == 0) ? bot1 : bot2;
                BotBrain.BotAction action = bot.computeBestAction(state);
                if (action == null) break;

                // Apply action
                if (action.type == BotBrain.BotAction.Type.MOVE) {
                    me.setPosition(action.moveTarget);
                } else {
                    action.wallToPlace.setOwnerIndex(playerIdx);
                    state.addWall(action.wallToPlace);
                }

                // Get race gap AFTER move
                int myDistAfter = PathFinder.aStarShortestPath(state, me);
                int oppDistAfter = PathFinder.aStarShortestPath(state, opp);
                int raceGapAfter = oppDistAfter - myDistAfter;

                // Calculate move quality
                double moveQuality = (raceGapAfter - raceGapBefore) / 4.0;
                moveQuality = Math.max(-1.0, Math.min(1.0, moveQuality));

                // Record state
                StateRecord record = new StateRecord(boardState, turn, moveQuality);
                if (playerIdx == 0) {
                    player0Records.add(record);
                } else {
                    player1Records.add(record);
                }

                state.checkWinCondition();
                if (!state.isGameOver()) {
                    state.nextTurn();
                }
            }

            // Determine winner
            boolean player0Won = false;
            boolean draw = true;

            if (state.isGameOver() && state.getWinner() != null) {
                draw = false;
                player0Won = (state.getWinner() == state.getPlayers()[0]);
            }

            // Add labeled samples
            addLabeledSamples(boardStates, labels, player0Records, player0Won, draw);
            addLabeledSamples(boardStates, labels, player1Records, !player0Won, draw);

            if ((game + 1) % 500 == 0) {
                System.out.println("  Simulated " + (game + 1) + "/" + numGames +
                        " games (" + boardStates.size() + " samples)");
            }
        }
    }

    /**
     * Adds labeled samples with temporal weighting and move quality.
     */
    private static void addLabeledSamples(List<INDArray> boardStates, List<Double> labels,
                                          List<StateRecord> records, boolean won, boolean draw) {
        int numMoves = records.size();

        for (int i = 0; i < numMoves; i++) {
            StateRecord record = records.get(i);

            // Base label from outcome
            double baseLabel;
            if (draw) {
                baseLabel = 0.5;
            } else {
                baseLabel = won ? WIN_BONUS : LOSE_PENALTY;
            }

            // Temporal weight: later moves matter more
            int distanceToEnd = numMoves - i;
            double temporalWeight = Math.pow(TEMPORAL_DISCOUNT, distanceToEnd);

            // Move quality adjustment
            double qualityAdjustment = record.moveQuality * MOVE_QUALITY_WEIGHT;

            // Final label
            double label = baseLabel + qualityAdjustment;
            label = 0.5 + (label - 0.5) * temporalWeight;
            label = Math.max(0.05, Math.min(0.95, label));

            boardStates.add(record.boardState);
            labels.add(label);
        }
    }

    /**
     * Loads game data from MongoDB and converts to CNN training format.
     */
    private static void loadMongoDBData(String mongoUri, List<INDArray> boardStates, List<Double> labels) {
        MongoClient client = MongoClients.create(mongoUri);
        MongoDatabase db = client.getDatabase("quoridor");
        MongoCollection<Document> collection = db.getCollection("games");

        int gameCount = 0;
        int sampleCount = 0;

        for (Document game : collection.find()) {
            int winner = game.getInteger("winner", -1);
            if (winner == -1) continue;

            List<Document> turns = game.getList("turns", Document.class);
            if (turns == null || turns.isEmpty()) continue;

            // Reconstruct game states from turn data
            GameState state = new GameState();
            int totalTurns = turns.size();

            List<StateRecord> player0Records = new ArrayList<>();
            List<StateRecord> player1Records = new ArrayList<>();

            for (int turnIdx = 0; turnIdx < totalTurns; turnIdx++) {
                Document turn = turns.get(turnIdx);

                // Skip early turns (less informative)
                if (turnIdx < 5) continue;

                int currentPlayer = turn.getInteger("currentPlayer", 0);

                // Get position data from turn
                Document p1Pos = turn.get("player1Position", Document.class);
                Document p2Pos = turn.get("player2Position", Document.class);

                if (p1Pos != null && p2Pos != null) {
                    // Reconstruct positions
                    int p1Row = p1Pos.getInteger("row", 8);
                    int p1Col = p1Pos.getInteger("col", 4);
                    int p2Row = p2Pos.getInteger("row", 0);
                    int p2Col = p2Pos.getInteger("col", 4);

                    state.getPlayers()[0].setPosition(new Position(p1Row, p1Col));
                    state.getPlayers()[1].setPosition(new Position(p2Row, p2Col));

                    // Set current player
                    while (state.getCurrentPlayerIndex() != currentPlayer) {
                        state.nextTurn();
                    }

                    // Encode board state
                    INDArray boardState = BoardEncoder.encode(state);

                    // Calculate simple move quality from feature data if available
                    Document featureDoc = turn.get("features", Document.class);
                    double moveQuality = 0.0;
                    if (featureDoc != null) {
                        Number raceGap = featureDoc.get("raceGap", Number.class);
                        if (raceGap != null) {
                            moveQuality = (currentPlayer == 0 ? raceGap.doubleValue() : -raceGap.doubleValue()) / 8.0;
                            moveQuality = Math.max(-1, Math.min(1, moveQuality));
                        }
                    }

                    StateRecord record = new StateRecord(boardState, turnIdx, moveQuality);
                    if (currentPlayer == 0) {
                        player0Records.add(record);
                    } else {
                        player1Records.add(record);
                    }
                }
            }

            // Label based on winner
            boolean player0Won = (winner == 0);
            addLabeledSamples(boardStates, labels, player0Records, player0Won, false);
            addLabeledSamples(boardStates, labels, player1Records, !player0Won, false);

            sampleCount += player0Records.size() + player1Records.size();
            gameCount++;

            if (gameCount % 5000 == 0) {
                System.out.println("    Processed " + gameCount + " games (" + sampleCount + " samples)");
            }

            // Limit for memory
            if (sampleCount > 500000) break;
        }

        client.close();
        System.out.println("    Loaded " + gameCount + " games, " + sampleCount + " samples from MongoDB");
    }

    /**
     * Trains the CNN using the collected data.
     */
    private static void trainCNN(QuoridorCNN cnn, List<INDArray> boardStates, List<Double> labelList) {
        // Shuffle data
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < boardStates.size(); i++) indices.add(i);
        Collections.shuffle(indices, new Random(42));

        // Split into train/validation
        int splitPoint = (int) (indices.size() * TRAIN_RATIO);

        System.out.println("Creating training dataset...");
        List<Integer> trainIndices = new ArrayList<>(indices.subList(0, splitPoint));
        List<Integer> valIndices = new ArrayList<>(indices.subList(splitPoint, indices.size()));

        System.out.println("Training samples: " + trainIndices.size());
        System.out.println("Validation samples: " + valIndices.size());

        // Add listener
        cnn.getNetwork().setListeners(new ScoreIterationListener(500));

        // Train with early stopping
        System.out.println("\nTraining CNN...");
        double bestValScore = Double.MAX_VALUE;
        int noImprovement = 0;

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            // Shuffle training data each epoch
            Collections.shuffle(trainIndices);

            // Train in batches
            for (int batch = 0; batch < trainIndices.size(); batch += BATCH_SIZE) {
                int end = Math.min(batch + BATCH_SIZE, trainIndices.size());
                int batchSize = end - batch;

                // Collect board states for this batch
                INDArray[] batchStates = new INDArray[batchSize];
                double[] batchLabelArr = new double[batchSize];

                for (int i = 0; i < batchSize; i++) {
                    int idx = trainIndices.get(batch + i);
                    batchStates[i] = boardStates.get(idx);
                    batchLabelArr[i] = labelList.get(idx);
                }

                // Stack into batch
                INDArray batchInput = Nd4j.stack(0, batchStates);
                INDArray batchLabels = Nd4j.create(batchLabelArr).reshape(batchSize, 1);

                DataSet ds = new DataSet(batchInput, batchLabels);
                cnn.getNetwork().fit(ds);
            }

            // Calculate validation score
            double valScore = calculateValidationScore(cnn, boardStates, labelList, valIndices);

            if (epoch % 5 == 0 || epoch == 1) {
                System.out.printf("Epoch %3d | Val Score: %.6f | Best: %.6f%n",
                        epoch, valScore, bestValScore);
            }

            if (valScore < bestValScore) {
                bestValScore = valScore;
                noImprovement = 0;
            } else {
                noImprovement++;
            }

            if (noImprovement >= PATIENCE) {
                System.out.println("Early stopping at epoch " + epoch);
                break;
            }
        }

        System.out.println("Best validation score: " + bestValScore);
    }

    /**
     * Calculates validation loss (MSE).
     */
    private static double calculateValidationScore(QuoridorCNN cnn, List<INDArray> boardStates,
                                                   List<Double> labels, List<Integer> valIndices) {
        double totalError = 0.0;

        for (int idx : valIndices) {
            INDArray input = boardStates.get(idx).reshape(1, BoardEncoder.CHANNELS,
                    BoardEncoder.BOARD_SIZE, BoardEncoder.BOARD_SIZE);
            INDArray output = cnn.getNetwork().output(input);

            double predicted = output.getDouble(0);
            double actual = labels.get(idx);
            double error = (predicted - actual) * (predicted - actual);
            totalError += error;
        }

        return totalError / valIndices.size();
    }

    /**
     * Tests the CNN against BotBrain using PURE ML (no heuristics).
     */
    private static int testAgainstBotBrain(QuoridorCNN cnn, int games) {
        int wins = 0;

        System.out.println("Testing CNN (PURE ML) vs BotBrain...");

        for (int g = 0; g < games; g++) {
            GameState state = new GameState();
            BotBrain bot = new BotBrain(0.0, 3);
            bot.setSilent(true);

            int cnnIdx = g % 2;  // Alternate sides

            for (int turn = 0; turn < 200; turn++) {
                if (state.isGameOver()) break;

                Player me = state.getCurrentPlayer();
                boolean isCNNTurn = (state.getCurrentPlayerIndex() == cnnIdx);

                if (isCNNTurn) {
                    // CNN plays - PURE ML, evaluate all actions
                    Position bestMove = null;
                    Wall bestWall = null;
                    double bestScore = Double.NEGATIVE_INFINITY;
                    boolean bestIsMove = true;

                    // Evaluate all valid moves
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

                    // Evaluate valid walls (sample for speed)
                    if (me.getWallsRemaining() > 0) {
                        List<Wall> validWalls = getValidWalls(state);
                        // Sample up to 50 walls for speed
                        if (validWalls.size() > 50) {
                            Collections.shuffle(validWalls);
                            validWalls = validWalls.subList(0, 50);
                        }

                        for (Wall wall : validWalls) {
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
                if (!state.isGameOver()) {
                    state.nextTurn();
                }
            }

            if (state.isGameOver() && state.getWinner() == state.getPlayers()[cnnIdx]) {
                wins++;
            }

            if ((g + 1) % 20 == 0) {
                System.out.println("  Game " + (g + 1) + "/" + games + ": " + wins + " wins");
            }
        }

        return wins;
    }

    /**
     * Gets all valid wall placements.
     */
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

    /**
     * Record of a board state with move quality.
     */
    private static class StateRecord {
        final INDArray boardState;
        final int turn;
        final double moveQuality;

        StateRecord(INDArray boardState, int turn, double moveQuality) {
            this.boardState = boardState;
            this.turn = turn;
            this.moveQuality = moveQuality;
        }
    }
}
