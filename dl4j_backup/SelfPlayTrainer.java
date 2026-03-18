package ml.cnn;

import bot.BotBrain;
import bot.WallEvaluator;
import logic.MoveValidator;
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
 * Self-Play Reinforcement Learning Trainer for Quoridor CNN.
 *
 * The CNN learns by playing against itself and earlier versions.
 * Winning positions are reinforced, losing positions are penalized.
 *
 * PURE ML: No heuristics - the network learns everything from experience.
 */
public class SelfPlayTrainer {

    // Training parameters - memory optimized
    private static final int INITIAL_BOTBRAIN_GAMES = 1500;  // Reduced for memory
    private static final int SELF_PLAY_ROUNDS = 3;           // Fewer rounds
    private static final int GAMES_PER_ROUND = 100;          // Faster self-play
    private static final int EPOCHS_PER_ROUND = 3;           // Quick epochs
    private static final int BATCH_SIZE = 32;                // Smaller batches
    private static final double EXPLORATION_RATE = 0.20;     // More exploration
    private static final int MAX_WALLS_TO_EVAL = 10;         // Minimal wall eval

    private QuoridorCNN currentModel;
    private Random random = new Random(42);

    public static void main(String[] args) {
        WallEvaluator.silent = true;
        new SelfPlayTrainer().train();
    }

    public void train() {
        System.out.println("=== Self-Play Reinforcement Learning ===");
        System.out.println("=== PURE ML - No Heuristics ===\n");

        // Check if GPU is available
        System.out.println("Backend: " + Nd4j.getBackend().getClass().getSimpleName());
        System.out.println("Device: " + Nd4j.getAffinityManager().getDeviceForCurrentThread());

        // Step 1: Bootstrap from BotBrain games
        System.out.println("\n[Phase 1] Bootstrapping from BotBrain games...");
        List<INDArray> allStates = new ArrayList<>();
        List<Double> allLabels = new ArrayList<>();
        generateBotBrainData(allStates, allLabels, INITIAL_BOTBRAIN_GAMES);
        System.out.println("Bootstrap samples: " + allStates.size());

        // Create and initial train
        currentModel = new QuoridorCNN();
        System.out.println("CNN Parameters: " + currentModel.getParameterCount());
        trainOnData(allStates, allLabels, 8);  // Balanced initial training

        // Test initial performance
        System.out.println("\n--- Initial Performance ---");
        testAgainstBotBrain(50);

        // Step 2: Self-play iterations
        for (int round = 1; round <= SELF_PLAY_ROUNDS; round++) {
            System.out.println("\n[Phase 2." + round + "] Self-Play Round " + round + "/" + SELF_PLAY_ROUNDS);

            // Generate self-play data
            List<INDArray> roundStates = new ArrayList<>();
            List<Double> roundLabels = new ArrayList<>();
            generateSelfPlayData(roundStates, roundLabels, GAMES_PER_ROUND);
            System.out.println("  Self-play samples: " + roundStates.size());

            // Add to training pool (keep recent data, limit size)
            allStates.addAll(roundStates);
            allLabels.addAll(roundLabels);

            // Limit total samples to prevent memory issues
            int maxSamples = 80000;
            if (allStates.size() > maxSamples) {
                int toRemove = allStates.size() - maxSamples;
                allStates.subList(0, toRemove).clear();
                allLabels.subList(0, toRemove).clear();
            }

            System.out.println("  Total training samples: " + allStates.size());

            // Train on combined data
            trainOnData(allStates, allLabels, EPOCHS_PER_ROUND);

            // Test performance
            System.out.println("  --- Round " + round + " Performance ---");
            int wins = testAgainstBotBrain(100);

            // Save checkpoint
            if (wins >= 50) {
                currentModel.save("quoridor_cnn_checkpoint_r" + round + ".zip");
            }

            // Early success?
            if (wins >= 75) {
                System.out.println("\n*** TARGET ACHIEVED: " + wins + "% win rate! ***");
                break;
            }
        }

        // Final save and test
        System.out.println("\n[Phase 3] Final Evaluation...");
        currentModel.save("quoridor_cnn.zip");

        System.out.println("\n--- Final Performance (200 games) ---");
        int finalWins = testAgainstBotBrain(200);
        double winRate = finalWins * 100.0 / 200;

        System.out.println("\n========================================");
        System.out.printf("FINAL WIN RATE: %d/200 (%.1f%%)%n", finalWins, winRate);
        System.out.println("========================================");

        if (winRate >= 75) {
            System.out.println("SUCCESS: Target 75% achieved!");
        } else if (winRate >= 50) {
            System.out.println("DECENT: Better than random, needs more training.");
        } else {
            System.out.println("NEEDS WORK: Continue training or adjust parameters.");
        }
    }

    /**
     * Generate training data from BotBrain vs BotBrain games.
     */
    private void generateBotBrainData(List<INDArray> states, List<Double> labels, int numGames) {
        for (int g = 0; g < numGames; g++) {
            GameState state = new GameState();
            BotBrain bot1 = new BotBrain(0.1, 5);
            BotBrain bot2 = new BotBrain(0.1, 5);
            bot1.setSilent(true);
            bot2.setSilent(true);

            List<INDArray> p0States = new ArrayList<>();
            List<INDArray> p1States = new ArrayList<>();

            for (int turn = 0; turn < 200 && !state.isGameOver(); turn++) {
                Player me = state.getCurrentPlayer();
                int idx = state.getCurrentPlayerIndex();

                // Record state
                INDArray encoded = BoardEncoder.encode(state);
                if (idx == 0) p0States.add(encoded);
                else p1States.add(encoded);

                // BotBrain makes move
                BotBrain bot = (idx == 0) ? bot1 : bot2;
                BotBrain.BotAction action = bot.computeBestAction(state);
                if (action == null) break;

                applyBotBrainAction(state, me, idx, action);
                state.checkWinCondition();
                if (!state.isGameOver()) state.nextTurn();
            }

            // Label based on outcome
            labelGameStates(states, labels, p0States, p1States, state);

            if ((g + 1) % 500 == 0) {
                System.out.println("  BotBrain games: " + (g + 1) + "/" + numGames);
            }
        }
    }

    /**
     * Generate self-play training data.
     */
    private void generateSelfPlayData(List<INDArray> states, List<Double> labels, int numGames) {
        int cnnWins = 0;

        for (int g = 0; g < numGames; g++) {
            GameState state = new GameState();

            List<INDArray> p0States = new ArrayList<>();
            List<INDArray> p1States = new ArrayList<>();

            for (int turn = 0; turn < 200 && !state.isGameOver(); turn++) {
                Player me = state.getCurrentPlayer();
                int idx = state.getCurrentPlayerIndex();

                // Record state
                INDArray encoded = BoardEncoder.encode(state);
                if (idx == 0) p0States.add(encoded);
                else p1States.add(encoded);

                // CNN makes move (with exploration)
                boolean explore = random.nextDouble() < EXPLORATION_RATE;
                Position bestMove = null;
                Wall bestWall = null;
                double bestScore = Double.NEGATIVE_INFINITY;
                boolean isMove = true;

                List<Position> validMoves = MoveValidator.getValidMoves(state, me);

                if (explore) {
                    // Random move for exploration
                    if (!validMoves.isEmpty()) {
                        bestMove = validMoves.get(random.nextInt(validMoves.size()));
                    }
                } else {
                    // Greedy CNN selection
                    for (Position move : validMoves) {
                        INDArray after = BoardEncoder.encodeAfterMove(state, move);
                        double score = currentModel.predict(after);
                        if (score > bestScore) {
                            bestScore = score;
                            bestMove = move;
                            isMove = true;
                        }
                    }

                    // Evaluate walls
                    if (me.getWallsRemaining() > 0) {
                        List<Wall> walls = getValidWalls(state);
                        if (walls.size() > MAX_WALLS_TO_EVAL) {
                            Collections.shuffle(walls, random);
                            walls = walls.subList(0, MAX_WALLS_TO_EVAL);
                        }
                        for (Wall wall : walls) {
                            INDArray after = BoardEncoder.encodeAfterWall(state, wall);
                            double score = currentModel.predict(after);
                            if (score > bestScore) {
                                bestScore = score;
                                bestWall = wall;
                                isMove = false;
                            }
                        }
                    }
                }

                // Apply action
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

            // Label and track wins
            boolean p0Won = state.isGameOver() && state.getWinner() == state.getPlayers()[0];
            labelGameStates(states, labels, p0States, p1States, state);

            if ((g + 1) % 100 == 0) {
                System.out.println("    Self-play: " + (g + 1) + "/" + numGames);
            }
        }
    }

    /**
     * Label game states based on outcome with temporal weighting.
     */
    private void labelGameStates(List<INDArray> states, List<Double> labels,
                                  List<INDArray> p0States, List<INDArray> p1States,
                                  GameState finalState) {
        boolean p0Won = finalState.isGameOver() && finalState.getWinner() == finalState.getPlayers()[0];
        boolean draw = !finalState.isGameOver() || finalState.getWinner() == null;

        // Temporal discount - later moves matter more
        double discount = 0.97;

        int n0 = p0States.size();
        for (int i = 0; i < n0; i++) {
            double weight = Math.pow(discount, n0 - 1 - i);
            double baseLabel = draw ? 0.5 : (p0Won ? 0.8 : 0.2);
            double label = 0.5 + (baseLabel - 0.5) * weight;
            states.add(p0States.get(i));
            labels.add(Math.max(0.05, Math.min(0.95, label)));
        }

        int n1 = p1States.size();
        for (int i = 0; i < n1; i++) {
            double weight = Math.pow(discount, n1 - 1 - i);
            double baseLabel = draw ? 0.5 : (p0Won ? 0.2 : 0.8);
            double label = 0.5 + (baseLabel - 0.5) * weight;
            states.add(p1States.get(i));
            labels.add(Math.max(0.05, Math.min(0.95, label)));
        }
    }

    /**
     * Train the CNN on collected data.
     */
    private void trainOnData(List<INDArray> states, List<Double> labels, int epochs) {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < states.size(); i++) indices.add(i);

        currentModel.getNetwork().setListeners(new ScoreIterationListener(500));

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

                currentModel.getNetwork().fit(new DataSet(input, output));
            }

            System.out.println("    Epoch " + epoch + "/" + epochs + " complete");
        }
    }

    /**
     * Test CNN against BotBrain.
     */
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
                        double score = currentModel.predict(after);
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
                            double score = currentModel.predict(after);
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
                    applyBotBrainAction(state, me, idx, action);
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

        System.out.println("  Win rate: " + wins + "/" + games + " (" + (wins * 100 / games) + "%)");
        return wins;
    }

    private void applyBotBrainAction(GameState state, Player me, int idx, BotBrain.BotAction action) {
        if (action.type == BotBrain.BotAction.Type.MOVE) {
            me.setPosition(action.moveTarget);
        } else {
            action.wallToPlace.setOwnerIndex(idx);
            state.addWall(action.wallToPlace);
            me.setWallsRemaining(me.getWallsRemaining() - 1);
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
}
