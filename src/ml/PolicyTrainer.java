package ml;

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
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.structure.NetworkCODEC;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.util.*;

/**
 * IMPROVED Policy Network Trainer for Quoridor.
 *
 * THREE KEY IMPROVEMENTS:
 *
 * 1. BETTER LABELS (Move Quality):
 *    - Instead of binary win/loss (1.0 or 0.0), use continuous quality score
 *    - Quality = base_win_label + move_quality_bonus + temporal_weight
 *    - Moves that improve position get higher labels even if player lost
 *    - Moves that hurt position get lower labels even if player won
 *
 * 2. REINFORCEMENT LEARNING STYLE:
 *    - Immediate reward based on race gap improvement
 *    - Temporal difference: later moves weighted more heavily
 *    - Discount factor for future rewards
 *
 * 3. SELF-PLAY:
 *    - Train NNBot vs NNBot to learn from itself
 *    - Iterative improvement: train -> self-play -> train
 *    - The NN learns to beat its previous version
 */
public class PolicyTrainer {

    // Training parameters - MORE BOTBRAIN GAMES, NO SELF-PLAY
    private static final int SIMULATION_GAMES = 3000;  // BotBrain vs BotBrain games (increased!)
    private static final int SELF_PLAY_GAMES = 0;      // DISABLED - self-play hurts performance
    private static final int SELF_PLAY_ITERATIONS = 0; // DISABLED - self-play hurts performance
    private static final int NEGATIVE_SAMPLES_PER_TURN = 2;
    private static final int EPOCHS = 300;
    private static final int PATIENCE = 40;
    private static final int NUM_RUNS = 5;
    private static final int OPENING_RANDOM_MOVES = 5;

    // RL-style parameters
    private static final double TEMPORAL_DISCOUNT = 0.95;  // γ for future rewards
    private static final double MOVE_QUALITY_WEIGHT = 0.3; // How much move quality affects label
    private static final double WIN_BONUS = 0.6;           // Base bonus for winning
    private static final double LOSE_PENALTY = 0.4;        // Base penalty for losing

    public static void main(String[] args) {
        // Suppress verbose wall evaluation output for faster training
        WallEvaluator.silent = true;

        String mongoUri = "mongodb+srv://EladMongo:Elad1234@eladprojectcluster.ebx4rpd.mongodb.net/?appName=EladProjectCluster";
        String outputPath = args.length > 0 ? args[0] : "quoridor_policy.eg";

        System.out.println("=== FAST Policy Network Trainer ===");
        System.out.println("=== With Better Labels + RL Rewards + Self-Play ===");
        System.out.println("Output: " + outputPath);

        // Step 1: Generate initial training data from BotBrain games
        System.out.println("\n[Step 1] Generating training data from BotBrain simulations...");
        List<double[]> features = new ArrayList<>();
        List<Double> labels = new ArrayList<>();
        generateImprovedSimulationData(features, labels, SIMULATION_GAMES, null);
        System.out.println("Initial samples: " + features.size());

        // Step 2: Load MongoDB data with improved labels
        System.out.println("\n[Step 2] Loading MongoDB data with improved labels...");
        try {
            int beforeSize = features.size();
            loadMongoDBDataImproved(mongoUri, features, labels);
            System.out.println("Added " + (features.size() - beforeSize) + " samples from MongoDB");
        } catch (Exception e) {
            System.out.println("MongoDB error: " + e.getMessage());
        }

        System.out.println("\nTotal initial samples: " + features.size());

        // Step 3: Initial training
        System.out.println("\n[Step 3] Initial training...");
        PolicyNetwork bestPN = trainNetwork(features, labels);

        // Step 4: Self-play iterations
        for (int iter = 1; iter <= SELF_PLAY_ITERATIONS; iter++) {
            System.out.println("\n========================================");
            System.out.println("[Step 4." + iter + "] SELF-PLAY ITERATION " + iter + "/" + SELF_PLAY_ITERATIONS);
            System.out.println("========================================");

            // Generate self-play data
            System.out.println("Generating " + SELF_PLAY_GAMES + " self-play games...");
            List<double[]> selfPlayFeatures = new ArrayList<>();
            List<Double> selfPlayLabels = new ArrayList<>();
            generateImprovedSimulationData(selfPlayFeatures, selfPlayLabels, SELF_PLAY_GAMES, bestPN);
            System.out.println("Self-play samples: " + selfPlayFeatures.size());

            // Add to training data (keep some old data for stability)
            int keepOld = Math.min(features.size(), 500000);  // Keep up to 500k old samples
            if (features.size() > keepOld) {
                // Randomly sample old data
                Random rand = new Random();
                List<double[]> oldFeatures = new ArrayList<>();
                List<Double> oldLabels = new ArrayList<>();
                List<Integer> indices = new ArrayList<>();
                for (int i = 0; i < features.size(); i++) indices.add(i);
                Collections.shuffle(indices, rand);
                for (int i = 0; i < keepOld; i++) {
                    oldFeatures.add(features.get(indices.get(i)));
                    oldLabels.add(labels.get(indices.get(i)));
                }
                features = oldFeatures;
                labels = oldLabels;
            }

            // Add self-play data
            features.addAll(selfPlayFeatures);
            labels.addAll(selfPlayLabels);
            System.out.println("Total samples for iteration " + iter + ": " + features.size());

            // Retrain
            System.out.println("Retraining with self-play data...");
            bestPN = trainNetwork(features, labels);

            // Test progress
            System.out.println("Testing iteration " + iter + " against BotBrain...");
            int wins = testAgainstBotBrain(bestPN, 100, 0.0);
            System.out.printf("Win rate: %d/100 (%.0f%%)%n", wins, wins * 1.0);
        }

        // Step 5: Save final model
        System.out.println("\n[Step 5] Saving final model...");
        bestPN.save(outputPath);

        // Step 6: Final evaluation
        System.out.println("\n[Step 6] Final evaluation against BotBrain...");
        double[] temps = {0.0, 0.3, 0.5};
        for (double temp : temps) {
            int wins = testAgainstBotBrain(bestPN, 100, temp);
            System.out.printf("  Temperature %.1f: %d/100 wins (%.0f%%)%n", temp, wins, wins * 1.0);
        }

        System.out.println("\n=== Training Complete ===");
    }

    /**
     * IMPROVED: Generates training data with better labels.
     *
     * Instead of binary win/loss, each move gets a quality score:
     *   label = base_win_label + move_quality + temporal_weight
     *
     * This teaches the NN which specific moves are good, not just
     * which player eventually won.
     */
    private static void generateImprovedSimulationData(List<double[]> features, List<Double> labels,
                                                        int numGames, PolicyNetwork selfPlayNN) {
        Random rand = new Random();
        boolean useSelfPlay = (selfPlayNN != null);

        for (int game = 0; game < numGames; game++) {
            GameState state = new GameState();

            // Create bots - either BotBrain or NNBot for self-play
            Object bot1, bot2;
            if (useSelfPlay) {
                NNBot nn1 = new NNBot(selfPlayNN);
                NNBot nn2 = new NNBot(selfPlayNN);
                nn1.setTemperature(0.3);  // Some exploration during self-play
                nn2.setTemperature(0.3);
                bot1 = nn1;
                bot2 = nn2;
            } else {
                BotBrain bb1 = new BotBrain(0.0, OPENING_RANDOM_MOVES);
                BotBrain bb2 = new BotBrain(0.0, OPENING_RANDOM_MOVES);
                bb1.setSilent(true);
                bb2.setSilent(true);
                bot1 = bb1;
                bot2 = bb2;
            }

            // Record game data with move qualities
            List<MoveRecord> game1Records = new ArrayList<>();  // Player 0
            List<MoveRecord> game2Records = new ArrayList<>();  // Player 1

            int prevRaceGap0 = 0, prevRaceGap1 = 0;  // Track race gap changes

            for (int turn = 0; turn < 200; turn++) {
                if (state.isGameOver()) break;

                Player me = state.getCurrentPlayer();
                Player opp = state.getOtherPlayer();
                int playerIdx = state.getCurrentPlayerIndex();

                // Get race gap BEFORE move
                int myDistBefore = PathFinder.aStarShortestPath(state, me);
                int oppDistBefore = PathFinder.aStarShortestPath(state, opp);
                int raceGapBefore = oppDistBefore - myDistBefore;

                // Get action from bot
                Object actionObj;
                if (useSelfPlay) {
                    NNBot nnBot = (playerIdx == 0) ? (NNBot) bot1 : (NNBot) bot2;
                    NNBot.BotAction action = nnBot.computeBestAction(state);
                    if (action == null) break;
                    actionObj = action;
                } else {
                    BotBrain bbBot = (playerIdx == 0) ? (BotBrain) bot1 : (BotBrain) bot2;
                    BotBrain.BotAction action = bbBot.computeBestAction(state);
                    if (action == null) break;
                    actionObj = action;
                }

                // Extract features BEFORE applying action
                double[] stateFeatures = GameFeatures.extract(state);
                double[] expandedState = NNTrainer.expandFeatures(stateFeatures);
                double[] normalizedState = NNTrainer.normalizeAll(expandedState);

                double[] actionFeatures;
                boolean isMove;
                Position moveTarget = null;
                Wall wallToPlace = null;

                if (useSelfPlay) {
                    NNBot.BotAction action = (NNBot.BotAction) actionObj;
                    isMove = (action.type == NNBot.BotAction.Type.MOVE);
                    if (isMove) {
                        moveTarget = action.moveTarget;
                        actionFeatures = ActionFeatures.extractMoveFeatures(state, moveTarget);
                    } else {
                        wallToPlace = action.wallToPlace;
                        actionFeatures = ActionFeatures.extractWallFeatures(state, wallToPlace);
                    }
                } else {
                    BotBrain.BotAction action = (BotBrain.BotAction) actionObj;
                    isMove = (action.type == BotBrain.BotAction.Type.MOVE);
                    if (isMove) {
                        moveTarget = action.moveTarget;
                        actionFeatures = ActionFeatures.extractMoveFeatures(state, moveTarget);
                    } else {
                        wallToPlace = action.wallToPlace;
                        actionFeatures = ActionFeatures.extractWallFeatures(state, wallToPlace);
                    }
                }

                double[] normalizedAction = ActionFeatures.normalize(actionFeatures);

                // Combine features
                double[] combined = new double[PolicyNetwork.INPUT_SIZE];
                System.arraycopy(normalizedState, 0, combined, 0, 18);
                System.arraycopy(normalizedAction, 0, combined, 18, 10);

                // Apply action
                if (isMove) {
                    me.setPosition(moveTarget);
                } else {
                    wallToPlace.setOwnerIndex(playerIdx);
                    state.addWall(wallToPlace);
                }

                // Get race gap AFTER move
                int myDistAfter = PathFinder.aStarShortestPath(state, me);
                int oppDistAfter = PathFinder.aStarShortestPath(state, opp);
                int raceGapAfter = oppDistAfter - myDistAfter;

                // Calculate move quality (how much this move improved position)
                // Positive = good move, negative = bad move
                double moveQuality = (raceGapAfter - raceGapBefore) / 4.0;  // Normalize to roughly [-1, 1]
                moveQuality = Math.max(-1.0, Math.min(1.0, moveQuality));

                // Record for later labeling
                MoveRecord record = new MoveRecord(combined, turn, moveQuality);
                if (playerIdx == 0) {
                    game1Records.add(record);
                } else {
                    game2Records.add(record);
                }

                // Generate negative examples (with lower quality)
                List<double[]> negatives = generateNegativeExamples(state, normalizedState,
                        isMove ? moveTarget : null, isMove ? null : wallToPlace, rand);
                for (double[] neg : negatives) {
                    features.add(neg);
                    labels.add(0.2);  // Low but not zero (they're valid moves, just not chosen)
                }

                state.checkWinCondition();
                if (!state.isGameOver()) {
                    state.nextTurn();
                }
            }

            // Determine winner and compute final labels
            int totalTurns = game1Records.size() + game2Records.size();
            boolean player0Won = false;
            boolean draw = true;

            if (state.isGameOver() && state.getWinner() != null) {
                draw = false;
                player0Won = (state.getWinner() == state.getPlayers()[0]);
            }

            // IMPROVED LABELING: Combine win/loss with move quality and temporal weight
            addLabeledSamples(features, labels, game1Records, player0Won, draw, totalTurns);
            addLabeledSamples(features, labels, game2Records, !player0Won, draw, totalTurns);

            if ((game + 1) % 500 == 0) {
                System.out.println("  Simulated " + (game + 1) + "/" + numGames +
                        " games (" + features.size() + " samples)" +
                        (useSelfPlay ? " [SELF-PLAY]" : ""));
            }
        }
    }

    /**
     * IMPROVED LABELING with move quality and temporal weighting.
     */
    private static void addLabeledSamples(List<double[]> features, List<Double> labels,
                                           List<MoveRecord> records, boolean won, boolean draw,
                                           int totalTurns) {
        int numMoves = records.size();

        for (int i = 0; i < numMoves; i++) {
            MoveRecord record = records.get(i);

            // Base label from outcome
            double baseLabel;
            if (draw) {
                baseLabel = 0.5;
            } else {
                baseLabel = won ? WIN_BONUS : (1.0 - LOSE_PENALTY);
            }

            // Temporal weight: later moves matter more (they're closer to the outcome)
            // Early moves get discounted by γ^(distance_to_end)
            int distanceToEnd = numMoves - i;
            double temporalWeight = Math.pow(TEMPORAL_DISCOUNT, distanceToEnd);

            // Move quality bonus/penalty
            double qualityAdjustment = record.moveQuality * MOVE_QUALITY_WEIGHT;

            // Final label: base + quality, weighted by temporal factor
            double label = baseLabel + qualityAdjustment;

            // Apply temporal weighting (blend toward 0.5 for early moves)
            label = 0.5 + (label - 0.5) * temporalWeight;

            // Clamp to valid range
            label = Math.max(0.05, Math.min(0.95, label));

            features.add(record.features);
            labels.add(label);
        }
    }

    /**
     * Generates negative examples (actions not taken).
     */
    private static List<double[]> generateNegativeExamples(GameState state, double[] normalizedState,
                                                            Position chosenMove, Wall chosenWall, Random rand) {
        List<double[]> negatives = new ArrayList<>();
        Player me = state.getCurrentPlayer();

        // Sample alternative moves
        List<Position> validMoves = MoveValidator.getValidMoves(state, me);
        List<Position> alternatives = new ArrayList<>();
        for (Position move : validMoves) {
            if (chosenMove == null || !move.equals(chosenMove)) {
                alternatives.add(move);
            }
        }

        Collections.shuffle(alternatives, rand);
        for (int i = 0; i < Math.min(NEGATIVE_SAMPLES_PER_TURN, alternatives.size()); i++) {
            Position alt = alternatives.get(i);
            double[] actionFeatures = ActionFeatures.extractMoveFeatures(state, alt);
            double[] normalizedAction = ActionFeatures.normalize(actionFeatures);

            double[] combined = new double[PolicyNetwork.INPUT_SIZE];
            System.arraycopy(normalizedState, 0, combined, 0, 18);
            System.arraycopy(normalizedAction, 0, combined, 18, 10);
            negatives.add(combined);
        }

        return negatives;
    }

    /**
     * Loads MongoDB data with improved labels.
     */
    private static void loadMongoDBDataImproved(String mongoUri, List<double[]> features, List<Double> labels) {
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

            int totalTurns = turns.size();

            for (int turnIdx = 0; turnIdx < totalTurns; turnIdx++) {
                Document turn = turns.get(turnIdx);
                int currentPlayer = turn.getInteger("currentPlayer", 0);
                Document featureDoc = turn.get("features", Document.class);
                if (featureDoc == null) continue;

                // Extract features
                String[] featureNames = GameFeatures.featureNames();
                double[] rawFeatures = new double[GameFeatures.FEATURE_COUNT];
                for (int f = 0; f < featureNames.length; f++) {
                    Number val = featureDoc.get(featureNames[f], Number.class);
                    rawFeatures[f] = (val != null) ? val.doubleValue() : 0.0;
                }

                double[] expanded = NNTrainer.expandFeatures(rawFeatures);
                double[] normalized = NNTrainer.normalizeAll(expanded);

                // Create synthetic action features
                double raceGap = rawFeatures[8];
                double[] syntheticAction = new double[10];
                syntheticAction[2] = raceGap > 0 ? 0.5 : 0.0;
                syntheticAction[4] = Math.max(-1, Math.min(1, raceGap / 5.0));
                syntheticAction = ActionFeatures.normalize(syntheticAction);

                double[] combined = new double[PolicyNetwork.INPUT_SIZE];
                System.arraycopy(normalized, 0, combined, 0, 18);
                System.arraycopy(syntheticAction, 0, combined, 18, 10);

                // Improved label with temporal weighting
                boolean won = (currentPlayer == winner);
                double baseLabel = won ? WIN_BONUS : (1.0 - LOSE_PENALTY);

                // Temporal weight
                int distanceToEnd = totalTurns - turnIdx;
                double temporalWeight = Math.pow(TEMPORAL_DISCOUNT, distanceToEnd);

                // Race gap as move quality proxy
                double qualityProxy = (currentPlayer == 0 ? raceGap : -raceGap) / 8.0;
                qualityProxy = Math.max(-1, Math.min(1, qualityProxy));

                double label = baseLabel + qualityProxy * MOVE_QUALITY_WEIGHT;
                label = 0.5 + (label - 0.5) * temporalWeight;
                label = Math.max(0.05, Math.min(0.95, label));

                // Skip very early turns (less informative)
                int turnNum = (int) rawFeatures[9];
                if (turnNum >= 5) {
                    features.add(combined);
                    labels.add(label);
                    sampleCount++;
                }
            }

            gameCount++;
            if (gameCount % 5000 == 0) {
                System.out.println("    Processed " + gameCount + " games (" + sampleCount + " samples)");
            }
        }

        client.close();
        System.out.println("    Loaded " + gameCount + " games, " + sampleCount + " samples from MongoDB");
    }

    /**
     * Trains the network and returns the best model.
     */
    private static PolicyNetwork trainNetwork(List<double[]> features, List<Double> labels) {
        // Shuffle and split
        long seed = System.currentTimeMillis();
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < features.size(); i++) indices.add(i);
        Collections.shuffle(indices, new Random(seed));

        int splitPoint = (int) (indices.size() * 0.85);
        BasicMLDataSet trainSet = buildDataSet(features, labels, indices, 0, splitPoint);
        BasicMLDataSet valSet = buildDataSet(features, labels, indices, splitPoint, indices.size());

        System.out.println("Training: " + trainSet.getRecordCount() + ", Validation: " + valSet.getRecordCount());

        double bestValError = Double.MAX_VALUE;
        double[] bestWeights = null;

        for (int run = 1; run <= NUM_RUNS; run++) {
            System.out.println("\n--- Run " + run + "/" + NUM_RUNS + " ---");

            PolicyNetwork pn = new PolicyNetwork();
            ResilientPropagation trainer = new ResilientPropagation(pn.getNetwork(), trainSet);

            double runBestValError = Double.MAX_VALUE;
            double[] runBestWeights = NetworkCODEC.networkToArray(pn.getNetwork());
            int noImprovement = 0;

            for (int epoch = 1; epoch <= EPOCHS; epoch++) {
                trainer.iteration();
                double trainError = trainer.getError();
                double valError = pn.getNetwork().calculateError(valSet);

                if (valError < runBestValError) {
                    runBestValError = valError;
                    runBestWeights = NetworkCODEC.networkToArray(pn.getNetwork());
                    noImprovement = 0;
                } else {
                    noImprovement++;
                }

                if (epoch % 30 == 0 || epoch == 1) {
                    System.out.printf("  Epoch %3d | Train: %.6f | Val: %.6f | Best: %.6f%n",
                            epoch, trainError, valError, runBestValError);
                }

                if (noImprovement >= PATIENCE) {
                    System.out.println("  Early stop at epoch " + epoch);
                    break;
                }
            }
            trainer.finishTraining();

            System.out.printf("  Run %d best: %.6f%n", run, runBestValError);

            if (runBestValError < bestValError) {
                bestValError = runBestValError;
                bestWeights = runBestWeights;
                System.out.println("  ** New best! **");
            }
        }

        PolicyNetwork bestPN = new PolicyNetwork();
        NetworkCODEC.arrayToNetwork(bestWeights, bestPN.getNetwork());
        return bestPN;
    }

    private static BasicMLDataSet buildDataSet(List<double[]> features, List<Double> labels,
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

    /**
     * Tests against BotBrain.
     */
    private static int testAgainstBotBrain(PolicyNetwork pn, int games, double temperature) {
        int wins = 0;

        for (int i = 0; i < games; i++) {
            NNBot nnBot = new NNBot(pn);
            nnBot.setTemperature(temperature);

            GameState state = new GameState();
            BotBrain bot = new BotBrain(0.0, 3);
            bot.setSilent(true);
            int pnIdx = i % 2;

            for (int turn = 0; turn < 200; turn++) {
                if (state.isGameOver()) break;

                Player me = state.getCurrentPlayer();
                boolean isPNTurn = (state.getCurrentPlayerIndex() == pnIdx);

                if (isPNTurn) {
                    NNBot.BotAction action = nnBot.computeBestAction(state);
                    if (action == null) break;

                    if (action.type == NNBot.BotAction.Type.MOVE) {
                        me.setPosition(action.moveTarget);
                    } else {
                        action.wallToPlace.setOwnerIndex(state.getCurrentPlayerIndex());
                        state.addWall(action.wallToPlace);
                    }
                } else {
                    BotBrain.BotAction action = bot.computeBestAction(state);
                    if (action == null) break;

                    if (action.type == BotBrain.BotAction.Type.MOVE) {
                        me.setPosition(action.moveTarget);
                    } else {
                        action.wallToPlace.setOwnerIndex(state.getCurrentPlayerIndex());
                        state.addWall(action.wallToPlace);
                    }
                }

                state.checkWinCondition();
                if (!state.isGameOver()) {
                    state.nextTurn();
                }
            }

            if (state.isGameOver() && state.getWinner() == state.getPlayers()[pnIdx]) {
                wins++;
            }
        }

        return wins;
    }

    /**
     * Record of a move with its quality metric.
     */
    private static class MoveRecord {
        final double[] features;
        final int turn;
        final double moveQuality;

        MoveRecord(double[] features, int turn, double moveQuality) {
            this.features = features;
            this.turn = turn;
            this.moveQuality = moveQuality;
        }
    }
}
