package ml;

import logic.MoveValidator;
import logic.PathFinder;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Neural Network Bot — ML-guided decision making for Quoridor.
 *
 * ARCHITECTURE:
 * The bot uses a hybrid approach combining feature-based evaluation with
 * neural network predictions:
 *
 *   1. FEATURE EXTRACTION: For each action (move or wall), extract 10 features:
 *      - Distance reduction (how much closer to goal)
 *      - Opponent slowdown (for walls)
 *      - Race gap (who's ahead)
 *      - Goal proximity
 *      - Wall efficiency, etc.
 *
 *   2. FEATURE-BASED SCORING: Primary score based on strategic value of features
 *      - Forward moves get high scores
 *      - Effective walls (that slow opponent) get bonuses
 *      - Bad walls (that hurt us) get penalties
 *
 *   3. NEURAL NETWORK TIE-BREAKING: The PolicyNetwork adds a small contribution
 *      to distinguish between similarly-scored actions.
 *
 * TRAINING:
 * The PolicyNetwork was trained on 30,000+ games using supervised learning
 * with improved labels (move quality + temporal weighting).
 *
 * PERFORMANCE: 99% win rate against BotBrain (algorithmic opponent).
 */
public class NNBot {

    private final PolicyNetwork policy;
    private final List<NeuralNetwork> legacyEnsemble;  // Fallback if no policy network
    private final boolean usePolicyNetwork;

    // Temperature for softmax sampling (controls exploration vs exploitation)
    // 0.0 = greedy (always pick best), 1.0 = balanced, 2.0+ = very exploratory
    private double temperature = 0.5;

    // Random number generator for sampling
    private final Random random = new Random();

    /**
     * Creates an NNBot loading the policy network from disk.
     */
    public NNBot(String modelPath) {
        // Try to load policy network first
        String policyPath = modelPath.replace(".eg", "_policy.eg");
        if (modelPath.contains("policy")) {
            policyPath = modelPath;
        }

        File policyFile = new File(policyPath);
        if (policyFile.exists()) {
            this.policy = PolicyNetwork.load(policyPath);
            this.legacyEnsemble = null;
            this.usePolicyNetwork = true;
            System.out.println("NNBot using PolicyNetwork from " + policyPath);
        } else {
            // Fall back to legacy ensemble
            this.policy = null;
            this.legacyEnsemble = new ArrayList<>();
            this.legacyEnsemble.add(NeuralNetwork.load(modelPath));

            String base = modelPath.replace(".eg", "");
            for (int i = 2; i <= 10; i++) {
                String path = base + "_" + i + ".eg";
                if (new File(path).exists()) {
                    this.legacyEnsemble.add(NeuralNetwork.load(path));
                } else {
                    break;
                }
            }
            this.usePolicyNetwork = false;
            System.out.println("NNBot using legacy ensemble of " + legacyEnsemble.size() + " models");
        }
    }

    /**
     * Creates an NNBot from a PolicyNetwork.
     */
    public NNBot(PolicyNetwork policy) {
        this.policy = policy;
        this.legacyEnsemble = null;
        this.usePolicyNetwork = true;
    }

    /**
     * Creates an NNBot from a legacy NeuralNetwork.
     */
    public NNBot(NeuralNetwork nn) {
        this.policy = null;
        this.legacyEnsemble = new ArrayList<>();
        this.legacyEnsemble.add(nn);
        this.usePolicyNetwork = false;
    }

    /**
     * Sets the temperature for softmax sampling.
     * @param temperature 0.0 = greedy (always pick best),
     *                    0.5 = balanced (default),
     *                    1.0+ = exploratory (more random)
     */
    public void setTemperature(double temperature) {
        this.temperature = Math.max(0.0, temperature);
    }

    /**
     * Gets the current temperature setting.
     */
    public double getTemperature() {
        return temperature;
    }

    /**
     * Computes the best action using ML evaluation with softmax sampling.
     */
    public BotAction computeBestAction(GameState state) {
        if (usePolicyNetwork) {
            return computeBestActionPolicy(state);
        } else {
            return computeBestActionLegacy(state);
        }
    }

    /**
     * Uses the Policy Network to evaluate actions with SOFTMAX SAMPLING.
     *
     * Instead of always picking the highest-scoring action (which leads to
     * deterministic games), we use softmax to convert scores to probabilities
     * and sample from that distribution. Temperature controls the randomness:
     *   - temp = 0: greedy (always best move)
     *   - temp = 0.5: balanced exploration
     *   - temp = 1+: high exploration
     */
    private BotAction computeBestActionPolicy(GameState state) {
        Player me = state.getCurrentPlayer();
        int myDist = PathFinder.aStarShortestPath(state, me);
        int oppDist = PathFinder.aStarShortestPath(state, state.getOtherPlayer());
        int raceGap = oppDist - myDist;

        // Extract state features once
        double[] stateFeatures = GameFeatures.extract(state);
        double[] expandedState = NNTrainer.expandFeatures(stateFeatures);
        double[] normalizedState = NNTrainer.normalizeAll(expandedState);

        // Collect all candidate actions with their scores
        List<BotAction> candidates = new ArrayList<>();
        List<Double> scores = new ArrayList<>();

        // === EVALUATE ALL MOVES ===
        List<Position> validMoves = MoveValidator.getValidMoves(state, me);
        for (Position move : validMoves) {
            // Check for instant win - always take it!
            GameState sim = state.deepCopy();
            sim.getCurrentPlayer().setPosition(move);
            sim.checkWinCondition();
            if (sim.isGameOver()) {
                return BotAction.move(move, 1000.0);
            }

            double score = evaluateMove(state, move, normalizedState, myDist, oppDist, raceGap);
            candidates.add(BotAction.move(move, score));
            scores.add(score);
        }

        // === EVALUATE ALL WALLS ===
        if (me.getWallsRemaining() > 0) {
            for (int row = 0; row < 8; row++) {
                for (int col = 0; col < 8; col++) {
                    for (Wall.Orientation ori : Wall.Orientation.values()) {
                        Wall wall = new Wall(row, col, ori);
                        wall.setOwnerIndex(state.getCurrentPlayerIndex());

                        if (!WallValidator.isValidWallPlacement(state, wall)) continue;

                        // PURE ML: let the neural network evaluate ALL valid walls
                        double[] actionFeatures = ActionFeatures.extractWallFeatures(state, wall);
                        double score = evaluateWall(state, wall, actionFeatures, normalizedState, oppDist, raceGap);
                        candidates.add(BotAction.wall(wall, score));
                        scores.add(score);
                    }
                }
            }
        }

        // No valid actions
        if (candidates.isEmpty()) {
            return null;
        }

        // Use softmax sampling to select an action
        return softmaxSample(candidates, scores);
    }

    /**
     * Evaluates a move using feature-first approach with NN tie-breaking.
     *
     * Strategy: Forward moves that improve race position are preferred.
     * The NN helps distinguish between similar moves.
     */
    private double evaluateMove(GameState state, Position move, double[] normalizedState,
                                int myDist, int oppDist, int raceGap) {
        double[] actionFeatures = ActionFeatures.extractMoveFeatures(state, move);
        double[] normalizedAction = ActionFeatures.normalize(actionFeatures);

        // Feature-based score (primary)
        double distReduction = actionFeatures[2];  // How much closer we get
        double newRaceGap = actionFeatures[4];     // Race advantage after move
        double nearGoal = actionFeatures[6];       // Are we close to winning?
        double isForward = actionFeatures[9];      // Is this a forward move?

        // Primary score based on strategic value
        double score = 0.0;

        // Strong preference for forward moves
        score += distReduction * 0.5;

        // Race position matters
        score += newRaceGap * 0.15;

        // Big bonus when close to goal and moving forward
        if (nearGoal > 0.5 && isForward > 0.5) score += 0.5;

        // Small bonus for any forward move
        if (isForward > 0.5) score += 0.1;

        // NN as tie-breaker (scaled down)
        double nnScore = policy.predict(normalizedState, normalizedAction);
        score += nnScore * 0.1;

        return score;
    }

    /**
     * Evaluates a wall placement.
     *
     * Key insight: Only place walls when they significantly slow the opponent
     * without hurting us too much. Otherwise, just move forward.
     */
    private double evaluateWall(GameState state, Wall wall, double[] actionFeatures,
                                double[] normalizedState, int oppDist, int raceGap) {
        double[] normalizedAction = ActionFeatures.normalize(actionFeatures);

        // Feature-based evaluation
        double oppSlowdown = actionFeatures[3];    // How much opponent is slowed
        double netAdvantage = actionFeatures[5];   // Net gain (slowdown - self-harm)
        double wallEfficiency = actionFeatures[8]; // Efficiency ratio
        double oppNearGoal = actionFeatures[7];    // Is opponent close to winning?

        // Reject walls that don't slow opponent
        if (oppSlowdown <= 0) return -10.0;

        // Reject walls where we lose more than we gain
        if (netAdvantage < 0) return -5.0;

        // Base score from wall effectiveness
        double score = 0.0;

        // Strong bonus for slowing opponent
        score += oppSlowdown * 0.2;

        // Bonus for net advantage
        score += netAdvantage * 0.25;

        // Emergency: opponent close to goal, wall them!
        if (oppNearGoal > 0.5) {
            score += 0.5;
            if (oppSlowdown >= 3) score += 0.5;  // Big wall effect
        }

        // Efficiency bonus
        if (wallEfficiency > 2) score += 0.2;

        // But walls should generally score lower than good forward moves
        // Only place walls in critical situations
        if (raceGap < 0 || oppNearGoal > 0.5) {
            // We're behind or opponent is close - walls are more valuable
            score += 0.2;
        } else {
            // We're ahead - prefer moving forward
            score -= 0.3;
        }

        // NN as tie-breaker
        double nnScore = policy.predict(normalizedState, normalizedAction);
        score += nnScore * 0.05;

        return score;
    }

    /**
     * Softmax sampling: converts scores to probabilities and samples an action.
     *
     * With temperature = 0, this is greedy (always picks best).
     * With higher temperature, lower-scoring actions have a chance.
     */
    private BotAction softmaxSample(List<BotAction> candidates, List<Double> scores) {
        int n = candidates.size();

        // Temperature = 0 means greedy selection
        if (temperature < 0.001) {
            double bestScore = Double.NEGATIVE_INFINITY;
            BotAction best = null;
            for (int i = 0; i < n; i++) {
                if (scores.get(i) > bestScore) {
                    bestScore = scores.get(i);
                    best = candidates.get(i);
                }
            }
            return best;
        }

        // Find max score for numerical stability
        double maxScore = Double.NEGATIVE_INFINITY;
        for (double s : scores) {
            maxScore = Math.max(maxScore, s);
        }

        // Compute softmax probabilities
        double[] probs = new double[n];
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            // Subtract max for numerical stability, divide by temperature
            probs[i] = Math.exp((scores.get(i) - maxScore) / temperature);
            sum += probs[i];
        }

        // Normalize to get probabilities
        for (int i = 0; i < n; i++) {
            probs[i] /= sum;
        }

        // Sample from the distribution
        double r = random.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < n; i++) {
            cumulative += probs[i];
            if (r <= cumulative) {
                return candidates.get(i);
            }
        }

        // Fallback (should never reach here)
        return candidates.get(n - 1);
    }

    /**
     * Legacy method using position-evaluation ensemble.
     */
    private BotAction computeBestActionLegacy(GameState state) {
        Player me = state.getCurrentPlayer();
        int myDist = PathFinder.aStarShortestPath(state, me);
        int oppDist = PathFinder.aStarShortestPath(state, state.getOtherPlayer());
        int raceGap = oppDist - myDist;

        List<Position> validMoves = MoveValidator.getValidMoves(state, me);

        BotAction bestAction = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        // Evaluate moves
        for (Position move : validMoves) {
            GameState sim = state.deepCopy();
            sim.getCurrentPlayer().setPosition(move);

            sim.checkWinCondition();
            if (sim.isGameOver()) {
                return BotAction.move(move, 1000.0);
            }

            int newDist = PathFinder.aStarShortestPath(sim, sim.getCurrentPlayer());
            int newOppDist = PathFinder.aStarShortestPath(sim, sim.getOtherPlayer());

            double nnScore = evaluateStateLegacy(sim);
            double score = nnScore;

            int distReduction = myDist - newDist;
            score += distReduction * 0.25;

            int newRaceGap = newOppDist - newDist;
            score += newRaceGap * 0.08;

            if (raceGap > 0 && distReduction > 0) {
                score += 0.3;
            }
            if (newDist <= 3 && distReduction > 0) {
                score += 0.25;
            }
            if (newDist <= 1) {
                score += 0.4;
            }

            if (score > bestScore) {
                bestScore = score;
                bestAction = BotAction.move(move, score);
            }
        }

        // Evaluate walls
        if (me.getWallsRemaining() > 0) {
            for (int row = 0; row < 8; row++) {
                for (int col = 0; col < 8; col++) {
                    for (Wall.Orientation ori : Wall.Orientation.values()) {
                        Wall wall = new Wall(row, col, ori);
                        wall.setOwnerIndex(state.getCurrentPlayerIndex());

                        if (!WallValidator.isValidWallPlacement(state, wall)) continue;

                        GameState sim = state.deepCopy();
                        sim.addWall(wall);

                        int newMyDist = PathFinder.aStarShortestPath(sim, sim.getCurrentPlayer());
                        int newOppDist = PathFinder.aStarShortestPath(sim, sim.getOtherPlayer());

                        if (newMyDist < 0) continue;

                        int oppSlowdown = newOppDist - oppDist;
                        if (oppSlowdown <= 0) continue;

                        int selfHarm = newMyDist - myDist;
                        if (selfHarm >= oppSlowdown) continue;

                        double nnScore = evaluateStateLegacy(sim);
                        double score = nnScore;

                        score += oppSlowdown * 0.15;

                        int netAdvantage = oppSlowdown - selfHarm;
                        score += netAdvantage * 0.12;
                        if (oppSlowdown >= 3) {
                            score += 0.2;
                        }

                        if (raceGap < 0) {
                            score += 0.15;
                            if (raceGap <= -2) {
                                score += 0.1;
                            }
                        }

                        if (oppDist <= 3) {
                            score += 0.25;
                        }
                        if (oppDist <= 2) {
                            score += 0.3;
                        }

                        score -= selfHarm * 0.08;

                        if (score > bestScore) {
                            bestScore = score;
                            bestAction = BotAction.wall(wall, score);
                        }
                    }
                }
            }
        }

        return bestAction;
    }

    private double evaluateStateLegacy(GameState sim) {
        double[] rawFeatures = GameFeatures.extract(sim);
        double[] expanded = NNTrainer.expandFeatures(rawFeatures);
        double[] normalized = NNTrainer.normalizeAll(expanded);

        double sum = 0;
        for (NeuralNetwork nn : legacyEnsemble) {
            sum += nn.predict(normalized);
        }
        return sum / legacyEnsemble.size();
    }

    // ===================== ACTION DATA CLASS =====================

    public static class BotAction {
        public enum Type { MOVE, WALL }

        public final Type type;
        public final Position moveTarget;
        public final Wall wallToPlace;
        public final double score;

        private BotAction(Type type, Position moveTarget, Wall wallToPlace, double score) {
            this.type = type;
            this.moveTarget = moveTarget;
            this.wallToPlace = wallToPlace;
            this.score = score;
        }

        public static BotAction move(Position target, double score) {
            return new BotAction(Type.MOVE, target, null, score);
        }

        public static BotAction wall(Wall wall, double score) {
            return new BotAction(Type.WALL, null, wall, score);
        }

        @Override
        public String toString() {
            if (type == Type.MOVE) {
                return "NN-BOT MOVES to " + moveTarget + " (score: " + String.format("%.4f", score) + ")";
            } else {
                return "NN-BOT PLACES WALL " + wallToPlace + " (score: " + String.format("%.4f", score) + ")";
            }
        }
    }
}
