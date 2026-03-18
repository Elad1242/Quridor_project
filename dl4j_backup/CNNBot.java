package ml.cnn;

import logic.MoveValidator;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * CNN-based Quoridor Bot using PURE Machine Learning.
 *
 * This bot evaluates all possible actions by predicting the quality of
 * the resulting board state using a trained CNN. NO heuristics or
 * hand-crafted features are used in decision making.
 *
 * The CNN has learned spatial patterns from thousands of games:
 * - Wall configurations that block/help paths
 * - Good vs bad pawn positions relative to goals
 * - Strategic resource management (walls remaining)
 */
public class CNNBot {

    private final QuoridorCNN cnn;
    private double temperature = 0.0;  // 0 = greedy, higher = more exploration
    private int maxWallsToEvaluate = 100;  // Limit for speed
    private final Random random = new Random();

    /**
     * Creates a CNNBot with a pre-trained model.
     *
     * @param modelPath Path to saved CNN model file
     */
    public CNNBot(String modelPath) {
        this.cnn = new QuoridorCNN(modelPath);
    }

    /**
     * Creates a CNNBot with an existing CNN instance.
     *
     * @param cnn Pre-loaded CNN model
     */
    public CNNBot(QuoridorCNN cnn) {
        this.cnn = cnn;
    }

    /**
     * Sets the exploration temperature.
     *
     * @param temperature 0.0 = always pick best, higher = more random exploration
     */
    public void setTemperature(double temperature) {
        this.temperature = temperature;
    }

    /**
     * Sets the maximum number of walls to evaluate per turn.
     *
     * @param max Maximum walls (lower = faster but might miss good walls)
     */
    public void setMaxWallsToEvaluate(int max) {
        this.maxWallsToEvaluate = max;
    }

    /**
     * Computes the best action using PURE ML evaluation.
     *
     * The CNN predicts the quality of each possible resulting state.
     * The action leading to the highest-quality state is chosen.
     *
     * @param state Current game state
     * @return Best action (move or wall) according to CNN
     */
    public BotAction computeBestAction(GameState state) {
        Player me = state.getCurrentPlayer();
        List<ScoredAction> scoredActions = new ArrayList<>();

        // Evaluate all valid moves
        List<Position> validMoves = MoveValidator.getValidMoves(state, me);
        for (Position move : validMoves) {
            INDArray stateAfter = BoardEncoder.encodeAfterMove(state, move);
            double score = cnn.predict(stateAfter);

            BotAction action = new BotAction(BotAction.Type.MOVE, move, null);
            scoredActions.add(new ScoredAction(action, score));
        }

        // Evaluate valid walls (with sampling for speed)
        if (me.getWallsRemaining() > 0) {
            List<Wall> validWalls = getValidWalls(state);

            // Sample if too many
            if (validWalls.size() > maxWallsToEvaluate) {
                Collections.shuffle(validWalls, random);
                validWalls = validWalls.subList(0, maxWallsToEvaluate);
            }

            for (Wall wall : validWalls) {
                INDArray stateAfter = BoardEncoder.encodeAfterWall(state, wall);
                double score = cnn.predict(stateAfter);

                BotAction action = new BotAction(BotAction.Type.WALL, null, wall);
                scoredActions.add(new ScoredAction(action, score));
            }
        }

        if (scoredActions.isEmpty()) {
            return null;
        }

        // Select action based on temperature
        if (temperature <= 0.0) {
            // Greedy: pick best
            ScoredAction best = scoredActions.get(0);
            for (ScoredAction sa : scoredActions) {
                if (sa.score > best.score) {
                    best = sa;
                }
            }
            return best.action;
        } else {
            // Softmax sampling with temperature
            return softmaxSample(scoredActions);
        }
    }

    /**
     * Gets all valid wall placements for current player.
     */
    private List<Wall> getValidWalls(GameState state) {
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
     * Softmax sampling for action selection.
     */
    private BotAction softmaxSample(List<ScoredAction> actions) {
        // Apply temperature scaling
        double maxScore = Double.NEGATIVE_INFINITY;
        for (ScoredAction sa : actions) {
            maxScore = Math.max(maxScore, sa.score);
        }

        double[] expScores = new double[actions.size()];
        double sumExp = 0.0;

        for (int i = 0; i < actions.size(); i++) {
            // Subtract max for numerical stability
            double scaled = (actions.get(i).score - maxScore) / temperature;
            expScores[i] = Math.exp(scaled);
            sumExp += expScores[i];
        }

        // Sample from distribution
        double rand = random.nextDouble() * sumExp;
        double cumSum = 0.0;

        for (int i = 0; i < actions.size(); i++) {
            cumSum += expScores[i];
            if (rand <= cumSum) {
                return actions.get(i).action;
            }
        }

        // Fallback to last action
        return actions.get(actions.size() - 1).action;
    }

    /**
     * Represents a bot action (move or wall placement).
     */
    public static class BotAction {
        public enum Type { MOVE, WALL }

        public final Type type;
        public final Position moveTarget;
        public final Wall wallToPlace;

        public BotAction(Type type, Position moveTarget, Wall wallToPlace) {
            this.type = type;
            this.moveTarget = moveTarget;
            this.wallToPlace = wallToPlace;
        }

        @Override
        public String toString() {
            if (type == Type.MOVE) {
                return "MOVE to " + moveTarget;
            } else {
                return "WALL at " + wallToPlace.getPosition() +
                        " (" + wallToPlace.getOrientation() + ")";
            }
        }
    }

    /**
     * Internal class for scored actions.
     */
    private static class ScoredAction {
        final BotAction action;
        final double score;

        ScoredAction(BotAction action, double score) {
            this.action = action;
            this.score = score;
        }
    }
}
