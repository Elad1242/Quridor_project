package ml;

import logic.MoveValidator;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.io.IOException;
import java.util.List;

/**
 * A bot that uses the trained neural network to evaluate positions and choose actions.
 *
 * Decision process:
 * 1. Generate all legal moves and wall placements
 * 2. For each action: simulate on a deep copy of the game state
 * 3. Extract features from the resulting state (from opponent's perspective)
 * 4. Run features through the NN to get a win probability
 * 5. Pick the action where the OPPONENT's win probability is LOWEST
 *    (which means OUR position is best)
 *
 * This is a "one-ply lookahead" evaluator — the NN acts as the evaluation function.
 */
public class NNBot {

    private final NeuralNetwork nn;

    /**
     * Creates an NNBot with the given trained network.
     */
    public NNBot(NeuralNetwork nn) {
        this.nn = nn;
    }

    /**
     * Creates an NNBot by loading weights from a file.
     */
    public NNBot(String weightsFile) throws IOException {
        this.nn = NeuralNetwork.load(weightsFile);
    }

    /**
     * Chooses the best action for the current player.
     *
     * @param state the current game state
     * @return the best action (move or wall), or null if no action available
     */
    public Action chooseBestAction(GameState state) {
        Player me = state.getCurrentPlayer();
        Player opp = state.getOtherPlayer();

        Action bestAction = null;
        double bestScore = -1.0;

        // === Evaluate all possible moves ===
        List<Position> validMoves = MoveValidator.getValidMoves(state, me);
        for (Position move : validMoves) {
            GameState copy = state.deepCopy();
            copy.getCurrentPlayer().setPosition(move);
            copy.nextTurn(); // now it's opponent's turn

            // Extract features from opponent's perspective and get THEIR win probability
            double[] rawFeatures = GameFeatures.extract(copy);
            double[] normalized = NNTrainer.normalizeSingle(rawFeatures);
            double oppWinProb = nn.predict(normalized);

            // Our score = 1 - opponent's win probability
            double myScore = 1.0 - oppWinProb;

            if (myScore > bestScore) {
                bestScore = myScore;
                bestAction = Action.move(move, myScore);
            }
        }

        // === Evaluate all possible wall placements ===
        if (me.getWallsRemaining() > 0) {
            for (int row = 0; row < 8; row++) {
                for (int col = 0; col < 8; col++) {
                    for (boolean horizontal : new boolean[]{true, false}) {
                        Wall wall = new Wall(row, col, horizontal, state.getCurrentPlayerIndex());
                        if (!WallValidator.isValidWallPlacement(state, wall)) continue;

                        GameState copy = state.deepCopy();
                        copy.addWall(wall);
                        copy.nextTurn();

                        double[] rawFeatures = GameFeatures.extract(copy);
                        double[] normalized = NNTrainer.normalizeSingle(rawFeatures);
                        double oppWinProb = nn.predict(normalized);

                        double myScore = 1.0 - oppWinProb;

                        if (myScore > bestScore) {
                            bestScore = myScore;
                            bestAction = Action.wall(wall, myScore);
                        }
                    }
                }
            }
        }

        return bestAction;
    }

    /**
     * Evaluates the current position's win probability for the current player.
     * Useful for debugging and analysis.
     */
    public double evaluatePosition(GameState state) {
        double[] rawFeatures = GameFeatures.extract(state);
        double[] normalized = NNTrainer.normalizeSingle(rawFeatures);
        return nn.predict(normalized);
    }

    // ==================== ACTION DATA CLASS ====================

    /**
     * Represents a bot action: either a move or a wall placement.
     */
    public static class Action {
        public enum Type { MOVE, WALL }

        public final Type type;
        public final Position moveTarget;
        public final Wall wall;
        public final double score;

        private Action(Type type, Position moveTarget, Wall wall, double score) {
            this.type = type;
            this.moveTarget = moveTarget;
            this.wall = wall;
            this.score = score;
        }

        public static Action move(Position target, double score) {
            return new Action(Type.MOVE, target, null, score);
        }

        public static Action wall(Wall wall, double score) {
            return new Action(Type.WALL, null, wall, score);
        }

        @Override
        public String toString() {
            if (type == Type.MOVE) {
                return "NN-MOVE to " + moveTarget + " (score: " + String.format("%.3f", score) + ")";
            } else {
                return "NN-WALL " + wall + " (score: " + String.format("%.3f", score) + ")";
            }
        }
    }
}
