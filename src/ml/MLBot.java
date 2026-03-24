package ml;

import logic.MoveValidator;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.ArrayList;
import java.util.List;

/**
 * Pure ML Bot — uses ONLY the trained CNN to make decisions.
 *
 * No PathFinder, no A*, no BFS, no heuristics.
 * Only uses MoveValidator/WallValidator to enumerate legal actions.
 *
 * Decision loop:
 *   1. Get all legal pawn moves and wall placements
 *   2. For each action, simulate it and encode the resulting board
 *   3. Evaluate with CNN from opponent's perspective (since it's their turn after)
 *   4. Pick action where opponent's value is LOWEST (= best for us)
 */
public class MLBot {

    private final ValueNetwork network;

    public MLBot(ValueNetwork network) {
        this.network = network;
    }

    /**
     * Creates MLBot by loading a trained model.
     */
    public MLBot(String modelDir, String modelName) throws Exception {
        this.network = new ValueNetwork();
        this.network.load(modelDir, modelName);
    }

    /**
     * Computes the best action using pure ML evaluation.
     *
     * @return BotAction (move or wall) with the highest CNN score
     */
    public Action computeBestAction(GameState state) {
        Player me = state.getCurrentPlayer();
        List<Action> candidates = new ArrayList<>();

        // 1. Evaluate all legal pawn moves
        List<Position> validMoves = MoveValidator.getValidMoves(state, me);
        for (Position move : validMoves) {
            // Check instant win
            if (move.getRow() == me.getGoalRow()) {
                return new Action(Action.Type.MOVE, move, null, 1000.0f);
            }

            // Encode state after move from OPPONENT's perspective
            float[][][] encoded = BoardEncoder.encodeAfterMove(state, move);
            float oppValue = network.evaluate(encoded);
            // Our value = 1 - opponent's value
            float score = 1.0f - oppValue;
            candidates.add(new Action(Action.Type.MOVE, move, null, score));
        }

        // 2. Evaluate all legal wall placements
        if (me.getWallsRemaining() > 0) {
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    for (Wall.Orientation ori : Wall.Orientation.values()) {
                        Wall wall = new Wall(r, c, ori);
                        if (!WallValidator.isValidWallPlacement(state, wall)) continue;

                        // Encode state after wall from OPPONENT's perspective
                        float[][][] encoded = BoardEncoder.encodeAfterWall(state, wall);
                        float oppValue = network.evaluate(encoded);
                        float score = 1.0f - oppValue;
                        candidates.add(new Action(Action.Type.WALL, null, wall, score));
                    }
                }
            }
        }

        // 3. Pick the action with the highest score
        Action best = null;
        for (Action a : candidates) {
            if (best == null || a.score > best.score) {
                best = a;
            }
        }

        return best;
    }

    public void close() {
        network.close();
    }

    /**
     * Represents a bot action (move or wall).
     */
    public static class Action {
        public enum Type { MOVE, WALL }

        public final Type type;
        public final Position moveTarget;
        public final Wall wall;
        public final float score;

        public Action(Type type, Position moveTarget, Wall wall, float score) {
            this.type = type;
            this.moveTarget = moveTarget;
            this.wall = wall;
            this.score = score;
        }

        @Override
        public String toString() {
            if (type == Type.MOVE) {
                return String.format("MOVE to %s (%.3f)", moveTarget, score);
            } else {
                return String.format("WALL at %s %s (%.3f)",
                        wall.getPosition(), wall.getOrientation(), score);
            }
        }
    }
}
