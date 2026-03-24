package ml;

import model.GameState;
import model.Position;
import model.Wall;

/**
 * Legacy NNBot stub — kept for UI compatibility.
 * The actual ML bot is MLBot (pure CNN, no algorithms).
 * This class does nothing useful — it's only here so GameController compiles.
 */
public class NNBot {

    private double temperature = 0.0;

    public NNBot(String modelPath) {
        System.out.println("[NNBot] Legacy stub — use MLBot instead. Model not loaded.");
    }

    public void setTemperature(double t) { this.temperature = t; }

    public BotAction computeBestAction(GameState state) {
        // Fallback: just move forward
        return null;
    }

    public static class BotAction {
        public enum Type { MOVE, WALL }
        public final Type type;
        public final Position moveTarget;
        public final Wall wallToPlace;
        public final double score;

        public BotAction(Type type, Position moveTarget, Wall wallToPlace, double score) {
            this.type = type;
            this.moveTarget = moveTarget;
            this.wallToPlace = wallToPlace;
            this.score = score;
        }
    }
}
