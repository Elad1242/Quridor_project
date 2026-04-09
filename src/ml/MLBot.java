package ml;

import model.GameState;
import model.Position;
import model.Wall;

/**
 * MLBot — GUI-facing wrapper around FeatureBot (pure-Java feedforward NN).
 *
 * Loads feature_model.bin (our from-scratch neural network trained on
 * 27 features: 22 global + 5 per-action). No external ML libraries.
 *
 * Keeps the exact API expected by GameController so the GUI needs no changes.
 */
public class MLBot {

    private final FeatureBot inner;

    /**
     * Legacy constructor kept for GameController compatibility.
     * Ignores the arguments and loads "feature_model.bin" from the working directory.
     */
    public MLBot(String modelDir, String modelName) throws Exception {
        this.inner = new FeatureBot("feature_model.bin");
    }

    public MLBot(FeatureBot bot) {
        this.inner = bot;
    }

    public Action computeBestAction(GameState state) {
        FeatureBot.Action a = inner.computeBestAction(state);
        if (a == null) return null;

        Action.Type t = (a.type == FeatureBot.Action.Type.MOVE)
                ? Action.Type.MOVE : Action.Type.WALL;
        return new Action(t, a.moveTarget, a.wall, (float) a.score);
    }

    public void close() {
        // FeatureBot holds a pure-Java NN, nothing to close
    }

    // === Action class — same shape as the old CNN MLBot.Action ===

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
            if (type == Type.MOVE) return String.format("MOVE to %s (%.3f)", moveTarget, score);
            return String.format("WALL at %s %s (%.3f)", wall.getPosition(), wall.getOrientation(), score);
        }
    }
}
