// v2.0 — refactored and cleaned, May 2026
package ml;

import model.GameState;
import model.Position;
import model.Wall;

// GUI-facing wrapper around FeatureBot. Keeps the API that GameController expects.
public class MLBot {

    private final FeatureBot inner;

    // modelDir and modelName are ignored — always loads feature_model.bin
    public MLBot(String modelDir, String modelName) throws Exception {
        this.inner = new FeatureBot("feature_model.bin");
    }

    public MLBot(FeatureBot bot) {
        this.inner = bot;
    }

    public Action computeBestAction(GameState state) {
        FeatureBot.Action a = inner.computeBestAction(state);
        if (a == null) return null;

        Action.Type t = (a.type == FeatureBot.Action.Type.MOVE) ? Action.Type.MOVE : Action.Type.WALL;
        return new Action(t, a.moveTarget, a.wall, (float) a.score);
    }

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
            if (type == Type.MOVE) return "MOVE to " + moveTarget + " (" + score + ")";
            return "WALL at " + wall.getPosition() + " " + wall.getOrientation() + " (" + score + ")";
        }
    }
}
