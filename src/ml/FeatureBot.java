// v2.0 — refactored and cleaned, May 2026
package ml;

import logic.MoveValidator;
import logic.PathFinder;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

// ML bot — scores every legal action with the neural network and picks the best.
public class FeatureBot {

    // pick randomly among actions within this margin of the top score (for variety)
    private static final double TOPK_THRESHOLD = 0.01;

    private final NeuralNetwork nn;
    private final Random rng = new Random();

    public FeatureBot(NeuralNetwork nn) {
        this.nn = nn;
    }

    public FeatureBot(String modelPath) throws Exception {
        this.nn = NeuralNetwork.load(modelPath);
    }

    public Action computeBestAction(GameState state) {
        Player me  = state.getCurrentPlayer();
        Player opp = state.getOtherPlayer();
        List<Position> validMoves = MoveValidator.getValidMoves(state, me);

        // instant win check
        for (Position move : validMoves) {
            if (move.getRow() == me.getGoalRow()) {
                return new Action(Action.Type.MOVE, move, null, 1000.0);
            }
        }

        // emergency blocking — opponent is about to win
        int oppDist = PathFinder.aStarShortestPath(state, opp);
        if (oppDist >= 0 && oppDist <= 2 && me.getWallsRemaining() > 0) {
            Action bestWall = evaluateBestWall(state, me, opp);
            if (bestWall != null) return bestWall;
        }

        List<Action> candidates = new ArrayList<>();
        double bestScore = Double.NEGATIVE_INFINITY;

        for (Position move : validMoves) {
            double[] features = GameFeatures.extractForMove(state, move);
            double score = nn.predict(features);
            candidates.add(new Action(Action.Type.MOVE, move, null, score));
            if (score > bestScore) bestScore = score;
        }

        // try all valid walls that actually slow the opponent down
        if (me.getWallsRemaining() > 0) {
            int myDist = PathFinder.aStarShortestPath(state, me);

            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    for (int orient = 0; orient < 2; orient++) {
                        Wall.Orientation o = (orient == 0)
                                ? Wall.Orientation.HORIZONTAL : Wall.Orientation.VERTICAL;
                        Wall wall = new Wall(r, c, o);
                        if (!WallValidator.isValidWallPlacement(state, wall)) continue;

                        int oppAfter = PathFinder.aStarWithWall(state, opp, wall);
                        int myAfter  = PathFinder.aStarWithWall(state, me,  wall);

                        // quick filter: skip walls that don't help or hurt us too much
                        if (oppAfter < 0 || myAfter < 0) continue;
                        if (oppAfter <= oppDist) continue;
                        if (myAfter - myDist >= 3) continue;

                        double[] features = GameFeatures.extractForWall(state, wall);
                        double score = nn.predict(features);
                        candidates.add(new Action(Action.Type.WALL, null, wall, score));
                        if (score > bestScore) bestScore = score;
                    }
                }
            }
        }

        if (candidates.isEmpty()) return new Action(Action.Type.MOVE, validMoves.get(0), null, 0);

        // pick randomly from near-best candidates for variety
        List<Action> topK = new ArrayList<>();
        for (Action a : candidates) {
            if (a.score >= bestScore - TOPK_THRESHOLD) topK.add(a);
        }

        return topK.get(rng.nextInt(topK.size()));
    }

    // wall-only evaluation for emergency blocking
    private Action evaluateBestWall(GameState state, Player me, Player opp) {
        Action bestWall = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                for (int orient = 0; orient < 2; orient++) {
                    Wall.Orientation o = (orient == 0)
                            ? Wall.Orientation.HORIZONTAL : Wall.Orientation.VERTICAL;
                    Wall wall = new Wall(r, c, o);
                    if (!WallValidator.isValidWallPlacement(state, wall)) continue;

                    int oppAfter = PathFinder.aStarWithWall(state, opp, wall);
                    int myAfter  = PathFinder.aStarWithWall(state, me,  wall);
                    if (oppAfter < 0 || myAfter < 0) continue;

                    double[] features = GameFeatures.extractForWall(state, wall);
                    double score = nn.predict(features);
                    if (score > bestScore) {
                        bestScore = score;
                        bestWall = new Action(Action.Type.WALL, null, wall, score);
                    }
                }
            }
        }

        return bestWall;
    }

    public static class Action {
        public enum Type { MOVE, WALL }

        public final Type type;
        public final Position moveTarget;
        public final Wall wall;
        public final double score;

        public Action(Type type, Position moveTarget, Wall wall, double score) {
            this.type = type;
            this.moveTarget = moveTarget;
            this.wall = wall;
            this.score = score;
        }

        public void applyTo(GameState state) {
            if (type == Type.MOVE) {
                state.getCurrentPlayer().setPosition(moveTarget);
            } else {
                wall.setOwnerIndex(state.getCurrentPlayerIndex());
                state.addWall(wall);
            }
        }
    }
}
