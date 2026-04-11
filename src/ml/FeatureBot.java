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

/**
 * Pure ML Bot using 27-feature evaluation (22 global + 5 per-action).
 *
 * For each legal action (move or wall):
 *   1. Compute 27 features (global position + action-specific quality metrics)
 *   2. NN predicts: how good is this action? (0=bad, 1=good)
 *   3. Pick the action with the highest score
 *
 * The per-action features tell the NN:
 *   - For moves: pathGain, rowAdvance, isOnAstarPath
 *   - For walls: pathDamage, selfHarm, netDamage
 *   - actionType: 0=move, 1=wall
 *
 * No 1-ply simulation needed. No wall-bias. Pure ML decision-making.
 * PathFinder is only used for feature extraction (input computation).
 */
public class FeatureBot {

    /** Top-k sampling: pick randomly from actions within this score of the best.
     *  Small enough to stay close to argmax (preserves playing strength),
     *  large enough to break ties between near-identical options so games vary.
     *  Can be overridden with -Dfeaturebot.topk=0 to force pure argmax for tests. */
    private static final double TOPK_THRESHOLD =
            Double.parseDouble(System.getProperty("featurebot.topk", "0.01"));

    private final NeuralNetwork nn;
    private final Random rng = new Random();

    public FeatureBot(NeuralNetwork nn) {
        this.nn = nn;
    }

    public FeatureBot(String modelPath) throws Exception {
        this.nn = NeuralNetwork.load(modelPath);
    }

    public Action computeBestAction(GameState state) {
        // Collect every candidate action with its NN score, then sample from
        // the near-best set instead of argmax. This produces visibly different
        // games without meaningfully weakening the bot, since we only sample
        // among actions whose score is within TOPK_THRESHOLD of the top.
        Player me = state.getCurrentPlayer();
        Player opp = state.getOtherPlayer();
        List<Position> validMoves = MoveValidator.getValidMoves(state, me);

        // === INSTANT WIN ===
        for (Position move : validMoves) {
            if (move.getRow() == me.getGoalRow()) {
                return new Action(Action.Type.MOVE, move, null, 1000.0);
            }
        }

        // === EMERGENCY BLOCKING ===
        int oppDist = PathFinder.aStarShortestPath(state, opp);
        if (oppDist >= 0 && oppDist <= 2 && me.getWallsRemaining() > 0) {
            // Still use NN to pick the best wall, but force wall placement
            Action bestWall = evaluateBestWall(state, me, opp);
            if (bestWall != null) return bestWall;
        }

        // === COLLECT ALL CANDIDATE ACTIONS WITH THEIR NN SCORES ===
        List<Action> candidates = new ArrayList<>();
        double bestScore = Double.NEGATIVE_INFINITY;

        // Evaluate all moves
        for (Position move : validMoves) {
            double[] features = GameFeatures.extractForMove(state, move);
            double score = nn.predict(features);
            candidates.add(new Action(Action.Type.MOVE, move, null, score));
            if (score > bestScore) bestScore = score;
        }

        // Evaluate walls (only if we have walls remaining)
        if (me.getWallsRemaining() > 0) {
            int myDist = PathFinder.aStarShortestPath(state, me);

            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    for (int orient = 0; orient < 2; orient++) {
                        Wall.Orientation o = (orient == 0)
                                ? Wall.Orientation.HORIZONTAL : Wall.Orientation.VERTICAL;
                        Wall wall = new Wall(r, c, o);
                        if (!WallValidator.isValidWallPlacement(state, wall)) continue;

                        // Quick filter: skip walls with no damage or too much self-harm
                        int oppAfter = PathFinder.aStarWithWall(state, opp, wall);
                        int myAfter = PathFinder.aStarWithWall(state, me, wall);
                        if (oppAfter < 0 || myAfter < 0) continue;
                        if (oppAfter <= oppDist) continue; // no damage
                        if (myAfter - myDist >= 3) continue; // too much self-harm

                        double[] features = GameFeatures.extractForWall(state, wall);
                        double score = nn.predict(features);
                        candidates.add(new Action(Action.Type.WALL, null, wall, score));
                        if (score > bestScore) bestScore = score;
                    }
                }
            }
        }

        if (candidates.isEmpty()) {
            return new Action(Action.Type.MOVE, validMoves.get(0), null, 0);
        }

        // Keep only the near-best candidates (within TOPK_THRESHOLD of the top).
        List<Action> topK = new ArrayList<>();
        for (Action a : candidates) {
            if (a.score >= bestScore - TOPK_THRESHOLD) topK.add(a);
        }

        // Sample uniformly from the top-k set. If only one candidate is near
        // the top, this is identical to argmax; if several are tied or close,
        // the bot picks among them randomly, which produces varied but still
        // strong play across games.
        return topK.get(rng.nextInt(topK.size()));
    }

    /**
     * Evaluates walls only (for emergency blocking).
     */
    private Action evaluateBestWall(GameState state, Player me, Player opp) {
        int myDist = PathFinder.aStarShortestPath(state, me);
        int oppDist = PathFinder.aStarShortestPath(state, opp);

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
                    int myAfter = PathFinder.aStarWithWall(state, me, wall);
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

    // === Action class ===

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
