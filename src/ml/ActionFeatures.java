package ml;

import logic.PathFinder;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

/**
 * Extracts features that describe an ACTION (move or wall placement).
 *
 * These features capture what makes an action good or bad:
 * - For moves: progress toward goal, effect on race
 * - For walls: opponent slowdown, self-harm, strategic position
 *
 * Combined with state features, this enables the policy network to
 * learn which actions are good in which situations.
 */
public class ActionFeatures {

    // Action feature indices
    public static final int ACTION_FEATURE_COUNT = 10;

    /**
     * Feature names for documentation.
     */
    public static String[] featureNames() {
        return new String[] {
            "isMove",           // 1 if move, 0 if wall
            "isWall",           // 1 if wall, 0 if move
            "distReduction",    // how much closer to goal (negative = farther)
            "oppDistChange",    // change in opponent's distance (positive = they're slower)
            "newRaceGap",       // oppDist - myDist after action (positive = we're ahead)
            "netAdvantage",     // oppDistChange - selfHarm
            "nearGoal",         // 1 if we're within 3 steps of goal after action
            "oppNearGoal",      // 1 if opponent is within 3 steps of their goal
            "wallEfficiency",   // oppSlowdown / (selfHarm + 1) for walls, 0 for moves
            "isForwardMove"     // 1 if move reduces distance, 0 otherwise
        };
    }

    /**
     * Extracts action features for a MOVE action.
     */
    public static double[] extractMoveFeatures(GameState state, Position moveTarget) {
        double[] features = new double[ACTION_FEATURE_COUNT];

        Player me = state.getCurrentPlayer();
        Player opp = state.getOtherPlayer();

        int myDistBefore = PathFinder.aStarShortestPath(state, me);
        int oppDistBefore = PathFinder.aStarShortestPath(state, opp);

        // Simulate move
        GameState sim = state.deepCopy();
        sim.getCurrentPlayer().setPosition(moveTarget);

        int myDistAfter = PathFinder.aStarShortestPath(sim, sim.getCurrentPlayer());
        int oppDistAfter = PathFinder.aStarShortestPath(sim, sim.getOtherPlayer());

        // Action type
        features[0] = 1.0;  // isMove
        features[1] = 0.0;  // isWall

        // Distance changes
        features[2] = myDistBefore - myDistAfter;  // distReduction (positive = good)
        features[3] = oppDistAfter - oppDistBefore;  // oppDistChange (positive = they're slower)
        features[4] = oppDistAfter - myDistAfter;  // newRaceGap (positive = we're ahead)
        features[5] = features[2];  // netAdvantage (for moves, same as distReduction)

        // Goal proximity
        features[6] = myDistAfter <= 3 ? 1.0 : 0.0;  // nearGoal
        features[7] = oppDistAfter <= 3 ? 1.0 : 0.0;  // oppNearGoal

        // Wall efficiency (N/A for moves)
        features[8] = 0.0;

        // Forward move indicator
        features[9] = features[2] > 0 ? 1.0 : 0.0;

        return features;
    }

    /**
     * Extracts action features for a WALL action.
     */
    public static double[] extractWallFeatures(GameState state, Wall wall) {
        double[] features = new double[ACTION_FEATURE_COUNT];

        Player me = state.getCurrentPlayer();
        Player opp = state.getOtherPlayer();

        int myDistBefore = PathFinder.aStarShortestPath(state, me);
        int oppDistBefore = PathFinder.aStarShortestPath(state, opp);

        // Simulate wall
        GameState sim = state.deepCopy();
        wall.setOwnerIndex(sim.getCurrentPlayerIndex());
        sim.addWall(wall);

        int myDistAfter = PathFinder.aStarShortestPath(sim, sim.getCurrentPlayer());
        int oppDistAfter = PathFinder.aStarShortestPath(sim, sim.getOtherPlayer());

        // Handle invalid walls (would trap someone)
        if (myDistAfter < 0 || oppDistAfter < 0) {
            // Return features that indicate a bad wall
            features[0] = 0.0;
            features[1] = 1.0;
            features[2] = -10.0;  // Very negative
            features[3] = 0.0;
            features[4] = -10.0;
            features[5] = -10.0;
            features[6] = 0.0;
            features[7] = 0.0;
            features[8] = 0.0;
            features[9] = 0.0;
            return features;
        }

        int selfHarm = myDistAfter - myDistBefore;
        int oppSlowdown = oppDistAfter - oppDistBefore;

        // Action type
        features[0] = 0.0;  // isMove
        features[1] = 1.0;  // isWall

        // Distance changes
        features[2] = -selfHarm;  // distReduction (negative selfHarm = we got farther)
        features[3] = oppSlowdown;  // oppDistChange
        features[4] = oppDistAfter - myDistAfter;  // newRaceGap
        features[5] = oppSlowdown - selfHarm;  // netAdvantage

        // Goal proximity
        features[6] = myDistAfter <= 3 ? 1.0 : 0.0;  // nearGoal
        features[7] = oppDistAfter <= 3 ? 1.0 : 0.0;  // oppNearGoal

        // Wall efficiency
        features[8] = oppSlowdown / (selfHarm + 1.0);

        // Forward move indicator (N/A for walls)
        features[9] = 0.0;

        return features;
    }

    /**
     * Normalization ranges for action features.
     */
    public static final double[] ACTION_FEATURE_MIN = {
        0, 0, -8, -4, -15, -10, 0, 0, 0, 0
    };

    public static final double[] ACTION_FEATURE_MAX = {
        1, 1, 8, 10, 15, 12, 1, 1, 10, 1
    };

    /**
     * Normalizes action features to [0, 1] range.
     */
    public static double[] normalize(double[] features) {
        double[] normalized = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            double min = ACTION_FEATURE_MIN[i];
            double max = ACTION_FEATURE_MAX[i];
            if (max == min) {
                normalized[i] = 0.5;
            } else {
                normalized[i] = (features[i] - min) / (max - min);
                normalized[i] = Math.max(0.0, Math.min(1.0, normalized[i]));
            }
        }
        return normalized;
    }
}
