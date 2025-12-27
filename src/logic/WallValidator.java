package logic;

import model.GameState;
import model.Player;
import model.Wall;

/**
 * Validates wall placement.
 * Walls must be in valid position, not overlap existing walls,
 * and not block any player's path to their goal.
 */
public class WallValidator {

    // Main validation - checks all rules
    public static boolean isValidWallPlacement(GameState state, Wall wall) {
        if (!wall.isValidPosition()) return false;
        if (!state.getCurrentPlayer().hasWalls()) return false;

        // check overlap with existing walls
        for (Wall existing : state.getWalls()) {
            if (wall.overlaps(existing)) return false;
        }

        // make sure both players can still reach their goal
        for (Player player : state.getPlayers()) {
            if (PathFinder.hasPathToGoalWithWall(state, player, wall)) {
                return false;
            }
        }

        return true;
    }

    // Returns reason why wall placement failed (for error messages)
    public static String getInvalidReason(GameState state, Wall wall) {
        if (!wall.isValidPosition()) {
            return "Wall position is outside the board";
        }

        if (!state.getCurrentPlayer().hasWalls()) {
            return "You have no walls remaining";
        }

        for (Wall existing : state.getWalls()) {
            if (wall.overlaps(existing)) {
                return "Wall overlaps with an existing wall";
            }
        }

        for (Player player : state.getPlayers()) {
            if (PathFinder.hasPathToGoalWithWall(state, player, wall)) {
                return "Wall would block " + player.getName() + "'s path to goal";
            }
        }

        return null;
    }
}
