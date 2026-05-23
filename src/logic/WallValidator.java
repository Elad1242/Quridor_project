// v2.0 — refactored and cleaned, May 2026
package logic;

import model.GameState;
import model.Player;
import model.Wall;

// Validates wall placement: in bounds, no overlap, doesn't cut off any player's path.
public class WallValidator {

    public static boolean isValidWallPlacement(GameState state, Wall wall) {
        if (!wall.isValidPosition()) return false;
        if (!state.getCurrentPlayer().hasWalls()) return false;

        for (Wall existing : state.getWalls()) {
            if (wall.overlaps(existing)) return false;
        }

        for (Player player : state.getPlayers()) {
            if (PathFinder.hasPathToGoalWithWall(state, player, wall)) return false;
        }

        return true;
    }

    // returns a human-readable reason why placement failed, or null if it's valid
    public static String getInvalidReason(GameState state, Wall wall) {
        if (!wall.isValidPosition()) return "Wall position is outside the board";
        if (!state.getCurrentPlayer().hasWalls()) return "You have no walls remaining";

        for (Wall existing : state.getWalls()) {
            if (wall.overlaps(existing)) return "Wall overlaps with an existing wall";
        }

        for (Player player : state.getPlayers()) {
            if (PathFinder.hasPathToGoalWithWall(state, player, wall)) {
                return "Wall would block " + player.getName() + "'s path to goal";
            }
        }

        return null;
    }
}
