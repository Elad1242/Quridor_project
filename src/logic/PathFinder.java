package logic;

import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.*;

/**
 * Uses BFS to check if players can reach their goal.
 * Used to validate wall placements don't trap anyone.
 */
public class PathFinder {

    // Checks if player can reach goal if a new wall is added
    public static boolean hasPathToGoalWithWall(GameState state, Player player, Wall newWall) {
        Position start = player.getPosition();
        int goalRow = player.getGoalRow();

        Set<Position> visited = new HashSet<>();
        Queue<Position> queue = new LinkedList<>();
        queue.add(start);
        visited.add(start);

        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        while (!queue.isEmpty()) {
            Position current = queue.poll();

            if (current.getRow() == goalRow) {
                return false;
            }

            for (int[] dir : directions) {
                Position next = current.move(dir[0], dir[1]);

                if (next.isValid() && !visited.contains(next)) {
                    boolean blockedByExisting = state.isBlocked(current, next);
                    boolean blockedByNew = newWall.blocksMove(current, next);

                    if (!blockedByExisting && !blockedByNew) {
                        visited.add(next);
                        queue.add(next);
                    }
                }
            }
        }

        return true;
    }

    // Returns shortest path length to goal (useful for AI)
    public static int getShortestPathLength(GameState state, Player player) {
        Position start = player.getPosition();
        int goalRow = player.getGoalRow();

        Map<Position, Integer> distance = new HashMap<>();
        Queue<Position> queue = new LinkedList<>();
        queue.add(start);
        distance.put(start, 0);

        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        while (!queue.isEmpty()) {
            Position current = queue.poll();
            int currentDist = distance.get(current);

            if (current.getRow() == goalRow) {
                return currentDist;
            }

            for (int[] dir : directions) {
                Position next = current.move(dir[0], dir[1]);

                if (next.isValid() &&
                    !distance.containsKey(next) &&
                    !state.isBlocked(current, next)) {

                    distance.put(next, currentDist + 1);
                    queue.add(next);
                }
            }
        }

        return -1;
    }
}
