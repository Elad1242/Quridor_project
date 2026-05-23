// v2.0 — refactored and cleaned, May 2026
package logic;

import bot.BoardGraph;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.*;

// BFS, A*, and Dijkstra for finding paths on the board.
// BFS for wall validation, A* for shortest path, Dijkstra for safest path.
public class PathFinder {

    private static final int[][] DIRECTIONS = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    // BFS shortest path from player's current position to their goal row
    public static int shortestPath(GameState state, Player player) {
        return shortestPathFrom(state, player.getPosition(), player.getGoalRow());
    }

    // BFS from any position to a goal row
    public static int shortestPathFrom(GameState state, Position start, int goalRow) {
        if (start.getRow() == goalRow) return 0;

        Set<Position> visited = new HashSet<>();
        Queue<int[]> queue = new LinkedList<>(); // [row, col, distance]
        queue.add(new int[]{start.getRow(), start.getCol(), 0});
        visited.add(start);

        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            Position currentPos = new Position(current[0], current[1]);
            int dist = current[2];

            for (int[] dir : DIRECTIONS) {
                Position next = currentPos.move(dir[0], dir[1]);
                if (!next.isValid() || visited.contains(next) || state.isBlocked(currentPos, next)) continue;
                if (next.getRow() == goalRow) return dist + 1;
                visited.add(next);
                queue.add(new int[]{next.getRow(), next.getCol(), dist + 1});
            }
        }

        return 999; // no path (shouldn't happen in a valid game)
    }

    // BFS that checks if the player can still reach their goal after a hypothetical wall placement
    public static boolean hasPathToGoalWithWall(GameState state, Player player, Wall newWall) {
        Position start = player.getPosition();
        int goalRow = player.getGoalRow();

        Set<Position> visited = new HashSet<>();
        Queue<Position> queue = new LinkedList<>();
        queue.add(start);
        visited.add(start);

        while (!queue.isEmpty()) {
            Position current = queue.poll();

            if (current.getRow() == goalRow) return false;

            for (int[] dir : DIRECTIONS) {
                Position next = current.move(dir[0], dir[1]);
                if (!next.isValid() || visited.contains(next)) continue;
                if (state.isBlocked(current, next) || newWall.blocksMove(current, next)) continue;
                visited.add(next);
                queue.add(next);
            }
        }

        return true; // no path found — wall blocks the player
    }

    // A* shortest path length (returns -1 if no path)
    public static int aStarShortestPath(GameState state, Player player) {
        List<Position> path = aStarPath(state, player.getPosition(), player.getGoalRow(), null);
        return (path == null) ? -1 : path.size() - 1;
    }

    // A* with a hypothetical extra wall (for testing wall impact)
    public static int aStarWithWall(GameState state, Player player, Wall hypotheticalWall) {
        List<Position> path = aStarPath(state, player.getPosition(), player.getGoalRow(), hypotheticalWall);
        return (path == null) ? -1 : path.size() - 1;
    }

    // returns the full A* path, not just its length
    public static List<Position> getAStarPath(GameState state, Player player) {
        return aStarPath(state, player.getPosition(), player.getGoalRow(), null);
    }

    // core A* — f = g + h, heuristic is manhattan distance to goal row
    private static List<Position> aStarPath(GameState state, Position start, int goalRow, Wall extraWall) {
        PriorityQueue<double[]> openSet = new PriorityQueue<>(Comparator.comparingDouble(a -> a[0]));
        Map<Position, Double> gScore = new HashMap<>();
        Map<Position, Position> cameFrom = new HashMap<>();
        Set<Position> closedSet = new HashSet<>();

        gScore.put(start, 0.0);
        openSet.add(new double[]{heuristic(start, goalRow), 0.0, start.getRow(), start.getCol()});

        while (!openSet.isEmpty()) {
            double[] current = openSet.poll();
            Position currentPos = new Position((int) current[2], (int) current[3]);

            if (closedSet.contains(currentPos)) continue;
            closedSet.add(currentPos);

            if (currentPos.getRow() == goalRow) return reconstructPath(cameFrom, currentPos);

            double currentG = gScore.get(currentPos);

            for (int[] dir : DIRECTIONS) {
                Position neighbor = currentPos.move(dir[0], dir[1]);
                if (!neighbor.isValid() || closedSet.contains(neighbor)) continue;
                if (state.isBlocked(currentPos, neighbor)) continue;
                if (extraWall != null && extraWall.blocksMove(currentPos, neighbor)) continue;

                double tentativeG = currentG + 1.0;
                if (tentativeG >= gScore.getOrDefault(neighbor, Double.MAX_VALUE)) continue;

                gScore.put(neighbor, tentativeG);
                cameFrom.put(neighbor, currentPos);
                double f = tentativeG + heuristic(neighbor, goalRow);
                openSet.add(new double[]{f, tentativeG, neighbor.getRow(), neighbor.getCol()});
            }
        }

        return null;
    }

    // manhattan distance to goal row — admissible heuristic since there are no diagonals
    private static double heuristic(Position pos, int goalRow) {
        return Math.abs(pos.getRow() - goalRow);
    }

    private static List<Position> reconstructPath(Map<Position, Position> cameFrom, Position goal) {
        List<Position> path = new ArrayList<>();
        Position current = goal;
        while (current != null) {
            path.add(current);
            current = cameFrom.get(current);
        }
        Collections.reverse(path);
        return path;
    }

    // Dijkstra on a weighted board graph — finds the "safest" path, not just shortest
    public static double dijkstraSafestPath(BoardGraph graph, Position start, int goalRow) {
        PriorityQueue<double[]> pq = new PriorityQueue<>(Comparator.comparingDouble(a -> a[0]));
        Map<Position, Double> dist = new HashMap<>();
        Set<Position> visited = new HashSet<>();

        dist.put(start, 0.0);
        pq.add(new double[]{0.0, start.getRow(), start.getCol()});

        while (!pq.isEmpty()) {
            double[] current = pq.poll();
            double currentDist = current[0];
            Position currentPos = new Position((int) current[1], (int) current[2]);

            if (visited.contains(currentPos)) continue;
            visited.add(currentPos);

            if (currentPos.getRow() == goalRow) return currentDist;

            Map<Position, Double> neighbors = graph.getNeighbors(currentPos);
            for (Map.Entry<Position, Double> entry : neighbors.entrySet()) {
                Position neighbor = entry.getKey();
                if (visited.contains(neighbor)) continue;

                double newDist = currentDist + entry.getValue();
                if (newDist < dist.getOrDefault(neighbor, Double.MAX_VALUE)) {
                    dist.put(neighbor, newDist);
                    pq.add(new double[]{newDist, neighbor.getRow(), neighbor.getCol()});
                }
            }
        }

        return -1.0; // no path
    }
}
