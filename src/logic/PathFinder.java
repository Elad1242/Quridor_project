package logic;

import bot.BoardGraph;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.*;

// Has BFS, A* and Dijkstra for finding paths on the board.
// BFS is used for wall validation, A* for shortest path, Dijkstra for safest path.
public class PathFinder {

    /**
     * Simple BFS shortest path from player's current position to their goal.
     */
    public static int shortestPath(GameState state, Player player) {
        return shortestPathFrom(state, player.getPosition(), player.getGoalRow());
    }

    /**
     * Simple BFS shortest path from any position to a goal row.
     */
    public static int shortestPathFrom(GameState state, Position start, int goalRow) {
        if (start.getRow() == goalRow) return 0;

        Set<Position> visited = new HashSet<>();
        Queue<int[]> queue = new LinkedList<>(); // [row, col, distance]
        queue.add(new int[]{start.getRow(), start.getCol(), 0});
        visited.add(start);

        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            Position currentPos = new Position(current[0], current[1]);
            int dist = current[2];

            for (int[] dir : directions) {
                Position next = currentPos.move(dir[0], dir[1]);
                if (next.isValid() && !visited.contains(next) && !state.isBlocked(currentPos, next)) {
                    if (next.getRow() == goalRow) {
                        return dist + 1;
                    }
                    visited.add(next);
                    queue.add(new int[]{next.getRow(), next.getCol(), dist + 1});
                }
            }
        }
        return 999; // No path (shouldn't happen in valid game)
    }

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

    // A* search - shortest path with heuristic
    public static int aStarShortestPath(GameState state, Player player) {
        List<Position> path = aStarPath(state, player.getPosition(), player.getGoalRow(), null);
        return (path == null) ? -1 : path.size() - 1; // -1 because path includes start
    }

    // A* but with an extra wall added (for testing wall impact)
    public static int aStarWithWall(GameState state, Player player, Wall hypotheticalWall) {
        List<Position> path = aStarPath(state, player.getPosition(), player.getGoalRow(), hypotheticalWall);
        return (path == null) ? -1 : path.size() - 1;
    }

    // returns the full path (not just length)
    public static List<Position> getAStarPath(GameState state, Player player) {
        return aStarPath(state, player.getPosition(), player.getGoalRow(), null);
    }

    // core A* - uses f = g + h with manhattan distance heuristic
    private static List<Position> aStarPath(GameState state, Position start, int goalRow, Wall extraWall) {
        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        PriorityQueue<double[]> openSet = new PriorityQueue<>(Comparator.comparingDouble(a -> a[0]));
        Map<Position, Double> gScore = new HashMap<>();
        Map<Position, Position> cameFrom = new HashMap<>();

        double startH = heuristic(start, goalRow, state);
        gScore.put(start, 0.0);
        openSet.add(new double[]{startH, 0.0, start.getRow(), start.getCol()});

        Set<Position> closedSet = new HashSet<>();

        while (!openSet.isEmpty()) {
            double[] current = openSet.poll();
            int row = (int) current[2];
            int col = (int) current[3];
            Position currentPos = new Position(row, col);

            if (!closedSet.contains(currentPos)) {
                closedSet.add(currentPos);

                // reached the goal
                if (currentPos.getRow() == goalRow) {
                    return reconstructPath(cameFrom, currentPos);
                }

                double currentG = gScore.get(currentPos);

                for (int[] dir : directions) {
                    Position neighbor = currentPos.move(dir[0], dir[1]);
                    if (neighbor.isValid()
                            && !closedSet.contains(neighbor)
                            && !state.isBlocked(currentPos, neighbor)
                            && (extraWall == null || !extraWall.blocksMove(currentPos, neighbor))) {

                        double tentativeG = currentG + 1.0; // All edges cost 1 in A*
                        double existingG = gScore.getOrDefault(neighbor, Double.MAX_VALUE);

                        if (tentativeG < existingG) {
                            gScore.put(neighbor, tentativeG);
                            cameFrom.put(neighbor, currentPos);
                            double f = tentativeG + heuristic(neighbor, goalRow, state);
                            openSet.add(new double[]{f, tentativeG, neighbor.getRow(), neighbor.getCol()});
                        }
                    }
                }
            }
        }

        return null; // No path exists
    }

    // heuristic: just manhattan distance to goal row (admissible)
    private static double heuristic(Position pos, int goalRow, GameState state) {
        return Math.abs(pos.getRow() - goalRow);
    }

    // trace back from goal to start using the cameFrom map
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

    // dijkstra on the weighted board graph - finds safest path (not shortest)
    public static double dijkstraSafestPath(BoardGraph graph, Position start, int goalRow) {
        // Priority queue: [0]=distance, [1]=row, [2]=col
        PriorityQueue<double[]> pq = new PriorityQueue<>(Comparator.comparingDouble(a -> a[0]));
        Map<Position, Double> dist = new HashMap<>();
        Set<Position> visited = new HashSet<>();

        dist.put(start, 0.0);
        pq.add(new double[]{0.0, start.getRow(), start.getCol()});

        while (!pq.isEmpty()) {
            double[] current = pq.poll();
            double currentDist = current[0];
            Position currentPos = new Position((int) current[1], (int) current[2]);

            if (!visited.contains(currentPos)) {
                visited.add(currentPos);

                // Reached the goal row - return the weighted distance
                if (currentPos.getRow() == goalRow) {
                    return currentDist;
                }

                // Explore neighbors using BoardGraph's weighted edges
                Map<Position, Double> neighbors = graph.getNeighbors(currentPos);
                for (Map.Entry<Position, Double> entry : neighbors.entrySet()) {
                    Position neighbor = entry.getKey();
                    double edgeWeight = entry.getValue();

                    if (!visited.contains(neighbor)) {
                        double newDist = currentDist + edgeWeight;
                        double existingDist = dist.getOrDefault(neighbor, Double.MAX_VALUE);

                        if (newDist < existingDist) {
                            dist.put(neighbor, newDist);
                            pq.add(new double[]{newDist, neighbor.getRow(), neighbor.getCol()});
                        }
                    }
                }
            }
        }

        return -1.0; // No path exists
    }
}
