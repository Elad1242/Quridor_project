package logic;

import bot.BoardGraph;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.*;

/**
 * Pathfinding toolkit for the Quoridor board.
 *
 * Provides three distinct algorithms, each serving a different purpose:
 *
 *   1. BFS (original)   — Unweighted shortest path. Used for wall validation
 *                          (fast, simple, guarantees correctness).
 *
 *   2. A* Search         — Shortest path with a heuristic that penalizes
 *                          wall-dense areas. Finds the OPTIMAL path while
 *                          being smarter about which nodes to explore first.
 *
 *   3. Dijkstra          — Weighted shortest path on the BoardGraph.
 *                          Finds the SAFEST path by considering risk factors
 *                          (wall proximity, opponent proximity, centrality).
 *
 * The bot compares A* (optimal) vs Dijkstra (safe) results to understand
 * how risky the current position is.
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

    // ==================== A* SEARCH ====================

    /**
     * A* finds the shortest path from a player's position to their goal row.
     *
     * Unlike BFS, A* uses a heuristic to prioritize exploration toward the goal.
     * The heuristic combines:
     *   - Manhattan distance to the goal row (admissible — never overestimates)
     *   - Wall density penalty: if many walls are near a cell, A* avoids
     *     exploring through congested areas first.
     *
     * This makes A* faster than BFS in practice (explores fewer nodes)
     * and gives us the actual path, not just the length.
     *
     * @return the shortest path length, or -1 if no path exists
     */
    public static int aStarShortestPath(GameState state, Player player) {
        List<Position> path = aStarPath(state, player.getPosition(), player.getGoalRow(), null);
        return (path == null) ? -1 : path.size() - 1; // -1 because path includes start
    }

    /**
     * A* with a hypothetical wall added — used by WallEvaluator to measure
     * the impact of a wall before actually placing it.
     */
    public static int aStarWithWall(GameState state, Player player, Wall hypotheticalWall) {
        List<Position> path = aStarPath(state, player.getPosition(), player.getGoalRow(), hypotheticalWall);
        return (path == null) ? -1 : path.size() - 1;
    }

    /**
     * Returns the actual path (list of positions) found by A*.
     * This is used by WallEvaluator to know WHERE the opponent walks,
     * so it can place walls that intersect that path.
     */
    public static List<Position> getAStarPath(GameState state, Player player) {
        return aStarPath(state, player.getPosition(), player.getGoalRow(), null);
    }

    /**
     * Core A* implementation.
     *
     * Uses a priority queue ordered by f(n) = g(n) + h(n):
     *   g(n) = actual distance from start to n
     *   h(n) = heuristic estimate from n to goal
     *
     * The heuristic is Manhattan distance to the goal row + a small penalty
     * for wall-dense areas (this keeps it admissible since the penalty is < 1).
     *
     * @param extraWall optional hypothetical wall to consider (null if none)
     */
    private static List<Position> aStarPath(GameState state, Position start, int goalRow, Wall extraWall) {
        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        // Priority queue: [0]=f score, [1]=g score. Lower f = higher priority.
        PriorityQueue<double[]> openSet = new PriorityQueue<>(Comparator.comparingDouble(a -> a[0]));
        Map<Position, Double> gScore = new HashMap<>();
        Map<Position, Position> cameFrom = new HashMap<>();

        double startH = heuristic(start, goalRow, state);
        gScore.put(start, 0.0);
        // Pack row,col into the array so we can retrieve the position
        openSet.add(new double[]{startH, 0.0, start.getRow(), start.getCol()});

        Set<Position> closedSet = new HashSet<>();

        while (!openSet.isEmpty()) {
            double[] current = openSet.poll();
            int row = (int) current[2];
            int col = (int) current[3];
            Position currentPos = new Position(row, col);

            // Skip if already processed (A* can add duplicates to PQ)
            if (closedSet.contains(currentPos)) continue;
            closedSet.add(currentPos);

            // Goal reached — reconstruct the path
            if (currentPos.getRow() == goalRow) {
                return reconstructPath(cameFrom, currentPos);
            }

            double currentG = gScore.get(currentPos);

            for (int[] dir : directions) {
                Position neighbor = currentPos.move(dir[0], dir[1]);
                if (!neighbor.isValid()) continue;
                if (closedSet.contains(neighbor)) continue;

                // Check wall blocking (existing walls + optional hypothetical wall)
                if (state.isBlocked(currentPos, neighbor)) continue;
                if (extraWall != null && extraWall.blocksMove(currentPos, neighbor)) continue;

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

        return null; // No path exists
    }

    /**
     * A* heuristic: Manhattan distance to the goal row.
     *
     * This is strictly admissible — it never overestimates the true distance
     * because the shortest possible path is a straight line to the goal row
     * (which takes exactly |row - goalRow| steps).
     *
     * We intentionally do NOT add wall penalties here to preserve admissibility.
     * Risk-based path evaluation is handled by Dijkstra on the weighted BoardGraph.
     */
    private static double heuristic(Position pos, int goalRow, GameState state) {
        return Math.abs(pos.getRow() - goalRow);
    }

    /**
     * Reconstructs the path from start to goal by following the cameFrom map.
     * Returns a list of positions from start to goal (inclusive).
     */
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

    // ==================== DIJKSTRA (WEIGHTED) ====================

    /**
     * Dijkstra's algorithm on the weighted BoardGraph.
     *
     * Unlike A* (which treats all edges as cost 1), Dijkstra uses the
     * dynamic weights from BoardGraph that encode RISK:
     *   - Edges near walls cost more (opponent can block you)
     *   - Edges near opponent cost more (opponent can interact)
     *   - Edges in central columns cost less (more escape routes)
     *
     * This finds the SAFEST path, not the shortest.
     *
     * @return weighted cost of the safest path, or -1.0 if no path exists
     */
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

            if (visited.contains(currentPos)) continue;
            visited.add(currentPos);

            // Reached the goal row — return the weighted distance
            if (currentPos.getRow() == goalRow) {
                return currentDist;
            }

            // Explore neighbors using BoardGraph's weighted edges
            Map<Position, Double> neighbors = graph.getNeighbors(currentPos);
            for (Map.Entry<Position, Double> entry : neighbors.entrySet()) {
                Position neighbor = entry.getKey();
                double edgeWeight = entry.getValue();

                if (visited.contains(neighbor)) continue;

                double newDist = currentDist + edgeWeight;
                double existingDist = dist.getOrDefault(neighbor, Double.MAX_VALUE);

                if (newDist < existingDist) {
                    dist.put(neighbor, newDist);
                    pq.add(new double[]{newDist, neighbor.getRow(), neighbor.getCol()});
                }
            }
        }

        return -1.0; // No path exists
    }
}
