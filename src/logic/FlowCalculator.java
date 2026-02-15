package logic;

import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.*;

/**
 * Computes Max Flow on the Quoridor board using the Edmonds-Karp algorithm
 * (BFS-based Ford-Fulkerson).
 *
 * PURPOSE: Measure how "flexible" a player's position is.
 * Max Flow = number of INDEPENDENT paths from the player to their goal row.
 *
 * Why this matters:
 *   - Max Flow = 1 → ONE wall can severely damage the player's route
 *   - Max Flow = 4 → The player has 4 independent ways to reach the goal,
 *                     making them very resilient to wall placements
 *
 * The bot uses this to:
 *   1. Evaluate its own vulnerability (low flow = danger)
 *   2. Evaluate wall placements (a wall that drops opponent's flow from 3→1 is devastating)
 *
 * GRAPH CONSTRUCTION:
 *   - Each board cell is a node
 *   - Each unblocked passage between cells is an edge with capacity 1
 *   - A virtual "super-sink" node connects to all cells in the goal row
 *   - Source = player's current position
 *   - Sink = super-sink
 *
 * COMPLEXITY: Edmonds-Karp is O(V * E²).
 * On a 9×9 board: V=81+1=82, E≈144+9=153 → very fast (<1ms).
 */
public class FlowCalculator {

    private static final int BOARD_SIZE = GameState.BOARD_SIZE;
    private static final int[][] DIRECTIONS = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    // We encode positions as integers for the flow network:
    // cell (r,c) → index r * BOARD_SIZE + c  (0..80)
    // super-sink → index 81
    private static final int SUPER_SINK = BOARD_SIZE * BOARD_SIZE; // = 81
    private static final int NODE_COUNT = SUPER_SINK + 1;          // = 82

    /**
     * Calculates the maximum flow from a player's position to their goal row.
     * This equals the number of node-independent paths to the goal.
     *
     * @return max flow value (number of independent paths)
     */
    public static int calculateMaxFlow(GameState state, Player player) {
        int[][] capacity = buildCapacityGraph(state, player.getGoalRow(), null);
        int source = posToIndex(player.getPosition());
        return edmondsKarp(capacity, source, SUPER_SINK);
    }

    /**
     * Same as above but with a hypothetical wall added.
     * Used to measure the IMPACT of a wall placement on flow.
     *
     * Example: if flow drops from 4 → 1 after placing this wall,
     * the wall effectively eliminates 3 independent paths.
     */
    public static int calculateMaxFlowWithWall(GameState state, Player player, Wall wall) {
        int[][] capacity = buildCapacityGraph(state, player.getGoalRow(), wall);
        int source = posToIndex(player.getPosition());
        return edmondsKarp(capacity, source, SUPER_SINK);
    }

    /**
     * Builds the capacity matrix for the flow network.
     *
     * Each unblocked passage gets capacity 1.
     * All goal-row cells connect to the super-sink with capacity 1.
     * An optional hypothetical wall can be included (treated as blocking).
     */
    private static int[][] buildCapacityGraph(GameState state, int goalRow, Wall extraWall) {
        int[][] capacity = new int[NODE_COUNT][NODE_COUNT];

        // Add edges for all valid passages on the board
        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                Position from = new Position(row, col);
                int fromIdx = posToIndex(from);

                for (int[] dir : DIRECTIONS) {
                    Position to = from.move(dir[0], dir[1]);
                    if (!to.isValid()) continue;

                    // Check if passage is blocked by existing walls
                    if (state.isBlocked(from, to)) continue;

                    // Check if passage is blocked by hypothetical wall
                    if (extraWall != null && extraWall.blocksMove(from, to)) continue;

                    int toIdx = posToIndex(to);
                    capacity[fromIdx][toIdx] = 1;
                }
            }
        }

        // Connect all goal-row cells to the super-sink
        for (int col = 0; col < BOARD_SIZE; col++) {
            int goalIdx = posToIndex(new Position(goalRow, col));
            capacity[goalIdx][SUPER_SINK] = 1;
        }

        return capacity;
    }

    /**
     * Edmonds-Karp algorithm: finds Max Flow using BFS to find augmenting paths.
     *
     * How it works:
     *   1. Use BFS to find a path from source to sink in the residual graph
     *   2. Find the bottleneck (minimum capacity) along that path
     *   3. Push flow along the path (reduce forward edges, increase reverse edges)
     *   4. Repeat until no augmenting path exists
     *
     * The total flow pushed equals the maximum flow.
     *
     * Using BFS (not DFS) guarantees O(V * E²) complexity, which is the
     * key improvement of Edmonds-Karp over basic Ford-Fulkerson.
     */
    private static int edmondsKarp(int[][] capacity, int source, int sink) {
        // Residual capacity graph (starts as a copy of the capacity graph)
        int[][] residual = new int[NODE_COUNT][NODE_COUNT];
        for (int i = 0; i < NODE_COUNT; i++) {
            System.arraycopy(capacity[i], 0, residual[i], 0, NODE_COUNT);
        }

        int totalFlow = 0;
        int[] parent = new int[NODE_COUNT]; // BFS parent array for path reconstruction

        // Keep finding augmenting paths until none exist
        while (bfs(residual, source, sink, parent)) {
            // Find the bottleneck capacity along the BFS path
            int pathFlow = Integer.MAX_VALUE;
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                pathFlow = Math.min(pathFlow, residual[u][v]);
            }

            // Update residual capacities along the path:
            //   - Forward edge: decrease by pathFlow
            //   - Reverse edge: increase by pathFlow (allows "undoing" flow)
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                residual[u][v] -= pathFlow;
                residual[v][u] += pathFlow;
            }

            totalFlow += pathFlow;
        }

        return totalFlow;
    }

    /**
     * BFS in the residual graph to find an augmenting path from source to sink.
     *
     * Only traverses edges with remaining capacity > 0.
     * Fills the parent[] array for path reconstruction.
     *
     * @return true if a path exists, false if source and sink are disconnected
     */
    private static boolean bfs(int[][] residual, int source, int sink, int[] parent) {
        Arrays.fill(parent, -1);
        parent[source] = source; // Mark source as visited

        Queue<Integer> queue = new LinkedList<>();
        queue.add(source);

        while (!queue.isEmpty()) {
            int current = queue.poll();

            for (int next = 0; next < NODE_COUNT; next++) {
                // Only traverse edges with remaining capacity
                if (parent[next] == -1 && residual[current][next] > 0) {
                    parent[next] = current;
                    if (next == sink) return true; // Found a path!
                    queue.add(next);
                }
            }
        }

        return false; // No augmenting path exists
    }

    /**
     * Converts a board Position to a node index in the flow network.
     * Position (r, c) → index r * 9 + c
     */
    private static int posToIndex(Position pos) {
        return pos.getRow() * BOARD_SIZE + pos.getCol();
    }
}
