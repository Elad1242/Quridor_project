package logic;

import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.*;

// Max flow using Edmonds-Karp. Counts how many independent paths a player
// has to their goal row - more paths means harder to block.
public class FlowCalculator {

    private static final int BOARD_SIZE = GameState.BOARD_SIZE;
    private static final int[][] DIRECTIONS = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    // each cell is r*9+c, super-sink is node 81
    private static final int SUPER_SINK = BOARD_SIZE * BOARD_SIZE;
    private static final int NODE_COUNT = SUPER_SINK + 1;

    // how many independent paths does this player have to goal
    public static int calculateMaxFlow(GameState state, Player player) {
        int[][] capacity = buildCapacityGraph(state, player.getGoalRow(), null);
        int source = posToIndex(player.getPosition());
        return edmondsKarp(capacity, source, SUPER_SINK);
    }

    // same but with an extra wall added
    public static int calculateMaxFlowWithWall(GameState state, Player player, Wall wall) {
        int[][] capacity = buildCapacityGraph(state, player.getGoalRow(), wall);
        int source = posToIndex(player.getPosition());
        return edmondsKarp(capacity, source, SUPER_SINK);
    }

    // build the capacity graph - each passage has capacity 1
    private static int[][] buildCapacityGraph(GameState state, int goalRow, Wall extraWall) {
        int[][] capacity = new int[NODE_COUNT][NODE_COUNT];

        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                Position from = new Position(row, col);
                int fromIdx = posToIndex(from);

                for (int[] dir : DIRECTIONS) {
                    Position to = from.move(dir[0], dir[1]);
                    if (to.isValid()
                            && !state.isBlocked(from, to)
                            && (extraWall == null || !extraWall.blocksMove(from, to))) {
                        int toIdx = posToIndex(to);
                        capacity[fromIdx][toIdx] = 1;
                    }
                }
            }
        }

        // goal row cells connect to super-sink
        for (int col = 0; col < BOARD_SIZE; col++) {
            int goalIdx = posToIndex(new Position(goalRow, col));
            capacity[goalIdx][SUPER_SINK] = 1;
        }

        return capacity;
    }

    // edmonds-karp: keep finding augmenting paths with BFS until there are none
    private static int edmondsKarp(int[][] capacity, int source, int sink) {
        int[][] residual = new int[NODE_COUNT][NODE_COUNT];
        for (int i = 0; i < NODE_COUNT; i++) {
            System.arraycopy(capacity[i], 0, residual[i], 0, NODE_COUNT);
        }

        int totalFlow = 0;
        int[] parent = new int[NODE_COUNT];

        while (bfs(residual, source, sink, parent)) {
            // find bottleneck
            int pathFlow = Integer.MAX_VALUE;
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                pathFlow = Math.min(pathFlow, residual[u][v]);
            }

            // update residual graph
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                residual[u][v] -= pathFlow;
                residual[v][u] += pathFlow;
            }

            totalFlow += pathFlow;
        }

        return totalFlow;
    }

    // BFS to find an augmenting path in the residual graph
    private static boolean bfs(int[][] residual, int source, int sink, int[] parent) {
        Arrays.fill(parent, -1);
        parent[source] = source;

        Queue<Integer> queue = new LinkedList<>();
        queue.add(source);

        while (!queue.isEmpty()) {
            int current = queue.poll();

            for (int next = 0; next < NODE_COUNT; next++) {
                if (parent[next] == -1 && residual[current][next] > 0) {
                    parent[next] = current;
                    if (next == sink) return true;
                    queue.add(next);
                }
            }
        }

        return false;
    }

    private static int posToIndex(Position pos) {
        return pos.getRow() * BOARD_SIZE + pos.getCol();
    }
}
