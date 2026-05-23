// v2.0 — refactored and cleaned, May 2026
package bot;

import model.GameState;
import model.Position;
import model.Wall;

import java.util.*;

// Weighted board graph for Dijkstra. Edges near opponent walls or the opponent pawn cost more.
public class BoardGraph {

    private static final int BOARD_SIZE = GameState.BOARD_SIZE;
    private static final double BASE_WEIGHT = 1.0;
    private static final double WALL_PROXIMITY_PENALTY = 0.3;
    private static final double OPPONENT_PROXIMITY_PENALTY = 0.2;
    private static final double CENTRALITY_BONUS = 0.1;

    private static final int[][] DIRECTIONS = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    private final Map<Position, Map<Position, Double>> adjacency;

    public BoardGraph() {
        this.adjacency = new HashMap<>();
    }

    public void buildFromState(GameState state, Position opponentPos) {
        adjacency.clear();

        int botPlayerIndex = state.getCurrentPlayerIndex();

        // only the opponent's walls create danger zones for us
        List<Wall> opponentWalls = new ArrayList<>();
        for (Wall w : state.getWalls()) {
            if (w.getOwnerIndex() != botPlayerIndex) {
                opponentWalls.add(w);
            }
        }

        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                Position from = new Position(row, col);

                for (int[] dir : DIRECTIONS) {
                    Position to = from.move(dir[0], dir[1]);
                    if (to.isValid() && !state.isBlocked(from, to)) {
                        double weight = calculateEdgeWeight(to, opponentWalls, opponentPos);
                        adjacency.computeIfAbsent(from, k -> new HashMap<>()).put(to, weight);
                    }
                }
            }
        }
    }

    // edge weight = base + wall proximity penalty + opponent proximity penalty - centrality bonus
    private double calculateEdgeWeight(Position to, List<Wall> opponentWalls, Position opponentPos) {
        double weight = BASE_WEIGHT;

        int nearbyWalls = countNearbyWalls(to, opponentWalls, 2);
        weight += nearbyWalls * WALL_PROXIMITY_PENALTY;

        // being close to the opponent pawn is risky
        int distToOpponent = manhattanDistance(to, opponentPos);
        if (distToOpponent <= 2) {
            weight += OPPONENT_PROXIMITY_PENALTY * (3 - distToOpponent);
        }

        // central columns have more escape routes
        int col = to.getCol();
        if (col >= 3 && col <= 5) {
            weight -= CENTRALITY_BONUS;
        }

        return Math.max(weight, 0.1); // Dijkstra needs positive weights
    }

    private int countNearbyWalls(Position pos, List<Wall> walls, int radius) {
        int count = 0;
        for (Wall wall : walls) {
            int wallRow = wall.getRow();
            int wallCol = wall.getCol();

            // each wall spans 2 cells
            if (wall.isHorizontal()) {
                if (manhattanDistance(pos, new Position(wallRow, wallCol)) <= radius
                        || manhattanDistance(pos, new Position(wallRow, wallCol + 1)) <= radius) {
                    count++;
                }
            } else {
                if (manhattanDistance(pos, new Position(wallRow, wallCol)) <= radius
                        || manhattanDistance(pos, new Position(wallRow + 1, wallCol)) <= radius) {
                    count++;
                }
            }
        }
        return count;
    }

    private int manhattanDistance(Position a, Position b) {
        return Math.abs(a.getRow() - b.getRow()) + Math.abs(a.getCol() - b.getCol());
    }

    public Map<Position, Double> getNeighbors(Position pos) {
        return adjacency.getOrDefault(pos, Collections.emptyMap());
    }
}
