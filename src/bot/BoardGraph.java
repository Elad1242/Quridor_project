package bot;

import model.GameState;
import model.Position;
import model.Wall;

import java.util.*;

// Weighted graph of the board for Dijkstra. Edges near opponent walls cost more.
public class BoardGraph {

    private static final int BOARD_SIZE = GameState.BOARD_SIZE; // 9
    private static final double BASE_WEIGHT = 1.0;
    private static final double WALL_PROXIMITY_PENALTY = 0.3;
    private static final double OPPONENT_PROXIMITY_PENALTY = 0.2;
    private static final double CENTRALITY_BONUS = 0.1;

    private final Map<Position, Map<Position, Double>> adjacency;
    private static final int[][] DIRECTIONS = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    private int botPlayerIndex = -1;

    public BoardGraph() {
        this.adjacency = new HashMap<>();
    }

    // Build the graph from current game state, weighting edges by risk
    public void buildFromState(GameState state, Position opponentPos) {
        adjacency.clear();

        // figure out which player is the bot
        this.botPlayerIndex = state.getCurrentPlayerIndex();

        // only care about opponent's walls for danger scoring
        List<Wall> opponentWalls = new ArrayList<>();
        for (Wall w : state.getWalls()) {
            if (w.getOwnerIndex() != botPlayerIndex) {
                opponentWalls.add(w);
            }
        }

        // create edges for all passages that aren't blocked
        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                Position from = new Position(row, col);

                for (int[] dir : DIRECTIONS) {
                    Position to = from.move(dir[0], dir[1]);
                    if (!to.isValid()) continue;

                    if (state.isBlocked(from, to)) continue;

                    // weight based on how dangerous this path is
                    double weight = calculateEdgeWeight(from, to, opponentWalls, opponentPos);

                    adjacency.computeIfAbsent(from, k -> new HashMap<>()).put(to, weight);
                }
            }
        }
    }

    // Edge weight based on opponent wall proximity, opponent pawn proximity, and centrality
    private double calculateEdgeWeight(Position from, Position to,
                                        List<Wall> opponentWalls, Position opponentPos) {
        double weight = BASE_WEIGHT;

        // penalty for being near opponent's walls (our own walls don't count)
        int nearbyWalls = countNearbyWalls(to, opponentWalls, 2);
        weight += nearbyWalls * WALL_PROXIMITY_PENALTY;

        // being near opponent is risky
        int distToOpponent = manhattanDistance(to, opponentPos);
        if (distToOpponent <= 2) {
            weight += OPPONENT_PROXIMITY_PENALTY * (3 - distToOpponent);
        }

        // central columns are safer (more escape routes)
        int col = to.getCol();
        if (col >= 3 && col <= 5) {
            weight -= CENTRALITY_BONUS;
        }

        // dijkstra needs positive weights
        return Math.max(weight, 0.1);
    }

    private int countNearbyWalls(Position pos, List<Wall> walls, int radius) {
        int count = 0;
        for (Wall wall : walls) {
            int wallRow = wall.getRow();
            int wallCol = wall.getCol();

            // each wall spans 2 cells
            if (wall.isHorizontal()) {
                if (manhattanDistance(pos, new Position(wallRow, wallCol)) <= radius ||
                    manhattanDistance(pos, new Position(wallRow, wallCol + 1)) <= radius) {
                    count++;
                }
            } else {
                if (manhattanDistance(pos, new Position(wallRow, wallCol)) <= radius ||
                    manhattanDistance(pos, new Position(wallRow + 1, wallCol)) <= radius) {
                    count++;
                }
            }
        }
        return count;
    }

    private int manhattanDistance(Position a, Position b) {
        return Math.abs(a.getRow() - b.getRow()) + Math.abs(a.getCol() - b.getCol());
    }

    // same thing but for a specific player (used by feature extraction)
    public void buildFromStateForPlayer(GameState state, Position opponentPos, int playerIndex) {
        adjacency.clear();
        this.botPlayerIndex = playerIndex;

        List<Wall> opponentWalls = new ArrayList<>();
        for (Wall w : state.getWalls()) {
            if (w.getOwnerIndex() != playerIndex) {
                opponentWalls.add(w);
            }
        }

        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                Position from = new Position(row, col);
                for (int[] dir : DIRECTIONS) {
                    Position to = from.move(dir[0], dir[1]);
                    if (!to.isValid()) continue;
                    if (state.isBlocked(from, to)) continue;
                    double weight = calculateEdgeWeight(from, to, opponentWalls, opponentPos);
                    adjacency.computeIfAbsent(from, k -> new HashMap<>()).put(to, weight);
                }
            }
        }
    }

    // get neighbors with their edge weights
    public Map<Position, Double> getNeighbors(Position pos) {
        return adjacency.getOrDefault(pos, Collections.emptyMap());
    }

}
