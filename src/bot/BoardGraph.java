package bot;

import model.GameState;
import model.Position;
import model.Wall;

import java.util.*;

/**
 * Represents the Quoridor board as a weighted directed graph.
 *
 * Each cell on the 9x9 board is a node. Each passage between adjacent cells
 * is an edge with a dynamic weight. Walls remove edges entirely.
 *
 * The weight system captures RISK — not just distance:
 *   - Base weight:          1.0 (normal passage)
 *   - Near OPPONENT walls:  +0.3 (opponent can easily extend blocks here)
 *   - Near opponent pawn:   +0.2 (opponent might jump or block you)
 *   - Central columns:      -0.1 (more escape routes = safer)
 *
 * CRITICAL FIX (v6): Wall proximity penalty only counts OPPONENT's walls.
 *   The bot's OWN walls are friendly — they don't make an area dangerous.
 *   Previously, the bot penalized paths near ALL walls, including its own,
 *   which caused the bot to avoid areas near its own wall formations and
 *   wander into longer, worse paths.
 *
 * This lets Dijkstra find the SAFEST path, not just the shortest.
 * A* finds the OPTIMAL path (ignoring risk). Comparing the two reveals
 * how "risky" the current board position is.
 */
public class BoardGraph {

    private static final int BOARD_SIZE = GameState.BOARD_SIZE; // 9
    private static final double BASE_WEIGHT = 1.0;
    private static final double WALL_PROXIMITY_PENALTY = 0.3;
    private static final double OPPONENT_PROXIMITY_PENALTY = 0.2;
    private static final double CENTRALITY_BONUS = 0.1;

    // Adjacency map: for each position, stores neighbor -> edge weight
    private final Map<Position, Map<Position, Double>> adjacency;

    // 4 cardinal directions: up, down, left, right
    private static final int[][] DIRECTIONS = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    // The bot's player index, used to distinguish own walls from opponent walls
    private int botPlayerIndex = -1;

    public BoardGraph() {
        this.adjacency = new HashMap<>();
    }

    /**
     * Builds the entire graph from the current game state.
     * This is called once per bot turn — it reads the board, places edges
     * where movement is allowed, and assigns weights based on risk factors.
     *
     * @param state       the current game state (walls, player positions)
     * @param opponentPos the opponent's position (used for proximity penalty)
     */
    public void buildFromState(GameState state, Position opponentPos) {
        adjacency.clear();

        // Determine bot's player index from the current state
        this.botPlayerIndex = state.getCurrentPlayerIndex();

        // Separate walls by owner for ownership-aware scoring
        List<Wall> opponentWalls = new ArrayList<>();
        for (Wall w : state.getWalls()) {
            if (w.getOwnerIndex() != botPlayerIndex) {
                opponentWalls.add(w);
            }
        }

        // Step 1: Create edges for all valid passages (not blocked by walls)
        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                Position from = new Position(row, col);

                for (int[] dir : DIRECTIONS) {
                    Position to = from.move(dir[0], dir[1]);
                    if (!to.isValid()) continue;

                    // If a wall blocks this passage, no edge exists
                    if (state.isBlocked(from, to)) continue;

                    // Calculate dynamic weight — ONLY penalize opponent walls
                    double weight = calculateEdgeWeight(from, to, opponentWalls, opponentPos);

                    adjacency.computeIfAbsent(from, k -> new HashMap<>()).put(to, weight);
                }
            }
        }
    }

    /**
     * Calculates the weight of an edge between two adjacent cells.
     * Higher weight = more dangerous to traverse.
     *
     * CRITICAL: Only OPPONENT walls increase danger. The bot's own walls
     * are friendly and should NOT make nearby paths seem risky.
     *
     * The weight factors:
     * 1. Opponent wall proximity — if opponent's walls are nearby, they can
     *    easily extend blocks in this area. Walking here is risky.
     * 2. Opponent pawn proximity — being near the opponent means they can jump
     *    over you or place a wall right next to you.
     * 3. Centrality — central columns (3,4,5) have more escape routes.
     *    Edge cells are easier to trap.
     */
    private double calculateEdgeWeight(Position from, Position to,
                                        List<Wall> opponentWalls, Position opponentPos) {
        double weight = BASE_WEIGHT;

        // --- Opponent wall proximity penalty ---
        // Only count OPPONENT's walls — bot's own walls are not a threat
        int nearbyWalls = countNearbyWalls(to, opponentWalls, 2);
        weight += nearbyWalls * WALL_PROXIMITY_PENALTY;

        // --- Opponent proximity penalty ---
        int distToOpponent = manhattanDistance(to, opponentPos);
        if (distToOpponent <= 2) {
            weight += OPPONENT_PROXIMITY_PENALTY * (3 - distToOpponent);
            // distance 0 → +0.6, distance 1 → +0.4, distance 2 → +0.2
        }

        // --- Centrality bonus (negative = good) ---
        // Columns 3,4,5 are central; columns 0,1,7,8 are edges
        int col = to.getCol();
        if (col >= 3 && col <= 5) {
            weight -= CENTRALITY_BONUS;
        }

        // Weight must be at least a small positive value for Dijkstra to work
        return Math.max(weight, 0.1);
    }

    /**
     * Counts how many walls are within a given Manhattan distance of a position.
     * Each wall occupies 2 cells, so we check both endpoints.
     *
     * NOTE: This method now receives a FILTERED wall list (opponent walls only).
     */
    private int countNearbyWalls(Position pos, List<Wall> walls, int radius) {
        int count = 0;
        for (Wall wall : walls) {
            int wallRow = wall.getRow();
            int wallCol = wall.getCol();

            // A wall spans 2 cells — check both endpoints
            if (wall.isHorizontal()) {
                // Horizontal wall covers (wallRow, wallCol) to (wallRow, wallCol+1)
                if (manhattanDistance(pos, new Position(wallRow, wallCol)) <= radius ||
                    manhattanDistance(pos, new Position(wallRow, wallCol + 1)) <= radius) {
                    count++;
                }
            } else {
                // Vertical wall covers (wallRow, wallCol) to (wallRow+1, wallCol)
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

    /**
     * Builds the graph from a specific player's perspective.
     * Used by GameFeatures to compute safety for either player.
     *
     * @param state       the current game state
     * @param opponentPos the opponent of playerIndex (used for proximity penalty)
     * @param playerIndex which player is "self" for wall ownership separation
     */
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

    // ===================== PUBLIC API =====================

    /**
     * Returns all neighbors of a position with their edge weights.
     * Used by Dijkstra to explore the graph.
     */
    public Map<Position, Double> getNeighbors(Position pos) {
        return adjacency.getOrDefault(pos, Collections.emptyMap());
    }

}
