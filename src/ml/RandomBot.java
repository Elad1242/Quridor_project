package ml;

import logic.MoveValidator;
import logic.PathFinder;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Random opponent bots for generating diverse training data.
 * BotBrain plays against these to produce varied board states.
 */
public abstract class RandomBot {

    protected final Random rng;

    protected RandomBot(long seed) {
        this.rng = new Random(seed);
    }

    /**
     * Choose an action: returns either a Position (pawn move) or a Wall.
     */
    public abstract Object chooseAction(GameState state);

    // --- Utility methods ---

    protected List<Position> getForwardAndSidewaysMoves(GameState state, Player player) {
        List<Position> all = MoveValidator.getValidMoves(state, player);
        List<Position> result = new ArrayList<>();
        int goalRow = player.getGoalRow();
        int currentRow = player.getPosition().getRow();
        int forwardDir = (goalRow < currentRow) ? -1 : 1;

        for (Position p : all) {
            int dr = p.getRow() - currentRow;
            if (dr == forwardDir || dr == 0) {
                result.add(p);
            }
        }
        return result.isEmpty() ? all : result;
    }

    protected List<Wall> getAllValidWalls(GameState state) {
        List<Wall> walls = new ArrayList<>();
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                Wall h = new Wall(r, c, Wall.Orientation.HORIZONTAL);
                if (WallValidator.isValidWallPlacement(state, h)) walls.add(h);
                Wall v = new Wall(r, c, Wall.Orientation.VERTICAL);
                if (WallValidator.isValidWallPlacement(state, v)) walls.add(v);
            }
        }
        return walls;
    }

    // ===== IMPLEMENTATIONS =====

    /**
     * Picks uniformly from ALL legal actions (moves + walls).
     * Creates chaotic boards with walls in bizarre positions.
     */
    public static class UniformRandom extends RandomBot {
        public UniformRandom(long seed) { super(seed); }

        @Override
        public Object chooseAction(GameState state) {
            Player me = state.getCurrentPlayer();
            List<Position> moves = MoveValidator.getValidMoves(state, me);
            List<Wall> walls = (me.getWallsRemaining() > 0) ? getAllValidWalls(state) : new ArrayList<>();

            int total = moves.size() + walls.size();
            if (total == 0) return moves.get(0); // shouldn't happen

            int pick = rng.nextInt(total);
            if (pick < moves.size()) {
                return moves.get(pick);
            } else {
                return walls.get(pick - moves.size());
            }
        }
    }

    /**
     * Picks randomly from forward/sideways pawn moves only (never walls, never backward).
     * Produces games with NO walls — pure movement races with varied pawn paths.
     */
    public static class ForwardRandom extends RandomBot {
        public ForwardRandom(long seed) { super(seed); }

        @Override
        public Object chooseAction(GameState state) {
            Player me = state.getCurrentPlayer();
            List<Position> moves = getForwardAndSidewaysMoves(state, me);
            return moves.get(rng.nextInt(moves.size()));
        }
    }

    /**
     * 50% chance random wall, 50% chance random forward/sideways pawn move.
     * Creates extremely congested boards with walls everywhere.
     */
    public static class RandomWaller extends RandomBot {
        public RandomWaller(long seed) { super(seed); }

        @Override
        public Object chooseAction(GameState state) {
            Player me = state.getCurrentPlayer();
            boolean tryWall = rng.nextDouble() < 0.5 && me.getWallsRemaining() > 0;

            if (tryWall) {
                List<Wall> walls = getAllValidWalls(state);
                if (!walls.isEmpty()) {
                    return walls.get(rng.nextInt(walls.size()));
                }
            }

            List<Position> moves = getForwardAndSidewaysMoves(state, me);
            return moves.get(rng.nextInt(moves.size()));
        }
    }

    /**
     * 70% follows shortest path, 30% random move. 15% chance of random wall.
     * Semi-competent but unpredictable opponent.
     */
    public static class SemiSmart extends RandomBot {
        public SemiSmart(long seed) { super(seed); }

        @Override
        public Object chooseAction(GameState state) {
            Player me = state.getCurrentPlayer();

            // 15% chance of random wall
            if (rng.nextDouble() < 0.15 && me.getWallsRemaining() > 0) {
                List<Wall> walls = getAllValidWalls(state);
                if (!walls.isEmpty()) {
                    return walls.get(rng.nextInt(walls.size()));
                }
            }

            List<Position> validMoves = MoveValidator.getValidMoves(state, me);

            // 70% follow shortest path direction
            if (rng.nextDouble() < 0.7) {
                // Pick the move that reduces A* distance the most
                int bestDist = Integer.MAX_VALUE;
                Position bestMove = validMoves.get(0);
                for (Position move : validMoves) {
                    GameState sim = state.deepCopy();
                    sim.getCurrentPlayer().setPosition(move);
                    int dist = PathFinder.aStarShortestPath(sim, sim.getCurrentPlayer());
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestMove = move;
                    }
                }
                return bestMove;
            }

            // 30% random move
            return validMoves.get(rng.nextInt(validMoves.size()));
        }
    }
}
