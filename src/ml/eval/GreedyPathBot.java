package ml.eval;
import ml.*;

import logic.MoveValidator;
import logic.PathFinder;
import model.GameState;
import model.Player;
import model.Position;

import java.util.List;

/**
 * Deterministic "weak but competent" baseline bot.
 *
 * Policy:
 *   - Never places a wall.
 *   - For each legal pawn move, simulates the move and picks the one that
 *     minimises the A* distance to the goal (tie-break: larger row advancement
 *     toward the goal, then lowest-col for full determinism).
 *
 * Strength ordering:
 *     UniformRandom < ForwardRandom < SemiSmart < GreedyPathBot < BotBrain
 *
 * Used as a baseline in evaluation harness to verify that FeatureBot beats
 * weak opponents convincingly.
 */
public final class GreedyPathBot {

    /** Returns the move this bot would take. Never returns null on a legal state. */
    public static Position chooseMove(GameState state) {
        Player me = state.getCurrentPlayer();
        List<Position> moves = MoveValidator.getValidMoves(state, me);
        if (moves.isEmpty()) return null;

        int goalRow = me.getGoalRow();
        int curRow = me.getPosition().getRow();
        int forwardDir = (goalRow < curRow) ? -1 : 1;

        Position best = moves.get(0);
        int bestDist = Integer.MAX_VALUE;
        int bestAdvance = Integer.MIN_VALUE;
        int bestCol = Integer.MAX_VALUE;

        for (Position p : moves) {
            // Simulate move and measure resulting A* distance.
            Position original = me.getPosition();
            me.setPosition(p);
            int dist = PathFinder.aStarShortestPath(state, me);
            me.setPosition(original);

            if (dist >= 0) {                            // unreachable (shouldn't happen) -> skip
                int advance = (p.getRow() - curRow) * forwardDir;  // positive = toward goal

                boolean better =
                        (dist < bestDist)
                        || (dist == bestDist && advance > bestAdvance)
                        || (dist == bestDist && advance == bestAdvance && p.getCol() < bestCol);

                if (better) {
                    best = p;
                    bestDist = dist;
                    bestAdvance = advance;
                    bestCol = p.getCol();
                }
            }
        }
        return best;
    }

    private GreedyPathBot() { /* no instances */ }
}
