// v2.0 — refactored and cleaned, May 2026
package ml.eval;
import ml.*;

import logic.MoveValidator;
import logic.PathFinder;
import model.GameState;
import model.Player;
import model.Position;

import java.util.List;

// Deterministic baseline bot — never places walls, always picks the move that minimizes A* distance.
// Tiebreak: largest row advancement toward goal, then lowest column (for full determinism).
//
// Strength ladder: UniformRandom < ForwardRandom < SemiSmart < GreedyPathBot < BotBrain
public final class GreedyPathBot {

    // never returns null on a valid game state
    public static Position chooseMove(GameState state) {
        Player me = state.getCurrentPlayer();
        List<Position> moves = MoveValidator.getValidMoves(state, me);
        if (moves.isEmpty()) return null;

        int goalRow    = me.getGoalRow();
        int curRow     = me.getPosition().getRow();
        int forwardDir = (goalRow < curRow) ? -1 : 1;

        Position best      = moves.get(0);
        int bestDist       = Integer.MAX_VALUE;
        int bestAdvance    = Integer.MIN_VALUE;
        int bestCol        = Integer.MAX_VALUE;

        for (Position p : moves) {
            Position original = me.getPosition();
            me.setPosition(p);
            int dist = PathFinder.aStarShortestPath(state, me);
            me.setPosition(original);

            if (dist >= 0) { // reachable — process this move

            int advance = (p.getRow() - curRow) * forwardDir; // positive = toward goal

            boolean better = (dist < bestDist)
                    || (dist == bestDist && advance > bestAdvance)
                    || (dist == bestDist && advance == bestAdvance && p.getCol() < bestCol);

            if (better) {
                best        = p;
                bestDist    = dist;
                bestAdvance = advance;
                bestCol     = p.getCol();
            }
            }
        }

        return best;
    }

    private GreedyPathBot() { /* no instances */ }
}
