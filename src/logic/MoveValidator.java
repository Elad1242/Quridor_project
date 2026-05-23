// v2.0 — refactored and cleaned, May 2026
package logic;

import model.GameState;
import model.Player;
import model.Position;

import java.util.ArrayList;
import java.util.List;

// Handles pawn movement — regular moves and jump moves over the opponent.
public class MoveValidator {

    // does the given target appear in the valid move list?
    public static boolean isValidMove(GameState state, Position from, Position to) {
        return getValidMoves(state, state.getCurrentPlayer()).contains(to);
    }

    public static List<Position> getValidMoves(GameState state, Player player) {
        List<Position> validMoves = new ArrayList<>();
        Position current = player.getPosition();
        Position opponent = state.getOtherPlayer().getPosition();

        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        for (int[] dir : directions) {
            Position next = current.move(dir[0], dir[1]);
            if (next.isValid() && !state.isBlocked(current, next)) {
                if (next.equals(opponent)) { //if the next is the position of the opponent we can jump above
                    handleJumpMoves(state, validMoves, next, dir[0], dir[1]);
                } else {
                    validMoves.add(next);
                }
            }
        }

        return validMoves;
    }

    // called when the opponent is adjacent — decides between straight jump and side jumps
    private static void handleJumpMoves(GameState state, List<Position> validMoves,
                                         Position opponentPos, int dRow, int dCol) {
        Position straightJump = opponentPos.move(dRow, dCol);
        boolean straightBlocked = !straightJump.isValid() || state.isBlocked(opponentPos, straightJump);

        if (!straightBlocked) {
            // straight jump only — square must be clear
            if (state.isOccupied(straightJump)) {
                validMoves.add(straightJump);
            }
            return;
        }

        // straight is blocked (wall or edge) — try side jumps
        int[][] sideDirs = (dRow != 0) ? new int[][]{{0, -1}, {0, 1}} : new int[][]{{-1, 0}, {1, 0}};

        for (int[] sideDir : sideDirs) {
            Position sideJump = opponentPos.move(sideDir[0], sideDir[1]);
            if (sideJump.isValid()
                    && !state.isBlocked(opponentPos, sideJump)
                    && state.isOccupied(sideJump)) {
                validMoves.add(sideJump);
            }
        }
    }
}
