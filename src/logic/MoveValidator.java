package logic;

import model.GameState;
import model.Player;
import model.Position;

import java.util.ArrayList;
import java.util.List;

/**
 * Handles pawn movement validation.
 * Pawns move one cell orthogonally, can jump over adjacent opponents,
 * and can do side jumps if the straight jump is blocked.
 */
public class MoveValidator {

    // Checks if moving from one position to another is valid
    public static boolean isValidMove(GameState state, Position from, Position to) {
        List<Position> validMoves = getValidMoves(state, state.getCurrentPlayer());
        return validMoves.contains(to);
    }

    // Returns all valid moves for a player
    public static List<Position> getValidMoves(GameState state, Player player) {
        List<Position> validMoves = new ArrayList<>();
        Position current = player.getPosition();
        Position opponent = state.getOtherPlayer().getPosition();

        // 4 directions: up, down, left, right
        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        for (int i = 0; i < directions.length; i++) {
            int dRow = directions[i][0];
            int dCol = directions[i][1];
            Position next = current.move(dRow, dCol);

            if (!next.isValid()) continue;
            if (state.isBlocked(current, next)) continue;

            if (next.equals(opponent)) {
                handleJumpMoves(state, validMoves, current, next, dRow, dCol);
            } else {
                validMoves.add(next);
            }
        }

        return validMoves;
    }

    // Handles jump moves when opponent is adjacent
    private static void handleJumpMoves(GameState state, List<Position> validMoves,
                                         Position current, Position opponentPos,
                                         int dRow, int dCol) {

        Position straightJump = opponentPos.move(dRow, dCol);

        boolean isOffBoard = !straightJump.isValid();
        boolean isBlockedByWall = state.isBlocked(opponentPos, straightJump);
        boolean canDoSideJumps = isOffBoard || isBlockedByWall;

        if (!canDoSideJumps) {
            // straight jump only
            if (state.isOccupied(straightJump)) {
                validMoves.add(straightJump);
            }
        } else {
            // side jumps allowed when straight is blocked
            int[][] sideDirs;
            if (dRow != 0) {
                sideDirs = new int[][]{{0, -1}, {0, 1}};
            } else {
                sideDirs = new int[][]{{-1, 0}, {1, 0}};
            }

            for (int[] sideDir : sideDirs) {
                Position sideJump = opponentPos.move(sideDir[0], sideDir[1]);

                if (sideJump.isValid() &&
                    !state.isBlocked(opponentPos, sideJump) &&
                        state.isOccupied(sideJump)) {
                    validMoves.add(sideJump);
                }
            }
        }
    }
}
