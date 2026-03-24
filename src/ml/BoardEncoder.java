package ml;

import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

/**
 * Encodes a Quoridor GameState into a 3D float array for CNN input.
 * No ML library dependencies — returns plain float[8][9][9].
 *
 * Channels:
 *   0: Current player pawn position (1.0 at pawn location)
 *   1: Opponent pawn position (1.0 at pawn location)
 *   2: Horizontal walls (marks blocked vertical edges)
 *   3: Vertical walls (marks blocked horizontal edges)
 *   4: Current player's goal row (1.0 for entire row)
 *   5: Opponent's goal row (1.0 for entire row)
 *   6: Current player's walls remaining (normalized 0-1, constant fill)
 *   7: Opponent's walls remaining (normalized 0-1, constant fill)
 */
public class BoardEncoder {

    public static final int CHANNELS = 8;
    public static final int BOARD_SIZE = 9;

    /**
     * Encodes the game state from the perspective of the current player.
     */
    public static float[][][] encode(GameState state) {
        float[][][] board = new float[CHANNELS][BOARD_SIZE][BOARD_SIZE];

        Player current = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();

        // Channel 0: Current player position
        Position myPos = current.getPosition();
        board[0][myPos.getRow()][myPos.getCol()] = 1.0f;

        // Channel 1: Opponent position
        Position oppPos = opponent.getPosition();
        board[1][oppPos.getRow()][oppPos.getCol()] = 1.0f;

        // Channel 2 & 3: Walls
        for (Wall wall : state.getWalls()) {
            encodeWall(board, wall);
        }

        // Channel 4: Current player's goal row
        int myGoal = current.getGoalRow();
        for (int c = 0; c < BOARD_SIZE; c++) {
            board[4][myGoal][c] = 1.0f;
        }

        // Channel 5: Opponent's goal row
        int oppGoal = opponent.getGoalRow();
        for (int c = 0; c < BOARD_SIZE; c++) {
            board[5][oppGoal][c] = 1.0f;
        }

        // Channel 6: Current player's walls remaining (normalized)
        float myWalls = current.getWallsRemaining() / 10.0f;
        fillChannel(board[6], myWalls);

        // Channel 7: Opponent's walls remaining (normalized)
        float oppWalls = opponent.getWallsRemaining() / 10.0f;
        fillChannel(board[7], oppWalls);

        return board;
    }

    /**
     * Encodes the state AFTER a pawn move, from the OPPONENT's perspective.
     * After our move, it becomes the opponent's turn — the CNN evaluates
     * from the perspective of whoever is about to play.
     */
    public static float[][][] encodeAfterMove(GameState state, Position move) {
        float[][][] board = new float[CHANNELS][BOARD_SIZE][BOARD_SIZE];

        Player current = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();

        // After our move, opponent becomes "current player" in the new state.
        // So we SWAP perspectives: opponent = ch0, us = ch1

        // Channel 0: Opponent pawn (they are now current player)
        Position oppPos = opponent.getPosition();
        board[0][oppPos.getRow()][oppPos.getCol()] = 1.0f;

        // Channel 1: Our pawn at the new position (we are now "opponent" in their view)
        board[1][move.getRow()][move.getCol()] = 1.0f;

        // Channel 2 & 3: Walls unchanged
        for (Wall wall : state.getWalls()) {
            encodeWall(board, wall);
        }

        // Channel 4: Opponent's goal row (they are now current)
        int oppGoal = opponent.getGoalRow();
        for (int c = 0; c < BOARD_SIZE; c++) {
            board[4][oppGoal][c] = 1.0f;
        }

        // Channel 5: Our goal row (we are now "opponent")
        int myGoal = current.getGoalRow();
        for (int c = 0; c < BOARD_SIZE; c++) {
            board[5][myGoal][c] = 1.0f;
        }

        // Channel 6: Opponent's walls remaining (they are now current)
        fillChannel(board[6], opponent.getWallsRemaining() / 10.0f);

        // Channel 7: Our walls remaining
        fillChannel(board[7], current.getWallsRemaining() / 10.0f);

        return board;
    }

    /**
     * Encodes the state AFTER a wall placement, from the OPPONENT's perspective.
     */
    public static float[][][] encodeAfterWall(GameState state, Wall wall) {
        float[][][] board = new float[CHANNELS][BOARD_SIZE][BOARD_SIZE];

        Player current = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();

        // After our wall, opponent becomes current player — swap perspectives

        // Channel 0: Opponent pawn (now current)
        Position oppPos = opponent.getPosition();
        board[0][oppPos.getRow()][oppPos.getCol()] = 1.0f;

        // Channel 1: Our pawn (now "opponent")
        Position myPos = current.getPosition();
        board[1][myPos.getRow()][myPos.getCol()] = 1.0f;

        // Channel 2 & 3: Existing walls + new wall
        for (Wall w : state.getWalls()) {
            encodeWall(board, w);
        }
        encodeWall(board, wall);

        // Channel 4: Opponent's goal row (now current)
        int oppGoal = opponent.getGoalRow();
        for (int c = 0; c < BOARD_SIZE; c++) {
            board[4][oppGoal][c] = 1.0f;
        }

        // Channel 5: Our goal row (now "opponent")
        int myGoal = current.getGoalRow();
        for (int c = 0; c < BOARD_SIZE; c++) {
            board[5][myGoal][c] = 1.0f;
        }

        // Channel 6: Opponent's walls remaining (now current)
        fillChannel(board[6], opponent.getWallsRemaining() / 10.0f);

        // Channel 7: Our walls remaining (decremented by 1)
        fillChannel(board[7], Math.max(0, current.getWallsRemaining() - 1) / 10.0f);

        return board;
    }

    private static void encodeWall(float[][][] board, Wall wall) {
        Position pos = wall.getPosition();
        int r = pos.getRow();
        int c = pos.getCol();

        if (wall.isHorizontal()) {
            if (r < BOARD_SIZE && c < BOARD_SIZE) board[2][r][c] = 1.0f;
            if (r < BOARD_SIZE && c + 1 < BOARD_SIZE) board[2][r][c + 1] = 1.0f;
        } else {
            if (r < BOARD_SIZE && c < BOARD_SIZE) board[3][r][c] = 1.0f;
            if (r + 1 < BOARD_SIZE && c < BOARD_SIZE) board[3][r + 1][c] = 1.0f;
        }
    }

    private static void fillChannel(float[][] channel, float value) {
        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                channel[r][c] = value;
            }
        }
    }
}
