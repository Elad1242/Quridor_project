package ml.cnn;

import model.GameState;
import model.Player;
import model.Position;
import model.Wall;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Encodes a Quoridor GameState into a 3D tensor suitable for CNN input.
 *
 * The board is represented as 8 channels x 9x9 grid:
 *
 * Channel 0: Current player pawn position (1.0 at pawn location)
 * Channel 1: Opponent pawn position (1.0 at pawn location)
 * Channel 2: Horizontal walls - marks blocked edges between rows
 * Channel 3: Vertical walls - marks blocked edges between columns
 * Channel 4: Current player's goal row (1.0 for entire goal row)
 * Channel 5: Opponent's goal row (1.0 for entire goal row)
 * Channel 6: Current player's walls remaining (normalized 0-1)
 * Channel 7: Opponent's walls remaining (normalized 0-1)
 */
public class BoardEncoder {

    public static final int CHANNELS = 8;
    public static final int BOARD_SIZE = 9;

    /**
     * Encodes the game state from the perspective of the current player.
     * The CNN always sees itself as "current player" regardless of which player index.
     *
     * @param state The current game state
     * @return INDArray of shape [channels, height, width] = [8, 9, 9]
     */
    public static INDArray encode(GameState state) {
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
        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                board[6][r][c] = myWalls;
            }
        }

        // Channel 7: Opponent's walls remaining (normalized)
        float oppWalls = opponent.getWallsRemaining() / 10.0f;
        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                board[7][r][c] = oppWalls;
            }
        }

        return Nd4j.create(board);
    }

    /**
     * Encodes a wall into the board representation.
     *
     * Horizontal walls block vertical movement (between rows).
     * Vertical walls block horizontal movement (between columns).
     *
     * Each wall spans 2 cells, so we mark both blocked edges.
     */
    private static void encodeWall(float[][][] board, Wall wall) {
        Position pos = wall.getPosition();
        int r = pos.getRow();
        int c = pos.getCol();

        if (wall.isHorizontal()) {
            // Horizontal wall at (r,c) blocks movement between rows r and r+1
            // for columns c and c+1
            // We mark in channel 2 at positions (r, c) and (r, c+1)
            // This indicates "cannot move vertically across this edge"
            if (r < BOARD_SIZE && c < BOARD_SIZE) {
                board[2][r][c] = 1.0f;
            }
            if (r < BOARD_SIZE && c + 1 < BOARD_SIZE) {
                board[2][r][c + 1] = 1.0f;
            }
        } else {
            // Vertical wall at (r,c) blocks movement between columns c and c+1
            // for rows r and r+1
            // We mark in channel 3 at positions (r, c) and (r+1, c)
            if (r < BOARD_SIZE && c < BOARD_SIZE) {
                board[3][r][c] = 1.0f;
            }
            if (r + 1 < BOARD_SIZE && c < BOARD_SIZE) {
                board[3][r + 1][c] = 1.0f;
            }
        }
    }

    /**
     * Encodes a hypothetical move action for the CNN to evaluate.
     * Returns a tensor representing the state AFTER the move would be made.
     *
     * @param state Current game state
     * @param move Target position for the move
     * @return Encoded state after move
     */
    public static INDArray encodeAfterMove(GameState state, Position move) {
        float[][][] board = new float[CHANNELS][BOARD_SIZE][BOARD_SIZE];

        Player current = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();

        // Channel 0: Current player's NEW position (after move)
        board[0][move.getRow()][move.getCol()] = 1.0f;

        // Channel 1: Opponent stays same
        Position oppPos = opponent.getPosition();
        board[1][oppPos.getRow()][oppPos.getCol()] = 1.0f;

        // Channel 2 & 3: Walls unchanged
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

        // Channel 6: Current player's walls (unchanged for move)
        float myWalls = current.getWallsRemaining() / 10.0f;
        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                board[6][r][c] = myWalls;
            }
        }

        // Channel 7: Opponent's walls
        float oppWalls = opponent.getWallsRemaining() / 10.0f;
        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                board[7][r][c] = oppWalls;
            }
        }

        return Nd4j.create(board);
    }

    /**
     * Encodes a hypothetical wall placement for the CNN to evaluate.
     * Returns a tensor representing the state AFTER the wall would be placed.
     *
     * @param state Current game state
     * @param wall Wall to place
     * @return Encoded state after wall placement
     */
    public static INDArray encodeAfterWall(GameState state, Wall wall) {
        float[][][] board = new float[CHANNELS][BOARD_SIZE][BOARD_SIZE];

        Player current = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();

        // Channel 0: Current player position unchanged
        Position myPos = current.getPosition();
        board[0][myPos.getRow()][myPos.getCol()] = 1.0f;

        // Channel 1: Opponent position unchanged
        Position oppPos = opponent.getPosition();
        board[1][oppPos.getRow()][oppPos.getCol()] = 1.0f;

        // Channel 2 & 3: Existing walls
        for (Wall w : state.getWalls()) {
            encodeWall(board, w);
        }
        // Add the new wall
        encodeWall(board, wall);

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

        // Channel 6: Current player's walls (decremented by 1)
        float myWalls = Math.max(0, current.getWallsRemaining() - 1) / 10.0f;
        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                board[6][r][c] = myWalls;
            }
        }

        // Channel 7: Opponent's walls
        float oppWalls = opponent.getWallsRemaining() / 10.0f;
        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                board[7][r][c] = oppWalls;
            }
        }

        return Nd4j.create(board);
    }

    /**
     * Creates a batch of encoded states.
     *
     * @param boards Array of encoded boards
     * @return INDArray of shape [batchSize, channels, height, width]
     */
    public static INDArray createBatch(INDArray[] boards) {
        if (boards.length == 0) {
            return Nd4j.empty();
        }

        INDArray batch = Nd4j.create(boards.length, CHANNELS, BOARD_SIZE, BOARD_SIZE);
        for (int i = 0; i < boards.length; i++) {
            batch.putRow(i, boards[i].reshape(1, CHANNELS, BOARD_SIZE, BOARD_SIZE).getRow(0));
        }
        return batch;
    }
}
