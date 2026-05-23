// v2.0 — refactored and cleaned, May 2026
package model;

import javafx.scene.paint.Color;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

// All game data: players, walls, whose turn it is, win state.
public class GameState {

    public static final int BOARD_SIZE = 9;

    private final Player[] players;
    private final List<Wall> walls;
    private int currentPlayerIndex;
    private int turnCount;
    private boolean gameOver;
    private Player winner;

    public GameState() {
        this("Player 1", "Player 2");
    }

    public GameState(String player1Name, String player2Name) {
        players = new Player[2];

        // player 1: starts bottom, goal is top (row 0)
        players[0] = new Player(
            player1Name,
            new Position(8, 4),
            0,
            Color.web("#2C3E50")
        );

        // player 2: starts top, goal is bottom (row 8)
        players[1] = new Player(
            player2Name,
            new Position(0, 4),
            8,
            Color.web("#ECF0F1")
        );

        walls = new ArrayList<>();
        currentPlayerIndex = 0;
        turnCount = 0;
        gameOver = false;
        winner = null;
    }

    public Player getCurrentPlayer() {
        return players[currentPlayerIndex];
    }

    public Player getOtherPlayer() {
        return players[1 - currentPlayerIndex];
    }

    public Player getPlayer(int index) {
        return players[index];
    }

    public Player[] getPlayers() {
        return players;
    }

    public List<Wall> getWalls() {
        return Collections.unmodifiableList(walls);
    }

    // places a wall and debits the current player's wall count
    public void addWall(Wall wall) {
        wall.setOwnerIndex(currentPlayerIndex);
        walls.add(wall);
        getCurrentPlayer().useWall();
    }

    // returns true if any placed wall blocks movement between the two cells
    public boolean isBlocked(Position from, Position to) {
        for (Wall wall : walls) {
            if (wall.blocksMove(from, to)) return true;
        }
        return false;
    }

    public void nextTurn() {
        currentPlayerIndex = 1 - currentPlayerIndex;
        turnCount++;
    }

    public int getTurnCount() {
        return turnCount;
    }

    public int getCurrentPlayerIndex() {
        return currentPlayerIndex;
    }

    public boolean isGameOver() {
        return gameOver;
    }

    public Player getWinner() {
        return winner;
    }

    public void setGameOver(Player winner) {
        this.gameOver = true;
        this.winner = winner;
    }

    public void checkWinCondition() {
        for (Player player : players) {
            if (player.hasReachedGoal()) {
                setGameOver(player);
                return;
            }
        }
    }

    // NOTE: despite the name, this returns true when the square is CLEAR (no player on it).
    // used in MoveValidator to decide if a jump target is available.
    public boolean isOccupied(Position pos) {
        for (Player player : players) {
            if (player.getPosition().equals(pos)) {
                return false;
            }
        }
        return true;
    }

    // deep copy for bot simulation — doesn't affect the real game
    public GameState deepCopy() {
        GameState copy = new GameState();

        for (int i = 0; i < 2; i++) {
            Player original = this.players[i];
            Player cloned = copy.players[i];
            cloned.setName(original.getName());
            cloned.setPosition(new Position(original.getPosition().getRow(), original.getPosition().getCol()));
            cloned.setWallsRemaining(original.getWallsRemaining());
        }

        for (Wall wall : this.walls) {
            Wall wallCopy = new Wall(wall.getRow(), wall.getCol(), wall.getOrientation());
            wallCopy.setOwnerIndex(wall.getOwnerIndex());
            copy.walls.add(wallCopy);
        }

        copy.currentPlayerIndex = this.currentPlayerIndex;
        copy.turnCount = this.turnCount;
        copy.gameOver = this.gameOver;

        if (this.winner != null) {
            int winnerIdx = (this.winner == this.players[0]) ? 0 : 1;
            copy.winner = copy.players[winnerIdx];
        }

        return copy;
    }

    public void reset() {
        players[0].reset(new Position(8, 4));
        players[1].reset(new Position(0, 4));
        walls.clear();
        currentPlayerIndex = 0;
        turnCount = 0;
        gameOver = false;
        winner = null;
    }
}
