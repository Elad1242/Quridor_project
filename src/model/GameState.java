package model;

import javafx.scene.paint.Color;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Holds all game data: players, walls, turns, and win state.
 */
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

        // Player 1: starts bottom, goal is top
        players[0] = new Player(
            player1Name,
            new Position(8, 4),
            0,
            Color.web("#2C3E50")
        );

        // Player 2: starts top, goal is bottom
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

    //return a list that cannot be modified
    public List<Wall> getWalls() {
        return Collections.unmodifiableList(walls);
    }

    // Adds wall and decrements player's wall count
    public void addWall(Wall wall) {
        wall.setOwnerIndex(currentPlayerIndex);
        walls.add(wall);
        getCurrentPlayer().useWall();
    }

    // Checks if any wall blocks movement between two cells
    public boolean isBlocked(Position from, Position to) {
        for (Wall wall : walls) {
            if (wall.blocksMove(from, to)) {
                return true;
            }
        }
        return false;
    }

    // Switches turn to the other player
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

    // Check if someone won
    public void checkWinCondition() {
        for (Player player : players) {
            if (player.hasReachedGoal()) {
                setGameOver(player);
                return;
            }
        }
    }

    public boolean isOccupied(Position pos) {
        for (Player player : players) {
            if (player.getPosition().equals(pos)) {
                return false;
            }
        }
        return true;
    }


    /**
     * Creates a deep copy of the entire game state for bot simulation.
     * The bot uses this to "imagine" future moves without affecting the real game.
     * All mutable objects (players, walls, positions) are cloned independently.
     */
    public GameState deepCopy() {
        GameState copy = new GameState();

        // Copy each player's current state (position, walls remaining)
        for (int i = 0; i < 2; i++) {
            Player original = this.players[i];
            Player cloned = copy.players[i];
            cloned.setName(original.getName());
            cloned.setPosition(new Position(original.getPosition().getRow(), original.getPosition().getCol()));
            cloned.setWallsRemaining(original.getWallsRemaining());
        }

        // Copy all walls on the board
        for (Wall wall : this.walls) {
            Wall wallCopy = new Wall(wall.getRow(), wall.getCol(), wall.getOrientation());
            wallCopy.setOwnerIndex(wall.getOwnerIndex());
            copy.walls.add(wallCopy);
        }

        copy.currentPlayerIndex = this.currentPlayerIndex;
        copy.turnCount = this.turnCount;
        copy.gameOver = this.gameOver;
        // winner reference points to copy's player array
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
