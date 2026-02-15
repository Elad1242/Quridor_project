package model;

import javafx.scene.paint.Color;

/**
 * Represents a player with position, goal row, color, and walls.
 */
public class Player {

    public static final int STARTING_WALLS = 10;

    private String name;
    private final int goalRow;
    private final Color color;
    private Position position;
    private int wallsRemaining;

    public Player(String name, Position startPosition, int goalRow, Color color) {
        this.name = name;
        this.position = startPosition;
        this.goalRow = goalRow;
        this.color = color;
        this.wallsRemaining = STARTING_WALLS;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Position getPosition() {
        return position;
    }

    public int getGoalRow() {
        return goalRow;
    }

    public Color getColor() {
        return color;
    }

    public int getWallsRemaining() {
        return wallsRemaining;
    }

    public void setPosition(Position newPosition) {
        this.position = newPosition;
    }

    public boolean hasWalls() {
        return wallsRemaining > 0;
    }

    public void useWall() {
        if (wallsRemaining > 0) {
            wallsRemaining--;
        }
    }

    public void setWallsRemaining(int wallsRemaining) {
        this.wallsRemaining = wallsRemaining;
    }

    /**
     * Creates a lightweight copy of this player for bot simulation.
     * Shares the immutable Color and goalRow, copies mutable position and walls count.
     */
    public Player copyOf() {
        Player copy = new Player(this.name, new Position(position.getRow(), position.getCol()), this.goalRow, this.color);
        copy.wallsRemaining = this.wallsRemaining;
        return copy;
    }

    public boolean hasReachedGoal() {
        return position.getRow() == goalRow;
    }

    public void reset(Position startPosition) {
        this.position = startPosition;
        this.wallsRemaining = STARTING_WALLS;
    }

    @Override
    public String toString() {
        return name + " at " + position + " (walls: " + wallsRemaining + ")";
    }
}
