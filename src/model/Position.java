package model;

import java.util.Objects;

/**
 * Represents a position on the 9x9 board.
 * Row 0 is top, row 8 is bottom.
 */
public class Position {

    private final int row;
    private final int col;

    public Position(int row, int col) {
        this.row = row;
        this.col = col;
    }

    public int getRow() {
        return row;
    }

    public int getCol() {
        return col;
    }

    // Returns new position offset by given amounts
    public Position move(int dRow, int dCol) {
        return new Position(row + dRow, col + dCol);
    }

    // Checks if position is within board bounds
    public boolean isValid() {
        return row >= 0 && row < 9 && col >= 0 && col < 9;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Position position = (Position) o;
        return row == position.row && col == position.col;
    }

    @Override
    public int hashCode() {
        return Objects.hash(row, col);
    }

    @Override
    public String toString() {
        return "(" + row + ", " + col + ")";
    }
}
