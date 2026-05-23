// v2.0 — refactored and cleaned, May 2026
package model;

import java.util.Objects;

// Represents a wall — spans 2 cells, anchored at the top-left corner (valid range: 0-7).
public class Wall {

    public enum Orientation {
        HORIZONTAL,  // blocks up/down movement
        VERTICAL     // blocks left/right movement
    }

    private final Position position;
    private final Orientation orientation;
    private int ownerIndex = -1;  // which player placed this (-1 if unset)

    public Wall(Position position, Orientation orientation) {
        this.position = position;
        this.orientation = orientation;
    }

    public Wall(int row, int col, Orientation orientation) {
        this(new Position(row, col), orientation);
    }

    public int getOwnerIndex() {
        return ownerIndex;
    }

    public void setOwnerIndex(int ownerIndex) {
        this.ownerIndex = ownerIndex;
    }

    public Position getPosition() {
        return position;
    }

    public int getRow() {
        return position.getRow();
    }

    public int getCol() {
        return position.getCol();
    }

    public Orientation getOrientation() {
        return orientation;
    }

    public boolean isHorizontal() {
        return orientation == Orientation.HORIZONTAL;
    }

    public boolean isVertical() {
        return orientation == Orientation.VERTICAL;
    }

    // does this wall block movement between two adjacent cells?
    public boolean blocksMove(Position from, Position to) {
        int fromRow = from.getRow();
        int fromCol = from.getCol();
        int toRow   = to.getRow();
        int toCol   = to.getCol();
        int wallRow = position.getRow();
        int wallCol = position.getCol();

        if (isHorizontal()) {
            // horizontal walls block vertical movement
            boolean sameCol = fromCol == toCol && (fromCol == wallCol || fromCol == wallCol + 1);
            if (!sameCol) return false;
            return (fromRow == wallRow && toRow == wallRow + 1)
                || (fromRow == wallRow + 1 && toRow == wallRow);
        } else {
            // vertical walls block horizontal movement
            boolean sameRow = fromRow == toRow && (fromRow == wallRow || fromRow == wallRow + 1);
            if (!sameRow) return false;
            return (fromCol == wallCol && toCol == wallCol + 1)
                || (fromCol == wallCol + 1 && toCol == wallCol);
        }
    }

    // does this wall overlap with another one? (same orientation touching, or crossing at the same cell)
    public boolean overlaps(Wall other) {
        int r1 = this.getRow();
        int c1 = this.getCol();
        int r2 = other.getRow();
        int c2 = other.getCol();

        if (this.orientation == other.orientation) {
            if (this.isHorizontal()) {
                return r1 == r2 && Math.abs(c1 - c2) <= 1;
            } else {
                return c1 == c2 && Math.abs(r1 - r2) <= 1;
            }
        }

        // different orientations only cross at the exact same anchor
        return r1 == r2 && c1 == c2;
    }

    public boolean isValidPosition() {
        int r = position.getRow();
        int c = position.getCol();
        return r >= 0 && r < 8 && c >= 0 && c < 8;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Wall wall = (Wall) o;
        return Objects.equals(position, wall.position) && orientation == wall.orientation;
    }

    @Override
    public int hashCode() {
        return Objects.hash(position, orientation);
    }

    @Override
    public String toString() {
        return "Wall{" + position + ", " + orientation + "}";
    }
}
