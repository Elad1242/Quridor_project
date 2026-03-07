package ui;

import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Rectangle;
import javafx.scene.effect.DropShadow;
import javafx.scene.effect.InnerShadow;
import model.*;
import logic.MoveValidator;

import java.util.List;

/**
 * Renders the game board - cells, walls, pawns, and valid move highlights.
 */
public class BoardView extends Pane {

    private static final int CELL_SIZE = 55;
    private static final int GAP_SIZE = 12;
    private static final int BOARD_SIZE = 9;
    private static final int PAWN_RADIUS = 22;

    // Colors
    private static final Color BOARD_BG = Color.web("#1a1a2e");
    private static final Color CELL_COLOR = Color.web("#4a4a5c");
    private static final Color CELL_BORDER = Color.web("#2d2d3a");
    private static final Color VALID_MOVE_COLOR = Color.web("#4ade80", 0.7);
    private static final Color HOVER_COLOR = Color.web("#ffffff", 0.2);
    private static final Color WALL_SLOT_COLOR = Color.web("#2d2d3a");
    private static final Color PLAYER1_WALL_COLOR = Color.web("#e74c3c");
    private static final Color PLAYER2_WALL_COLOR = Color.web("#f1c40f");
    private static final Color WALL_PREVIEW_COLOR = Color.web("#888888", 0.7);
    private static final Color WALL_INVALID_COLOR = Color.web("#ff0000", 0.4);

    private GameController controller;
    private GameState gameState;

    private Rectangle[][] cells;
    private Circle[] pawns;
    private Rectangle wallPreview;
    private Position previewWallPos;
    private boolean previewHorizontal;
    private Rectangle selectedWallRect;
    private Wall currentSelectedWall;

    public BoardView() {
        int totalSize = BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * GAP_SIZE + 40;

        setPrefSize(totalSize, totalSize);
        setMinSize(totalSize, totalSize);
        setMaxSize(totalSize, totalSize);
        setStyle("-fx-background-color: #1a1a2e; -fx-background-radius: 10;");

        cells = new Rectangle[BOARD_SIZE][BOARD_SIZE];
        pawns = new Circle[2];

        initializeBoard();
        initializePawns();
        initializeWallPreview();
        initializeSelectedWallRect();
    }

    public void setController(GameController controller) {
        this.controller = controller;
    }

    public void setGameState(GameState gameState) {
        this.gameState = gameState;
    }

    // Creates the 9x9 grid and wall slots
    private void initializeBoard() {
        int padding = 20;

        // Create cells
        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                Rectangle cell = new Rectangle(CELL_SIZE, CELL_SIZE);
                cell.setFill(CELL_COLOR);
                cell.setStroke(CELL_BORDER);
                cell.setStrokeWidth(1);
                cell.setArcWidth(8);
                cell.setArcHeight(8);

                InnerShadow innerShadow = new InnerShadow();
                innerShadow.setRadius(3);
                innerShadow.setColor(Color.rgb(0, 0, 0, 0.3));
                cell.setEffect(innerShadow);

                double x = padding + col * (CELL_SIZE + GAP_SIZE);
                double y = padding + row * (CELL_SIZE + GAP_SIZE);
                cell.setLayoutX(x);
                cell.setLayoutY(y);

                final int r = row;
                final int c = col;

                cell.setOnMouseClicked(e -> {
                    if (controller != null) {
                        controller.onCellClicked(r, c);
                    }
                });

                cell.setOnMouseEntered(e -> {
                    if (controller != null && !controller.isWallMode()) {
                        cell.setStroke(Color.web("#ffffff", 0.5));
                        cell.setStrokeWidth(2);
                    }
                });

                cell.setOnMouseExited(e -> {
                    cell.setStroke(CELL_BORDER);
                    cell.setStrokeWidth(1);
                });

                cells[row][col] = cell;
                getChildren().add(cell);
            }
        }

        // Create wall slots
        for (int row = 0; row < BOARD_SIZE - 1; row++) {
            for (int col = 0; col < BOARD_SIZE - 1; col++) {
                Rectangle hSlot = createWallSlot(row, col, true, padding);
                Rectangle vSlot = createWallSlot(row, col, false, padding);
                getChildren().addAll(hSlot, vSlot);
            }
        }
    }

    private Rectangle createWallSlot(int row, int col, boolean horizontal, int padding) {
        Rectangle slot;
        double x, y;

        if (horizontal) {
            slot = new Rectangle(CELL_SIZE * 2 + GAP_SIZE, GAP_SIZE);
            x = padding + col * (CELL_SIZE + GAP_SIZE);
            y = padding + (row + 1) * CELL_SIZE + row * GAP_SIZE;
        } else {
            slot = new Rectangle(GAP_SIZE, CELL_SIZE * 2 + GAP_SIZE);
            x = padding + (col + 1) * CELL_SIZE + col * GAP_SIZE;
            y = padding + row * (CELL_SIZE + GAP_SIZE);
        }

        slot.setFill(Color.TRANSPARENT);
        slot.setLayoutX(x);
        slot.setLayoutY(y);

        final int r = row;
        final int c = col;
        final boolean isHorizontal = horizontal;

        slot.setOnMouseEntered(e -> {
            if (controller != null && controller.isWallMode()) {
                showWallPreview(r, c, isHorizontal);
            }
        });

        slot.setOnMouseExited(e -> hideWallPreview());

        slot.setOnMouseClicked(e -> {
            if (controller != null && controller.isWallMode()) {
                controller.onWallSlotClicked(r, c, isHorizontal);
                hideWallPreview();
            }
        });

        return slot;
    }

    private void initializePawns() {
        for (int i = 0; i < 2; i++) {
            Circle pawn = new Circle(PAWN_RADIUS);
            pawn.setStroke(Color.web("#1a1a2e"));
            pawn.setStrokeWidth(3);

            DropShadow shadow = new DropShadow();
            shadow.setRadius(8);
            shadow.setOffsetY(3);
            shadow.setColor(Color.rgb(0, 0, 0, 0.5));
            pawn.setEffect(shadow);

            pawns[i] = pawn;
            getChildren().add(pawn);
        }
    }

    private void initializeWallPreview() {
        wallPreview = new Rectangle();
        wallPreview.setArcWidth(4);
        wallPreview.setArcHeight(4);
        wallPreview.setVisible(false);
        wallPreview.setMouseTransparent(true);
        getChildren().add(wallPreview);
    }

    private void initializeSelectedWallRect() {
        selectedWallRect = new Rectangle();
        selectedWallRect.setArcWidth(4);
        selectedWallRect.setArcHeight(4);
        selectedWallRect.setVisible(false);
        selectedWallRect.setMouseTransparent(true);
        selectedWallRect.setFill(Color.web("#00ff00", 0.8));
        selectedWallRect.setStroke(Color.web("#ffffff"));
        selectedWallRect.setStrokeWidth(2);
        getChildren().add(selectedWallRect);
    }

    // Refreshes the board display
    public void update() {
        if (gameState == null) return;

        int padding = 20;

        // Reset cells
        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                cells[row][col].setFill(CELL_COLOR);
            }
        }

        // Highlight valid moves
        if (controller != null && !controller.isWallMode() && !gameState.isGameOver()) {
            List<Position> validMoves = MoveValidator.getValidMoves(gameState, gameState.getCurrentPlayer());
            for (Position pos : validMoves) {
                cells[pos.getRow()][pos.getCol()].setFill(VALID_MOVE_COLOR);
            }
        }

        // Update pawns
        for (int i = 0; i < 2; i++) {
            Player player = gameState.getPlayer(i);
            Position pos = player.getPosition();

            double x = padding + pos.getCol() * (CELL_SIZE + GAP_SIZE) + CELL_SIZE / 2.0;
            double y = padding + pos.getRow() * (CELL_SIZE + GAP_SIZE) + CELL_SIZE / 2.0;

            pawns[i].setCenterX(x);
            pawns[i].setCenterY(y);
            pawns[i].setFill(player.getColor());
        }

        // Redraw walls
        getChildren().removeIf(node ->
            node instanceof Rectangle && "wall".equals(node.getUserData())
        );

        for (Wall wall : gameState.getWalls()) {
            drawWall(wall, padding);
        }
    }

    private void drawWall(Wall wall, int padding) {
        Rectangle wallRect = new Rectangle();
        wallRect.setArcWidth(4);
        wallRect.setArcHeight(4);
        wallRect.setUserData("wall");

        Color wallColor = (wall.getOwnerIndex() == 0) ? PLAYER1_WALL_COLOR : PLAYER2_WALL_COLOR;
        wallRect.setFill(wallColor);

        DropShadow glow = new DropShadow();
        glow.setRadius(5);
        glow.setColor(Color.rgb(255, 255, 255, 0.3));
        wallRect.setEffect(glow);

        int row = wall.getRow();
        int col = wall.getCol();

        if (wall.isHorizontal()) {
            wallRect.setWidth(CELL_SIZE * 2 + GAP_SIZE);
            wallRect.setHeight(GAP_SIZE - 2);
            wallRect.setLayoutX(padding + col * (CELL_SIZE + GAP_SIZE));
            wallRect.setLayoutY(padding + (row + 1) * CELL_SIZE + row * GAP_SIZE + 1);
        } else {
            wallRect.setWidth(GAP_SIZE - 2);
            wallRect.setHeight(CELL_SIZE * 2 + GAP_SIZE);
            wallRect.setLayoutX(padding + (col + 1) * CELL_SIZE + col * GAP_SIZE + 1);
            wallRect.setLayoutY(padding + row * (CELL_SIZE + GAP_SIZE));
        }

        getChildren().add(wallRect);
    }

    private void showWallPreview(int row, int col, boolean horizontal) {
        if (currentSelectedWall != null) return;

        Wall.Orientation orientation = horizontal ?
            Wall.Orientation.HORIZONTAL : Wall.Orientation.VERTICAL;
        Position newPos = new Position(row, col);

        if (previewWallPos != null && previewWallPos.equals(newPos) &&
            previewHorizontal == horizontal && wallPreview.isVisible()) {
            return;
        }

        previewWallPos = newPos;
        previewHorizontal = horizontal;

        int padding = 20;
        Wall previewWall = new Wall(row, col, orientation);

        boolean isValid = logic.WallValidator.isValidWallPlacement(gameState, previewWall);
        wallPreview.setFill(isValid ? WALL_PREVIEW_COLOR : WALL_INVALID_COLOR);

        if (horizontal) {
            wallPreview.setWidth(CELL_SIZE * 2 + GAP_SIZE);
            wallPreview.setHeight(GAP_SIZE - 2);
            wallPreview.setLayoutX(padding + col * (CELL_SIZE + GAP_SIZE));
            wallPreview.setLayoutY(padding + (row + 1) * CELL_SIZE + row * GAP_SIZE + 1);
        } else {
            wallPreview.setWidth(GAP_SIZE - 2);
            wallPreview.setHeight(CELL_SIZE * 2 + GAP_SIZE);
            wallPreview.setLayoutX(padding + (col + 1) * CELL_SIZE + col * GAP_SIZE + 1);
            wallPreview.setLayoutY(padding + row * (CELL_SIZE + GAP_SIZE));
        }

        wallPreview.setVisible(true);
    }

    private void hideWallPreview() {
        if (currentSelectedWall != null) return;
        wallPreview.setVisible(false);
        previewWallPos = null;
    }

    public void showSelectedWall(Wall wall) {
        currentSelectedWall = wall;
        int padding = 20;
        int row = wall.getRow();
        int col = wall.getCol();

        if (wall.isHorizontal()) {
            selectedWallRect.setWidth(CELL_SIZE * 2 + GAP_SIZE);
            selectedWallRect.setHeight(GAP_SIZE - 2);
            selectedWallRect.setLayoutX(padding + col * (CELL_SIZE + GAP_SIZE));
            selectedWallRect.setLayoutY(padding + (row + 1) * CELL_SIZE + row * GAP_SIZE + 1);
        } else {
            selectedWallRect.setWidth(GAP_SIZE - 2);
            selectedWallRect.setHeight(CELL_SIZE * 2 + GAP_SIZE);
            selectedWallRect.setLayoutX(padding + (col + 1) * CELL_SIZE + col * GAP_SIZE + 1);
            selectedWallRect.setLayoutY(padding + row * (CELL_SIZE + GAP_SIZE));
        }

        selectedWallRect.setVisible(true);
        hideWallPreview();
    }

    public void clearSelectedWall() {
        currentSelectedWall = null;
        selectedWallRect.setVisible(false);
    }

}
