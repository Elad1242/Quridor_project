// v2.0 — refactored and cleaned, May 2026
package ui;

import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.geometry.Rectangle2D;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Modality;
import javafx.stage.Screen;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.util.Duration;
import logic.MoveValidator;
import logic.WallValidator;
import model.*;
import bot.BotBrain;
import ml.MLBot;

import java.util.List;

// Main controller: UI, user input, timer, and game flow.
public class GameController {

    private static final int TURN_TIME_SECONDS = 15;

    private GameState gameState;
    private BoardView boardView;
    private boolean wallMode = false;

    private Wall selectedWall = null;
    private boolean wallConfirmPending = false;

    private Timeline turnTimer;
    private int timeRemaining;
    private Label timerLabel;

    private Label turnLabel;
    private Label player1WallsLabel;
    private Label player2WallsLabel;
    private Label statusLabel;
    private Button placeWallButton;
    private Button movePawnButton;

    private VBox player1Panel;
    private VBox player2Panel;
    private HBox[] player1WallIndicators;
    private HBox[] player2WallIndicators;

    private Stage primaryStage;
    private String player1Name = "Player 1";
    private String player2Name = "Player 2";

    // bot mode flags
    private boolean player2IsBot   = false;
    private boolean player1IsBot   = false; // bot vs bot mode
    private BotBrain botBrain;              // brain for player 2
    private BotBrain botBrain1;             // brain for player 1 (bot vs bot)
    private MLBot mlBot;
    private boolean player1IsMLBot = false;
    private boolean player2IsMLBot = false;
    private boolean botThinking    = false; // true while bot is computing in background

    // bot vs bot controls
    private boolean botVsBotPaused  = false;
    private int botVsBotDelayMs     = 600;
    private Label speedLabel;
    private Button pauseButton;

    // MLBot vs BotBrain win tracking
    private int mlBotWins    = 0;
    private int botBrainWins = 0;
    private Label scoreLabel;

    // incremented on restart to invalidate stale bot callbacks
    private int gameGeneration = 0;
    private Timeline pendingBotDelay = null;

    public GameController(Stage stage) {
        this.primaryStage = stage;

        showPlayerNameDialog();

        gameState = new GameState(player1Name, player2Name);
        initBots();

        Rectangle2D screenBounds = Screen.getPrimary().getVisualBounds();
        double screenW = screenBounds.getWidth();
        double screenH = screenBounds.getHeight();

        // fit board within screen, leaving room for panels
        double availableForBoard = Math.min(screenH - 280, screenW - 400);
        availableForBoard = Math.max(400, Math.min(650, availableForBoard));

        boardView = new BoardView(availableForBoard);
        boardView.setController(this);
        boardView.setGameState(gameState);

        BorderPane root = createMainLayout();

        double sceneW = Math.min(screenW * 0.92, availableForBoard + 400);
        double sceneH = Math.min(screenH * 0.92, availableForBoard + 280);
        Scene scene = new Scene(root, sceneW, sceneH);
        stage.setScene(scene);
        stage.setTitle("Quoridor - Strategy Board Game");
        stage.setResizable(true);
        stage.setMinWidth(800);
        stage.setMinHeight(600);

        boardView.update();
        updateUI();
        initializeTimer();

        if (player1IsBot && player2IsBot) {
            stopTimer();
            setBotVsBotTimerLabel();
            // short delay so the UI finishes rendering before the first bot move
            new Timeline(new KeyFrame(Duration.millis(500), e -> triggerBotTurnIfNeeded())).play();
        } else {
            startTimer();
            triggerBotTurnIfNeeded();
        }
    }

    // sets up bots based on the flags set by the name dialog
    private void initBots() {
        if (player1IsBot && player2IsBot) {
            if (player1IsMLBot) {
                mlBot    = loadMLBot();
                botBrain = new BotBrain(6);
                botBrain.setSilent(true);
                botBrain1 = null;
            } else {
                botBrain  = new BotBrain(6);
                botBrain1 = new BotBrain(6);
                botBrain.setSilent(true);
                botBrain1.setSilent(true);
            }
        } else if (player2IsBot) {
            if (player2IsMLBot) {
                mlBot    = loadMLBot();
                botBrain = null;
            } else {
                botBrain = new BotBrain();
                botBrain.setSilent(true);
            }
        }
    }

    private MLBot loadMLBot() {
        try {
            return new MLBot("models", "best-model");
        } catch (Exception ex) {
            System.err.println("Failed to load MLBot: " + ex.getMessage());
            return null;
        }
    }

    private void setBotVsBotTimerLabel() {
        if (player1IsMLBot) {
            timerLabel.setText("🤖 MLBot (NN) vs BotBrain 🤖");
            timerLabel.setTextFill(Color.web("#e74c3c"));
        } else {
            timerLabel.setText("🤖 Bot vs Bot");
            timerLabel.setTextFill(Color.web("#9b59b6"));
        }
    }

    private void showPlayerNameDialog() {
        Stage dialog = new Stage();
        dialog.initModality(Modality.APPLICATION_MODAL);
        dialog.initOwner(primaryStage);
        dialog.initStyle(StageStyle.UNDECORATED);
        dialog.setTitle("Enter Player Names");

        VBox mainContainer = new VBox(20);
        mainContainer.setAlignment(Pos.CENTER);
        mainContainer.setPadding(new Insets(30));
        mainContainer.setStyle("-fx-background-color: linear-gradient(to bottom, #1a1a2e, #16213e); " +
                              "-fx-background-radius: 15; -fx-border-radius: 15; " +
                              "-fx-border-color: #3498db; -fx-border-width: 2;");

        Label titleLabel = new Label("QUORIDOR");
        titleLabel.setFont(Font.font("Arial", FontWeight.BOLD, 32));
        titleLabel.setTextFill(Color.web("#ecf0f1"));

        Label subtitleLabel = new Label("Enter Player Names");
        subtitleLabel.setFont(Font.font("Arial", FontWeight.NORMAL, 16));
        subtitleLabel.setTextFill(Color.web("#95a5a6"));

        VBox player1Box = new VBox(5);
        player1Box.setAlignment(Pos.CENTER_LEFT);
        HBox player1Header = new HBox(10);
        player1Header.setAlignment(Pos.CENTER_LEFT);
        Circle p1Icon = new Circle(12);
        p1Icon.setFill(Color.web("#2C3E50"));
        p1Icon.setStroke(Color.web("#ecf0f1"));
        p1Icon.setStrokeWidth(2);
        Label p1Label = new Label("Player 1 (Dark - starts at bottom):");
        p1Label.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        p1Label.setTextFill(Color.web("#ecf0f1"));
        player1Header.getChildren().addAll(p1Icon, p1Label);
        TextField player1Field = new TextField("Player 1");
        player1Field.setFont(Font.font("Arial", 14));
        player1Field.setPrefWidth(250);
        player1Field.setStyle("-fx-background-color: #34495e; -fx-text-fill: #ecf0f1; " +
                             "-fx-background-radius: 5; -fx-padding: 10;");
        player1Box.getChildren().addAll(player1Header, player1Field);

        VBox player2Box = new VBox(5);
        player2Box.setAlignment(Pos.CENTER_LEFT);
        HBox player2Header = new HBox(10);
        player2Header.setAlignment(Pos.CENTER_LEFT);
        Circle p2Icon = new Circle(12);
        p2Icon.setFill(Color.web("#ECF0F1"));
        p2Icon.setStroke(Color.web("#2c3e50"));
        p2Icon.setStrokeWidth(2);
        Label p2Label = new Label("Player 2 (Light - starts at top):");
        p2Label.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        p2Label.setTextFill(Color.web("#ecf0f1"));
        player2Header.getChildren().addAll(p2Icon, p2Label);
        TextField player2Field = new TextField("Player 2");
        player2Field.setFont(Font.font("Arial", 14));
        player2Field.setPrefWidth(250);
        player2Field.setStyle("-fx-background-color: #34495e; -fx-text-fill: #ecf0f1; " +
                             "-fx-background-radius: 5; -fx-padding: 10;");
        player2Box.getChildren().addAll(player2Header, player2Field);

        CheckBox botCheckbox         = new CheckBox("🤖 Play against BotBrain");
        CheckBox botVsBotCheckbox    = new CheckBox("🤖 Watch BotBrain vs BotBrain");
        CheckBox mlBotCheckbox       = new CheckBox("🤖 Watch MLBot vs BotBrain");
        CheckBox humanVsMLBotCheckbox = new CheckBox("🎮 Play against MLBot");

        styleCheckbox(botCheckbox,          "#e67e22", player2IsBot && !player2IsMLBot && !player1IsBot);
        styleCheckbox(botVsBotCheckbox,     "#9b59b6", player1IsBot && player2IsBot && !player1IsMLBot);
        styleCheckbox(mlBotCheckbox,        "#e74c3c", player1IsMLBot);
        styleCheckbox(humanVsMLBotCheckbox, "#16a085", player2IsMLBot);

        botCheckbox.setOnAction(e -> {
            if (!botCheckbox.isSelected()) { player2Field.setText("Player 2"); player2Field.setDisable(false); return; }
            botVsBotCheckbox.setSelected(false); mlBotCheckbox.setSelected(false); humanVsMLBotCheckbox.setSelected(false);
            player1Field.setText("Player 1"); player1Field.setDisable(false);
            player2Field.setText("BotBrain"); player2Field.setDisable(true);
        });

        botVsBotCheckbox.setOnAction(e -> {
            if (!botVsBotCheckbox.isSelected()) {
                player1Field.setText("Player 1"); player1Field.setDisable(false);
                player2Field.setText("Player 2"); player2Field.setDisable(false);
                return;
            }
            botCheckbox.setSelected(false); mlBotCheckbox.setSelected(false); humanVsMLBotCheckbox.setSelected(false);
            player1Field.setText("BotBrain A"); player1Field.setDisable(true);
            player2Field.setText("BotBrain B"); player2Field.setDisable(true);
        });

        mlBotCheckbox.setOnAction(e -> {
            if (!mlBotCheckbox.isSelected()) {
                player1Field.setText("Player 1"); player1Field.setDisable(false);
                player2Field.setText("Player 2"); player2Field.setDisable(false);
                return;
            }
            botCheckbox.setSelected(false); botVsBotCheckbox.setSelected(false); humanVsMLBotCheckbox.setSelected(false);
            player1Field.setText("MLBot (NN)"); player1Field.setDisable(true);
            player2Field.setText("BotBrain");   player2Field.setDisable(true);
        });

        humanVsMLBotCheckbox.setOnAction(e -> {
            if (!humanVsMLBotCheckbox.isSelected()) { player2Field.setText("Player 2"); player2Field.setDisable(false); return; }
            botCheckbox.setSelected(false); botVsBotCheckbox.setSelected(false); mlBotCheckbox.setSelected(false);
            player1Field.setText("Player 1");      player1Field.setDisable(false);
            player2Field.setText("MLBot (Java NN)"); player2Field.setDisable(true);
        });

        // restore dialog state on restart
        if (player1IsMLBot) {
            mlBotCheckbox.setSelected(true);
            player1Field.setText("MLBot (NN)"); player1Field.setDisable(true);
            player2Field.setText("BotBrain");   player2Field.setDisable(true);
        } else if (player2IsMLBot) {
            humanVsMLBotCheckbox.setSelected(true);
            player2Field.setText("MLBot (Java NN)"); player2Field.setDisable(true);
        } else if (player1IsBot && player2IsBot) {
            botVsBotCheckbox.setSelected(true);
            player1Field.setText("Bot A"); player1Field.setDisable(true);
            player2Field.setText("Bot B"); player2Field.setDisable(true);
        } else if (player2IsBot) {
            botCheckbox.setSelected(true);
            player2Field.setText("BotBrain"); player2Field.setDisable(true);
        }

        Button startButton = new Button("Start Game");
        startButton.setFont(Font.font("Arial", FontWeight.BOLD, 16));
        startButton.setPrefSize(200, 45);
        startButton.setStyle("-fx-background-color: #27ae60; -fx-text-fill: white; -fx-background-radius: 5; -fx-cursor: hand;");
        startButton.setOnMouseEntered(e -> startButton.setStyle("-fx-background-color: #2ecc71; -fx-text-fill: white; -fx-background-radius: 5; -fx-cursor: hand;"));
        startButton.setOnMouseExited(e ->  startButton.setStyle("-fx-background-color: #27ae60; -fx-text-fill: white; -fx-background-radius: 5; -fx-cursor: hand;"));

        startButton.setOnAction(e -> {
            String name1 = player1Field.getText().trim();
            String name2 = player2Field.getText().trim();
            player1Name = name1.isEmpty() ? "Player 1" : name1;
            player2Name = name2.isEmpty() ? "Player 2" : name2;
            player2IsBot   = botCheckbox.isSelected() || botVsBotCheckbox.isSelected()
                          || mlBotCheckbox.isSelected() || humanVsMLBotCheckbox.isSelected();
            player1IsBot   = botVsBotCheckbox.isSelected() || mlBotCheckbox.isSelected();
            player1IsMLBot = mlBotCheckbox.isSelected();
            player2IsMLBot = humanVsMLBotCheckbox.isSelected();
            dialog.close();
        });

        player2Field.setOnAction(e -> startButton.fire());
        player1Field.setOnAction(e -> player2Field.requestFocus());

        mainContainer.getChildren().addAll(titleLabel, subtitleLabel,
                player1Box, player2Box, botCheckbox, botVsBotCheckbox,
                mlBotCheckbox, humanVsMLBotCheckbox, startButton);

        Scene dialogScene = new Scene(mainContainer, 380, 650);
        dialog.setScene(dialogScene);
        dialog.centerOnScreen();
        dialog.showAndWait();
    }

    private void styleCheckbox(CheckBox cb, String color, boolean selected) {
        cb.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        cb.setTextFill(Color.web(color));
        cb.setSelected(selected);
        cb.setStyle("-fx-cursor: hand;");
    }

    private BorderPane createMainLayout() {
        BorderPane root = new BorderPane();
        root.setStyle("-fx-background-color: linear-gradient(to bottom, #1a1a2e, #16213e);");
        root.setPadding(new Insets(10));

        root.setTop(createTopPanel());
        root.setCenter(new StackPane(boardView) {{ setPadding(new Insets(10)); }});
        root.setLeft(player1Panel  = createPlayerPanel(0));
        root.setRight(player2Panel = createPlayerPanel(1));
        root.setBottom(createBottomPanel());

        return root;
    }

    private VBox createTopPanel() {
        VBox panel = new VBox(5);
        panel.setAlignment(Pos.CENTER);
        panel.setPadding(new Insets(5, 10, 10, 10));

        Label title = new Label("QUORIDOR");
        title.setFont(Font.font("Arial", FontWeight.BOLD, 28));
        title.setTextFill(Color.web("#ecf0f1"));

        turnLabel = new Label("Player 1's Turn");
        turnLabel.setFont(Font.font("Arial", FontWeight.BOLD, 18));
        turnLabel.setTextFill(Color.web("#3498db"));

        timerLabel = new Label("Time: 15");
        timerLabel.setFont(Font.font("Arial", FontWeight.BOLD, 20));
        timerLabel.setTextFill(Color.web("#e74c3c"));
        timerLabel.setStyle("-fx-background-color: #2c3e50; -fx-padding: 6 16; -fx-background-radius: 5;");

        statusLabel = new Label("");
        statusLabel.setFont(Font.font("Arial", FontWeight.NORMAL, 13));
        statusLabel.setTextFill(Color.web("#f39c12"));

        panel.getChildren().addAll(title, turnLabel, timerLabel, statusLabel);
        return panel;
    }

    private VBox createPlayerPanel(int playerIndex) {
        VBox panel = new VBox(10);
        panel.setAlignment(Pos.TOP_CENTER);
        panel.setPadding(new Insets(15, 10, 15, 10));
        panel.setPrefWidth(160);
        panel.setMinWidth(140);
        panel.setStyle("-fx-background-color: #2c3e50; -fx-background-radius: 10;");

        Player player = gameState.getPlayer(playerIndex);

        Circle icon = new Circle(30);
        icon.setFill(player.getColor());
        icon.setStroke(Color.web("#ecf0f1"));
        icon.setStrokeWidth(3);

        Label nameLabel = new Label(player.getName());
        nameLabel.setFont(Font.font("Arial", FontWeight.BOLD, 16));
        nameLabel.setTextFill(Color.web("#ecf0f1"));

        Label goalLabel = new Label(playerIndex == 0 ? "Goal: Top ↑" : "Goal: Bottom ↓");
        goalLabel.setFont(Font.font("Arial", 12));
        goalLabel.setTextFill(Color.web("#95a5a6"));

        Label wallsTitle = new Label("Walls");
        wallsTitle.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        wallsTitle.setTextFill(Color.web("#ecf0f1"));

        Label wallsLabel = new Label("10");
        wallsLabel.setFont(Font.font("Arial", FontWeight.BOLD, 24));
        wallsLabel.setTextFill(Color.web("#3498db"));

        if (playerIndex == 0) player1WallsLabel = wallsLabel;
        else                  player2WallsLabel = wallsLabel;

        Color wallColor = (playerIndex == 0) ? Color.web("#e74c3c") : Color.web("#f1c40f");
        VBox wallIndicators = new VBox(3);
        wallIndicators.setAlignment(Pos.CENTER);

        for (int i = 0; i < 2; i++) {
            HBox row = new HBox(3);
            row.setAlignment(Pos.CENTER);
            for (int j = 0; j < 5; j++) {
                Rectangle wallBlock = new Rectangle(18, 8);
                wallBlock.setFill(wallColor);
                wallBlock.setArcWidth(2);
                wallBlock.setArcHeight(2);
                row.getChildren().add(wallBlock);
            }
            wallIndicators.getChildren().add(row);
        }

        if (playerIndex == 0) {
            player1WallIndicators = new HBox[]{
                (HBox) wallIndicators.getChildren().get(0),
                (HBox) wallIndicators.getChildren().get(1)
            };
        } else {
            player2WallIndicators = new HBox[]{
                (HBox) wallIndicators.getChildren().get(0),
                (HBox) wallIndicators.getChildren().get(1)
            };
        }

        Region spacer = new Region();
        VBox.setVgrow(spacer, Priority.ALWAYS);
        panel.getChildren().addAll(icon, nameLabel, goalLabel, spacer, wallsTitle, wallsLabel, wallIndicators);

        return panel;
    }

    private HBox createBottomPanel() {
        HBox panel = new HBox(15);
        panel.setAlignment(Pos.CENTER);
        panel.setPadding(new Insets(10, 15, 10, 15));
        panel.setStyle("-fx-background-color: #2c3e50; -fx-background-radius: 10;");
        panel.setMinHeight(55);

        movePawnButton = new Button("Move Pawn");
        movePawnButton.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        movePawnButton.setPrefSize(140, 40);
        styleButton(movePawnButton, "#27ae60");
        movePawnButton.setOnAction(e -> setMoveMode());

        placeWallButton = new Button("Place Wall");
        placeWallButton.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        placeWallButton.setPrefSize(140, 40);
        styleButton(placeWallButton, "#3498db");
        placeWallButton.setOnAction(e -> setWallMode());

        Button restartButton = new Button("New Game");
        restartButton.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        restartButton.setPrefSize(120, 40);
        styleButton(restartButton, "#e74c3c");
        restartButton.setOnAction(e -> restart());

        updateModeButtons();
        panel.getChildren().addAll(movePawnButton, placeWallButton, restartButton);

        if (player1IsBot && player2IsBot) {
            movePawnButton.setVisible(false);
            movePawnButton.setManaged(false);
            placeWallButton.setVisible(false);
            placeWallButton.setManaged(false);

            Region spacer = new Region();
            spacer.setPrefWidth(20);

            pauseButton = new Button("⏸ Pause");
            pauseButton.setFont(Font.font("Arial", FontWeight.BOLD, 13));
            pauseButton.setPrefSize(100, 40);
            styleButton(pauseButton, "#9b59b6");
            pauseButton.setOnAction(e -> toggleBotVsBotPause());

            Button slowerButton = new Button("🐢 Slower");
            slowerButton.setFont(Font.font("Arial", FontWeight.BOLD, 13));
            slowerButton.setPrefSize(95, 40);
            styleButton(slowerButton, "#e67e22");
            slowerButton.setOnAction(e -> changeBotSpeed(+300));

            Button fasterButton = new Button("⚡ Faster");
            fasterButton.setFont(Font.font("Arial", FontWeight.BOLD, 13));
            fasterButton.setPrefSize(95, 40);
            styleButton(fasterButton, "#2ecc71");
            fasterButton.setOnAction(e -> changeBotSpeed(-200));

            speedLabel = new Label("Speed: 0.6s");
            speedLabel.setFont(Font.font("Arial", FontWeight.BOLD, 14));
            speedLabel.setTextFill(Color.web("#ecf0f1"));

            panel.getChildren().addAll(spacer, pauseButton, slowerButton, fasterButton, speedLabel);

            if (player1IsMLBot) {
                Region spacer2 = new Region();
                spacer2.setPrefWidth(30);
                scoreLabel = new Label("Score: MLBot " + mlBotWins + " - " + botBrainWins + " BotBrain");
                scoreLabel.setFont(Font.font("Arial", FontWeight.BOLD, 14));
                scoreLabel.setTextFill(Color.web("#f1c40f"));
                scoreLabel.setStyle("-fx-background-color: #34495e; -fx-padding: 8 15; -fx-background-radius: 5;");
                panel.getChildren().addAll(spacer2, scoreLabel);
            }
        }

        return panel;
    }

    private void styleButton(Button button, String color) {
        button.setStyle("-fx-background-color: " + color + "; -fx-text-fill: white; -fx-background-radius: 5; -fx-cursor: hand;");
        button.setOnMouseEntered(e -> button.setStyle("-fx-background-color: derive(" + color + ", 20%); -fx-text-fill: white; -fx-background-radius: 5; -fx-cursor: hand;"));
        button.setOnMouseExited(e  -> button.setStyle("-fx-background-color: " + color + "; -fx-text-fill: white; -fx-background-radius: 5; -fx-cursor: hand;"));
    }

    private void initializeTimer() {
        turnTimer = new Timeline(new KeyFrame(Duration.seconds(1), e -> {
            timeRemaining--;
            updateTimerDisplay();
            if (timeRemaining <= 0) onTimeUp();
        }));
        turnTimer.setCycleCount(Timeline.INDEFINITE);
    }

    private void startTimer() {
        turnTimer.stop();
        timeRemaining = TURN_TIME_SECONDS;
        updateTimerDisplay();
        turnTimer.play();
    }

    private void stopTimer() {
        turnTimer.stop();
    }

    private void updateTimerDisplay() {
        timerLabel.setText("Time: " + timeRemaining);
        if (timeRemaining <= 5) {
            timerLabel.setTextFill(Color.web("#e74c3c"));
        } else if (timeRemaining <= 10) {
            timerLabel.setTextFill(Color.web("#f39c12"));
        } else {
            timerLabel.setTextFill(Color.web("#2ecc71"));
        }
    }

    private void onTimeUp() {
        if (gameState.isGameOver()) return;
        if (botThinking) return;
        if (player1IsBot && player2IsBot) return;

        statusLabel.setText(gameState.getCurrentPlayer().getName() + " ran out of time!");
        gameState.nextTurn();
        wallMode = false;
        updateModeButtons();
        boardView.update();
        updateUI();
        startTimer();
        triggerBotTurnIfNeeded();
    }

    public void onCellClicked(int row, int col) {
        if (gameState.isGameOver()) return;
        if (wallMode) return;
        if (botThinking) return;

        Position target = new Position(row, col);
        Player current  = gameState.getCurrentPlayer();
        List<Position> validMoves = MoveValidator.getValidMoves(gameState, current);

        if (!validMoves.contains(target)) {
            statusLabel.setText("Invalid move! Click a green cell.");
            return;
        }

        current.setPosition(target);
        gameState.checkWinCondition();

        if (gameState.isGameOver()) {
            stopTimer();
            showWinner();
        } else {
            gameState.nextTurn();
            startTimer();
            triggerBotTurnIfNeeded();
        }

        boardView.update();
        updateUI();
        statusLabel.setText("");
    }

    // two-click wall placement: first click selects, second click confirms
    public void onWallSlotClicked(int row, int col, boolean horizontal) {
        if (gameState.isGameOver()) return;
        if (!wallMode) return;
        if (botThinking) return;

        Wall.Orientation orientation = horizontal ? Wall.Orientation.HORIZONTAL : Wall.Orientation.VERTICAL;
        Wall wall = new Wall(row, col, orientation);

        if (wallConfirmPending && selectedWall != null && selectedWall.equals(wall)) {
            // second click — try to place
            if (WallValidator.isValidWallPlacement(gameState, wall)) {
                gameState.addWall(wall);
                selectedWall = null;
                wallConfirmPending = false;
                gameState.nextTurn();
                wallMode = false;
                updateModeButtons();
                startTimer();
                boardView.update();
                updateUI();
                statusLabel.setText("");
                triggerBotTurnIfNeeded();
            } else {
                statusLabel.setText("Can't place wall: " + WallValidator.getInvalidReason(gameState, wall));
                selectedWall = null;
                wallConfirmPending = false;
                boardView.clearSelectedWall();
            }
            return;
        }

        // first click — select
        if (WallValidator.isValidWallPlacement(gameState, wall)) {
            selectedWall = wall;
            wallConfirmPending = true;
            boardView.showSelectedWall(wall);
            statusLabel.setText("Click again to confirm wall placement");
        } else {
            statusLabel.setText("Can't place wall: " + WallValidator.getInvalidReason(gameState, wall));
            selectedWall = null;
            wallConfirmPending = false;
            boardView.clearSelectedWall();
        }
    }

    public boolean isWallMode() {
        return wallMode;
    }

    private void setMoveMode() {
        if (gameState.isGameOver()) return;
        wallMode = false;
        selectedWall = null;
        wallConfirmPending = false;
        boardView.clearSelectedWall();
        updateModeButtons();
        boardView.update();
        statusLabel.setText("Click a green cell to move your pawn");
    }

    private void setWallMode() {
        if (gameState.isGameOver()) return;
        if (!gameState.getCurrentPlayer().hasWalls()) {
            statusLabel.setText("No walls remaining!");
            return;
        }
        wallMode = true;
        selectedWall = null;
        wallConfirmPending = false;
        boardView.clearSelectedWall();
        updateModeButtons();
        boardView.update();
        statusLabel.setText("Click on a gap to select wall position");
    }

    private void updateModeButtons() {
        if (wallMode) {
            styleButtonActive(placeWallButton, "#27ae60");
            styleButton(movePawnButton, "#3498db");
        } else {
            styleButtonActive(movePawnButton, "#27ae60");
            styleButton(placeWallButton, "#3498db");
        }
    }

    private void styleButtonActive(Button button, String color) {
        button.setStyle("-fx-background-color: " + color + "; -fx-text-fill: white; -fx-background-radius: 5; " +
                        "-fx-border-color: white; -fx-border-width: 3; -fx-border-radius: 5; -fx-cursor: hand;");
    }

    private void updateUI() {
        Player current = gameState.getCurrentPlayer();

        turnLabel.setText(current.getName() + "'s Turn");
        turnLabel.setTextFill(current.getColor());

        player1WallsLabel.setText(String.valueOf(gameState.getPlayer(0).getWallsRemaining()));
        player2WallsLabel.setText(String.valueOf(gameState.getPlayer(1).getWallsRemaining()));

        updateWallIndicators(player1WallIndicators, gameState.getPlayer(0).getWallsRemaining(), 0);
        updateWallIndicators(player2WallIndicators, gameState.getPlayer(1).getWallsRemaining(), 1);

        if (gameState.getCurrentPlayerIndex() == 0) {
            player1Panel.setStyle("-fx-background-color: #34495e; -fx-background-radius: 10; -fx-border-color: #3498db; -fx-border-width: 3; -fx-border-radius: 10;");
            player2Panel.setStyle("-fx-background-color: #2c3e50; -fx-background-radius: 10;");
        } else {
            player1Panel.setStyle("-fx-background-color: #2c3e50; -fx-background-radius: 10;");
            player2Panel.setStyle("-fx-background-color: #34495e; -fx-background-radius: 10; -fx-border-color: #3498db; -fx-border-width: 3; -fx-border-radius: 10;");
        }

        if (wallMode && !current.hasWalls()) {
            wallMode = false;
            updateModeButtons();
        }
    }

    private void updateWallIndicators(HBox[] rows, int wallsRemaining, int playerIndex) {
        if (rows == null) return;
        Color availableColor = (playerIndex == 0) ? Color.web("#e74c3c") : Color.web("#f1c40f");

        int wallIndex = 0;
        for (HBox row : rows) {
            for (javafx.scene.Node node : row.getChildren()) {
                if (!(node instanceof Rectangle)) continue;
                Rectangle block = (Rectangle) node;
                block.setFill(wallIndex < wallsRemaining ? availableColor : Color.web("#555555"));
                wallIndex++;
            }
        }
    }

    private void showWinner() {
        Player winner = gameState.getWinner();

        if (player1IsMLBot) {
            if (winner == gameState.getPlayer(0)) mlBotWins++;
            else botBrainWins++;
            if (scoreLabel != null) {
                scoreLabel.setText("Score: MLBot " + mlBotWins + " - " + botBrainWins + " BotBrain");
            }
        }

        Stage dialog = new Stage();
        dialog.initModality(Modality.APPLICATION_MODAL);
        dialog.initOwner(primaryStage);
        dialog.initStyle(StageStyle.UNDECORATED);
        dialog.setTitle("Game Over!");

        VBox container = new VBox(20);
        container.setAlignment(Pos.CENTER);
        container.setPadding(new Insets(40));
        container.setStyle("-fx-background-color: linear-gradient(to bottom, #1a1a2e, #16213e); " +
                          "-fx-background-radius: 15; -fx-border-radius: 15; " +
                          "-fx-border-color: #f1c40f; -fx-border-width: 3;");

        Label titleLabel = new Label("GAME OVER!");
        titleLabel.setFont(Font.font("Arial", FontWeight.BOLD, 28));
        titleLabel.setTextFill(Color.web("#ecf0f1"));

        Label winnerLabel = new Label("🏆 " + winner.getName() + " Wins! 🏆");
        winnerLabel.setFont(Font.font("Arial", FontWeight.BOLD, 24));
        winnerLabel.setTextFill(Color.web("#f1c40f"));

        Label messageLabel = new Label("Congratulations! " + winner.getName() + " has reached their goal row!");
        messageLabel.setFont(Font.font("Arial", 14));
        messageLabel.setTextFill(Color.web("#95a5a6"));

        HBox buttonBox = new HBox(20);
        buttonBox.setAlignment(Pos.CENTER);

        Button nextGameButton = new Button("Next Game");
        nextGameButton.setFont(Font.font("Arial", FontWeight.BOLD, 16));
        nextGameButton.setPrefSize(150, 45);
        nextGameButton.setStyle("-fx-background-color: #27ae60; -fx-text-fill: white; -fx-background-radius: 5; -fx-cursor: hand;");
        nextGameButton.setOnMouseEntered(e -> nextGameButton.setStyle("-fx-background-color: #2ecc71; -fx-text-fill: white; -fx-background-radius: 5; -fx-cursor: hand;"));
        nextGameButton.setOnMouseExited(e  -> nextGameButton.setStyle("-fx-background-color: #27ae60; -fx-text-fill: white; -fx-background-radius: 5; -fx-cursor: hand;"));
        nextGameButton.setOnAction(e -> {
            dialog.close();
            Platform.runLater(this::restart); // defer to avoid nested showAndWait
        });

        Button quitButton = new Button("Quit");
        quitButton.setFont(Font.font("Arial", FontWeight.BOLD, 16));
        quitButton.setPrefSize(150, 45);
        quitButton.setStyle("-fx-background-color: #e74c3c; -fx-text-fill: white; -fx-background-radius: 5; -fx-cursor: hand;");
        quitButton.setOnMouseEntered(e -> quitButton.setStyle("-fx-background-color: #c0392b; -fx-text-fill: white; -fx-background-radius: 5; -fx-cursor: hand;"));
        quitButton.setOnMouseExited(e  -> quitButton.setStyle("-fx-background-color: #e74c3c; -fx-text-fill: white; -fx-background-radius: 5; -fx-cursor: hand;"));
        quitButton.setOnAction(e -> { dialog.close(); primaryStage.close(); });

        buttonBox.getChildren().addAll(nextGameButton, quitButton);
        container.getChildren().addAll(titleLabel, winnerLabel, messageLabel, buttonBox);

        Scene dialogScene = new Scene(container, 400, 250);
        dialog.setScene(dialogScene);
        dialog.centerOnScreen();
        dialog.showAndWait();
    }

    private void restart() {
        try {
            gameGeneration++;
            if (pendingBotDelay != null) { pendingBotDelay.stop(); pendingBotDelay = null; }
            stopTimer();

            showPlayerNameDialog();

            gameState.getPlayer(0).setName(player1Name);
            gameState.getPlayer(1).setName(player2Name);

            // re-initialize bots based on dialog selection
            if (player1IsBot && player2IsBot) {
                if (player1IsMLBot) {
                    mlBot    = loadMLBot();
                    botBrain = new BotBrain(6); botBrain.setSilent(true);
                    botBrain1 = null;
                } else {
                    botBrain  = new BotBrain(6); botBrain.setSilent(true);
                    botBrain1 = new BotBrain(6); botBrain1.setSilent(true);
                    mlBot = null;
                }
            } else if (player2IsBot) {
                if (player2IsMLBot) {
                    mlBot = loadMLBot(); botBrain = null;
                } else {
                    botBrain = new BotBrain(); botBrain.setSilent(true); mlBot = null;
                }
                botBrain1 = null;
            } else {
                botBrain = null; botBrain1 = null; mlBot = null;
            }

            botThinking    = false;
            botVsBotPaused = false;

            gameState.reset();
            wallMode = false;
            selectedWall = null;
            wallConfirmPending = false;
            boardView.clearSelectedWall();
            updateModeButtons();
            rebuildPlayerPanels();
            rebuildBottomPanel();
            boardView.update();
            updateUI();
            statusLabel.setText("");

            if (player1IsBot && player2IsBot) {
                stopTimer();
                setBotVsBotTimerLabel();
                new Timeline(new KeyFrame(Duration.millis(500), e -> triggerBotTurnIfNeeded())).play();
            } else {
                startTimer();
            }
        } catch (Exception ex) {
            System.err.println("[RESTART ERROR] " + ex.getClass().getName() + ": " + ex.getMessage());
            ex.printStackTrace();
        }
    }

    private void rebuildPlayerPanels() {
        BorderPane root = (BorderPane) primaryStage.getScene().getRoot();
        player1Panel = createPlayerPanel(0); root.setLeft(player1Panel);
        player2Panel = createPlayerPanel(1); root.setRight(player2Panel);
    }

    private void rebuildBottomPanel() {
        BorderPane root = (BorderPane) primaryStage.getScene().getRoot();
        root.setBottom(createBottomPanel());
    }

    private void toggleBotVsBotPause() {
        botVsBotPaused = !botVsBotPaused;
        if (pauseButton == null) return;

        if (botVsBotPaused) {
            pauseButton.setText("▶ Play");
            statusLabel.setText("⏸ Paused - click Play to continue");
            statusLabel.setTextFill(Color.web("#f39c12"));
        } else {
            pauseButton.setText("⏸ Pause");
            statusLabel.setText("");
            triggerBotTurnIfNeeded();
        }
    }

    private void changeBotSpeed(int deltaMs) {
        botVsBotDelayMs = Math.max(100, Math.min(2000, botVsBotDelayMs + deltaMs));
        if (speedLabel != null) {
            speedLabel.setText("Speed: " + String.format("%.1f", botVsBotDelayMs / 1000.0) + "s");
        }
    }

    // checks whose turn it is and schedules the bot's move on a background thread
    private void triggerBotTurnIfNeeded() {
        if (gameState.isGameOver()) return;
        if (botVsBotPaused) return;

        int currentIdx = gameState.getCurrentPlayerIndex();

        boolean useMLBot = (currentIdx == 0 && player1IsMLBot && mlBot != null)
                        || (currentIdx == 1 && player2IsMLBot && mlBot != null);

        BotBrain activeBrain = null;
        if (!useMLBot) {
            if (currentIdx == 0 && player1IsBot && botBrain1 != null) activeBrain = botBrain1;
            else if (currentIdx == 1 && player2IsBot && !player2IsMLBot && botBrain != null) activeBrain = botBrain;
        }

        if (activeBrain == null && !useMLBot) return;

        botThinking = true;
        stopTimer();

        String botName = gameState.getCurrentPlayer().getName();
        boolean isBotVsBot = player1IsBot && player2IsBot;
        statusLabel.setText(isBotVsBot ? (botName + " is thinking...") : "Bot is thinking...");
        statusLabel.setTextFill(Color.web("#e67e22"));

        int delayMs = isBotVsBot ? botVsBotDelayMs : 800;
        final BotBrain brainToUse = activeBrain;
        final boolean useML = useMLBot;
        final int expectedGeneration = gameGeneration;

        if (pendingBotDelay != null) pendingBotDelay.stop();

        pendingBotDelay = new Timeline(new KeyFrame(Duration.millis(delayMs), e -> {
            if (expectedGeneration != gameGeneration) return;

            Thread botThread = new Thread(() -> {
                if (useML) {
                    MLBot.Action mlAction = mlBot.computeBestAction(gameState);
                    Platform.runLater(() -> {
                        if (expectedGeneration != gameGeneration) return;
                        executeMLBotAction(mlAction);
                    });
                } else {
                    BotBrain.BotAction action = brainToUse.computeBestAction(gameState);
                    Platform.runLater(() -> {
                        if (expectedGeneration != gameGeneration) return;
                        executeBotAction(action);
                    });
                }
            });
            botThread.setDaemon(true);
            botThread.start();
        }));
        pendingBotDelay.play();
    }

    // runs on the JavaFX thread — safe to touch game state and UI
    private void executeBotAction(BotBrain.BotAction action) {
        try {
            String botName = gameState.getCurrentPlayer().getName();

            if (action == null) {
                statusLabel.setText(botName + " couldn't decide, turn skipped");
                advanceTurn();
                return;
            }

            if (action.type == BotBrain.BotAction.Type.MOVE) {
                gameState.getCurrentPlayer().setPosition(action.moveTarget);
                statusLabel.setText(botName + " moved to " + action.moveTarget);
                statusLabel.setTextFill(Color.web("#2ecc71"));

                gameState.checkWinCondition();
                if (gameState.isGameOver()) {
                    botThinking = false; stopTimer(); boardView.update(); updateUI(); showWinner();
                    return;
                }

            } else if (action.type == BotBrain.BotAction.Type.WALL) {
                Wall wall = action.wallToPlace;
                if (WallValidator.isValidWallPlacement(gameState, wall)) {
                    gameState.addWall(wall);
                    String orientStr = wall.isHorizontal() ? "horizontal" : "vertical";
                    statusLabel.setText(botName + " placed " + orientStr + " wall at ("
                                       + wall.getRow() + "," + wall.getCol() + ")");
                    statusLabel.setTextFill(Color.web("#e67e22"));
                } else {
                    statusLabel.setText(botName + " wall was invalid, turn skipped");
                }
            }

            advanceTurn();
        } catch (Exception ex) {
            System.err.println("Bot action error: " + ex.getMessage());
            ex.printStackTrace();
            botThinking = false;
        }
    }

    private void executeMLBotAction(MLBot.Action action) {
        try {
            String botName = gameState.getCurrentPlayer().getName();

            if (action == null) {
                statusLabel.setText(botName + " couldn't decide, turn skipped");
                advanceTurn();
                return;
            }

            if (action.type == MLBot.Action.Type.MOVE) {
                gameState.getCurrentPlayer().setPosition(action.moveTarget);
                statusLabel.setText(botName + " moved to " + action.moveTarget
                        + " (NN: " + String.format("%.3f", action.score) + ")");
                statusLabel.setTextFill(Color.web("#e74c3c"));

                gameState.checkWinCondition();
                if (gameState.isGameOver()) {
                    botThinking = false; stopTimer(); boardView.update(); updateUI(); showWinner();
                    return;
                }

            } else if (action.type == MLBot.Action.Type.WALL) {
                Wall wall = action.wall;
                if (WallValidator.isValidWallPlacement(gameState, wall)) {
                    gameState.addWall(wall);
                    String orientStr = wall.isHorizontal() ? "horizontal" : "vertical";
                    statusLabel.setText(botName + " placed " + orientStr + " wall at ("
                            + wall.getRow() + "," + wall.getCol() + ")"
                            + " (NN: " + String.format("%.3f", action.score) + ")");
                    statusLabel.setTextFill(Color.web("#e74c3c"));
                } else {
                    statusLabel.setText(botName + " wall was invalid, turn skipped");
                }
            }

            advanceTurn();
        } catch (Exception ex) {
            System.err.println("[MLBOT ACTION ERROR] " + ex.getMessage());
            ex.printStackTrace();
            botThinking = false;
        }
    }

    // shared post-action cleanup — advance turn, refresh UI, chain next bot move
    private void advanceTurn() {
        gameState.nextTurn();
        botThinking = false;
        wallMode = false;
        updateModeButtons();
        boardView.update();
        updateUI();
        startTimer();
        triggerBotTurnIfNeeded();
    }
}
