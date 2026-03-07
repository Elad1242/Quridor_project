package ui;

import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.util.Duration;
import logic.MoveValidator;
import logic.WallValidator;
import model.*;
import bot.BotBrain;

import java.util.List;

/**
 * Main controller - handles UI, user input, timer, and game flow.
 */
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

    // --- Bot integration ---
    // Player 2 can be controlled by the algorithmic bot.
    // When active, human input is blocked during the bot's turn and
    // the bot computes its action in a background thread.
    private boolean player2IsBot = false;
    private boolean player1IsBot = false; // Bot vs Bot mode
    private BotBrain botBrain;    // Brain for player 2
    private BotBrain botBrain1;   // Brain for player 1 (Bot vs Bot mode)
    private boolean botThinking = false; // True while bot is computing

    // --- Bot vs Bot controls ---
    private boolean botVsBotPaused = false;
    private int botVsBotDelayMs = 600; // Delay between moves in ms
    private Label speedLabel;
    private Button pauseButton;

    // --- Game generation counter (prevents stale bot actions after restart) ---
    private int gameGeneration = 0;
    private Timeline pendingBotDelay; // Track pending bot delay for cancellation

    public GameController(Stage stage) {
        this.primaryStage = stage;

        showPlayerNameDialog();

        gameState = new GameState(player1Name, player2Name);

        // Initialize bots if AI-controlled
        if (player1IsBot && player2IsBot) {
            // Bot vs Bot: additive noise (±8 pts) + 3 random opening moves
            // Creates diverse but high-quality games for ML training
            botBrain  = new BotBrain(0.0, 3); // Bot B
            botBrain1 = new BotBrain(0.0, 3); // Bot A
        } else {
            if (player2IsBot) {
                botBrain = new BotBrain(); // Deterministic for Human vs Bot
            }
        }

        boardView = new BoardView();
        boardView.setController(this);
        boardView.setGameState(gameState);

        BorderPane root = createMainLayout();

        Scene scene = new Scene(root, 1200, 950);
        stage.setScene(scene);
        stage.setTitle("Quoridor - Strategy Board Game");
        stage.setResizable(true);
        stage.setMinWidth(1100);
        stage.setMinHeight(900);

        boardView.update();
        updateUI();

        initializeTimer();

        if (player1IsBot && player2IsBot) {
            // Bot vs Bot: hide timer and start bot 1's turn
            stopTimer();
            timerLabel.setText("🤖 Bot vs Bot");
            timerLabel.setTextFill(Color.web("#9b59b6"));
            // Use a short delay to let the UI finish rendering
            Timeline startDelay = new Timeline(new KeyFrame(Duration.millis(500), e -> {
                triggerBotTurnIfNeeded();
            }));
            startDelay.play();
        } else {
            startTimer();
            triggerBotTurnIfNeeded(); // In case player1 is not bot but player2 is (shouldn't happen with current setup)
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

        // --- Bot toggle checkbox ---
        CheckBox botCheckbox = new CheckBox("Player 2 is Bot (AI)");
        botCheckbox.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        botCheckbox.setTextFill(Color.web("#e67e22"));
        botCheckbox.setSelected(player2IsBot);
        botCheckbox.setStyle("-fx-cursor: hand;");

        // --- Bot vs Bot checkbox ---
        CheckBox botVsBotCheckbox = new CheckBox("\uD83E\uDD16 Bot vs Bot (watch AI play)");
        botVsBotCheckbox.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        botVsBotCheckbox.setTextFill(Color.web("#9b59b6"));
        botVsBotCheckbox.setSelected(player1IsBot && player2IsBot);
        botVsBotCheckbox.setStyle("-fx-cursor: hand;");

        // When bot is enabled, auto-fill player 2 name and disable the field
        botCheckbox.setOnAction(e -> {
            if (botCheckbox.isSelected()) {
                player2Field.setText("Bot (AI)");
                player2Field.setDisable(true);
            } else {
                player2Field.setText("Player 2");
                player2Field.setDisable(false);
                botVsBotCheckbox.setSelected(false);
            }
        });

        botVsBotCheckbox.setOnAction(e -> {
            if (botVsBotCheckbox.isSelected()) {
                // Enable both bots
                botCheckbox.setSelected(true);
                player1Field.setText("Bot A");
                player1Field.setDisable(true);
                player2Field.setText("Bot B");
                player2Field.setDisable(true);
            } else {
                player1Field.setText("Player 1");
                player1Field.setDisable(false);
                // Keep player 2 as bot if checkbox is still checked
                if (!botCheckbox.isSelected()) {
                    player2Field.setText("Player 2");
                    player2Field.setDisable(false);
                }
            }
        });

        // Apply initial state if bot was already selected (e.g. on restart)
        if (player1IsBot && player2IsBot) {
            botVsBotCheckbox.setSelected(true);
            player1Field.setText("Bot A");
            player1Field.setDisable(true);
            player2Field.setText("Bot B");
            player2Field.setDisable(true);
        } else if (player2IsBot) {
            player2Field.setText("Bot (AI)");
            player2Field.setDisable(true);
        }

        Button startButton = new Button("Start Game");
        startButton.setFont(Font.font("Arial", FontWeight.BOLD, 16));
        startButton.setPrefSize(200, 45);
        startButton.setStyle("-fx-background-color: #27ae60; -fx-text-fill: white; " +
                            "-fx-background-radius: 5; -fx-cursor: hand;");

        startButton.setOnMouseEntered(e ->
            startButton.setStyle("-fx-background-color: #2ecc71; -fx-text-fill: white; " +
                                "-fx-background-radius: 5; -fx-cursor: hand;")
        );
        startButton.setOnMouseExited(e ->
            startButton.setStyle("-fx-background-color: #27ae60; -fx-text-fill: white; " +
                                "-fx-background-radius: 5; -fx-cursor: hand;")
        );

        startButton.setOnAction(e -> {
            String name1 = player1Field.getText().trim();
            String name2 = player2Field.getText().trim();
            player1Name = name1.isEmpty() ? "Player 1" : name1;
            player2Name = name2.isEmpty() ? "Player 2" : name2;
            player2IsBot = botCheckbox.isSelected() || botVsBotCheckbox.isSelected();
            player1IsBot = botVsBotCheckbox.isSelected();
            dialog.close();
        });

        player2Field.setOnAction(e -> startButton.fire());
        player1Field.setOnAction(e -> player2Field.requestFocus());

        mainContainer.getChildren().addAll(titleLabel, subtitleLabel,
                                           player1Box, player2Box, botCheckbox, botVsBotCheckbox, startButton);

        Scene dialogScene = new Scene(mainContainer, 380, 450);
        dialog.setScene(dialogScene);
        dialog.centerOnScreen();
        dialog.showAndWait();
    }

    private BorderPane createMainLayout() {
        BorderPane root = new BorderPane();
        root.setStyle("-fx-background-color: linear-gradient(to bottom, #1a1a2e, #16213e);");
        root.setPadding(new Insets(20));

        VBox topPanel = createTopPanel();
        root.setTop(topPanel);

        StackPane centerPane = new StackPane(boardView);
        centerPane.setPadding(new Insets(10));
        root.setCenter(centerPane);

        player1Panel = createPlayerPanel(0);
        root.setLeft(player1Panel);

        player2Panel = createPlayerPanel(1);
        root.setRight(player2Panel);

        HBox bottomPanel = createBottomPanel();
        root.setBottom(bottomPanel);

        return root;
    }

    private VBox createTopPanel() {
        VBox panel = new VBox(10);
        panel.setAlignment(Pos.CENTER);
        panel.setPadding(new Insets(10, 10, 20, 10));

        Label title = new Label("QUORIDOR");
        title.setFont(Font.font("Arial", FontWeight.BOLD, 36));
        title.setTextFill(Color.web("#ecf0f1"));

        turnLabel = new Label("Player 1's Turn");
        turnLabel.setFont(Font.font("Arial", FontWeight.BOLD, 20));
        turnLabel.setTextFill(Color.web("#3498db"));

        timerLabel = new Label("Time: 15");
        timerLabel.setFont(Font.font("Arial", FontWeight.BOLD, 24));
        timerLabel.setTextFill(Color.web("#e74c3c"));
        timerLabel.setStyle("-fx-background-color: #2c3e50; -fx-padding: 10 20; -fx-background-radius: 5;");

        statusLabel = new Label("");
        statusLabel.setFont(Font.font("Arial", FontWeight.NORMAL, 14));
        statusLabel.setTextFill(Color.web("#f39c12"));

        panel.getChildren().addAll(title, turnLabel, timerLabel, statusLabel);
        return panel;
    }

    private VBox createPlayerPanel(int playerIndex) {
        VBox panel = new VBox(15);
        panel.setAlignment(Pos.TOP_CENTER);
        panel.setPadding(new Insets(20));
        panel.setPrefWidth(180);
        panel.setMinWidth(180);
        panel.setStyle("-fx-background-color: #2c3e50; -fx-background-radius: 10;");

        Player player = gameState.getPlayer(playerIndex);

        Circle icon = new Circle(30);
        icon.setFill(player.getColor());
        icon.setStroke(Color.web("#ecf0f1"));
        icon.setStrokeWidth(3);

        Label nameLabel = new Label(player.getName());
        nameLabel.setFont(Font.font("Arial", FontWeight.BOLD, 16));
        nameLabel.setTextFill(Color.web("#ecf0f1"));

        String goalText = playerIndex == 0 ? "Goal: Top ↑" : "Goal: Bottom ↓";
        Label goalLabel = new Label(goalText);
        goalLabel.setFont(Font.font("Arial", 12));
        goalLabel.setTextFill(Color.web("#95a5a6"));

        Label wallsTitle = new Label("Walls");
        wallsTitle.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        wallsTitle.setTextFill(Color.web("#ecf0f1"));

        Label wallsLabel = new Label("10");
        wallsLabel.setFont(Font.font("Arial", FontWeight.BOLD, 24));
        wallsLabel.setTextFill(Color.web("#3498db"));

        if (playerIndex == 0) {
            player1WallsLabel = wallsLabel;
        } else {
            player2WallsLabel = wallsLabel;
        }

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
            player1WallIndicators = new HBox[2];
            player1WallIndicators[0] = (HBox) wallIndicators.getChildren().get(0);
            player1WallIndicators[1] = (HBox) wallIndicators.getChildren().get(1);
        } else {
            player2WallIndicators = new HBox[2];
            player2WallIndicators[0] = (HBox) wallIndicators.getChildren().get(0);
            player2WallIndicators[1] = (HBox) wallIndicators.getChildren().get(1);
        }

        panel.getChildren().addAll(icon, nameLabel, goalLabel,
                new Region(), wallsTitle, wallsLabel, wallIndicators);

        Region spacer = new Region();
        VBox.setVgrow(spacer, Priority.ALWAYS);
        panel.getChildren().add(3, spacer);

        return panel;
    }

    private HBox createBottomPanel() {
        HBox panel = new HBox(20);
        panel.setAlignment(Pos.CENTER);
        panel.setPadding(new Insets(20));
        panel.setStyle("-fx-background-color: #2c3e50; -fx-background-radius: 10;");
        panel.setMinHeight(80);

        movePawnButton = new Button("Move Pawn");
        movePawnButton.setFont(Font.font("Arial", FontWeight.BOLD, 16));
        movePawnButton.setPrefSize(180, 50);
        styleButton(movePawnButton, "#27ae60");
        movePawnButton.setOnAction(e -> setMoveMode());

        placeWallButton = new Button("Place Wall");
        placeWallButton.setFont(Font.font("Arial", FontWeight.BOLD, 16));
        placeWallButton.setPrefSize(180, 50);
        styleButton(placeWallButton, "#3498db");
        placeWallButton.setOnAction(e -> setWallMode());

        Button restartButton = new Button("New Game");
        restartButton.setFont(Font.font("Arial", FontWeight.BOLD, 16));
        restartButton.setPrefSize(150, 50);
        styleButton(restartButton, "#e74c3c");
        restartButton.setOnAction(e -> restart());

        updateModeButtons();

        panel.getChildren().addAll(movePawnButton, placeWallButton, restartButton);

        // Bot vs Bot speed controls
        if (player1IsBot && player2IsBot) {
            // Hide move/wall buttons (not needed in bot mode)
            movePawnButton.setVisible(false);
            movePawnButton.setManaged(false);
            placeWallButton.setVisible(false);
            placeWallButton.setManaged(false);

            Region spacer = new Region();
            spacer.setPrefWidth(20);

            pauseButton = new Button("⏸ Pause");
            pauseButton.setFont(Font.font("Arial", FontWeight.BOLD, 14));
            pauseButton.setPrefSize(120, 50);
            styleButton(pauseButton, "#9b59b6");
            pauseButton.setOnAction(e -> toggleBotVsBotPause());

            Button slowerButton = new Button("🐢 Slower");
            slowerButton.setFont(Font.font("Arial", FontWeight.BOLD, 14));
            slowerButton.setPrefSize(110, 50);
            styleButton(slowerButton, "#e67e22");
            slowerButton.setOnAction(e -> changeBotSpeed(+300));

            Button fasterButton = new Button("⚡ Faster");
            fasterButton.setFont(Font.font("Arial", FontWeight.BOLD, 14));
            fasterButton.setPrefSize(110, 50);
            styleButton(fasterButton, "#2ecc71");
            fasterButton.setOnAction(e -> changeBotSpeed(-200));

            speedLabel = new Label("Speed: 0.6s");
            speedLabel.setFont(Font.font("Arial", FontWeight.BOLD, 14));
            speedLabel.setTextFill(Color.web("#ecf0f1"));

            panel.getChildren().addAll(spacer, pauseButton, slowerButton, fasterButton, speedLabel);
        }

        return panel;
    }

    private void styleButton(Button button, String color) {
        button.setStyle(
                "-fx-background-color: " + color + ";" +
                "-fx-text-fill: white;" +
                "-fx-background-radius: 5;" +
                "-fx-cursor: hand;"
        );

        button.setOnMouseEntered(e ->
                button.setStyle(
                        "-fx-background-color: derive(" + color + ", 20%);" +
                        "-fx-text-fill: white;" +
                        "-fx-background-radius: 5;" +
                        "-fx-cursor: hand;"
                )
        );

        button.setOnMouseExited(e ->
                button.setStyle(
                        "-fx-background-color: " + color + ";" +
                        "-fx-text-fill: white;" +
                        "-fx-background-radius: 5;" +
                        "-fx-cursor: hand;"
                )
        );
    }

    private void initializeTimer() {
        turnTimer = new Timeline(new KeyFrame(Duration.seconds(1), e -> {
            timeRemaining--;
            updateTimerDisplay();
            if (timeRemaining <= 0) {
                onTimeUp();
            }
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
        if (botThinking) return; // Don't timeout while bot is computing
        if (player1IsBot && player2IsBot) return; // No timeout in bot vs bot

        statusLabel.setText(gameState.getCurrentPlayer().getName() + " ran out of time!");

        gameState.nextTurn();
        wallMode = false;
        updateModeButtons();
        boardView.update();
        updateUI();
        startTimer();
        triggerBotTurnIfNeeded(); // If it's now the bot's turn, let it play
    }

    // Handle cell clicks for pawn movement
    public void onCellClicked(int row, int col) {
        if (gameState.isGameOver()) return;
        if (wallMode) return;
        if (botThinking) return; // Block input while bot computes

        Position target = new Position(row, col);
        Player current = gameState.getCurrentPlayer();

        List<Position> validMoves = MoveValidator.getValidMoves(gameState, current);

        if (validMoves.contains(target)) {
            current.setPosition(target);
            gameState.checkWinCondition();

            if (gameState.isGameOver()) {
                stopTimer();
                showWinner();
            } else {
                gameState.nextTurn();
                startTimer();
                triggerBotTurnIfNeeded(); // Let the bot play if it's their turn
            }

            boardView.update();
            updateUI();
            statusLabel.setText("");
        } else {
            statusLabel.setText("Invalid move! Click a green cell.");
        }
    }

    // Handle wall slot clicks (two-click placement)
    public void onWallSlotClicked(int row, int col, boolean horizontal) {
        if (gameState.isGameOver()) return;
        if (!wallMode) return;
        if (botThinking) return; // Block input while bot computes

        Wall.Orientation orientation = horizontal ?
                Wall.Orientation.HORIZONTAL : Wall.Orientation.VERTICAL;
        Wall wall = new Wall(row, col, orientation);

        // Second click - confirm placement
        if (wallConfirmPending && selectedWall != null && selectedWall.equals(wall)) {
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
                triggerBotTurnIfNeeded(); // Let the bot play if it's their turn
            } else {
                String reason = WallValidator.getInvalidReason(gameState, wall);
                statusLabel.setText("Can't place wall: " + reason);
                selectedWall = null;
                wallConfirmPending = false;
                boardView.clearSelectedWall();
            }
        } else {
            // First click - select wall position
            if (WallValidator.isValidWallPlacement(gameState, wall)) {
                selectedWall = wall;
                wallConfirmPending = true;
                boardView.showSelectedWall(wall);
                statusLabel.setText("Click again to confirm wall placement");
            } else {
                String reason = WallValidator.getInvalidReason(gameState, wall);
                statusLabel.setText("Can't place wall: " + reason);
                selectedWall = null;
                wallConfirmPending = false;
                boardView.clearSelectedWall();
            }
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
        button.setStyle(
                "-fx-background-color: " + color + ";" +
                "-fx-text-fill: white;" +
                "-fx-background-radius: 5;" +
                "-fx-border-color: white;" +
                "-fx-border-width: 3;" +
                "-fx-border-radius: 5;" +
                "-fx-cursor: hand;"
        );
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
                if (node instanceof Rectangle) {
                    Rectangle block = (Rectangle) node;
                    if (wallIndex < wallsRemaining) {
                        block.setFill(availableColor);
                    } else {
                        block.setFill(Color.web("#555555"));
                    }
                    wallIndex++;
                }
            }
        }
    }

    private void showWinner() {
        Player winner = gameState.getWinner();

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

        Label messageLabel = new Label("Congratulations! " + winner.getName() +
                                       " has reached their goal row!");
        messageLabel.setFont(Font.font("Arial", 14));
        messageLabel.setTextFill(Color.web("#95a5a6"));

        HBox buttonBox = new HBox(20);
        buttonBox.setAlignment(Pos.CENTER);

        Button nextGameButton = new Button("Next Game");
        nextGameButton.setFont(Font.font("Arial", FontWeight.BOLD, 16));
        nextGameButton.setPrefSize(150, 45);
        nextGameButton.setStyle("-fx-background-color: #27ae60; -fx-text-fill: white; " +
                               "-fx-background-radius: 5; -fx-cursor: hand;");
        nextGameButton.setOnMouseEntered(e ->
            nextGameButton.setStyle("-fx-background-color: #2ecc71; -fx-text-fill: white; " +
                                   "-fx-background-radius: 5; -fx-cursor: hand;"));
        nextGameButton.setOnMouseExited(e ->
            nextGameButton.setStyle("-fx-background-color: #27ae60; -fx-text-fill: white; " +
                                   "-fx-background-radius: 5; -fx-cursor: hand;"));
        nextGameButton.setOnAction(e -> {
            dialog.close();
            // Defer restart to avoid nested showAndWait() (dialog inside dialog)
            Platform.runLater(() -> restart());
        });

        Button quitButton = new Button("Quit");
        quitButton.setFont(Font.font("Arial", FontWeight.BOLD, 16));
        quitButton.setPrefSize(150, 45);
        quitButton.setStyle("-fx-background-color: #e74c3c; -fx-text-fill: white; " +
                           "-fx-background-radius: 5; -fx-cursor: hand;");
        quitButton.setOnMouseEntered(e ->
            quitButton.setStyle("-fx-background-color: #c0392b; -fx-text-fill: white; " +
                               "-fx-background-radius: 5; -fx-cursor: hand;"));
        quitButton.setOnMouseExited(e ->
            quitButton.setStyle("-fx-background-color: #e74c3c; -fx-text-fill: white; " +
                               "-fx-background-radius: 5; -fx-cursor: hand;"));
        quitButton.setOnAction(e -> {
            dialog.close();
            primaryStage.close();
        });

        buttonBox.getChildren().addAll(nextGameButton, quitButton);
        container.getChildren().addAll(titleLabel, winnerLabel, messageLabel, buttonBox);

        Scene dialogScene = new Scene(container, 400, 250);
        dialog.setScene(dialogScene);
        dialog.centerOnScreen();
        dialog.showAndWait();
    }

    private void restart() {
        try {
            // Invalidate any in-flight bot computations from the previous game
            gameGeneration++;
            if (pendingBotDelay != null) {
                pendingBotDelay.stop();
                pendingBotDelay = null;
            }
            stopTimer();

            showPlayerNameDialog();

            gameState.getPlayer(0).setName(player1Name);
            gameState.getPlayer(1).setName(player2Name);

            // Re-initialize or remove bots based on dialog choice
            if (player1IsBot && player2IsBot) {
                // Bot vs Bot: random openings for diverse ML data
                botBrain  = new BotBrain(0.0, 3);
                botBrain1 = new BotBrain(0.0, 3);
            } else if (player2IsBot) {
                botBrain = new BotBrain(); // Deterministic for Human vs Bot
                botBrain1 = null;
            } else {
                botBrain = null;
                botBrain1 = null;
            }
            botThinking = false;
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
                timerLabel.setText("\uD83E\uDD16 Bot vs Bot");
                timerLabel.setTextFill(Color.web("#9b59b6"));
                Timeline startDelay = new Timeline(new KeyFrame(Duration.millis(500), e -> {
                    triggerBotTurnIfNeeded();
                }));
                startDelay.play();
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
        player1Panel = createPlayerPanel(0);
        root.setLeft(player1Panel);
        player2Panel = createPlayerPanel(1);
        root.setRight(player2Panel);
    }

    private void rebuildBottomPanel() {
        BorderPane root = (BorderPane) primaryStage.getScene().getRoot();
        HBox bottomPanel = createBottomPanel();
        root.setBottom(bottomPanel);
    }

    // ==================== BOT VS BOT CONTROLS ====================

    private void toggleBotVsBotPause() {
        botVsBotPaused = !botVsBotPaused;
        if (pauseButton != null) {
            if (botVsBotPaused) {
                pauseButton.setText("▶ Play");
                statusLabel.setText("⏸ Paused — click Play to continue");
                statusLabel.setTextFill(Color.web("#f39c12"));
            } else {
                pauseButton.setText("⏸ Pause");
                statusLabel.setText("");
                // Resume by triggering next bot turn
                triggerBotTurnIfNeeded();
            }
        }
    }

    private void changeBotSpeed(int deltaMs) {
        botVsBotDelayMs = Math.max(100, Math.min(2000, botVsBotDelayMs + deltaMs));
        if (speedLabel != null) {
            speedLabel.setText("Speed: " + String.format("%.1f", botVsBotDelayMs / 1000.0) + "s");
        }
    }

    // ==================== BOT INTEGRATION ====================

    /**
     * Checks if the current turn belongs to the bot and triggers its move.
     *
     * The bot runs in a BACKGROUND THREAD to avoid freezing the UI while
     * it computes. A small delay (0.8s) is added before execution so the
     * human can see the board state before the bot plays.
     *
     * Flow:
     *   1. Check if current player is the bot (player index 1)
     *   2. Show "thinking" status
     *   3. Run BotBrain.computeBestAction() in a background thread
     *   4. When done, use Platform.runLater() to apply the action on the JavaFX thread
     */
    private void triggerBotTurnIfNeeded() {
        if (gameState.isGameOver()) return;
        if (botVsBotPaused) return;

        int currentIdx = gameState.getCurrentPlayerIndex();

        // Determine which brain to use
        BotBrain activeBrain = null;
        if (currentIdx == 0 && player1IsBot && botBrain1 != null) {
            activeBrain = botBrain1;
        } else if (currentIdx == 1 && player2IsBot && botBrain != null) {
            activeBrain = botBrain;
        }

        if (activeBrain == null) return;

        botThinking = true;
        stopTimer(); // Don't tick timer while bot thinks

        String botName = gameState.getCurrentPlayer().getName();
        boolean isBotVsBot = player1IsBot && player2IsBot;

        if (isBotVsBot) {
            statusLabel.setText(botName + " is thinking...");
        } else {
            statusLabel.setText("Bot is thinking...");
        }
        statusLabel.setTextFill(Color.web("#e67e22"));

        // Delay: shorter for bot vs bot, longer for human vs bot
        int delayMs = isBotVsBot ? botVsBotDelayMs : 800;
        final BotBrain brainToUse = activeBrain;
        final int expectedGeneration = gameGeneration; // Capture current generation

        // Cancel any pending bot delay from previous turn
        if (pendingBotDelay != null) {
            pendingBotDelay.stop();
        }

        pendingBotDelay = new Timeline(new KeyFrame(Duration.millis(delayMs), e -> {

            // Check if game was restarted while waiting
            if (expectedGeneration != gameGeneration) return;

            // Run bot computation in a background thread to keep UI responsive
            Thread botThread = new Thread(() -> {
                BotBrain.BotAction action = brainToUse.computeBestAction(gameState);

                // Apply the action back on the JavaFX Application Thread
                Platform.runLater(() -> {
                    // Check if game was restarted while computing
                    if (expectedGeneration != gameGeneration) return;
                    executeBotAction(action);
                });
            });
            botThread.setDaemon(true); // Thread dies when app closes
            botThread.start();
        }));
        pendingBotDelay.play();
    }

    /**
     * Executes the bot's chosen action on the game state.
     *
     * This runs on the JavaFX thread (via Platform.runLater) so it's safe
     * to modify UI elements and game state here.
     */
    private void executeBotAction(BotBrain.BotAction action) {
        try {
        String botName = gameState.getCurrentPlayer().getName();

        if (action == null) {
            statusLabel.setText(botName + " couldn't decide — turn skipped");
            gameState.nextTurn();
            botThinking = false;
            boardView.update();
            updateUI();
            startTimer();
            triggerBotTurnIfNeeded(); // Next bot's turn in Bot vs Bot
            return;
        }

        if (action.type == BotBrain.BotAction.Type.MOVE) {
            Player bot = gameState.getCurrentPlayer();
            bot.setPosition(action.moveTarget);
            statusLabel.setText(botName + " moved to " + action.moveTarget);
            statusLabel.setTextFill(Color.web("#2ecc71"));

            gameState.checkWinCondition();
            if (gameState.isGameOver()) {
                botThinking = false;
                stopTimer();
                boardView.update();
                updateUI();
                showWinner();
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
                statusLabel.setText(botName + " wall was invalid — turn skipped");
            }
        }

        // Advance to next turn
        gameState.nextTurn();
        botThinking = false;
        wallMode = false;
        updateModeButtons();
        boardView.update();
        updateUI();
        startTimer();
        triggerBotTurnIfNeeded(); // Chain to next bot in Bot vs Bot
        } catch (Exception ex) {
            System.err.println("[BOT ACTION ERROR] " + ex.getClass().getName() + ": " + ex.getMessage());
            ex.printStackTrace();
            botThinking = false;
        }
    }

}
