package ml;

import bot.BotBrain;
import bot.WallEvaluator;
import logic.MoveValidator;
import logic.PathFinder;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.io.File;
import java.util.List;

/**
 * Sanity tests for the ML pipeline components.
 * Run with: java ml.SanityTest
 */
public class SanityTest {

    static int passed = 0, failed = 0;

    public static void main(String[] args) throws Exception {
        WallEvaluator.silent = true;

        System.out.println("=== ML Pipeline Sanity Tests ===\n");

        testRandomBots();
        testBoardEncoder();
        testTrainingDataIO();
        testDataGeneratorSmall();

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) System.exit(1);
    }

    // ========== TEST 1: RandomBots ==========

    static void testRandomBots() {
        System.out.println("--- Test 1: RandomBots ---");

        String[] names = {"UniformRandom", "ForwardRandom", "RandomWaller", "SemiSmart"};
        RandomBot[] bots = {
            new RandomBot.UniformRandom(42),
            new RandomBot.ForwardRandom(42),
            new RandomBot.RandomWaller(42),
            new RandomBot.SemiSmart(42)
        };

        for (int b = 0; b < bots.length; b++) {
            int gamesPlayed = 0;
            int botMoves = 0;
            int crashes = 0;

            for (int g = 0; g < 50; g++) {
                try {
                    GameState state = new GameState();
                    BotBrain brain = new BotBrain();
                    brain.setSilent(true);
                    int turns = 0;

                    while (!state.isGameOver() && turns < 200) {
                        if (state.getCurrentPlayerIndex() == 0) {
                            // BotBrain plays P1
                            BotBrain.BotAction action = brain.computeBestAction(state);
                            if (action == null) break;
                            if (action.type == BotBrain.BotAction.Type.MOVE) {
                                state.getCurrentPlayer().setPosition(action.moveTarget);
                            } else {
                                state.addWall(action.wallToPlace);
                            }
                        } else {
                            // RandomBot plays P2
                            Object action = bots[b].chooseAction(state);
                            if (action instanceof Position) {
                                Position move = (Position) action;
                                // Verify it's a valid move
                                List<Position> valid = MoveValidator.getValidMoves(state, state.getCurrentPlayer());
                                if (!valid.contains(move)) {
                                    System.out.println("  INVALID MOVE by " + names[b] + ": " + move);
                                    crashes++;
                                    break;
                                }
                                state.getCurrentPlayer().setPosition(move);
                            } else if (action instanceof Wall) {
                                Wall wall = (Wall) action;
                                if (!WallValidator.isValidWallPlacement(state, wall)) {
                                    System.out.println("  INVALID WALL by " + names[b] + ": " + wall);
                                    crashes++;
                                    break;
                                }
                                state.addWall(wall);
                            }
                            botMoves++;
                        }
                        state.checkWinCondition();
                        if (!state.isGameOver()) state.nextTurn();
                        turns++;
                    }
                    gamesPlayed++;
                } catch (Exception e) {
                    crashes++;
                    System.out.println("  CRASH in " + names[b] + " game " + g + ": " + e.getMessage());
                }
            }

            if (crashes == 0) {
                check(names[b] + ": 50 games, 0 crashes, " + botMoves + " bot moves", true);
            } else {
                check(names[b] + ": " + crashes + " crashes!", false);
            }
        }
    }

    // ========== TEST 2: BoardEncoder ==========

    static void testBoardEncoder() {
        System.out.println("\n--- Test 2: BoardEncoder ---");

        // Test initial state encoding
        GameState state = new GameState();
        float[][][] encoded = BoardEncoder.encode(state);

        check("Encoding shape: " + encoded.length + "x" + encoded[0].length + "x" + encoded[0][0].length,
              encoded.length == 8 && encoded[0].length == 9 && encoded[0][0].length == 9);

        // P1 starts at (8,4), is current player -> ch0
        check("Ch0 P1 at (8,4): " + encoded[0][8][4], encoded[0][8][4] == 1.0f);
        check("Ch0 empty at (0,0): " + encoded[0][0][0], encoded[0][0][0] == 0.0f);

        // P2 at (0,4) -> ch1
        check("Ch1 P2 at (0,4): " + encoded[1][0][4], encoded[1][0][4] == 1.0f);

        // P1 goal is row 0 -> ch4
        check("Ch4 goal row 0: " + encoded[4][0][0], encoded[4][0][0] == 1.0f);
        check("Ch4 non-goal row 5: " + encoded[4][5][0], encoded[4][5][0] == 0.0f);

        // Walls remaining = 10/10 = 1.0 -> ch6, ch7
        check("Ch6 walls remaining: " + encoded[6][0][0], encoded[6][0][0] == 1.0f);
        check("Ch7 opp walls: " + encoded[7][0][0], encoded[7][0][0] == 1.0f);

        // No walls -> ch2, ch3 should be all zeros
        float wallSum = 0;
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++)
                wallSum += encoded[2][r][c] + encoded[3][r][c];
        check("No walls (ch2+ch3 sum=0): " + wallSum, wallSum == 0.0f);

        // Test encodeAfterMove — perspective swap
        Position moveForward = new Position(7, 4); // P1 moves from (8,4) to (7,4)
        float[][][] afterMove = BoardEncoder.encodeAfterMove(state, moveForward);

        // After P1's move, it's P2's turn. P2 is now "current" in ch0
        check("AfterMove ch0 = P2 at (0,4): " + afterMove[0][0][4], afterMove[0][0][4] == 1.0f);
        check("AfterMove ch1 = P1 at (7,4): " + afterMove[1][7][4], afterMove[1][7][4] == 1.0f);
        // P2's goal is row 8 -> ch4 (since P2 is now current)
        check("AfterMove ch4 = P2 goal row 8: " + afterMove[4][8][0], afterMove[4][8][0] == 1.0f);

        // Test encodeAfterWall
        Wall testWall = new Wall(3, 3, Wall.Orientation.HORIZONTAL);
        float[][][] afterWall = BoardEncoder.encodeAfterWall(state, testWall);

        // Wall should appear in ch2 (horizontal)
        check("AfterWall ch2 at (3,3): " + afterWall[2][3][3], afterWall[2][3][3] == 1.0f);
        check("AfterWall ch2 at (3,4): " + afterWall[2][3][4], afterWall[2][3][4] == 1.0f);

        // After wall placement, our walls go from 10 to 9. In afterWall encoding,
        // WE are now "opponent" (ch7), so ch7 = 9/10 = 0.9
        check("AfterWall ch7 (our walls, now opp) = 0.9: " + afterWall[7][0][0],
              Math.abs(afterWall[7][0][0] - 0.9f) < 0.001f);
    }

    // ========== TEST 3: TrainingDataWriter I/O ==========

    static void testTrainingDataIO() throws Exception {
        System.out.println("\n--- Test 3: TrainingData I/O ---");

        String testFile = "test_data_tmp.qdat";

        // Write some data
        TrainingDataWriter writer = new TrainingDataWriter(testFile);
        GameState state = new GameState();

        float[][][] board1 = BoardEncoder.encode(state);
        writer.write(board1, 0.5f);

        state.getCurrentPlayer().setPosition(new Position(7, 4));
        float[][][] board2 = BoardEncoder.encode(state);
        writer.write(board2, 0.73f);

        writer.close();
        check("Writer wrote 2 records", writer.getRecordCount() == 2);

        // Read it back
        TrainingDataWriter.TrainingData data = TrainingDataWriter.read(testFile);
        check("Reader got 2 samples", data.size() == 2);
        check("Label 1 = 0.5: " + data.labels[0], Math.abs(data.labels[0] - 0.5f) < 0.001f);
        check("Label 2 = 0.73: " + data.labels[1], Math.abs(data.labels[1] - 0.73f) < 0.001f);

        // Verify board data roundtrips correctly
        check("Board1 ch0 (8,4) = 1.0: " + data.boards[0][0][8][4],
              data.boards[0][0][8][4] == 1.0f);
        check("Board2 ch0 (7,4) = 1.0: " + data.boards[1][0][7][4],
              data.boards[1][0][7][4] == 1.0f);

        // Test split
        TrainingDataWriter.TrainingData[] split = data.split(0.5, new java.util.Random(42));
        check("Split train size: " + split[0].size(), split[0].size() == 1);
        check("Split val size: " + split[1].size(), split[1].size() == 1);

        // Cleanup
        new File(testFile).delete();
    }

    // ========== TEST 4: DataGenerator (small batch) ==========

    static void testDataGeneratorSmall() throws Exception {
        System.out.println("\n--- Test 4: DataGenerator (10 games) ---");

        String testDir = "test_training_tmp";
        new File(testDir).mkdirs();

        // Run a small generation directly
        long startTime = System.currentTimeMillis();

        // Test each bot type with 2 games each
        String[] types = {"uniform", "forward", "waller", "semismart"};
        int totalSamples = 0;

        for (int t = 0; t < 4; t++) {
            String filePath = testDir + "/" + types[t] + "_test.qdat";
            TrainingDataWriter writer = new TrainingDataWriter(filePath);

            for (int g = 0; g < 3; g++) {
                GameState state = new GameState();
                BotBrain bot = new BotBrain();
                bot.setSilent(true);
                RandomBot opponent = createOpponent(t, System.nanoTime());
                boolean botIsP1 = (g % 2 == 0);
                int botIdx = botIsP1 ? 0 : 1;

                int turns = 0;
                while (!state.isGameOver() && turns < 200) {
                    // Encode and label
                    float[][][] encoded = BoardEncoder.encode(state);
                    Player current = state.getCurrentPlayer();
                    Player other = state.getOtherPlayer();
                    int myDist = PathFinder.shortestPath(state, current);
                    int oppDist = PathFinder.shortestPath(state, other);
                    float advantage = (oppDist - myDist) / 8.0f;
                    float label = (float) (1.0 / (1.0 + Math.exp(-advantage)));
                    label = Math.max(0.02f, Math.min(0.98f, label));

                    writer.write(encoded, label);
                    totalSamples++;

                    // Execute move
                    if (state.getCurrentPlayerIndex() == botIdx) {
                        BotBrain.BotAction action = bot.computeBestAction(state);
                        if (action == null) break;
                        if (action.type == BotBrain.BotAction.Type.MOVE)
                            state.getCurrentPlayer().setPosition(action.moveTarget);
                        else
                            state.addWall(action.wallToPlace);
                    } else {
                        Object action = opponent.chooseAction(state);
                        if (action instanceof Position)
                            state.getCurrentPlayer().setPosition((Position) action);
                        else
                            state.addWall((Wall) action);
                    }

                    state.checkWinCondition();
                    if (!state.isGameOver()) state.nextTurn();
                    turns++;
                }
            }

            writer.close();
            check(types[t] + " wrote " + writer.getRecordCount() + " samples", writer.getRecordCount() > 0);
        }

        long elapsed = System.currentTimeMillis() - startTime;
        System.out.println("  12 games generated " + totalSamples + " samples in " + elapsed + "ms");

        // Read back and verify labels
        File[] files = new File(testDir).listFiles((d, n) -> n.endsWith(".qdat"));
        TrainingDataWriter.TrainingData allData = TrainingDataWriter.readAll(
            java.util.Arrays.stream(files).map(File::getAbsolutePath).toArray(String[]::new)
        );

        check("Total samples readable: " + allData.size(), allData.size() == totalSamples);

        // Check label distribution
        float sumLabels = 0;
        float minLabel = 1, maxLabel = 0;
        for (float l : allData.labels) {
            sumLabels += l;
            minLabel = Math.min(minLabel, l);
            maxLabel = Math.max(maxLabel, l);
        }
        float meanLabel = sumLabels / allData.size();
        System.out.println("  Labels — mean: " + String.format("%.3f", meanLabel)
                + ", min: " + String.format("%.3f", minLabel)
                + ", max: " + String.format("%.3f", maxLabel));

        check("Mean label > 0.3 (BotBrain should mostly win)", meanLabel > 0.3);
        check("Mean label < 0.9 (not all extreme)", meanLabel < 0.9);
        check("Min label >= 0.02", minLabel >= 0.019f);
        check("Max label <= 0.98", maxLabel <= 0.981f);

        // Cleanup
        for (File f : new File(testDir).listFiles()) f.delete();
        new File(testDir).delete();
    }

    static RandomBot createOpponent(int type, long seed) {
        switch (type) {
            case 0: return new RandomBot.UniformRandom(seed);
            case 1: return new RandomBot.ForwardRandom(seed);
            case 2: return new RandomBot.RandomWaller(seed);
            case 3: return new RandomBot.SemiSmart(seed);
            default: return new RandomBot.UniformRandom(seed);
        }
    }

    static void check(String desc, boolean passed_) {
        if (passed_) {
            System.out.println("  PASS: " + desc);
            passed++;
        } else {
            System.out.println("  FAIL: " + desc);
            failed++;
        }
    }
}
