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
import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Generates training data by running BotBrain vs RandomBot games.
 *
 * Usage: java ml.DataGenerator [totalGames] [outputDir]
 * Default: 100000 games, output to ./training_data/
 *
 * Label: sigmoid((oppDist - myDist) / 8.0) from BotBrain's perspective.
 * Endgame anchors: winning in <=3 moves -> 0.95, losing in <=3 -> 0.05.
 */
public class DataGenerator {

    private static final int MAX_TURNS = 200; // prevent infinite games
    private static final int REPORT_INTERVAL = 5000;

    public static void main(String[] args) throws Exception {
        int totalGames = (args.length > 0) ? Integer.parseInt(args[0]) : 100000;
        String outputDir = (args.length > 1) ? args[1] : "training_data";

        new File(outputDir).mkdirs();
        WallEvaluator.silent = true;

        System.out.println("=== Quoridor Training Data Generator ===");
        System.out.println("Total games: " + totalGames);
        System.out.println("Output dir: " + outputDir);

        // Distribution: 40% Uniform, 20% Forward, 20% Waller, 20% SemiSmart
        int uniformGames = (int) (totalGames * 0.4);
        int forwardGames = (int) (totalGames * 0.2);
        int wallerGames = (int) (totalGames * 0.2);
        int semiSmartGames = totalGames - uniformGames - forwardGames - wallerGames;

        long startTime = System.currentTimeMillis();

        int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
        System.out.println("Using " + threads + " threads");

        // Generate each opponent type
        generateBatch("uniform", outputDir, uniformGames, threads, 0);
        generateBatch("forward", outputDir, forwardGames, threads, 1);
        generateBatch("waller", outputDir, wallerGames, threads, 2);
        generateBatch("semismart", outputDir, semiSmartGames, threads, 3);

        long elapsed = (System.currentTimeMillis() - startTime) / 1000;
        System.out.println("\nDone! Total time: " + elapsed + "s");
    }

    private static void generateBatch(String name, String outputDir, int numGames,
                                       int threads, int opponentType) throws Exception {
        System.out.println("\n--- Generating " + numGames + " games vs " + name + " ---");

        AtomicInteger gamesCompleted = new AtomicInteger(0);
        AtomicInteger totalSamples = new AtomicInteger(0);
        AtomicInteger botWins = new AtomicInteger(0);

        // Split games across threads, each thread writes its own file
        int gamesPerThread = numGames / threads;
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        CountDownLatch latch = new CountDownLatch(threads);

        for (int t = 0; t < threads; t++) {
            final int threadId = t;
            final int threadGames = (t == threads - 1) ? (numGames - gamesPerThread * (threads - 1)) : gamesPerThread;

            executor.submit(() -> {
                try {
                    String filePath = outputDir + "/" + name + "_" + threadId + ".qdat";
                    TrainingDataWriter writer = new TrainingDataWriter(filePath);

                    for (int g = 0; g < threadGames; g++) {
                        long seed = System.nanoTime() ^ (threadId * 1_000_000L + g);

                        // BotBrain alternates sides
                        boolean botIsP1 = (g % 2 == 0);
                        int samples = playGame(writer, seed, opponentType, botIsP1);
                        totalSamples.addAndGet(samples);

                        // Track if BotBrain won (for sanity check)
                        int completed = gamesCompleted.incrementAndGet();
                        if (completed % REPORT_INTERVAL == 0) {
                            System.out.printf("  [%s] %d/%d games, %d samples so far%n",
                                    name, completed, numGames, totalSamples.get());
                        }
                    }

                    writer.close();
                    System.out.printf("  Thread %d: wrote %d records to %s%n",
                            threadId, writer.getRecordCount(), filePath);
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    latch.countDown();
                }
            });
        }

        latch.await();
        executor.shutdown();

        System.out.printf("  %s complete: %d games, %d samples%n",
                name, gamesCompleted.get(), totalSamples.get());
    }

    /**
     * Plays one game and writes all board states with labels.
     * Returns the number of samples written.
     */
    private static int playGame(TrainingDataWriter writer, long seed,
                                 int opponentType, boolean botIsP1) throws IOException {
        GameState state = new GameState();
        BotBrain bot = new BotBrain();
        bot.setSilent(true);

        RandomBot opponent = createOpponent(opponentType, seed);
        int botPlayerIndex = botIsP1 ? 0 : 1;

        int turnCount = 0;
        int samplesWritten = 0;

        while (!state.isGameOver() && turnCount < MAX_TURNS) {
            boolean isBotTurn = (state.getCurrentPlayerIndex() == botPlayerIndex);

            // Record board state from current player's perspective BEFORE the move
            float[][][] boardBefore = BoardEncoder.encode(state);

            // Compute label: distance advantage from current player's perspective
            Player current = state.getCurrentPlayer();
            Player other = state.getOtherPlayer();
            int myDist = PathFinder.shortestPath(state, current);
            int oppDist = PathFinder.shortestPath(state, other);

            float label;
            if (myDist <= 0) {
                // Current player already at goal (shouldn't happen mid-game)
                label = 0.95f;
            } else if (oppDist <= 0) {
                label = 0.05f;
            } else if (myDist <= 3 && myDist < oppDist) {
                // Close to winning with advantage
                label = 0.90f + (3 - myDist) * 0.017f; // 0.90, 0.917, 0.934, 0.95
            } else if (oppDist <= 3 && oppDist < myDist) {
                // Opponent close to winning
                label = 0.10f - (3 - oppDist) * 0.017f; // 0.10, 0.083, 0.066, 0.05
            } else {
                // General case: sigmoid of distance advantage
                float advantage = (oppDist - myDist) / 8.0f;
                label = sigmoid(advantage);
            }

            // Clamp to [0.02, 0.98]
            label = Math.max(0.02f, Math.min(0.98f, label));

            writer.write(boardBefore, label);
            samplesWritten++;

            // Execute the move
            if (isBotTurn) {
                BotBrain.BotAction action = bot.computeBestAction(state);
                if (action == null) break;
                applyAction(state, action);
            } else {
                Object action = opponent.chooseAction(state);
                if (action instanceof Position) {
                    state.getCurrentPlayer().setPosition((Position) action);
                } else if (action instanceof Wall) {
                    state.addWall((Wall) action);
                }
            }

            state.checkWinCondition();
            if (!state.isGameOver()) {
                state.nextTurn();
            }
            turnCount++;
        }

        return samplesWritten;
    }

    private static void applyAction(GameState state, BotBrain.BotAction action) {
        if (action.type == BotBrain.BotAction.Type.MOVE) {
            state.getCurrentPlayer().setPosition(action.moveTarget);
        } else {
            state.addWall(action.wallToPlace);
        }
    }

    private static RandomBot createOpponent(int type, long seed) {
        switch (type) {
            case 0: return new RandomBot.UniformRandom(seed);
            case 1: return new RandomBot.ForwardRandom(seed);
            case 2: return new RandomBot.RandomWaller(seed);
            case 3: return new RandomBot.SemiSmart(seed);
            default: return new RandomBot.UniformRandom(seed);
        }
    }

    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }
}
