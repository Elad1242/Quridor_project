package DataBaseBuild;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Main entry point for generating training data via batch Bot vs Bot simulations.
 *
 * Runs games in parallel using a thread pool, collects results, and exports
 * them to MongoDB Atlas in batches.
 *
 * Usage:
 *   java TrainingDataGenerator [numGames] [mongoUri]
 *
 * Default: 10000 games, prompts for MongoDB URI if not provided.
 */
public class TrainingDataGenerator {

    private static final int DEFAULT_NUM_GAMES = 10000;
    private static final int THREAD_COUNT = 6;
    private static final int BATCH_SIZE = 50; // export every 50 games
    private static final String DEFAULT_DB_NAME = "quoridor";

    public static void main(String[] args) {
        int numGames = DEFAULT_NUM_GAMES;
        String mongoUri = null;

        // Parse command line arguments
        if (args.length >= 1) {
            try {
                numGames = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.out.println("Invalid number of games, using default: " + DEFAULT_NUM_GAMES);
            }
        }
        if (args.length >= 2) {
            mongoUri = args[1];
        }

        if (mongoUri == null || mongoUri.isEmpty()) {
            System.out.println("Usage: TrainingDataGenerator [numGames] [mongoUri]");
            System.out.println("Example: TrainingDataGenerator 10000 mongodb+srv://user:pass@cluster.mongodb.net/");
            System.out.println();
            System.out.println("No MongoDB URI provided. Running in dry-run mode (simulations only, no export).");
            runDryMode(numGames);
        } else {
            runWithMongo(numGames, mongoUri);
        }
    }

    /**
     * Runs simulations and exports to MongoDB.
     */
    private static void runWithMongo(int numGames, String mongoUri) {
        System.out.println("=== Quoridor Training Data Generator ===");
        System.out.println("Games to simulate: " + numGames);
        System.out.println("Threads: " + THREAD_COUNT);
        System.out.println("Connecting to MongoDB...");

        MongoExporter exporter = new MongoExporter(mongoUri, DEFAULT_DB_NAME);
        long existingGames = exporter.getGameCount();
        int startId = (int) existingGames + 1;
        System.out.println("Existing games in DB: " + existingGames);
        System.out.println("Starting from gameId: " + startId);
        System.out.println();

        ExecutorService executor = Executors.newFixedThreadPool(THREAD_COUNT);
        AtomicInteger completed = new AtomicInteger(0);
        AtomicInteger wins0 = new AtomicInteger(0);
        AtomicInteger wins1 = new AtomicInteger(0);
        AtomicInteger draws = new AtomicInteger(0);

        long startTime = System.currentTimeMillis();

        // Submit all game tasks
        List<Future<GameSimulator.GameRecord>> futures = new ArrayList<>();
        for (int i = 0; i < numGames; i++) {
            futures.add(executor.submit(GameSimulator::simulate));
        }

        // Collect results in batches and export
        List<GameSimulator.GameRecord> batch = new ArrayList<>();
        int batchStartId = startId;

        for (int i = 0; i < futures.size(); i++) {
            try {
                GameSimulator.GameRecord record = futures.get(i).get();
                batch.add(record);

                // Track stats
                int done = completed.incrementAndGet();
                if (record.winnerIndex == 0) wins0.incrementAndGet();
                else if (record.winnerIndex == 1) wins1.incrementAndGet();
                else draws.incrementAndGet();

                // Export batch when full
                if (batch.size() >= BATCH_SIZE || done == numGames) {
                    exporter.exportBatch(batch, batchStartId);
                    batchStartId += batch.size();
                    batch.clear();
                }

                // Progress report every 100 games
                if (done % 100 == 0 || done == numGames) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    double gamesPerSec = done * 1000.0 / elapsed;
                    int remaining = numGames - done;
                    long etaMs = (long) (remaining / gamesPerSec * 1000);
                    System.out.printf("Game %d/%d (%.1f games/sec, ETA: %s) | P1: %d, P2: %d, Draw: %d%n",
                            done, numGames, gamesPerSec, formatTime(etaMs),
                            wins0.get(), wins1.get(), draws.get());
                }
            } catch (Exception e) {
                System.err.println("Error in game " + (i + 1) + ": " + e.getMessage());
            }
        }

        executor.shutdown();
        exporter.close();

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.println();
        System.out.println("=== COMPLETE ===");
        System.out.println("Total games: " + completed.get());
        System.out.println("Total time: " + formatTime(totalTime));
        System.out.println("Player 1 wins: " + wins0.get());
        System.out.println("Player 2 wins: " + wins1.get());
        System.out.println("Draws/Timeouts: " + draws.get());
    }

    /**
     * Runs simulations without MongoDB (for testing or when no URI is provided).
     * Prints results to console.
     */
    private static void runDryMode(int numGames) {
        System.out.println("=== Quoridor Training Data Generator (DRY RUN) ===");
        System.out.println("Games to simulate: " + numGames);
        System.out.println("Threads: " + THREAD_COUNT);
        System.out.println();

        ExecutorService executor = Executors.newFixedThreadPool(THREAD_COUNT);
        AtomicInteger completed = new AtomicInteger(0);
        AtomicInteger wins0 = new AtomicInteger(0);
        AtomicInteger wins1 = new AtomicInteger(0);
        AtomicInteger draws = new AtomicInteger(0);
        AtomicInteger totalTurns = new AtomicInteger(0);

        long startTime = System.currentTimeMillis();

        List<Future<GameSimulator.GameRecord>> futures = new ArrayList<>();
        for (int i = 0; i < numGames; i++) {
            futures.add(executor.submit(GameSimulator::simulate));
        }

        for (int i = 0; i < futures.size(); i++) {
            try {
                GameSimulator.GameRecord record = futures.get(i).get();
                int done = completed.incrementAndGet();
                totalTurns.addAndGet(record.totalTurns);

                if (record.winnerIndex == 0) wins0.incrementAndGet();
                else if (record.winnerIndex == 1) wins1.incrementAndGet();
                else draws.incrementAndGet();

                if (done % 100 == 0 || done == numGames) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    double gamesPerSec = done * 1000.0 / elapsed;
                    int remaining = numGames - done;
                    long etaMs = (long) (remaining / gamesPerSec * 1000);
                    System.out.printf("Game %d/%d (%.1f games/sec, ETA: %s) | P1: %d, P2: %d, Draw: %d%n",
                            done, numGames, gamesPerSec, formatTime(etaMs),
                            wins0.get(), wins1.get(), draws.get());
                }
            } catch (Exception e) {
                System.err.println("Error in game " + (i + 1) + ": " + e.getMessage());
            }
        }

        executor.shutdown();

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.println();
        System.out.println("=== COMPLETE (DRY RUN - no data exported) ===");
        System.out.println("Total games: " + completed.get());
        System.out.println("Total time: " + formatTime(totalTime));
        System.out.println("Avg turns/game: " + (totalTurns.get() / Math.max(1, completed.get())));
        System.out.println("Player 1 wins: " + wins0.get());
        System.out.println("Player 2 wins: " + wins1.get());
        System.out.println("Draws/Timeouts: " + draws.get());
        System.out.println();
        System.out.println("To export to MongoDB, run with URI:");
        System.out.println("  TrainingDataGenerator " + numGames + " mongodb+srv://user:pass@cluster.mongodb.net/");
    }

    private static String formatTime(long ms) {
        long seconds = ms / 1000;
        if (seconds < 60) return seconds + "s";
        long minutes = seconds / 60;
        seconds = seconds % 60;
        return minutes + "m " + seconds + "s";
    }
}
