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

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Generates training data with 27 features.
 * BotBrain's chosen action gets label 0.9, random alternatives get 0.1.
 */
public class FeatureDataGenerator {

    private static final int MAX_TURNS = 200;
    private static final int RANDOM_ALTERNATIVES = 2;

    public static void main(String[] args) throws Exception {
        int numGames = (args.length > 0) ? Integer.parseInt(args[0]) : 50000;
        String outputFile = (args.length > 1) ? args[1] : "training_features.dat";

        WallEvaluator.silent = true;
        System.out.println("Pairwise Preference Data Generator (27 features)");
        System.out.println("Games: " + numGames);

        int threads = Runtime.getRuntime().availableProcessors();
        System.out.println("Threads: " + threads);

        AtomicInteger completed = new AtomicInteger(0);
        AtomicInteger totalSamples = new AtomicInteger(0);

        List<double[]> allFeatures = new ArrayList<>();
        List<Double> allLabels = new ArrayList<>();

        int perThread = numGames / threads;
        ExecutorService exec = Executors.newFixedThreadPool(threads);
        CountDownLatch latch = new CountDownLatch(threads);
        long start = System.currentTimeMillis();

        for (int t = 0; t < threads; t++) {
            final int tid = t;
            final int tGames = (t == threads - 1) ? (numGames - perThread * (threads - 1)) : perThread;
            exec.submit(() -> {
                Random rng = new Random(tid * 12345L + System.nanoTime());
                try {
                    for (int g = 0; g < tGames; g++) {
                        int oppType = (tid * perThread + g) % 5;
                        playAndRecord(oppType, rng, allFeatures, allLabels, totalSamples);

                        int c = completed.incrementAndGet();
                        if (c % 5000 == 0) {
                            System.out.println("  " + c + "/" + numGames + " games, " + totalSamples.get() + " samples");
                        }
                    }
                } catch (Exception e) { e.printStackTrace(); }
                finally { latch.countDown(); }
            });
        }

        latch.await();
        exec.shutdown();

        long elapsed = (System.currentTimeMillis() - start) / 1000;
        System.out.println("Done in " + elapsed + "s. Samples: " + allFeatures.size());

        writeData(outputFile, allFeatures, allLabels);
        System.out.println("Written to " + outputFile);
    }

    private static void playAndRecord(int oppType, Random rng,
                                        List<double[]> allFeatures, List<Double> allLabels,
                                        AtomicInteger totalSamples) {
        GameState state = new GameState();
        BotBrain bot = new BotBrain(0, 3);
        bot.setSilent(true);
        Object opponent = createOpponent(oppType);

        int turns = 0;
        boolean running = true;
        while (running && !state.isGameOver() && turns < MAX_TURNS) {
            if (state.getCurrentPlayerIndex() == 0) {
                // BotBrain's turn - record chosen action + random alternatives
                try {
                    Player me = state.getCurrentPlayer();
                    List<Position> validMoves = MoveValidator.getValidMoves(state, me);

                    BotBrain.BotAction chosen = bot.computeBestAction(state);
                    if (chosen == null) {
                        running = false;
                    } else {
                        // chosen action gets label 0.9
                        double[] chosenFeatures = computeActionFeatures(state, chosen);
                        if (chosenFeatures != null) {
                            allFeatures.add(chosenFeatures);
                            allLabels.add(0.9);
                            totalSamples.incrementAndGet();
                        }

                        // random alternatives get label 0.1
                        List<Object> alternatives = getAlternativeActions(state, me, chosen, rng);
                        for (Object alt : alternatives) {
                            double[] altFeatures = computeAltFeatures(state, alt);
                            if (altFeatures != null) {
                                allFeatures.add(altFeatures);
                                allLabels.add(0.1);
                                totalSamples.incrementAndGet();
                            }
                        }

                        // apply BotBrain's chosen action
                        if (chosen.type == BotBrain.BotAction.Type.MOVE) {
                            state.getCurrentPlayer().setPosition(chosen.moveTarget);
                        } else {
                            chosen.wallToPlace.setOwnerIndex(0);
                            state.addWall(chosen.wallToPlace);
                        }
                    }
                } catch (Exception e) { running = false; }
            } else {
                applyOpponentAction(state, opponent);
            }

            if (running) {
                state.checkWinCondition();
                if (!state.isGameOver()) state.nextTurn();
                turns++;
            }
        }
    }

    private static double[] computeActionFeatures(GameState state, BotBrain.BotAction action) {
        if (action.type == BotBrain.BotAction.Type.MOVE) {
            return GameFeatures.extractForMove(state, action.moveTarget);
        } else {
            return GameFeatures.extractForWall(state, action.wallToPlace);
        }
    }

    private static double[] computeAltFeatures(GameState state, Object action) {
        if (action instanceof Position) {
            return GameFeatures.extractForMove(state, (Position) action);
        } else if (action instanceof Wall) {
            return GameFeatures.extractForWall(state, (Wall) action);
        }
        return null;
    }

    /** Gets 2 random alternative actions that are different from BotBrain's choice. */
    private static List<Object> getAlternativeActions(GameState state, Player me,
                                                        BotBrain.BotAction chosen, Random rng) {
        List<Object> alts = new ArrayList<>();
        List<Object> candidates = new ArrayList<>();

        // add all legal moves (except chosen)
        List<Position> moves = MoveValidator.getValidMoves(state, me);
        for (Position p : moves) {
            if (!(chosen.type == BotBrain.BotAction.Type.MOVE && p.equals(chosen.moveTarget))) {
                candidates.add(p);
            }
        }

        // add some legal walls (except chosen)
        if (me.getWallsRemaining() > 0) {
            List<Wall> walls = new ArrayList<>();
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    for (int o = 0; o < 2; o++) {
                        Wall.Orientation orient = (o == 0) ?
                                Wall.Orientation.HORIZONTAL : Wall.Orientation.VERTICAL;
                        Wall w = new Wall(r, c, orient);
                        if (WallValidator.isValidWallPlacement(state, w)) {
                            boolean isChosen = false;
                            if (chosen.type == BotBrain.BotAction.Type.WALL) {
                                Wall cw = chosen.wallToPlace;
                                if (cw.getPosition().getRow() == r && cw.getPosition().getCol() == c
                                        && cw.getOrientation() == orient) {
                                    isChosen = true;
                                }
                            }
                            if (!isChosen) {
                                walls.add(w);
                            }
                        }
                    }
                }
            }
            // sample up to 5 random walls (not all 128)
            for (int i = 0; i < Math.min(5, walls.size()); i++) {
                int idx = rng.nextInt(walls.size());
                candidates.add(walls.get(idx));
                walls.remove(idx);
            }
        }

        // pick RANDOM_ALTERNATIVES from candidates
        for (int i = 0; i < RANDOM_ALTERNATIVES && !candidates.isEmpty(); i++) {
            int idx = rng.nextInt(candidates.size());
            alts.add(candidates.get(idx));
            candidates.remove(idx);
        }

        return alts;
    }

    private static Object createOpponent(int type) {
        switch (type) {
            case 0: return new RandomBot.UniformRandom(System.nanoTime());
            case 1: return new RandomBot.ForwardRandom(System.nanoTime());
            case 2: return new RandomBot.RandomWaller(System.nanoTime());
            case 3: return new RandomBot.SemiSmart(System.nanoTime());
            case 4:
                BotBrain b = new BotBrain(0, 5);
                b.setSilent(true);
                return b;
            default: return new RandomBot.UniformRandom(System.nanoTime());
        }
    }

    private static void applyOpponentAction(GameState state, Object opponent) {
        if (opponent instanceof BotBrain) {
            BotBrain brain = (BotBrain) opponent;
            BotBrain.BotAction a = brain.computeBestAction(state);
            if (a == null) return;
            if (a.type == BotBrain.BotAction.Type.MOVE)
                state.getCurrentPlayer().setPosition(a.moveTarget);
            else {
                a.wallToPlace.setOwnerIndex(state.getCurrentPlayerIndex());
                state.addWall(a.wallToPlace);
            }
        } else {
            RandomBot rbot = (RandomBot) opponent;
            Object action = rbot.chooseAction(state);
            if (action instanceof Position) state.getCurrentPlayer().setPosition((Position) action);
            else {
                Wall w = (Wall) action;
                w.setOwnerIndex(state.getCurrentPlayerIndex());
                state.addWall(w);
            }
        }
    }

    // I/O

    public static void writeData(String path, List<double[]> features, List<Double> labels) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))) {
            out.writeInt(features.size());
            out.writeInt(GameFeatures.TOTAL_FEATURES);
            for (int i = 0; i < features.size(); i++) {
                for (double f : features.get(i)) out.writeDouble(f);
                out.writeDouble(labels.get(i));
            }
        }
    }

    public static TrainingData readData(String path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
            int count = in.readInt();
            int numFeats = in.readInt();
            double[][] features = new double[count][numFeats];
            double[] labels = new double[count];
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < numFeats; j++) features[i][j] = in.readDouble();
                labels[i] = in.readDouble();
            }
            return new TrainingData(features, labels);
        }
    }

    public static class TrainingData {
        public final double[][] features;
        public final double[] labels;
        public TrainingData(double[][] features, double[] labels) {
            this.features = features;
            this.labels = labels;
        }
        public int size() { return labels.length; }
        public void shuffle(Random rng) {
            for (int i = labels.length - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                double[] tmpF = features[i]; features[i] = features[j]; features[j] = tmpF;
                double tmpL = labels[i]; labels[i] = labels[j]; labels[j] = tmpL;
            }
        }
    }
}
