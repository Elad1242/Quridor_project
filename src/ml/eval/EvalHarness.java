package ml.eval;
import ml.*;

import bot.BotBrain;
import bot.WallEvaluator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

/**
 * Rigorous evaluation harness for head-to-head bot matchups.
 *
 * Improvements over the original FeatureEvalHarness:
 *   1. P1/P2 breakdown — the "fake 50%" trap from the journey is caught here.
 *      Reports wins as Player 1 AND as Player 2 separately, so a bot that only
 *      wins via the first-move tempo advantage is exposed.
 *   2. Wilson 95% confidence intervals — every reported rate has a CI so the
 *      reader can judge statistical significance (a 52% win rate over 500 games
 *      is NOT significantly better than 50%).
 *   3. Uniform Participant adapter — BotBrain, FeatureBot, GreedyPathBot and
 *      any RandomBot subclass are all driven through a single interface, so
 *      arbitrary matchups are a single line to set up.
 *   4. Draw / truncation tracking — games that hit MAX_TURNS without a winner
 *      are counted separately and excluded from the win-rate denominator.
 *   5. Clean tabular output ready to drop into RESULTS.md.
 *
 * Usage:
 *   java ml.EvalHarness [botA] [botB] [numGames] [modelPath]
 *
 *   botA / botB ∈ {featurebot, botbrain, greedy, semismart, forwardrandom,
 *                  uniformrandom}
 *   numGames    default 500
 *   modelPath   default feature_model.bin (only used if a bot is "featurebot")
 *
 * Example:
 *   java ml.EvalHarness featurebot botbrain 500
 *   java ml.EvalHarness featurebot greedy   500
 */
public class EvalHarness {

    private static final int MAX_TURNS = 200;
    private static final double Z_95 = 1.959963984540054; // ≈ 1.96

    // ---------------------------------------------------------------- main ---

    public static void main(String[] args) throws Exception {
        String botAName = (args.length > 0) ? args[0].toLowerCase() : "featurebot";
        String botBName = (args.length > 1) ? args[1].toLowerCase() : "botbrain";
        int numGames    = (args.length > 2) ? Integer.parseInt(args[2]) : 500;
        String modelPath = (args.length > 3) ? args[3] : "feature_model.bin";

        WallEvaluator.silent = true;
        System.out.printf("%n=== Head-to-head: %s vs %s (%d games) ===%n",
                botAName, botBName, numGames);

        Result r = run(botAName, botBName, numGames, modelPath);
        r.printReport(botAName, botBName);
    }

    // --------------------------------------------------------------- engine ---

    /** Runs numGames between botA and botB, alternating who starts as Player 1. */
    public static Result run(String botAName, String botBName, int numGames, String modelPath)
            throws Exception {
        Participant pA = Participant.create(botAName, modelPath);
        Participant pB = Participant.create(botBName, modelPath);

        Result r = new Result();
        long start = System.currentTimeMillis();

        for (int g = 0; g < numGames; g++) {
            boolean aIsP1 = (g % 2 == 0);
            Participant p1 = aIsP1 ? pA : pB;
            Participant p2 = aIsP1 ? pB : pA;

            Outcome o = playOneGame(p1, p2);

            if (o == Outcome.DRAW) {
                r.draws++;
            } else {
                r.played++;
                boolean aWon = (o == Outcome.P1_WIN) == aIsP1;
                if (aIsP1) {
                    r.aAsP1Played++;
                    if (aWon) { r.aWins++; r.aAsP1Wins++; }
                    else      { r.bWins++; }
                } else {
                    r.aAsP2Played++;
                    if (aWon) { r.aWins++; r.aAsP2Wins++; }
                    else      { r.bWins++; }
                }
            }

            if ((g + 1) % 50 == 0) {
                double winPct = r.played > 0 ? 100.0 * r.aWins / r.played : 0;
                System.out.printf("  %d/%d — %s at %.1f%%%n",
                        g + 1, numGames, botAName, winPct);
            }
        }

        r.elapsedSec = (System.currentTimeMillis() - start) / 1000;
        return r;
    }

    private enum Outcome { P1_WIN, P2_WIN, DRAW }

    private static Outcome playOneGame(Participant p1, Participant p2) {
        GameState state = new GameState();
        p1.reset();
        p2.reset();

        int turns = 0;
        while (!state.isGameOver() && turns < MAX_TURNS) {
            Participant current = (state.getCurrentPlayerIndex() == 0) ? p1 : p2;
            boolean ok = current.playOneTurn(state);
            if (!ok) return Outcome.DRAW; // pathological: no legal action produced
            state.checkWinCondition();
            if (!state.isGameOver()) state.nextTurn();
            turns++;
        }

        if (!state.isGameOver() || state.getWinner() == null) return Outcome.DRAW;
        return (state.getWinner() == state.getPlayer(0)) ? Outcome.P1_WIN : Outcome.P2_WIN;
    }

    // ------------------------------------------------------- Participant -----
    /**
     * Uniform adapter for any bot type. Each implementation knows how to
     * compute + apply its move on the current GameState. Kept in-file to
     * avoid touching BotBrain / FeatureBot / RandomBot signatures.
     */
    static abstract class Participant {
        /** Plays the current player's turn. Returns false if no action possible. */
        abstract boolean playOneTurn(GameState state);
        void reset() { /* stateless by default */ }

        static Participant create(String name, String modelPath) throws Exception {
            switch (name) {
                case "featurebot": {
                    final FeatureBot bot = new FeatureBot(modelPath);
                    return new Participant() {
                        @Override boolean playOneTurn(GameState state) {
                            FeatureBot.Action a = bot.computeBestAction(state);
                            if (a == null) return false;
                            a.applyTo(state);
                            return true;
                        }
                    };
                }
                case "botbrain": {
                    final BotBrain brain = new BotBrain();
                    brain.setSilent(true);
                    return new Participant() {
                        @Override boolean playOneTurn(GameState state) {
                            BotBrain.BotAction a = brain.computeBestAction(state);
                            if (a == null) return false;
                            applyBrainAction(state, a);
                            return true;
                        }
                    };
                }
                case "botbrain6": {
                    // BotBrain with 6 random opening moves — same as the GUI uses.
                    // A fresh instance is created per game in EvalHarness.run (one
                    // Participant is reused across games, so this matters: the
                    // turnCount inside BotBrain persists across games, which means
                    // after the first game the random-opening phase has already
                    // ended. We therefore create a FRESH BotBrain per turn call
                    // with 6 random openings — the state resets between games
                    // because Participant.reset() is called by the engine).
                    return new Participant() {
                        BotBrain brain = freshBrain();
                        private BotBrain freshBrain() {
                            BotBrain b = new BotBrain(0.0, 6);
                            b.setSilent(true);
                            return b;
                        }
                        @Override void reset() { brain = freshBrain(); }
                        @Override boolean playOneTurn(GameState state) {
                            BotBrain.BotAction a = brain.computeBestAction(state);
                            if (a == null) return false;
                            applyBrainAction(state, a);
                            return true;
                        }
                    };
                }
                case "greedy": {
                    return new Participant() {
                        @Override boolean playOneTurn(GameState state) {
                            Position p = GreedyPathBot.chooseMove(state);
                            if (p == null) return false;
                            state.getCurrentPlayer().setPosition(p);
                            return true;
                        }
                    };
                }
                case "semismart":
                    return wrapRandom(new RandomBot.SemiSmart(System.nanoTime()));
                case "forwardrandom":
                    return wrapRandom(new RandomBot.ForwardRandom(System.nanoTime()));
                case "uniformrandom":
                    return wrapRandom(new RandomBot.UniformRandom(System.nanoTime()));
                default:
                    throw new IllegalArgumentException("Unknown bot: " + name);
            }
        }

        private static Participant wrapRandom(final RandomBot rbot) {
            return new Participant() {
                @Override boolean playOneTurn(GameState state) {
                    Object action = rbot.chooseAction(state);
                    if (action == null) return false;
                    if (action instanceof Position) {
                        state.getCurrentPlayer().setPosition((Position) action);
                    } else {
                        Wall w = (Wall) action;
                        w.setOwnerIndex(state.getCurrentPlayerIndex());
                        state.addWall(w);
                    }
                    return true;
                }
            };
        }

        private static void applyBrainAction(GameState state, BotBrain.BotAction a) {
            if (a.type == BotBrain.BotAction.Type.MOVE) {
                state.getCurrentPlayer().setPosition(a.moveTarget);
            } else {
                a.wallToPlace.setOwnerIndex(state.getCurrentPlayerIndex());
                state.addWall(a.wallToPlace);
            }
        }
    }

    // -------------------------------------------------------------- Result ---

    public static class Result {
        public int aWins = 0, bWins = 0, played = 0, draws = 0;
        public int aAsP1Wins = 0, aAsP1Played = 0;
        public int aAsP2Wins = 0, aAsP2Played = 0;
        public long elapsedSec = 0;

        public double rate()     { return played == 0     ? 0 : (double) aWins     / played; }
        public double rateP1()   { return aAsP1Played==0  ? 0 : (double) aAsP1Wins / aAsP1Played; }
        public double rateP2()   { return aAsP2Played==0  ? 0 : (double) aAsP2Wins / aAsP2Played; }

        public void printReport(String nameA, String nameB) {
            System.out.printf("%n------- %s vs %s -------%n", nameA, nameB);
            System.out.printf("Decisive games: %d   Draws (truncated): %d   Time: %ds%n",
                    played, draws, elapsedSec);
            System.out.println();
            System.out.printf("  Overall:  %-12s %s%n", nameA, formatRate(aWins, played));
            System.out.printf("            %-12s %s%n", nameB, formatRate(bWins, played));
            System.out.println();
            System.out.printf("  %s as P1:  %s%n", nameA, formatRate(aAsP1Wins, aAsP1Played));
            System.out.printf("  %s as P2:  %s%n", nameA, formatRate(aAsP2Wins, aAsP2Played));
            System.out.println();
            System.out.println(interpret(nameA, nameB));
        }

        /** Detects the "fake 50%" pattern (wins all P1, loses all P2). */
        private String interpret(String nameA, String nameB) {
            if (played == 0) return "  [no decisive games]";
            double p1 = rateP1(), p2 = rateP2();
            if (p1 >= 0.95 && p2 <= 0.05) {
                return "  [!] FAKE-50% DETECTED: " + nameA + " wins as P1, loses as P2."
                        + "\n      This is the first-move tempo advantage, NOT learning."
                        + "\n      Both bots are effectively playing the same policy.";
            }
            if (p1 <= 0.05 && p2 >= 0.95) {
                return "  [!] FAKE-50% DETECTED (mirrored): " + nameA + " wins as P2, loses as P1."
                        + "\n      " + nameB + " holds the tempo advantage; " + nameA + " only wins"
                        + "\n      when it can react. Not a genuine strength advantage.";
            }
            if (p1 >= 0.55 && p2 >= 0.45) {
                return "  [+] Genuine advantage: " + nameA + " wins on BOTH sides of the board.";
            }
            if (p1 <= 0.45 && p2 <= 0.45) {
                return "  " + nameB + " clearly dominates on both sides.";
            }
            return "  Asymmetric result — interpret with care (see P1/P2 breakdown).";
        }
    }

    // ------------------------------------------------ Wilson 95% CI helper ---

    /** Returns "p̂=X.X% 95%CI[lo, hi] (k/n)" — Wilson score interval. */
    public static String formatRate(int k, int n) {
        if (n == 0) return "  — (0/0)";
        double p = (double) k / n;
        double z = Z_95, z2 = z * z;
        double denom = 1 + z2 / n;
        double center = (p + z2 / (2.0 * n)) / denom;
        double half = (z * Math.sqrt(p * (1 - p) / n + z2 / (4.0 * n * n))) / denom;
        double lo = Math.max(0, center - half);
        double hi = Math.min(1, center + half);
        return String.format("%.1f%% 95%%CI[%.1f, %.1f]  (%d/%d)",
                p * 100, lo * 100, hi * 100, k, n);
    }
}
