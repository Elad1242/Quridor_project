package ml;

import bot.WallEvaluator;

/**
 * Round-robin tournament across all available bots, producing:
 *   1. A pairwise win-rate matrix (each cell is row-bot win% vs column-bot).
 *   2. An Elo leaderboard computed by iterating all individual game results
 *      through the standard Elo update rule (K=32, start rating 1000).
 *
 * Why Elo? A single number per bot that is internally consistent across the
 * full field makes the "how strong is FeatureBot really?" question answerable
 * in one line. The pairwise matrix lets you double-check whether any Elo
 * ordering is actually supported by direct games.
 *
 * Usage:
 *   java ml.Tournament [gamesPerPair] [modelPath]
 *
 * Output is tabular and copy-paste ready for RESULTS.md.
 */
public class Tournament {

    private static final int MAX_TURNS = 200;
    private static final double K = 32.0;
    private static final double START_RATING = 1000.0;

    // Bots in strength order (weakest → strongest, as a hypothesis).
    private static final String[] BOTS = {
            "uniformrandom",
            "forwardrandom",
            "semismart",
            "greedy",
            "featurebot",
            "botbrain"
    };

    public static void main(String[] args) throws Exception {
        int gamesPerPair = (args.length > 0) ? Integer.parseInt(args[0]) : 100;
        String modelPath = (args.length > 1) ? args[1] : "feature_model.bin";

        WallEvaluator.silent = true;

        int n = BOTS.length;
        int[][] wins = new int[n][n];       // wins[i][j] = games where BOTS[i] beat BOTS[j]
        int[][] played = new int[n][n];     // decisive games between BOTS[i] and BOTS[j]
        double[] elo = new double[n];
        for (int i = 0; i < n; i++) elo[i] = START_RATING;

        System.out.printf("=== Round-robin tournament — %d games per pair ===%n", gamesPerPair);
        System.out.println("Participants: " + String.join(", ", BOTS));
        System.out.println();

        long t0 = System.currentTimeMillis();

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                System.out.printf("  %s vs %s ... ", BOTS[i], BOTS[j]);
                System.out.flush();

                EvalHarness.Result r =
                        EvalHarness.run(BOTS[i], BOTS[j], gamesPerPair, modelPath);

                wins[i][j] = r.aWins;
                wins[j][i] = r.bWins;
                played[i][j] = played[j][i] = r.played;

                // Iteratively apply Elo updates per individual game so the
                // ratings converge sensibly even with a small games-per-pair.
                // We feed aWins games with result 1.0 and bWins with 0.0.
                for (int w = 0; w < r.aWins; w++) eloUpdate(elo, i, j, 1.0);
                for (int w = 0; w < r.bWins; w++) eloUpdate(elo, i, j, 0.0);

                System.out.printf("%d-%d (draws: %d)%n", r.aWins, r.bWins, r.draws);
            }
        }

        long elapsed = (System.currentTimeMillis() - t0) / 1000;
        System.out.printf("%nTournament done in %ds.%n%n", elapsed);

        printMatrix(wins, played);
        printEloLeaderboard(elo);
    }

    /** Standard Elo update for one game. outcomeA = 1 (A won), 0 (A lost), 0.5 (draw). */
    private static void eloUpdate(double[] elo, int i, int j, double outcomeA) {
        double expectedI = 1.0 / (1.0 + Math.pow(10.0, (elo[j] - elo[i]) / 400.0));
        elo[i] += K * (outcomeA - expectedI);
        elo[j] += K * ((1.0 - outcomeA) - (1.0 - expectedI));
    }

    private static void printMatrix(int[][] wins, int[][] played) {
        int n = BOTS.length;
        System.out.println("--- Pairwise win-rate matrix (row bot win% vs column bot) ---");
        System.out.print(pad("", 16));
        for (String b : BOTS) System.out.print(pad(b, 14));
        System.out.println();

        for (int i = 0; i < n; i++) {
            System.out.print(pad(BOTS[i], 16));
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    System.out.print(pad("  —", 14));
                } else if (played[i][j] == 0) {
                    System.out.print(pad("  -", 14));
                } else {
                    double pct = 100.0 * wins[i][j] / played[i][j];
                    System.out.print(pad(String.format("%.1f%% (%d/%d)",
                            pct, wins[i][j], played[i][j]), 14));
                }
            }
            System.out.println();
        }
        System.out.println();
    }

    private static void printEloLeaderboard(double[] elo) {
        int n = BOTS.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (a, b) -> Double.compare(elo[b], elo[a]));

        System.out.println("--- Elo leaderboard (K=32, start=1000) ---");
        System.out.printf("%-4s  %-16s  %s%n", "Rank", "Bot", "Elo");
        for (int rank = 0; rank < n; rank++) {
            int i = idx[rank];
            System.out.printf("%-4d  %-16s  %.0f%n", rank + 1, BOTS[i], elo[i]);
        }
        System.out.println();
        System.out.println("Interpretation: +400 Elo ≈ 10× expected win odds.");
        System.out.println("If FeatureBot is within a few Elo of BotBrain, that reflects");
        System.out.println("the imitation ceiling — same policy, tempo-bound outcomes.");
    }

    private static String pad(String s, int w) {
        if (s.length() >= w) return s.substring(0, w);
        StringBuilder sb = new StringBuilder(s);
        while (sb.length() < w) sb.append(' ');
        return sb.toString();
    }
}
