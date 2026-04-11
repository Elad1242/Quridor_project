package ml;

import bot.WallEvaluator;

/**
 * The "imitation-ceiling" ablation — the experiment that converts the
 * PROJECT_JOURNEY.md ceiling CLAIM into empirical EVIDENCE.
 *
 * Hypothesis (from the journey):
 *     "A supervised-learning student cannot surpass its teacher. Its
 *      achievable strength is upper-bounded by the teacher's strength."
 *
 * Prediction derived from the hypothesis:
 *     If we measure FeatureBot's win rate against a LADDER of opponents
 *     ordered by strength (weakest → strongest), FeatureBot should win
 *     decisively against everything weaker than its teacher (BotBrain),
 *     and then plateau right at BotBrain's level.
 *
 *     The shape of the curve (high-wins → cliff → ~50%) is the signature
 *     of an imitation ceiling. A shape that kept climbing past BotBrain
 *     would refute it. A shape that was flat everywhere would imply the
 *     NN wasn't learning anything at all.
 *
 * What this experiment outputs:
 *     A single table with one row per opponent in the ladder, showing both
 *     FeatureBot's and BotBrain's win rates (overall + P1/P2 split) against
 *     that opponent. If the rows track each other almost exactly — and if
 *     the last row (vs BotBrain) shows the fake-50% pattern — the ceiling
 *     hypothesis is empirically supported.
 *
 * Why this is the single highest-leverage experiment for the writeup:
 *   - Reuses the existing trained model (no retraining needed).
 *   - Produces a clean, publication-ready table.
 *   - Directly demonstrates that the NN *does* learn (it beats weak bots),
 *     and that its failure against BotBrain is structural, not a coding bug.
 *
 * Usage:
 *   java ml.SkillGradientExperiment [gamesPerOpponent] [modelPath]
 */
public class SkillGradientExperiment {

    /** The skill ladder — ordered weakest → strongest by a priori reasoning. */
    private static final String[] LADDER = {
            "uniformrandom",   // chaos baseline
            "forwardrandom",   // never walls, random forward moves
            "semismart",       // 70% shortest-path, 15% random walls
            "greedy",          // deterministic A* follower, no walls
            "botbrain"         // full-strength teacher
    };

    public static void main(String[] args) throws Exception {
        int games = (args.length > 0) ? Integer.parseInt(args[0]) : 200;
        String modelPath = (args.length > 1) ? args[1] : "feature_model.bin";

        WallEvaluator.silent = true;

        System.out.println("========================================================");
        System.out.println("  SKILL-GRADIENT ABLATION — testing the imitation ceiling");
        System.out.println("========================================================");
        System.out.printf("Games per opponent: %d    Model: %s%n%n", games, modelPath);

        EvalHarness.Result[] featureRes = new EvalHarness.Result[LADDER.length];
        EvalHarness.Result[] brainRes   = new EvalHarness.Result[LADDER.length];

        for (int i = 0; i < LADDER.length; i++) {
            String opp = LADDER[i];
            System.out.printf("[%d/%d] vs %s ...%n", i + 1, LADDER.length, opp);

            // Skip self-play for BotBrain-vs-BotBrain (it's measuring nothing useful,
            // and it's dominated by the tempo advantage).
            System.out.printf("      FeatureBot vs %s%n", opp);
            featureRes[i] = EvalHarness.run("featurebot", opp, games, modelPath);

            if (!opp.equals("botbrain")) {
                System.out.printf("      BotBrain   vs %s%n", opp);
                brainRes[i] = EvalHarness.run("botbrain", opp, games, modelPath);
            }
        }

        // ---- Print summary table ------------------------------------------------
        System.out.println();
        System.out.println("=================== SKILL-GRADIENT RESULTS ===================");
        System.out.printf("%-16s | %-28s | %-28s%n",
                "Opponent", "FeatureBot win% (CI)", "BotBrain win% (CI)");
        System.out.println("-----------------+------------------------------+------------------------------");
        for (int i = 0; i < LADDER.length; i++) {
            String opp = LADDER[i];
            String fbCell = cell(featureRes[i]);
            String bbCell = (brainRes[i] == null) ? "  —  (self)" : cell(brainRes[i]);
            System.out.printf("%-16s | %-28s | %-28s%n", opp, fbCell, bbCell);
        }
        System.out.println("=============================================================");
        System.out.println();

        // ---- Print P1/P2 breakdown for the key matchup -------------------------
        System.out.println("--- Key row: FeatureBot vs BotBrain (P1 / P2 split) ---");
        featureRes[LADDER.length - 1].printReport("featurebot", "botbrain");

        System.out.println();
        System.out.println("INTERPRETATION GUIDE:");
        System.out.println("  • If FeatureBot's win% is HIGH against weak opponents and COLLAPSES");
        System.out.println("    to ~50% (fake) against BotBrain, the imitation ceiling is confirmed:");
        System.out.println("    the NN learned to play LIKE BotBrain, so it inherits BotBrain's level.");
        System.out.println("  • If FeatureBot's win% is flat low across the board, the NN didn't learn.");
        System.out.println("  • If FeatureBot's win% against BotBrain is >55% on BOTH sides (P1 & P2),");
        System.out.println("    the student surpassed the teacher and the hypothesis is refuted.");
    }

    private static String cell(EvalHarness.Result r) {
        if (r == null) return "  —";
        return EvalHarness.formatRate(r.aWins, r.played);
    }
}
