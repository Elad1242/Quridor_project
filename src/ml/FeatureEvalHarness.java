package ml;

import bot.BotBrain;
import bot.WallEvaluator;
import model.GameState;
import model.Wall;

/**
 * Evaluates FeatureBot vs BotBrain.
 * Usage: java ml.FeatureEvalHarness [modelPath] [numGames]
 */
public class FeatureEvalHarness {

    private static final int MAX_TURNS = 200;

    public static void main(String[] args) throws Exception {
        String modelPath = (args.length > 0) ? args[0] : "feature_model.bin";
        int numGames = (args.length > 1) ? Integer.parseInt(args[1]) : 500;

        WallEvaluator.silent = true;
        System.out.println("=== FeatureBot vs BotBrain ===");
        System.out.println("Model: " + modelPath + ", Games: " + numGames);

        float winRate = evaluate(modelPath, numGames);

        if (winRate >= 0.10f) System.out.println("\n>>> PROGRESS! Win rate above 10%!");
        if (winRate >= 0.30f) System.out.println(">>> GREAT! Win rate above 30%!");
        if (winRate >= 0.65f) System.out.println(">>> TARGET MET! 65%+!");
    }

    public static float evaluate(String modelPath, int numGames) throws Exception {
        NeuralNetwork nn = NeuralNetwork.load(modelPath);
        FeatureBot mlBot = new FeatureBot(nn);

        int wins = 0, played = 0, draws = 0;
        long start = System.currentTimeMillis();

        for (int g = 0; g < numGames; g++) {
            boolean mlP1 = (g % 2 == 0);
            int mlIdx = mlP1 ? 0 : 1;

            GameState state = new GameState();
            BotBrain brain = new BotBrain();
            brain.setSilent(true);

            int turns = 0;
            while (!state.isGameOver() && turns < MAX_TURNS) {
                if (state.getCurrentPlayerIndex() == mlIdx) {
                    FeatureBot.Action a = mlBot.computeBestAction(state);
                    if (a == null) break;
                    a.applyTo(state);
                } else {
                    BotBrain.BotAction a = brain.computeBestAction(state);
                    if (a == null) break;
                    if (a.type == BotBrain.BotAction.Type.MOVE)
                        state.getCurrentPlayer().setPosition(a.moveTarget);
                    else {
                        a.wallToPlace.setOwnerIndex(state.getCurrentPlayerIndex());
                        state.addWall(a.wallToPlace);
                    }
                }
                state.checkWinCondition();
                if (!state.isGameOver()) state.nextTurn();
                turns++;
            }

            if (state.isGameOver() && state.getWinner() != null) {
                int w = (state.getWinner() == state.getPlayer(0)) ? 0 : 1;
                played++;
                if (w == mlIdx) wins++;
            } else {
                draws++;
            }

            if ((g + 1) % 50 == 0) {
                System.out.printf("  %d/%d — %.1f%% (%d/%d)%n",
                        g + 1, numGames, played > 0 ? 100.0 * wins / played : 0, wins, played);
            }
        }

        long elapsed = (System.currentTimeMillis() - start) / 1000;
        float rate = played > 0 ? (float) wins / played : 0;

        System.out.printf("\n=== RESULT: %.1f%% (%d/%d), draws=%d, time=%ds ===%n",
                rate * 100, wins, played, draws, elapsed);
        return rate;
    }
}
