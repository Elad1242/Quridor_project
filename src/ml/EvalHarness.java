package ml;

import bot.BotBrain;
import bot.WallEvaluator;
import model.GameState;
import model.Position;
import model.Wall;

/**
 * Measures MLBot win rate against BotBrain.
 *
 * Usage: java ml.EvalHarness [modelDir] [modelName] [numGames]
 * Default: models/best-model, 1000 games
 */
public class EvalHarness {

    private static final int MAX_TURNS = 200;

    public static void main(String[] args) throws Exception {
        String modelDir = (args.length > 0) ? args[0] : "models";
        String modelName = (args.length > 1) ? args[1] : "best-model";
        int numGames = (args.length > 2) ? Integer.parseInt(args[2]) : 1000;

        WallEvaluator.silent = true;

        System.out.println("=== MLBot vs BotBrain Evaluation ===");
        System.out.println("Model: " + modelDir + "/" + modelName);
        System.out.println("Games: " + numGames);

        MLBot mlBot = new MLBot(modelDir, modelName);

        int mlWinsAsP1 = 0, mlWinsAsP2 = 0;
        int gamesAsP1 = 0, gamesAsP2 = 0;
        int totalTurns = 0;
        int draws = 0;

        for (int g = 0; g < numGames; g++) {
            boolean mlIsP1 = (g % 2 == 0);
            int mlPlayerIndex = mlIsP1 ? 0 : 1;

            GameState state = new GameState();
            BotBrain bot = new BotBrain();
            bot.setSilent(true);

            int turns = 0;
            while (!state.isGameOver() && turns < MAX_TURNS) {
                boolean isMLTurn = (state.getCurrentPlayerIndex() == mlPlayerIndex);

                if (isMLTurn) {
                    MLBot.Action action = mlBot.computeBestAction(state);
                    if (action == null) break;
                    if (action.type == MLBot.Action.Type.MOVE) {
                        state.getCurrentPlayer().setPosition(action.moveTarget);
                    } else {
                        Wall wall = action.wall;
                        wall.setOwnerIndex(state.getCurrentPlayerIndex());
                        state.addWall(wall);
                    }
                } else {
                    BotBrain.BotAction action = bot.computeBestAction(state);
                    if (action == null) break;
                    if (action.type == BotBrain.BotAction.Type.MOVE) {
                        state.getCurrentPlayer().setPosition(action.moveTarget);
                    } else {
                        Wall wall = action.wallToPlace;
                        wall.setOwnerIndex(state.getCurrentPlayerIndex());
                        state.addWall(wall);
                    }
                }

                state.checkWinCondition();
                if (!state.isGameOver()) {
                    state.nextTurn();
                }
                turns++;
            }

            totalTurns += turns;

            if (state.isGameOver() && state.getWinner() != null) {
                int winnerIndex = (state.getWinner() == state.getPlayer(0)) ? 0 : 1;
                boolean mlWon = (winnerIndex == mlPlayerIndex);

                if (mlIsP1) {
                    gamesAsP1++;
                    if (mlWon) mlWinsAsP1++;
                } else {
                    gamesAsP2++;
                    if (mlWon) mlWinsAsP2++;
                }
            } else {
                draws++;
            }

            // Progress report
            if ((g + 1) % 100 == 0) {
                int totalWins = mlWinsAsP1 + mlWinsAsP2;
                int totalPlayed = gamesAsP1 + gamesAsP2;
                double winRate = (totalPlayed > 0) ? (100.0 * totalWins / totalPlayed) : 0;
                System.out.printf("  Game %d/%d — MLBot win rate: %.1f%% (%d/%d)%n",
                        g + 1, numGames, winRate, totalWins, totalPlayed);
            }
        }

        // Final results
        int totalWins = mlWinsAsP1 + mlWinsAsP2;
        int totalPlayed = gamesAsP1 + gamesAsP2;
        double overallWinRate = (totalPlayed > 0) ? (100.0 * totalWins / totalPlayed) : 0;
        double p1WinRate = (gamesAsP1 > 0) ? (100.0 * mlWinsAsP1 / gamesAsP1) : 0;
        double p2WinRate = (gamesAsP2 > 0) ? (100.0 * mlWinsAsP2 / gamesAsP2) : 0;
        double avgTurns = (double) totalTurns / numGames;

        System.out.println("\n=== RESULTS ===");
        System.out.printf("Overall win rate: %.1f%% (%d/%d)%n", overallWinRate, totalWins, totalPlayed);
        System.out.printf("As Player 1:      %.1f%% (%d/%d)%n", p1WinRate, mlWinsAsP1, gamesAsP1);
        System.out.printf("As Player 2:      %.1f%% (%d/%d)%n", p2WinRate, mlWinsAsP2, gamesAsP2);
        System.out.printf("Draws:            %d%n", draws);
        System.out.printf("Avg game length:  %.1f turns%n", avgTurns);

        if (overallWinRate >= 65.0) {
            System.out.println("\n>>> TARGET MET! MLBot achieves 65%+ win rate! <<<");
        } else {
            System.out.println("\n>>> Target NOT met. Need more training or different approach.");
        }

        mlBot.close();
    }
}
