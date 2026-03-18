package ml.cnn;

import bot.BotBrain;
import bot.WallEvaluator;
import model.GameState;
import model.Player;

/**
 * Tests the CNN Bot against BotBrain.
 *
 * PURE ML TEST: The CNN makes decisions solely based on learned
 * patterns, with NO heuristic assistance.
 */
public class TestCNNBot {

    public static void main(String[] args) {
        WallEvaluator.silent = true;

        String modelPath = args.length > 0 ? args[0] : "quoridor_cnn.zip";
        int games = 100;

        System.out.println("=== CNN Bot Test (PURE ML) ===");
        System.out.println("Model: " + modelPath);
        System.out.println("Games: " + games + " per side");
        System.out.println();

        CNNBot cnnBot;
        try {
            cnnBot = new CNNBot(modelPath);
            cnnBot.setTemperature(0.0);  // Greedy for best performance
        } catch (Exception e) {
            System.err.println("Error loading model: " + e.getMessage());
            return;
        }

        // Test as Player 1
        System.out.println("--- CNN as Player 1 (moves first) ---");
        int winsAsP1 = testAsPlayer(cnnBot, 0, games);
        System.out.printf("Result: %d/%d (%.1f%%)%n", winsAsP1, games, winsAsP1 * 100.0 / games);

        // Test as Player 2
        System.out.println("\n--- CNN as Player 2 (moves second) ---");
        int winsAsP2 = testAsPlayer(cnnBot, 1, games);
        System.out.printf("Result: %d/%d (%.1f%%)%n", winsAsP2, games, winsAsP2 * 100.0 / games);

        // Overall results
        int totalWins = winsAsP1 + winsAsP2;
        int totalGames = games * 2;
        double winRate = totalWins * 100.0 / totalGames;

        System.out.println("\n========== RESULTS ==========");
        System.out.printf("As Player 1: %d/%d (%.1f%%)%n", winsAsP1, games, winsAsP1 * 100.0 / games);
        System.out.printf("As Player 2: %d/%d (%.1f%%)%n", winsAsP2, games, winsAsP2 * 100.0 / games);
        System.out.printf("OVERALL: %d/%d (%.1f%%)%n", totalWins, totalGames, winRate);
        System.out.println("=============================");

        // Interpretation
        if (winRate >= 75) {
            System.out.println("SUCCESS: Target win rate of 75%+ achieved!");
        } else if (winRate >= 50) {
            System.out.println("DECENT: CNN performs better than random.");
        } else {
            System.out.println("NEEDS WORK: CNN underperforms BotBrain.");
        }

        // Warning for suspiciously high results
        if (winRate >= 95) {
            System.out.println("\nWARNING: Win rate is suspiciously high (>95%).");
            System.out.println("Please verify this is truly pure ML with no hidden heuristics.");
        }
    }

    /**
     * Tests CNN bot as a specific player against BotBrain.
     *
     * @param cnnBot The CNN bot to test
     * @param cnnPlayerIdx Which player index the CNN plays as (0 or 1)
     * @param games Number of games to play
     * @return Number of wins
     */
    private static int testAsPlayer(CNNBot cnnBot, int cnnPlayerIdx, int games) {
        int wins = 0;

        for (int g = 0; g < games; g++) {
            GameState state = new GameState();
            BotBrain botBrain = new BotBrain();
            botBrain.setSilent(true);

            int turns = 0;
            while (!state.isGameOver() && turns < 200) {
                Player current = state.getCurrentPlayer();
                int currentIdx = state.getCurrentPlayerIndex();

                if (currentIdx == cnnPlayerIdx) {
                    // CNN plays
                    CNNBot.BotAction action = cnnBot.computeBestAction(state);
                    if (action == null) break;

                    if (action.type == CNNBot.BotAction.Type.MOVE) {
                        current.setPosition(action.moveTarget);
                    } else {
                        action.wallToPlace.setOwnerIndex(currentIdx);
                        state.addWall(action.wallToPlace);
                        current.setWallsRemaining(current.getWallsRemaining() - 1);
                    }
                } else {
                    // BotBrain plays
                    BotBrain.BotAction action = botBrain.computeBestAction(state);
                    if (action == null) break;

                    if (action.type == BotBrain.BotAction.Type.MOVE) {
                        current.setPosition(action.moveTarget);
                    } else if (action.type == BotBrain.BotAction.Type.WALL) {
                        action.wallToPlace.setOwnerIndex(currentIdx);
                        state.addWall(action.wallToPlace);
                        current.setWallsRemaining(current.getWallsRemaining() - 1);
                    }
                }

                state.checkWinCondition();
                if (!state.isGameOver()) {
                    state.nextTurn();
                }
                turns++;
            }

            if (state.isGameOver()) {
                Player winner = state.getWinner();
                if (winner != null && winner == state.getPlayers()[cnnPlayerIdx]) {
                    wins++;
                }
            }

            if ((g + 1) % 20 == 0) {
                System.out.printf("  Game %d/%d: %d wins so far%n", g + 1, games, wins);
            }
        }

        return wins;
    }
}
