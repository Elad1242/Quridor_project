package ml;

import bot.BotBrain;
import bot.WallEvaluator;
import model.GameState;
import model.Player;

/**
 * Quick test of NNBot vs BotBrain win rate.
 */
public class TestNNBot {

    public static void main(String[] args) {
        WallEvaluator.silent = true;

        String modelPath = args.length > 0 ? args[0] : "quoridor_policy.eg";
        NNBot nnBot = new NNBot(modelPath);
        nnBot.setTemperature(0.0);  // Greedy mode

        int wins = 0;
        int games = 100;

        System.out.println("Testing NNBot vs BotBrain...");
        System.out.println("Model: " + modelPath);
        System.out.println("Temperature: 0.0 (greedy)");
        System.out.println();

        // Test NNBot as Player 1
        System.out.println("--- NNBot as Player 1 ---");
        int winsAsP1 = 0;
        for (int g = 0; g < games; g++) {
            GameState state = new GameState();
            int turns = 0;

            while (!state.isGameOver() && turns < 200) {
                if (state.getCurrentPlayerIndex() == 0) {
                    // NNBot plays as Player 1
                    NNBot.BotAction action = nnBot.computeBestAction(state);
                    if (action == null) break;

                    if (action.type == NNBot.BotAction.Type.MOVE) {
                        state.getCurrentPlayer().setPosition(action.moveTarget);
                    } else {
                        action.wallToPlace.setOwnerIndex(0);
                        state.addWall(action.wallToPlace);
                        Player p = state.getCurrentPlayer();
                        p.setWallsRemaining(p.getWallsRemaining() - 1);
                    }
                } else {
                    // BotBrain plays as Player 2
                    BotBrain brain = new BotBrain();
                    brain.setSilent(true);
                    BotBrain.BotAction action = brain.computeBestAction(state);

                    if (action.type == BotBrain.BotAction.Type.MOVE) {
                        state.getCurrentPlayer().setPosition(action.moveTarget);
                    } else if (action.type == BotBrain.BotAction.Type.WALL) {
                        action.wallToPlace.setOwnerIndex(1);
                        state.addWall(action.wallToPlace);
                        Player p = state.getCurrentPlayer();
                        p.setWallsRemaining(p.getWallsRemaining() - 1);
                    }
                }

                state.checkWinCondition();
                state.nextTurn();
                turns++;
            }

            if (state.isGameOver()) {
                Player winner = state.getWinner();
                if (winner != null && winner == state.getPlayers()[0]) {
                    winsAsP1++;
                }
            }

            if ((g + 1) % 20 == 0) {
                System.out.println("  Game " + (g + 1) + "/" + games + ": " + winsAsP1 + " wins");
            }
        }

        // Test NNBot as Player 2
        System.out.println("\n--- NNBot as Player 2 ---");
        int winsAsP2 = 0;
        for (int g = 0; g < games; g++) {
            GameState state = new GameState();
            int turns = 0;

            while (!state.isGameOver() && turns < 200) {
                if (state.getCurrentPlayerIndex() == 1) {
                    // NNBot plays as Player 2
                    NNBot.BotAction action = nnBot.computeBestAction(state);
                    if (action == null) break;

                    if (action.type == NNBot.BotAction.Type.MOVE) {
                        state.getCurrentPlayer().setPosition(action.moveTarget);
                    } else {
                        action.wallToPlace.setOwnerIndex(1);
                        state.addWall(action.wallToPlace);
                        Player p = state.getCurrentPlayer();
                        p.setWallsRemaining(p.getWallsRemaining() - 1);
                    }
                } else {
                    // BotBrain plays as Player 1
                    BotBrain brain = new BotBrain();
                    brain.setSilent(true);
                    BotBrain.BotAction action = brain.computeBestAction(state);

                    if (action.type == BotBrain.BotAction.Type.MOVE) {
                        state.getCurrentPlayer().setPosition(action.moveTarget);
                    } else if (action.type == BotBrain.BotAction.Type.WALL) {
                        action.wallToPlace.setOwnerIndex(0);
                        state.addWall(action.wallToPlace);
                        Player p = state.getCurrentPlayer();
                        p.setWallsRemaining(p.getWallsRemaining() - 1);
                    }
                }

                state.checkWinCondition();
                state.nextTurn();
                turns++;
            }

            if (state.isGameOver()) {
                Player winner = state.getWinner();
                if (winner != null && winner == state.getPlayers()[1]) {
                    winsAsP2++;
                }
            }

            if ((g + 1) % 20 == 0) {
                System.out.println("  Game " + (g + 1) + "/" + games + ": " + winsAsP2 + " wins");
            }
        }

        wins = winsAsP1 + winsAsP2;

        System.out.println();
        System.out.println("=== RESULTS ===");
        System.out.println("As Player 1: " + winsAsP1 + "/" + games + " (" + String.format("%.1f%%", winsAsP1 * 100.0 / games) + ")");
        System.out.println("As Player 2: " + winsAsP2 + "/" + games + " (" + String.format("%.1f%%", winsAsP2 * 100.0 / games) + ")");
        System.out.println("Overall: " + wins + "/" + (2 * games) + " (" + String.format("%.1f%%", wins * 100.0 / (2 * games)) + ")");
    }
}
