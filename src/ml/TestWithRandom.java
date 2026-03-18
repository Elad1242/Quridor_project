package ml;

import bot.BotBrain;
import bot.WallEvaluator;
import model.GameState;
import model.Player;

public class TestWithRandom {
    public static void main(String[] args) {
        WallEvaluator.silent = true;
        int games = 100;

        String modelPath = args.length > 0 ? args[0] : "quoridor_policy.eg";
        NNBot nnBot = new NNBot(modelPath);
        nnBot.setTemperature(0.3);  // Some randomness

        System.out.println("Testing with randomness enabled...");
        System.out.println();

        // NNBot as P1
        int winsAsP1 = 0;
        System.out.println("--- NNBot (P1) vs BotBrain (P2) ---");
        for (int g = 0; g < games; g++) {
            GameState state = new GameState();
            int turns = 0;

            while (!state.isGameOver() && turns < 200) {
                if (state.getCurrentPlayerIndex() == 0) {
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
                    BotBrain brain = new BotBrain(5.0, 3);  // Random openings
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
        }
        System.out.println("NNBot P1 wins: " + winsAsP1 + "/" + games);

        // NNBot as P2
        int winsAsP2 = 0;
        System.out.println("\n--- BotBrain (P1) vs NNBot (P2) ---");
        for (int g = 0; g < games; g++) {
            GameState state = new GameState();
            int turns = 0;

            while (!state.isGameOver() && turns < 200) {
                if (state.getCurrentPlayerIndex() == 1) {
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
                    BotBrain brain = new BotBrain(5.0, 3);  // Random openings
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
        }
        System.out.println("NNBot P2 wins: " + winsAsP2 + "/" + games);

        System.out.println("\n=== FINAL RESULTS ===");
        int total = winsAsP1 + winsAsP2;
        double rate = total * 100.0 / (2 * games);
        System.out.println("As P1: " + winsAsP1 + "/" + games);
        System.out.println("As P2: " + winsAsP2 + "/" + games);
        System.out.println("Overall: " + total + "/" + (2*games) + " (" + String.format("%.1f%%", rate) + ")");
    }
}
