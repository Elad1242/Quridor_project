package ml;

import bot.BotBrain;
import bot.WallEvaluator;
import model.GameState;
import model.Player;

public class TestBotBrain {
    public static void main(String[] args) {
        WallEvaluator.silent = true;
        int games = 100;
        int p1Wins = 0;

        System.out.println("Testing BotBrain vs BotBrain...");
        for (int g = 0; g < games; g++) {
            GameState state = new GameState();
            int turns = 0;

            while (!state.isGameOver() && turns < 200) {
                BotBrain brain = new BotBrain();
                brain.setSilent(true);
                BotBrain.BotAction action = brain.computeBestAction(state);

                if (action.type == BotBrain.BotAction.Type.MOVE) {
                    state.getCurrentPlayer().setPosition(action.moveTarget);
                } else if (action.type == BotBrain.BotAction.Type.WALL) {
                    action.wallToPlace.setOwnerIndex(state.getCurrentPlayerIndex());
                    state.addWall(action.wallToPlace);
                    Player p = state.getCurrentPlayer();
                    p.setWallsRemaining(p.getWallsRemaining() - 1);
                }

                state.checkWinCondition();
                state.nextTurn();
                turns++;
            }

            if (state.isGameOver()) {
                Player winner = state.getWinner();
                if (winner != null && winner == state.getPlayers()[0]) {
                    p1Wins++;
                }
            }

            if ((g + 1) % 20 == 0) {
                System.out.println("  Game " + (g + 1) + "/" + games + ": P1 wins " + p1Wins);
            }
        }

        System.out.println("\nBotBrain P1 wins: " + p1Wins + "/" + games);
    }
}
