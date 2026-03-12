package DataBaseBuild;

import bot.BoardGraph;
import bot.BotBrain;
import bot.WallEvaluator;
import ml.GameFeatures;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.ArrayList;
import java.util.List;

/**
 * Runs headless (no UI) Bot vs Bot Quoridor games for training data generation.
 *
 * Each game produces a GameRecord containing:
 * - Metadata: winner index, total turns, timestamp
 * - Per-turn data: 12 features + the action taken + current player index
 *
 * Both bots use BotBrain with exploration noise for diverse training data.
 * Games are capped at MAX_TURNS to prevent infinite loops.
 */
public class GameSimulator {

    private static final int MAX_TURNS = 200; // safety cap

    /**
     * Simulates one complete game and returns the record.
     * Both players use BotBrain with some randomness for varied games.
     */
    public static GameRecord simulate() {
        GameState state = new GameState("Bot0", "Bot1");

        // Both bots get exploration noise for training diversity
        BotBrain bot0 = new BotBrain(0.0, 4); // 4 random opening moves
        bot0.setSilent(true);
        BotBrain bot1 = new BotBrain(0.0, 4);
        bot1.setSilent(true);
        WallEvaluator.silent = true;
        BotBrain[] bots = { bot0, bot1 };

        List<TurnRecord> turns = new ArrayList<>();

        while (!state.isGameOver() && state.getTurnCount() < MAX_TURNS) {
            int currentIdx = state.getCurrentPlayerIndex();

            // Extract features BEFORE the action (captures the board state the bot sees)
            double[] features = GameFeatures.extract(state);

            // Bot decides its action
            BotBrain.BotAction action = bots[currentIdx].computeBestAction(state);

            // Record the action details
            ActionRecord actionRecord;
            if (action.type == BotBrain.BotAction.Type.MOVE) {
                actionRecord = ActionRecord.move(
                    action.moveTarget.getRow(),
                    action.moveTarget.getCol()
                );
            } else {
                actionRecord = ActionRecord.wall(
                    action.wallToPlace.getRow(),
                    action.wallToPlace.getCol(),
                    action.wallToPlace.isHorizontal() ? "H" : "V"
                );
            }

            // Store turn data
            turns.add(new TurnRecord(currentIdx, features, actionRecord));

            // Apply the action to the game state
            applyAction(state, action);

            // Check win condition
            state.checkWinCondition();
        }

        // Determine winner
        int winnerIndex = -1; // draw/timeout
        if (state.isGameOver() && state.getWinner() != null) {
            winnerIndex = (state.getWinner() == state.getPlayer(0)) ? 0 : 1;
        }

        return new GameRecord(winnerIndex, turns.size(), turns);
    }

    /**
     * Applies a BotAction to the game state (move or wall placement).
     */
    private static void applyAction(GameState state, BotBrain.BotAction action) {
        if (action.type == BotBrain.BotAction.Type.MOVE) {
            state.getCurrentPlayer().setPosition(action.moveTarget);
        } else {
            state.addWall(action.wallToPlace);
        }
        state.nextTurn();
    }

    // ===================== DATA CLASSES =====================

    /**
     * Complete record of one game.
     */
    public static class GameRecord {
        public final int winnerIndex;   // 0, 1, or -1 for draw/timeout
        public final int totalTurns;
        public final List<TurnRecord> turns;

        public GameRecord(int winnerIndex, int totalTurns, List<TurnRecord> turns) {
            this.winnerIndex = winnerIndex;
            this.totalTurns = totalTurns;
            this.turns = turns;
        }
    }

    /**
     * Record of a single turn: who played, board features, and what they did.
     */
    public static class TurnRecord {
        public final int currentPlayer;
        public final double[] features;    // 12 features
        public final ActionRecord action;

        public TurnRecord(int currentPlayer, double[] features, ActionRecord action) {
            this.currentPlayer = currentPlayer;
            this.features = features;
            this.action = action;
        }
    }

    /**
     * Record of an action: either a move or a wall placement.
     */
    public static class ActionRecord {
        public final String type;    // "move" or "wall"
        public final int row;
        public final int col;
        public final String orientation; // "H" or "V" for walls, null for moves

        private ActionRecord(String type, int row, int col, String orientation) {
            this.type = type;
            this.row = row;
            this.col = col;
            this.orientation = orientation;
        }

        public static ActionRecord move(int row, int col) {
            return new ActionRecord("move", row, col, null);
        }

        public static ActionRecord wall(int row, int col, String orientation) {
            return new ActionRecord("wall", row, col, orientation);
        }
    }
}
