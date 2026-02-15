package bot;

import logic.FlowCalculator;
import logic.MoveValidator;
import logic.PathFinder;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.List;

/**
 * Headless test for the bot — runs without JavaFX UI.
 * Simulates a few turns and prints bot decisions to verify correctness.
 *
 * Run: java -cp out/production/Quridor_project bot.BotTest
 * (No JavaFX needed since we bypass the UI)
 */
public class BotTest {

    public static void main(String[] args) {
        System.out.println("=== QUORIDOR BOT TEST ===\n");

        // --- Test 1: Basic bot decision on empty board ---
        System.out.println("--- Test 1: Bot first move (empty board) ---");
        GameState state = new GameState("Human", "Bot");

        // Bot is player 2 (index 1), starts at (0,4), goal row 8
        // Human is player 1 (index 0), starts at (8,4), goal row 0
        // Human moves first, so switch to bot's turn
        state.nextTurn();

        printBoardState(state);

        BotBrain brain1 = new BotBrain();
        BotBrain.BotAction action1 = brain1.computeBestAction(state);
        System.out.println("Bot decision: " + action1);
        System.out.println();

        // --- Test 2: Bot decision after some moves ---
        System.out.println("--- Test 2: Mid-game scenario ---");
        GameState state2 = new GameState("Human", "Bot");

        // Simulate a few moves
        state2.getPlayer(0).setPosition(new Position(6, 4)); // Human advanced 2 rows
        state2.getPlayer(1).setPosition(new Position(2, 4)); // Bot advanced 2 rows
        state2.nextTurn(); // Bot's turn

        printBoardState(state2);

        BotBrain.BotAction action2 = new BotBrain().computeBestAction(state2);
        System.out.println("Bot decision: " + action2);
        System.out.println();

        // --- Test 3: Bot with walls blocking its path ---
        System.out.println("--- Test 3: Bot with wall blocking ---");
        GameState state3 = new GameState("Human", "Bot");
        state3.getPlayer(1).setPosition(new Position(3, 4));
        // Place a horizontal wall blocking bot's path at row 3
        Wall blockWall = new Wall(3, 3, Wall.Orientation.HORIZONTAL);
        blockWall.setOwnerIndex(0);
        state3.getPlayer(0).useWall(); // Account for wall usage
        state3.nextTurn();

        // Need to add wall directly to test
        // We have to use addWall through proper channel
        state3.nextTurn(); // Switch back to player 0
        state3.addWall(blockWall);
        state3.nextTurn(); // Switch to bot (player 1)

        printBoardState(state3);

        BotBrain.BotAction action3 = new BotBrain().computeBestAction(state3);
        System.out.println("Bot decision: " + action3);
        System.out.println();

        // --- Test 4: Bot near goal (should always move) ---
        System.out.println("--- Test 4: Bot near goal ---");
        GameState state4 = new GameState("Human", "Bot");
        state4.getPlayer(1).setPosition(new Position(7, 4)); // 1 step from goal row 8
        state4.nextTurn();

        printBoardState(state4);

        BotBrain.BotAction action4 = new BotBrain().computeBestAction(state4);
        System.out.println("Bot decision: " + action4);
        System.out.println("(Should be MOVE since bot is 1 step from winning)");
        System.out.println();

        // --- Test 5: Path and Flow calculations ---
        System.out.println("--- Test 5: Algorithm outputs ---");
        GameState state5 = new GameState("Human", "Bot");
        Player bot5 = state5.getPlayer(1);

        int astarPath = PathFinder.aStarShortestPath(state5, bot5);
        int bfsPath = PathFinder.getShortestPathLength(state5, bot5);
        int maxFlow = FlowCalculator.calculateMaxFlow(state5, bot5);

        System.out.println("Bot at " + bot5.getPosition() + ", goal row " + bot5.getGoalRow());
        System.out.println("A* shortest path:  " + astarPath + " steps");
        System.out.println("BFS shortest path: " + bfsPath + " steps");
        System.out.println("Max Flow:          " + maxFlow + " independent paths");
        System.out.println("(A* should equal BFS on empty board: both should be 8)");
        System.out.println("(Max Flow should be > 1 on empty board)");
        System.out.println();

        // --- Test 6: Full game simulation (30 turns) ---
        System.out.println("--- Test 6: Simulated game (30 turns) ---");
        GameState gameSim = new GameState("Human", "Bot");
        BotBrain brain = new BotBrain();  // Fresh brain for simulation

        for (int turn = 1; turn <= 30; turn++) {
            Player current = gameSim.getCurrentPlayer();
            int playerIdx = gameSim.getCurrentPlayerIndex();

            if (playerIdx == 0) {
                // Human: just move toward goal (simple strategy)
                List<Position> moves = MoveValidator.getValidMoves(gameSim, current);
                Position bestMove = null;
                int bestDist = Integer.MAX_VALUE;
                for (Position m : moves) {
                    int dist = Math.abs(m.getRow() - current.getGoalRow());
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestMove = m;
                    }
                }
                if (bestMove != null) {
                    current.setPosition(bestMove);
                    System.out.println("Turn " + turn + " [Human] moves to " + bestMove);
                }
            } else {
                // Bot: use BotBrain
                BotBrain.BotAction action = brain.computeBestAction(gameSim);
                if (action != null) {
                    if (action.type == BotBrain.BotAction.Type.MOVE) {
                        current.setPosition(action.moveTarget);
                    } else {
                        gameSim.addWall(action.wallToPlace);
                    }
                    System.out.println("Turn " + turn + " [Bot]   " + action);
                }
            }

            gameSim.checkWinCondition();
            if (gameSim.isGameOver()) {
                System.out.println("GAME OVER! Winner: " + gameSim.getWinner().getName());
                break;
            }
            gameSim.nextTurn();
        }

        System.out.println("\nFinal state:");
        System.out.println("  Human: " + gameSim.getPlayer(0).getPosition()
                         + " | walls left: " + gameSim.getPlayer(0).getWallsRemaining());
        System.out.println("  Bot:   " + gameSim.getPlayer(1).getPosition()
                         + " | walls left: " + gameSim.getPlayer(1).getWallsRemaining());
        System.out.println("  Total walls on board: " + gameSim.getWalls().size());
        System.out.println();

        System.out.println("=== ALL TESTS COMPLETE ===");
    }

    private static void printBoardState(GameState state) {
        Player p0 = state.getPlayer(0);
        Player p1 = state.getPlayer(1);
        System.out.println("  Human (P1): " + p0.getPosition() + " → goal row " + p0.getGoalRow()
                         + " | walls: " + p0.getWallsRemaining());
        System.out.println("  Bot   (P2): " + p1.getPosition() + " → goal row " + p1.getGoalRow()
                         + " | walls: " + p1.getWallsRemaining());
        System.out.println("  Walls on board: " + state.getWalls().size());
        System.out.println("  Current turn: Player " + (state.getCurrentPlayerIndex() + 1));
    }
}
