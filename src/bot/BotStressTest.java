package bot;

import logic.MoveValidator;
import logic.PathFinder;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.List;

/**
 * Stress test: runs multiple full games of bot vs opponents of varying skill.
 * Tests THREE opponent types:
 *   1. GREEDY: Always picks the move closest to the goal row (naive).
 *   2. A*-GUIDED: Follows the A* shortest path, navigating around walls intelligently.
 *   3. A*+WALLS: Follows A* path AND places walls to block the bot (hardest).
 *
 * Checks for crashes, illegal moves, and measures win rate.
 */
public class BotStressTest {

    public static void main(String[] args) {
        System.out.println("========== BOT STRESS TEST ==========\n");

        // --- Round 1: Bot vs Greedy opponent ---
        System.out.println("--- ROUND 1: Bot vs GREEDY opponent (10 games) ---");
        runSuite(10, 0);

        System.out.println();

        // --- Round 2: Bot vs A*-guided opponent (smarter) ---
        System.out.println("--- ROUND 2: Bot vs A*-GUIDED opponent (10 games) ---");
        runSuite(10, 1);

        System.out.println();

        // --- Round 3: Bot vs A*+Walls opponent (hardest) ---
        System.out.println("--- ROUND 3: Bot vs A*+WALLS opponent (10 games) ---");
        runSuite(10, 2);
    }

    private static void runSuite(int totalGames, int opponentType) {
        int botWins = 0;
        int humanWins = 0;
        int draws = 0;

        for (int game = 1; game <= totalGames; game++) {
            String result = playOneGame(game, opponentType);
            if (result.equals("BOT")) botWins++;
            else if (result.equals("HUMAN")) humanWins++;
            else draws++;
        }

        System.out.println("  Bot wins:   " + botWins + "/" + totalGames);
        System.out.println("  Human wins: " + humanWins + "/" + totalGames);
        System.out.println("  Draws:      " + draws + "/" + totalGames);
    }

    private static String playOneGame(int gameNum, int opponentType) {
        GameState state = new GameState("Human", "Bot");
        BotBrain brain = new BotBrain();
        int maxTurns = 200;
        int humanWallsUsed = 0;

        for (int turn = 1; turn <= maxTurns; turn++) {
            Player current = state.getCurrentPlayer();
            int playerIdx = state.getCurrentPlayerIndex();

            try {
                if (playerIdx == 0) {
                    // Human turn
                    boolean placed = false;

                    // A*+Walls opponent: sometimes places walls instead of moving
                    if (opponentType == 2 && current.hasWalls() && humanWallsUsed < 6) {
                        placed = tryPlaceBlockingWall(state, current);
                        if (placed) humanWallsUsed++;
                    }

                    if (!placed) {
                        Position bestMove;
                        if (opponentType >= 1) {
                            bestMove = pickSmartMove(state, current);
                        } else {
                            bestMove = pickGreedyMove(state, current);
                        }

                        if (bestMove != null) {
                            current.setPosition(bestMove);
                        } else {
                            state.nextTurn();
                            continue;
                        }
                    }
                } else {
                    // Bot: full BotBrain computation
                    long startTime = System.currentTimeMillis();
                    BotBrain.BotAction action = brain.computeBestAction(state);
                    long elapsed = System.currentTimeMillis() - startTime;

                    if (elapsed > 5000) {
                        System.out.println("  WARNING: Bot took " + elapsed + "ms on turn " + turn);
                    }

                    if (action == null) {
                        state.nextTurn();
                        continue;
                    }

                    if (action.type == BotBrain.BotAction.Type.MOVE) {
                        List<Position> validMoves = MoveValidator.getValidMoves(state, current);
                        if (!validMoves.contains(action.moveTarget)) {
                            System.out.println("  ERROR: Bot made ILLEGAL MOVE to " + action.moveTarget + " on turn " + turn);
                            System.out.println("  Valid moves: " + validMoves);
                            return "ERROR";
                        }
                        current.setPosition(action.moveTarget);
                    } else {
                        if (!WallValidator.isValidWallPlacement(state, action.wallToPlace)) {
                            System.out.println("  ERROR: Bot placed ILLEGAL WALL " + action.wallToPlace + " on turn " + turn);
                            return "ERROR";
                        }
                        state.addWall(action.wallToPlace);
                    }
                }

                state.checkWinCondition();
                if (state.isGameOver()) {
                    String winner = state.getWinner().getName();
                    System.out.println("  Game " + gameNum + ": " + winner + " wins in " + turn + " turns"
                        + " (walls on board: " + state.getWalls().size() + ")");
                    return winner.equals("Bot") ? "BOT" : "HUMAN";
                }

                state.nextTurn();

            } catch (Exception e) {
                System.out.println("  CRASH on turn " + turn + ": " + e.getMessage());
                e.printStackTrace();
                return "CRASH";
            }
        }

        System.out.println("  Game " + gameNum + ": Draw (reached " + maxTurns + " turns)");
        return "DRAW";
    }

    /**
     * Greedy opponent: always picks the move that gets closest to the goal row.
     */
    private static Position pickGreedyMove(GameState state, Player current) {
        List<Position> moves = MoveValidator.getValidMoves(state, current);
        Position bestMove = null;
        int bestDist = Integer.MAX_VALUE;
        for (Position m : moves) {
            int dist = Math.abs(m.getRow() - current.getGoalRow());
            if (dist < bestDist) {
                bestDist = dist;
                bestMove = m;
            }
        }
        return bestMove;
    }

    /**
     * A*-guided opponent: uses A* shortest path to decide the best next step.
     */
    private static Position pickSmartMove(GameState state, Player current) {
        List<Position> path = PathFinder.getAStarPath(state, current);
        List<Position> validMoves = MoveValidator.getValidMoves(state, current);

        if (validMoves.isEmpty()) return null;

        if (path != null && path.size() >= 2) {
            Position nextStep = path.get(1);
            if (validMoves.contains(nextStep)) {
                return nextStep;
            }
        }

        Position bestMove = null;
        int bestPath = Integer.MAX_VALUE;

        for (Position m : validMoves) {
            Position original = current.getPosition();
            current.setPosition(m);
            int pathLen = PathFinder.aStarShortestPath(state, current);
            current.setPosition(original);

            if (pathLen >= 0 && pathLen < bestPath) {
                bestPath = pathLen;
                bestMove = m;
            }
        }

        return bestMove;
    }

    /**
     * Wall-placing opponent: tries to find and place a wall that maximizes
     * damage to the bot's path. Only places a wall if it adds 2+ steps.
     * Returns true if a wall was placed, false if should move instead.
     */
    private static boolean tryPlaceBlockingWall(GameState state, Player human) {
        Player bot = state.getOtherPlayer();
        int botPathBefore = PathFinder.aStarShortestPath(state, bot);
        if (botPathBefore < 0) return false;

        int humanPathBefore = PathFinder.aStarShortestPath(state, human);

        // Wall if: bot is close to winning, or bot is at same distance or closer than us
        if (botPathBefore > 6 && humanPathBefore <= botPathBefore) return false;

        Wall bestWall = null;
        int bestDamage = 1; // Must add at least 2 steps

        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                for (Wall.Orientation orient : Wall.Orientation.values()) {
                    Wall candidate = new Wall(r, c, orient);
                    if (!WallValidator.isValidWallPlacement(state, candidate)) continue;

                    int botPathAfter = PathFinder.aStarWithWall(state, bot, candidate);
                    if (botPathAfter < 0) continue;

                    int damage = botPathAfter - botPathBefore;

                    // Also check self-harm
                    int humanPathAfter = PathFinder.aStarWithWall(state, human, candidate);
                    if (humanPathAfter < 0) continue;
                    int selfHarm = humanPathAfter - humanPathBefore;

                    // Only place if net positive
                    if (damage - selfHarm > bestDamage) {
                        bestDamage = damage - selfHarm;
                        bestWall = candidate;
                    }
                }
            }
        }

        if (bestWall != null) {
            state.addWall(bestWall);
            return true;
        }

        return false;
    }
}
