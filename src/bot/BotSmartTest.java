package bot;

import logic.MoveValidator;
import logic.PathFinder;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.*;

/**
 * Advanced bot testing: Bot vs Bot (same algorithm) and Bot vs Smart opponent.
 *
 * Tests the bot in challenging scenarios:
 *   1. Bot vs Bot: Both sides use BotBrain — tests symmetry and wall wars
 *   2. Bot vs Strategic Human: Human uses A* + aggressive wall placement
 *   3. Specific scenarios: Tests critical game situations
 */
public class BotSmartTest {

    public static void main(String[] args) {
        System.out.println("========== SMART BOT TEST v5 ==========\n");

        // Test 1: Bot vs Bot (same BotBrain algorithm)
        System.out.println("=== TEST 1: Bot vs Bot (10 games) ===");
        System.out.println("Both players use BotBrain. Player 1 has tempo advantage.");
        runBotVsBot(10);

        System.out.println();

        // Test 2: Bot vs Aggressive Wall Opponent
        System.out.println("=== TEST 2: Bot vs AGGRESSIVE wall opponent (10 games) ===");
        runBotVsAggressiveWaller(10);

        System.out.println();

        // Test 3: Critical scenarios
        System.out.println("=== TEST 3: Critical Scenarios ===");
        testCriticalScenarios();

        System.out.println();

        // Test 4: Bot vs Random-path human (non-deterministic)
        System.out.println("=== TEST 4: Bot vs RANDOM-PATH opponent (20 games) ===");
        runBotVsRandomPath(20);

        System.out.println();

        // Test 5: EXPLORATION bots quality check (ML training mode)
        System.out.println("=== TEST 5: Exploration Bot QUALITY CHECK ===");
        System.out.println("Bots use noise=0.0, openingRandom=3 (same as Bot vs Bot UI)");
        runExplorationQualityTest();
    }

    // =================================================================
    // TEST 1: Bot vs Bot
    // =================================================================
    private static void runBotVsBot(int totalGames) {
        int p1Wins = 0, p2Wins = 0, draws = 0;

        for (int game = 1; game <= totalGames; game++) {
            GameState state = new GameState("BotA", "BotB");
            BotBrain brainA = new BotBrain();
            BotBrain brainB = new BotBrain();
            int maxTurns = 200;

            String winner = "DRAW";
            int finalTurn = maxTurns;

            for (int turn = 1; turn <= maxTurns; turn++) {
                Player current = state.getCurrentPlayer();
                int playerIdx = state.getCurrentPlayerIndex();

                try {
                    BotBrain brain = (playerIdx == 0) ? brainA : brainB;
                    BotBrain.BotAction action = brain.computeBestAction(state);

                    if (action == null) {
                        state.nextTurn();
                        continue;
                    }

                    if (action.type == BotBrain.BotAction.Type.MOVE) {
                        current.setPosition(action.moveTarget);
                    } else {
                        state.addWall(action.wallToPlace);
                    }

                    state.checkWinCondition();
                    if (state.isGameOver()) {
                        winner = state.getWinner().getName();
                        finalTurn = turn;
                        break;
                    }

                    state.nextTurn();
                } catch (Exception e) {
                    System.out.println("  CRASH game " + game + " turn " + turn + ": " + e.getMessage());
                    winner = "CRASH";
                    break;
                }
            }

            if (winner.equals("BotA")) p1Wins++;
            else if (winner.equals("BotB")) p2Wins++;
            else draws++;

            System.out.println("  Game " + game + ": " + winner + " wins in " + finalTurn + " turns"
                    + " (walls: " + state.getWalls().size() + ")"
                    + " P1walls=" + state.getPlayer(0).getWallsRemaining()
                    + " P2walls=" + state.getPlayer(1).getWallsRemaining());
        }

        System.out.println("  --- Results ---");
        System.out.println("  Player 1 (tempo advantage): " + p1Wins + "/" + totalGames);
        System.out.println("  Player 2:                    " + p2Wins + "/" + totalGames);
        System.out.println("  Draws:                       " + draws + "/" + totalGames);
    }

    // =================================================================
    // TEST 2: Bot vs Aggressive Wall Opponent
    // =================================================================
    private static void runBotVsAggressiveWaller(int totalGames) {
        int botWins = 0, humanWins = 0, draws = 0;

        for (int game = 1; game <= totalGames; game++) {
            GameState state = new GameState("AggWaller", "Bot");
            BotBrain brain = new BotBrain();
            int maxTurns = 200;
            int humanWallsPlaced = 0;

            String winner = "DRAW";
            int finalTurn = maxTurns;

            for (int turn = 1; turn <= maxTurns; turn++) {
                Player current = state.getCurrentPlayer();
                int playerIdx = state.getCurrentPlayerIndex();

                try {
                    if (playerIdx == 0) {
                        // Aggressive waller: places walls every other turn, targeting bot's path
                        boolean placed = false;
                        if (current.hasWalls() && humanWallsPlaced < 8 && turn % 2 == 1) {
                            placed = placeAggressiveWall(state, current);
                            if (placed) humanWallsPlaced++;
                        }
                        if (!placed) {
                            Position move = pickSmartMove(state, current);
                            if (move != null) current.setPosition(move);
                        }
                    } else {
                        // Bot
                        BotBrain.BotAction action = brain.computeBestAction(state);
                        if (action == null) {
                            state.nextTurn();
                            continue;
                        }
                        if (action.type == BotBrain.BotAction.Type.MOVE) {
                            current.setPosition(action.moveTarget);
                        } else {
                            state.addWall(action.wallToPlace);
                        }
                    }

                    state.checkWinCondition();
                    if (state.isGameOver()) {
                        winner = state.getWinner().getName();
                        finalTurn = turn;
                        break;
                    }

                    state.nextTurn();
                } catch (Exception e) {
                    System.out.println("  CRASH game " + game + " turn " + turn + ": " + e.getMessage());
                    winner = "CRASH";
                    break;
                }
            }

            if (winner.equals("Bot")) botWins++;
            else if (winner.equals("AggWaller")) humanWins++;
            else draws++;

            System.out.println("  Game " + game + ": " + winner + " in " + finalTurn + " turns"
                    + " (total walls: " + state.getWalls().size() + ")");
        }

        System.out.println("  --- Results ---");
        System.out.println("  Bot wins:      " + botWins + "/" + totalGames);
        System.out.println("  Opponent wins: " + humanWins + "/" + totalGames);
        System.out.println("  Draws:         " + draws + "/" + totalGames);
    }

    // =================================================================
    // TEST 3: Critical Scenarios
    // =================================================================
    private static void testCriticalScenarios() {
        // Scenario A: Opponent 1 step from winning, bot has walls
        System.out.println("\n  --- Scenario A: Opponent 1 step from winning ---");
        {
            GameState state = new GameState("Human", "Bot");
            state.getPlayer(0).setPosition(new Position(1, 4)); // Human 1 step from goal (row 0)
            state.getPlayer(1).setPosition(new Position(5, 4)); // Bot far from goal
            state.nextTurn(); // Bot's turn

            BotBrain brain = new BotBrain();
            BotBrain.BotAction action = brain.computeBestAction(state);
            System.out.println("    Bot at (5,4), Opp at (1,4) 1 step from winning");
            System.out.println("    Decision: " + action);
            System.out.println("    Expected: WALL (must block!)");
            System.out.println("    " + (action != null && action.type == BotBrain.BotAction.Type.WALL ? "PASS" : "FAIL"));
        }

        // Scenario B: Opponent 2 steps from winning, bot has walls
        System.out.println("\n  --- Scenario B: Opponent 2 steps from winning ---");
        {
            GameState state = new GameState("Human", "Bot");
            state.getPlayer(0).setPosition(new Position(2, 4)); // Human 2 steps from goal
            state.getPlayer(1).setPosition(new Position(5, 4));
            state.nextTurn();

            BotBrain brain = new BotBrain();
            BotBrain.BotAction action = brain.computeBestAction(state);
            System.out.println("    Bot at (5,4), Opp at (2,4) 2 steps from winning");
            System.out.println("    Decision: " + action);
            System.out.println("    Expected: WALL (should block!)");
            System.out.println("    " + (action != null && action.type == BotBrain.BotAction.Type.WALL ? "PASS" : "FAIL"));
        }

        // Scenario C: Bot 1 step from winning (should ALWAYS move)
        System.out.println("\n  --- Scenario C: Bot 1 step from winning ---");
        {
            GameState state = new GameState("Human", "Bot");
            state.getPlayer(1).setPosition(new Position(7, 4)); // Bot 1 step from goal (row 8)
            state.getPlayer(0).setPosition(new Position(2, 4));
            state.nextTurn();

            BotBrain brain = new BotBrain();
            BotBrain.BotAction action = brain.computeBestAction(state);
            System.out.println("    Bot at (7,4) 1 step from winning");
            System.out.println("    Decision: " + action);
            System.out.println("    Expected: MOVE (instant win!)");
            System.out.println("    " + (action != null && action.type == BotBrain.BotAction.Type.MOVE ? "PASS" : "FAIL"));
        }

        // Scenario D: Even race, straight corridor — bot should use walls
        System.out.println("\n  --- Scenario D: Even race, both at distance 5 ---");
        {
            GameState state = new GameState("Human", "Bot");
            state.getPlayer(0).setPosition(new Position(5, 4)); // Human dist=5 to row 0
            state.getPlayer(1).setPosition(new Position(3, 4)); // Bot dist=5 to row 8
            state.nextTurn();

            BotBrain brain = new BotBrain();
            BotBrain.BotAction action = brain.computeBestAction(state);
            System.out.println("    Both at distance 5 — even race");
            System.out.println("    Decision: " + action);
            System.out.println("    Note: Bot (P2) has tempo disadvantage, should wall to compensate");
        }

        // Scenario E: Bot no walls left — must move
        System.out.println("\n  --- Scenario E: Bot has 0 walls ---");
        {
            GameState state = new GameState("Human", "Bot");
            state.getPlayer(1).setPosition(new Position(4, 4));
            state.getPlayer(1).setWallsRemaining(0);
            state.nextTurn();

            BotBrain brain = new BotBrain();
            BotBrain.BotAction action = brain.computeBestAction(state);
            System.out.println("    Bot at (4,4), 0 walls remaining");
            System.out.println("    Decision: " + action);
            System.out.println("    Expected: MOVE (no walls available)");
            System.out.println("    " + (action != null && action.type == BotBrain.BotAction.Type.MOVE ? "PASS" : "FAIL"));
        }
    }

    // =================================================================
    // TEST 4: Bot vs Random-path Human
    // =================================================================
    private static void runBotVsRandomPath(int totalGames) {
        Random rng = new Random(42);
        int botWins = 0, humanWins = 0, draws = 0;

        for (int game = 1; game <= totalGames; game++) {
            GameState state = new GameState("RandHuman", "Bot");
            BotBrain brain = new BotBrain();
            int maxTurns = 200;

            String winner = "DRAW";
            int finalTurn = maxTurns;

            for (int turn = 1; turn <= maxTurns; turn++) {
                Player current = state.getCurrentPlayer();
                int playerIdx = state.getCurrentPlayerIndex();

                try {
                    if (playerIdx == 0) {
                        // Random-path human: sometimes picks A* move, sometimes random valid move
                        // Also places walls randomly sometimes
                        Position move = null;
                        List<Position> validMoves = MoveValidator.getValidMoves(state, current);

                        // 20% chance to place a random valid wall
                        if (rng.nextDouble() < 0.2 && current.hasWalls()) {
                            Wall randomWall = pickRandomWall(state, rng);
                            if (randomWall != null) {
                                state.addWall(randomWall);
                                state.checkWinCondition();
                                if (state.isGameOver()) {
                                    winner = state.getWinner().getName();
                                    finalTurn = turn;
                                    break;
                                }
                                state.nextTurn();
                                continue;
                            }
                        }

                        if (rng.nextDouble() < 0.6) {
                            // 60% A* guided move
                            move = pickSmartMove(state, current);
                        } else {
                            // 40% random valid move
                            if (!validMoves.isEmpty()) {
                                move = validMoves.get(rng.nextInt(validMoves.size()));
                            }
                        }

                        if (move != null) current.setPosition(move);
                    } else {
                        BotBrain.BotAction action = brain.computeBestAction(state);
                        if (action == null) {
                            state.nextTurn();
                            continue;
                        }
                        if (action.type == BotBrain.BotAction.Type.MOVE) {
                            current.setPosition(action.moveTarget);
                        } else {
                            state.addWall(action.wallToPlace);
                        }
                    }

                    state.checkWinCondition();
                    if (state.isGameOver()) {
                        winner = state.getWinner().getName();
                        finalTurn = turn;
                        break;
                    }
                    state.nextTurn();
                } catch (Exception e) {
                    System.out.println("  CRASH game " + game + " turn " + turn + ": " + e.getMessage());
                    winner = "CRASH";
                    break;
                }
            }

            if (winner.equals("Bot")) botWins++;
            else if (winner.equals("RandHuman")) humanWins++;
            else draws++;

            System.out.println("  Game " + game + ": " + winner + " in " + finalTurn + " turns");
        }

        System.out.println("  --- Results ---");
        System.out.println("  Bot wins:      " + botWins + "/" + totalGames);
        System.out.println("  Opponent wins: " + humanWins + "/" + totalGames);
        System.out.println("  Draws:         " + draws + "/" + totalGames);
    }

    // =================================================================
    // TEST 5: Exploration Bot Quality Check
    // Verifies that bots with exploration noise still play at a HIGH level.
    // =================================================================
    private static void runExplorationQualityTest() {
        // Part A: Exploration bot vs AggWaller (must still win!)
        System.out.println("\n  --- Part A: Exploration Bot vs AggWaller (10 games) ---");
        int botWins = 0, humanWins = 0;
        for (int game = 1; game <= 10; game++) {
            GameState state = new GameState("AggWaller", "ExploBot");
            BotBrain brain = new BotBrain(0.0, 3); // Same params as UI
            int maxTurns = 200;
            int humanWallsPlaced = 0;
            String winner = "DRAW";
            int finalTurn = maxTurns;

            for (int turn = 1; turn <= maxTurns; turn++) {
                Player current = state.getCurrentPlayer();
                int playerIdx = state.getCurrentPlayerIndex();
                try {
                    if (playerIdx == 0) {
                        boolean placed = false;
                        if (current.hasWalls() && humanWallsPlaced < 8 && turn % 2 == 1) {
                            placed = placeAggressiveWall(state, current);
                            if (placed) humanWallsPlaced++;
                        }
                        if (!placed) {
                            Position move = pickSmartMove(state, current);
                            if (move != null) current.setPosition(move);
                        }
                    } else {
                        BotBrain.BotAction action = brain.computeBestAction(state);
                        if (action == null) { state.nextTurn(); continue; }
                        if (action.type == BotBrain.BotAction.Type.MOVE) {
                            current.setPosition(action.moveTarget);
                        } else {
                            state.addWall(action.wallToPlace);
                        }
                    }
                    state.checkWinCondition();
                    if (state.isGameOver()) { winner = state.getWinner().getName(); finalTurn = turn; break; }
                    state.nextTurn();
                } catch (Exception e) {
                    System.out.println("  CRASH game " + game + ": " + e.getMessage());
                    winner = "CRASH"; break;
                }
            }
            if (winner.equals("ExploBot")) botWins++;
            else if (winner.equals("AggWaller")) humanWins++;
            System.out.println("    Game " + game + ": " + winner + " in " + finalTurn + " turns");
        }
        System.out.println("    ExploBot wins: " + botWins + "/10 (target: >= 7 for quality)");
        System.out.println("    " + (botWins >= 7 ? "PASS" : "WARN — exploration may be too noisy"));

        // Part B: Bot vs Bot diversity check (10 games, check all are different)
        System.out.println("\n  --- Part B: Exploration Bot vs Bot DIVERSITY (10 games) ---");
        Set<String> gameFingerprints = new HashSet<>();
        int totalTurns = 0;
        int wallGames = 0; // games with any walls placed
        for (int game = 1; game <= 10; game++) {
            GameState state = new GameState("ExploA", "ExploB");
            BotBrain brainA = new BotBrain(0.0, 3);
            BotBrain brainB = new BotBrain(0.0, 3);
            int maxTurns = 200;
            String winner = "DRAW";
            int finalTurn = maxTurns;

            // Track move sequence for fingerprint
            StringBuilder moveSeq = new StringBuilder();

            for (int turn = 1; turn <= maxTurns; turn++) {
                int playerIdx = state.getCurrentPlayerIndex();
                Player current = state.getCurrentPlayer();
                try {
                    BotBrain brain = (playerIdx == 0) ? brainA : brainB;
                    BotBrain.BotAction action = brain.computeBestAction(state);
                    if (action == null) { state.nextTurn(); continue; }
                    if (action.type == BotBrain.BotAction.Type.MOVE) {
                        current.setPosition(action.moveTarget);
                        moveSeq.append("M").append(action.moveTarget).append(";");
                    } else {
                        state.addWall(action.wallToPlace);
                        moveSeq.append("W").append(action.wallToPlace).append(";");
                    }
                    state.checkWinCondition();
                    if (state.isGameOver()) { winner = state.getWinner().getName(); finalTurn = turn; break; }
                    state.nextTurn();
                } catch (Exception e) {
                    System.out.println("  CRASH game " + game + ": " + e.getMessage());
                    winner = "CRASH"; break;
                }
            }
            totalTurns += finalTurn;
            if (state.getWalls().size() > 0) wallGames++;
            String fingerprint = moveSeq.toString();
            gameFingerprints.add(fingerprint);
            System.out.println("    Game " + game + ": " + winner + " in " + finalTurn + " turns"
                    + " (walls: " + state.getWalls().size() + ")");
        }
        int uniqueGames = gameFingerprints.size();
        double avgTurns = totalTurns / 10.0;
        System.out.println("    Unique games: " + uniqueGames + "/10 (must be >= 7 for diversity)");
        System.out.println("    Avg turns: " + String.format("%.1f", avgTurns));
        System.out.println("    Games with walls: " + wallGames + "/10");
        System.out.println("    " + (uniqueGames >= 7 ? "PASS" : "FAIL — not enough diversity!"));
    }

    // =================================================================
    // Helper methods
    // =================================================================

    private static Position pickSmartMove(GameState state, Player current) {
        List<Position> path = PathFinder.getAStarPath(state, current);
        List<Position> validMoves = MoveValidator.getValidMoves(state, current);

        if (validMoves.isEmpty()) return null;

        if (path != null && path.size() >= 2) {
            Position nextStep = path.get(1);
            if (validMoves.contains(nextStep)) return nextStep;
        }

        // Fallback: pick move that minimizes path
        Position bestMove = null;
        int bestPath = Integer.MAX_VALUE;
        for (Position m : validMoves) {
            Position orig = current.getPosition();
            current.setPosition(m);
            int pLen = PathFinder.aStarShortestPath(state, current);
            current.setPosition(orig);
            if (pLen >= 0 && pLen < bestPath) {
                bestPath = pLen;
                bestMove = m;
            }
        }
        return bestMove;
    }

    /**
     * Aggressive wall placement: finds the wall that maximizes damage to the bot.
     */
    private static boolean placeAggressiveWall(GameState state, Player human) {
        Player bot = state.getOtherPlayer();
        int botPathBefore = PathFinder.aStarShortestPath(state, bot);
        if (botPathBefore < 0) return false;

        int humanPathBefore = PathFinder.aStarShortestPath(state, human);

        Wall bestWall = null;
        int bestNetDamage = 0;

        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                for (Wall.Orientation orient : Wall.Orientation.values()) {
                    Wall candidate = new Wall(r, c, orient);
                    if (!WallValidator.isValidWallPlacement(state, candidate)) continue;

                    int botPathAfter = PathFinder.aStarWithWall(state, bot, candidate);
                    if (botPathAfter < 0) continue;

                    int humanPathAfter = PathFinder.aStarWithWall(state, human, candidate);
                    if (humanPathAfter < 0) continue;

                    int damage = botPathAfter - botPathBefore;
                    int selfHarm = humanPathAfter - humanPathBefore;
                    int netDamage = damage - selfHarm;

                    if (netDamage > bestNetDamage) {
                        bestNetDamage = netDamage;
                        bestWall = candidate;
                    }
                }
            }
        }

        if (bestWall != null && bestNetDamage >= 1) {
            state.addWall(bestWall);
            return true;
        }
        return false;
    }

    private static Wall pickRandomWall(GameState state, Random rng) {
        List<Wall> validWalls = new ArrayList<>();
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                for (Wall.Orientation orient : Wall.Orientation.values()) {
                    Wall w = new Wall(r, c, orient);
                    if (WallValidator.isValidWallPlacement(state, w)) {
                        validWalls.add(w);
                    }
                }
            }
        }
        if (validWalls.isEmpty()) return null;
        return validWalls.get(rng.nextInt(validWalls.size()));
    }
}
