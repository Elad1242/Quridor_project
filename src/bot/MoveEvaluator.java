package bot;

import logic.FlowCalculator;
import logic.MoveValidator;
import logic.PathFinder;
import model.GameState;
import model.Player;
import model.Position;

import java.util.List;

/**
 * Evaluates pawn movement options for the bot.
 *
 * CRITICAL PRINCIPLE: ALWAYS ADVANCE TOWARD GOAL ROW.
 *   The bot MUST move toward its goal row whenever possible.
 *   Row advancement is the dominant scoring factor — it CANNOT be overridden
 *   by safety, flexibility, or A* path following.
 *
 *   This prevents the catastrophic "sideways oscillation" bug where the bot
 *   followed A* step-by-step, but A* changed every turn due to opponent walls,
 *   causing the bot to zigzag on the same row for 8+ turns.
 *
 * SCORING HIERARCHY:
 *   1. ROW ADVANCE: 50.0 for moves that advance toward goal row (DOMINANT)
 *      - Only given when the move actually goes to a closer row AND doesn't
 *        make botDist catastrophically worse (pathGain >= -2)
 *   2. PATH GAIN: 10.0 per step of botDist reduction (STRONG)
 *      - Ensures moves that reduce shortest-path distance are preferred
 *   3. A* TIEBREAK: 5.0 if move is the next step on A* path (TIEBREAKER)
 *      - Among equal moves, prefer the one A* recommends
 *   4. SAFETY: 0.15 bonus for safer positions (TIEBREAKER)
 *   5. FLEXIBILITY: 0.1 bonus for more escape routes (TIEBREAKER)
 *   6. OPPONENT: 0.2 penalty if move lets opponent advance (TIEBREAKER)
 *
 * EXAMPLES:
 *   Forward move on A* path, pathGain=1:  50 + 10 + 5 + bonuses ≈ 65
 *   Forward move off A* path, pathGain=1: 50 + 10 + 0 + bonuses ≈ 60
 *   Forward move, pathGain=-1 (minor dead end): 25 - 10 + bonuses ≈ 15
 *   Sideways move on A* path, pathGain=1: 0 + 10 + 5 + bonuses ≈ 15
 *   Sideways move, pathGain=0: 0 + 0 + 5 + bonuses ≈ 5
 *   Retreat: 0 - 10 + bonuses ≈ -10
 *
 *   This ensures: forward > sideways (always), forward+A* > forward (tiebreak).
 */
public class MoveEvaluator {

    // ROW ADVANCE is the #1 factor — moving toward the goal row is paramount.
    // This ensures a row-advancing move ALWAYS beats a sideways move,
    // preventing the oscillation bug where A* path-following caused zigzagging.
    private static final double W_ROW_ADVANCE = 50.0;

    // Partial row advance bonus when the forward move increases botDist slightly.
    // This is still better than going sideways, but not as good as optimal forward.
    private static final double W_ROW_ADVANCE_PARTIAL = 25.0;

    // PATH GAIN is the #2 factor — each step of botDist reduction is worth 10 points.
    private static final double W_PATH_GAIN = 10.0;

    // A* path tiebreaker — demoted from dominant to tiebreaker.
    // Among moves with equal row advancement and path gain, prefer the A* step.
    private static final double W_ASTAR_TIEBREAK = 5.0;

    // Tiebreakers — only matter when comparing moves with the same major scores.
    private static final double W_SAFETY = 0.15;
    private static final double W_FLEXIBILITY = 0.1;
    private static final double W_OPPONENT_THREAT = 0.2;

    // === NEW: JUMP-RELATED WEIGHTS ===
    // JUMP CREATION: Big bonus for moves that set up a jump opportunity next turn
    private static final double W_JUMP_SETUP = 20.0;

    // JUMP AVOIDANCE: Penalty for moves that let opponent jump over us
    private static final double W_GIVE_JUMP_PENALTY = 35.0;

    // DIRECT JUMP: Huge bonus for actually executing a jump (2 row advancement)
    private static final double W_DIRECT_JUMP = 25.0;

    /**
     * Result of evaluating a move: the target position and its score.
     */
    public static class ScoredMove {
        public final Position target;
        public final double score;

        public ScoredMove(Position target, double score) {
            this.target = target;
            this.score = score;
        }

        @Override
        public String toString() {
            return "Move to " + target + " (score: " + String.format("%.2f", score) + ")";
        }
    }

    /**
     * Evaluates all valid moves and returns the best one (deterministic version).
     */
    public static ScoredMove findBestMove(GameState state, BoardGraph graph) {
        return findBestMove(state, graph, null);
    }

    /**
     * Evaluates all valid moves and returns the best one.
     * The botBrain parameter is accepted for API compatibility but
     * move selection is fully deterministic (diversity comes from
     * random openings in BotBrain, not from the evaluator).
     */
    public static ScoredMove findBestMove(GameState state, BoardGraph graph, BotBrain botBrain) {
        Player bot = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();
        List<Position> validMoves = MoveValidator.getValidMoves(state, bot);

        if (validMoves.isEmpty()) return null;

        // Pre-compute current state metrics (only computed once per turn)
        int currentPath = PathFinder.aStarShortestPath(state, bot);
        double currentSafety = PathFinder.dijkstraSafestPath(graph, bot.getPosition(), bot.getGoalRow());
        int currentFlow = FlowCalculator.calculateMaxFlow(state, bot);

        // Get the A* shortest path to identify the NEXT optimal step (tiebreaker only)
        List<Position> shortestPath = PathFinder.getAStarPath(state, bot);
        Position nextStepOnPath = null;
        if (shortestPath != null && shortestPath.size() >= 2) {
            nextStepOnPath = shortestPath.get(1); // index 0 is current position
        }

        ScoredMove bestMove = null;

        for (Position target : validMoves) {
            double score = evaluateMove(state, graph, bot, opponent, target,
                                        currentPath, currentSafety, currentFlow,
                                        nextStepOnPath);

            if (bestMove == null || score > bestMove.score) {
                bestMove = new ScoredMove(target, score);
            }
        }

        return bestMove;
    }

    /**
     * Scores a single move. Row advancement dominates; path gain is secondary;
     * A* following and other factors are tiebreakers.
     * Now includes jump awareness for better tactical play.
     */
    private static double evaluateMove(GameState state, BoardGraph graph,
                                        Player bot, Player opponent,
                                        Position target,
                                        int currentPath, double currentSafety,
                                        int currentFlow,
                                        Position nextStepOnPath) {

        Position botPos = bot.getPosition();
        Position oppPos = opponent.getPosition();

        // --- Simulate the move ---
        GameState simState = state.deepCopy();
        Player simBot = simState.getCurrentPlayer();
        Player simOpponent = simState.getOtherPlayer();
        simBot.setPosition(target);

        // --- Factor 1: Path Gain (A*) ---
        int newPath = PathFinder.aStarShortestPath(simState, simBot);

        // Winning move!
        if (newPath == 0) return 1000.0;

        // No path — catastrophic
        if (newPath < 0) return -500.0;
        if (currentPath < 0) currentPath = 20;

        double pathGain = currentPath - newPath; // +1 = advanced, 0 = sideways, -1 = retreated

        // --- Factor 0: Row Advancement (DOMINANT) ---
        int rowAdvancement = computeRowAdvancement(botPos, target, bot.getGoalRow());

        double rowAdvanceBonus = 0;
        if (rowAdvancement > 0) {
            if (pathGain >= 0) {
                // Best case: advancing row AND reducing/maintaining botDist
                rowAdvanceBonus = W_ROW_ADVANCE;
            } else if (pathGain >= -2) {
                // Advancing row but botDist gets slightly worse (minor detour)
                // Still better than going sideways
                rowAdvanceBonus = W_ROW_ADVANCE_PARTIAL;
            } else {
                // Advancing row but botDist gets MUCH worse (dead end)
                // Don't reward this — let path gain handle it
                rowAdvanceBonus = 0;
            }
        }

        // --- JUMP BONUS: Detect if this is a jump move (2 rows advanced) ---
        double jumpBonus = 0.0;
        if (Math.abs(rowAdvancement) == 2) {
            // This is a jump move! Big bonus
            jumpBonus = W_DIRECT_JUMP;
        }

        // --- Factor 2: A* Path Tiebreaker ---
        double astarTiebreak = 0;
        if (nextStepOnPath != null && target.equals(nextStepOnPath)) {
            astarTiebreak = W_ASTAR_TIEBREAK;
        }

        // --- Factor 3: Safety (Dijkstra weighted path cost) ---
        BoardGraph simGraph = new BoardGraph();
        simGraph.buildFromState(simState, simOpponent.getPosition());
        double newSafety = PathFinder.dijkstraSafestPath(simGraph, target, simBot.getGoalRow());
        double safetyScore = 0;
        if (newSafety > 0 && currentSafety > 0) {
            safetyScore = currentSafety - newSafety;
        }

        // --- Factor 4: Flexibility (Max Flow) ---
        int newFlow = FlowCalculator.calculateMaxFlow(simState, simBot);
        double flexScore = newFlow - currentFlow;

        // --- Factor 5: Opponent Response (1-step lookahead) ---
        double opponentThreat = evaluateOpponentResponse(simState, simOpponent);

        // --- NEW Factor 6: Jump Avoidance ---
        // Penalize moves that make us adjacent to opponent and give them a jump
        double jumpPenalty = evaluateJumpRisk(state, bot, opponent, target);

        // --- NEW Factor 7: Jump Setup ---
        // Bonus for moves that position us to jump next turn
        double jumpSetup = evaluateJumpSetup(simState, simBot, simOpponent);

        // --- Combine: Row advance > Path gain > Jump factors > A* tiebreak > rest ---
        double totalScore = rowAdvanceBonus
                          + jumpBonus
                          + W_PATH_GAIN * pathGain
                          + astarTiebreak
                          + W_SAFETY * safetyScore
                          + W_FLEXIBILITY * flexScore
                          - W_OPPONENT_THREAT * opponentThreat
                          - jumpPenalty
                          + jumpSetup;

        return totalScore;
    }

    /**
     * Evaluates the risk of giving opponent a jump opportunity.
     * Returns a penalty score if the move puts us in a bad adjacent position.
     */
    private static double evaluateJumpRisk(GameState state, Player bot, Player opponent, Position target) {
        Position oppPos = opponent.getPosition();
        int botGoalRow = bot.getGoalRow();
        int oppGoalRow = opponent.getGoalRow();

        // Check if move makes us adjacent to opponent
        int rowDiff = Math.abs(target.getRow() - oppPos.getRow());
        int colDiff = Math.abs(target.getCol() - oppPos.getCol());
        boolean adjacent = (rowDiff + colDiff == 1);

        if (!adjacent) return 0.0;

        // We're adjacent - check if this helps opponent
        int oppDirection = (oppGoalRow > oppPos.getRow()) ? 1 : -1;
        int botDirection = (botGoalRow > target.getRow()) ? 1 : -1;

        // CRITICAL: If opponent is directly behind us in their travel direction,
        // they can jump OVER us toward their goal
        boolean oppCanJumpOverUs = false;
        if (rowDiff == 1 && colDiff == 0) {
            // We're in same column, one row apart
            // Check if opponent would jump in their goal direction
            int jumpRow = target.getRow() + (target.getRow() - oppPos.getRow());
            if ((jumpRow - oppPos.getRow()) * oppDirection > 0) {
                // The jump would advance opponent toward their goal
                oppCanJumpOverUs = true;
            }
        }

        if (oppCanJumpOverUs) {
            // Calculate race state
            int botDist = PathFinder.aStarShortestPath(state, bot);
            int oppDist = PathFinder.aStarShortestPath(state, opponent);
            int raceGap = oppDist - botDist; // positive = we're ahead

            // If we're behind or tied, giving a jump is very bad
            if (raceGap <= 0) {
                return W_GIVE_JUMP_PENALTY;
            } else if (raceGap == 1) {
                return W_GIVE_JUMP_PENALTY * 0.7;
            } else if (raceGap == 2) {
                return W_GIVE_JUMP_PENALTY * 0.4;
            } else {
                // We're far ahead, less risky
                return W_GIVE_JUMP_PENALTY * 0.2;
            }
        }

        return 0.0;
    }

    /**
     * Evaluates if the move sets up a jump opportunity for next turn.
     * Returns a bonus if we'll be able to jump over opponent.
     */
    private static double evaluateJumpSetup(GameState simState, Player simBot, Player simOpponent) {
        Position newPos = simBot.getPosition();
        Position oppPos = simOpponent.getPosition();
        int botGoalRow = simBot.getGoalRow();

        // Check if we're now adjacent to opponent
        int rowDiff = Math.abs(newPos.getRow() - oppPos.getRow());
        int colDiff = Math.abs(newPos.getCol() - oppPos.getCol());
        boolean adjacent = (rowDiff + colDiff == 1);

        if (!adjacent) return 0.0;

        // We're adjacent - check if we could jump toward our goal
        int botDirection = (botGoalRow > newPos.getRow()) ? 1 : -1;

        if (rowDiff == 1 && colDiff == 0) {
            // Same column, check if opponent is in our way toward goal
            int oppRelativePos = (oppPos.getRow() - newPos.getRow()) * botDirection;
            if (oppRelativePos > 0) {
                // Opponent is between us and our goal!
                // Check if jump square is valid (in bounds and not blocked by wall)
                int jumpRow = oppPos.getRow() + botDirection;
                if (jumpRow >= 0 && jumpRow <= 8) {
                    // This is a potential jump setup
                    // The actual jump validity depends on walls, but this is a good indicator
                    return W_JUMP_SETUP;
                }
            }
        }

        return 0.0;
    }

    /**
     * Computes how many rows closer to the goal this move gets us.
     *
     * @return +1 if advancing toward goal, 0 if sideways, -1 if retreating
     */
    private static int computeRowAdvancement(Position from, Position to, int goalRow) {
        if (goalRow > from.getRow()) {
            // Goal is below (higher row number): advancing means row increases
            return to.getRow() - from.getRow();
        } else {
            // Goal is above (lower row number): advancing means row decreases
            return from.getRow() - to.getRow();
        }
    }

    /**
     * Estimates how much the opponent benefits from their best response.
     */
    private static double evaluateOpponentResponse(GameState simState, Player simOpponent) {
        int opponentCurrentPath = PathFinder.aStarShortestPath(simState, simOpponent);
        if (opponentCurrentPath < 0) return 0;

        simState.nextTurn();
        List<Position> opponentMoves = MoveValidator.getValidMoves(simState, simOpponent);
        simState.nextTurn();

        int bestOpponentPath = opponentCurrentPath;

        for (Position oppMove : opponentMoves) {
            Position originalPos = simOpponent.getPosition();
            simOpponent.setPosition(oppMove);
            int newPath = PathFinder.aStarShortestPath(simState, simOpponent);
            simOpponent.setPosition(originalPos);

            if (newPath >= 0 && newPath < bestOpponentPath) {
                bestOpponentPath = newPath;
            }
        }

        return opponentCurrentPath - bestOpponentPath;
    }
}
