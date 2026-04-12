package bot;

import logic.FlowCalculator;
import logic.MoveValidator;
import logic.PathFinder;
import model.GameState;
import model.Player;
import model.Position;

import java.util.List;

// Scores pawn moves. Row advancement is the most important factor.
public class MoveEvaluator {

    // row advance is the biggest factor - always prefer going forward
    private static final double W_ROW_ADVANCE = 50.0;
    // partial bonus when forward move slightly increases path length
    private static final double W_ROW_ADVANCE_PARTIAL = 25.0;
    // path gain from A*
    private static final double W_PATH_GAIN = 10.0;
    // tiebreaker for following A* path
    private static final double W_ASTAR_TIEBREAK = 5.0;
    // minor tiebreakers
    private static final double W_SAFETY = 0.15;
    private static final double W_FLEXIBILITY = 0.1;
    private static final double W_OPPONENT_THREAT = 0.2;

    // jump-related weights
    private static final double W_JUMP_SETUP = 20.0;
    private static final double W_GIVE_JUMP_PENALTY = 35.0;
    private static final double W_DIRECT_JUMP = 25.0;

    /** A move target with its computed score. */
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

    /** Finds the best move (deterministic). */
    public static ScoredMove findBestMove(GameState state, BoardGraph graph) {
        return findBestMove(state, graph, null);
    }

    /** Finds the best move. botBrain param is for API compat, selection is deterministic. */
    public static ScoredMove findBestMove(GameState state, BoardGraph graph, BotBrain botBrain) {
        Player bot = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();
        List<Position> validMoves = MoveValidator.getValidMoves(state, bot);

        if (validMoves.isEmpty()) return null;

        // pre-compute current state stuff
        int currentPath = PathFinder.aStarShortestPath(state, bot);
        double currentSafety = PathFinder.dijkstraSafestPath(graph, bot.getPosition(), bot.getGoalRow());
        int currentFlow = FlowCalculator.calculateMaxFlow(state, bot);

        // get A* path for tiebreaking
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

    /** Scores a single move. Forward movement dominates, everything else is a tiebreaker. */
    private static double evaluateMove(GameState state, BoardGraph graph,
                                        Player bot, Player opponent,
                                        Position target,
                                        int currentPath, double currentSafety,
                                        int currentFlow,
                                        Position nextStepOnPath) {

        Position botPos = bot.getPosition();
        Position oppPos = opponent.getPosition();

        // simulate the move
        GameState simState = state.deepCopy();
        Player simBot = simState.getCurrentPlayer();
        Player simOpponent = simState.getOtherPlayer();
        simBot.setPosition(target);

        // path gain from A*
        int newPath = PathFinder.aStarShortestPath(simState, simBot);

        // winning move
        if (newPath == 0) return 1000.0;

        // no path - really bad
        if (newPath < 0) return -500.0;
        if (currentPath < 0) currentPath = 20;

        double pathGain = currentPath - newPath;

        // row advancement - the most important factor
        int rowAdvancement = computeRowAdvancement(botPos, target, bot.getGoalRow());

        double rowAdvanceBonus = 0;
        if (rowAdvancement > 0) {
            if (pathGain >= 0) {
                // best case: forward and path gets shorter
                rowAdvanceBonus = W_ROW_ADVANCE;
            } else if (pathGain >= -2) {
                // forward but path gets a bit worse, still better than sideways
                rowAdvanceBonus = W_ROW_ADVANCE_PARTIAL;
            } else {
                // dead end, don't reward
                rowAdvanceBonus = 0;
            }
        }

        // bonus for jump moves (2 rows at once)
        double jumpBonus = 0.0;
        if (Math.abs(rowAdvancement) == 2) {
            jumpBonus = W_DIRECT_JUMP;
        }

        // A* path tiebreaker
        double astarTiebreak = 0;
        if (nextStepOnPath != null && target.equals(nextStepOnPath)) {
            astarTiebreak = W_ASTAR_TIEBREAK;
        }

        // safety from dijkstra weighted path
        BoardGraph simGraph = new BoardGraph();
        simGraph.buildFromState(simState, simOpponent.getPosition());
        double newSafety = PathFinder.dijkstraSafestPath(simGraph, target, simBot.getGoalRow());
        double safetyScore = 0;
        if (newSafety > 0 && currentSafety > 0) {
            safetyScore = currentSafety - newSafety;
        }

        // flexibility from max flow
        int newFlow = FlowCalculator.calculateMaxFlow(simState, simBot);
        double flexScore = newFlow - currentFlow;

        // opponent response lookahead
        double opponentThreat = evaluateOpponentResponse(simState, simOpponent);

        // penalize giving opponent a jump
        double jumpPenalty = evaluateJumpRisk(state, bot, opponent, target);

        // bonus for setting up our own jump next turn
        double jumpSetup = evaluateJumpSetup(simState, simBot, simOpponent);

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

    // Penalty for giving opponent a jump opportunity
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

        // if opponent is behind us they can jump over us toward their goal
        boolean oppCanJumpOverUs = false;
        if (rowDiff == 1 && colDiff == 0) {
            // same column, one row apart - check if jump goes toward their goal
            int jumpRow = target.getRow() + (target.getRow() - oppPos.getRow());
            if ((jumpRow - oppPos.getRow()) * oppDirection > 0) {
                oppCanJumpOverUs = true;
            }
        }

        if (oppCanJumpOverUs) {
            int botDist = PathFinder.aStarShortestPath(state, bot);
            int oppDist = PathFinder.aStarShortestPath(state, opponent);
            int raceGap = oppDist - botDist;

            // giving a jump when behind or tied is really bad
            if (raceGap <= 0) {
                return W_GIVE_JUMP_PENALTY;
            } else if (raceGap == 1) {
                return W_GIVE_JUMP_PENALTY * 0.7;
            } else if (raceGap == 2) {
                return W_GIVE_JUMP_PENALTY * 0.4;
            } else {
                return W_GIVE_JUMP_PENALTY * 0.2;
            }
        }

        return 0.0;
    }

    // Bonus if this move sets up a jump for next turn
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
            // check if opponent is between us and our goal
            int oppRelativePos = (oppPos.getRow() - newPos.getRow()) * botDirection;
            if (oppRelativePos > 0) {
                // opponent is in the way - check if jump square is in bounds
                int jumpRow = oppPos.getRow() + botDirection;
                if (jumpRow >= 0 && jumpRow <= 8) {
                    return W_JUMP_SETUP;
                }
            }
        }

        return 0.0;
    }

    // How many rows closer to goal: +1 forward, 0 sideways, -1 back
    private static int computeRowAdvancement(Position from, Position to, int goalRow) {
        if (goalRow > from.getRow()) {
            // goal is below
            return to.getRow() - from.getRow();
        } else {
            // goal is above
            return from.getRow() - to.getRow();
        }
    }

    // How much the opponent gains from their best response move
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
