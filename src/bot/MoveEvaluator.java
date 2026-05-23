// v2.0 — refactored and cleaned, May 2026
package bot;

import logic.FlowCalculator;
import logic.MoveValidator;
import logic.PathFinder;
import model.GameState;
import model.Player;
import model.Position;

import java.util.List;

// Scores pawn moves. Row advancement is the dominant factor; everything else is a tiebreaker.
public class MoveEvaluator {

    private static final double W_ROW_ADVANCE         = 50.0;
    private static final double W_ROW_ADVANCE_PARTIAL = 25.0; // forward but path gets slightly worse
    private static final double W_PATH_GAIN           = 10.0;
    private static final double W_ASTAR_TIEBREAK      = 5.0;  // bonus for following the A* path
    private static final double W_SAFETY              = 0.15;
    private static final double W_FLEXIBILITY         = 0.1;
    private static final double W_OPPONENT_THREAT     = 0.2;

    private static final double W_JUMP_SETUP      = 20.0;
    private static final double W_GIVE_JUMP_PENALTY = 35.0;
    private static final double W_DIRECT_JUMP     = 25.0;

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

    public static ScoredMove findBestMove(GameState state, BoardGraph graph) {
        Player bot = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();
        List<Position> validMoves = MoveValidator.getValidMoves(state, bot);

        if (validMoves.isEmpty()) return null;

        int currentPath = PathFinder.aStarShortestPath(state, bot);
        double currentSafety = PathFinder.dijkstraSafestPath(graph, bot.getPosition(), bot.getGoalRow());
        int currentFlow = FlowCalculator.calculateMaxFlow(state, bot);

        // next step on the A* path — used as a tiebreaker
        List<Position> shortestPath = PathFinder.getAStarPath(state, bot);
        Position nextStepOnPath = (shortestPath != null && shortestPath.size() >= 2)
                ? shortestPath.get(1) : null;

        ScoredMove bestMove = null;

        for (Position target : validMoves) {
            double score = evaluateMove(state, graph, bot, opponent, target,
                                        currentPath, currentSafety, currentFlow, nextStepOnPath);
            if (bestMove == null || score > bestMove.score) {
                bestMove = new ScoredMove(target, score);
            }
        }

        return bestMove;
    }

    // scores a single move; forward movement dominates, rest are tiebreakers
    private static double evaluateMove(GameState state, BoardGraph graph,
                                        Player bot, Player opponent,
                                        Position target,
                                        int currentPath, double currentSafety,
                                        int currentFlow, Position nextStepOnPath) {
        Position botPos = bot.getPosition();

        GameState simState = state.deepCopy();
        Player simBot = simState.getCurrentPlayer();
        Player simOpponent = simState.getOtherPlayer();
        simBot.setPosition(target);

        int newPath = PathFinder.aStarShortestPath(simState, simBot);

        if (newPath == 0) return 1000.0;  // winning move
        if (newPath < 0) return -500.0;   // no path — really bad
        if (currentPath < 0) currentPath = 20;

        double pathGain = currentPath - newPath;
        int rowAdvancement = computeRowAdvancement(botPos, target, bot.getGoalRow());

        double rowAdvanceBonus = 0;
        if (rowAdvancement > 0) {
            if (pathGain >= 0) {
                rowAdvanceBonus = W_ROW_ADVANCE;         // forward + path shorter — best case
            } else if (pathGain >= -2) {
                rowAdvanceBonus = W_ROW_ADVANCE_PARTIAL; // forward but slight detour
            }
            // else: dead end, no reward
        }

        // jumping 2 rows at once is extra valuable
        double jumpBonus = (Math.abs(rowAdvancement) == 2) ? W_DIRECT_JUMP : 0.0;

        double astarTiebreak = (nextStepOnPath != null && target.equals(nextStepOnPath))
                ? W_ASTAR_TIEBREAK : 0;

        BoardGraph simGraph = new BoardGraph();
        simGraph.buildFromState(simState, simOpponent.getPosition());
        double newSafety = PathFinder.dijkstraSafestPath(simGraph, target, simBot.getGoalRow());
        double safetyScore = (newSafety > 0 && currentSafety > 0) ? (currentSafety - newSafety) : 0;

        int newFlow = FlowCalculator.calculateMaxFlow(simState, simBot);
        double flexScore = newFlow - currentFlow;

        double opponentThreat = evaluateOpponentResponse(simState, simOpponent);
        double jumpPenalty = evaluateJumpRisk(state, bot, opponent, target);
        double jumpSetup = evaluateJumpSetup(simState, simBot, simOpponent);

        return rowAdvanceBonus
             + jumpBonus
             + W_PATH_GAIN * pathGain
             + astarTiebreak
             + W_SAFETY * safetyScore
             + W_FLEXIBILITY * flexScore
             - W_OPPONENT_THREAT * opponentThreat
             - jumpPenalty
             + jumpSetup;
    }

    // penalty for moving adjacent to the opponent and giving them a free jump toward their goal
    private static double evaluateJumpRisk(GameState state, Player bot, Player opponent, Position target) {
        Position oppPos = opponent.getPosition();
        int rowDiff = Math.abs(target.getRow() - oppPos.getRow());
        int colDiff = Math.abs(target.getCol() - oppPos.getCol());

        if (rowDiff + colDiff != 1) return 0.0; // not adjacent

        int oppGoalRow = opponent.getGoalRow();
        int oppDirection = (oppGoalRow > oppPos.getRow()) ? 1 : -1;

        // check if the opponent can jump over us toward their goal
        boolean oppCanJumpOverUs = false;
        if (rowDiff == 1 && colDiff == 0) {
            int jumpRow = target.getRow() + (target.getRow() - oppPos.getRow());
            if ((jumpRow - oppPos.getRow()) * oppDirection > 0) {
                oppCanJumpOverUs = true;
            }
        }

        if (!oppCanJumpOverUs) return 0.0;

        int botDist = PathFinder.aStarShortestPath(state, bot);
        int oppDist = PathFinder.aStarShortestPath(state, opponent);
        int raceGap = oppDist - botDist; // positive = we're ahead

        // giving a jump when behind or tied is very bad
        if (raceGap <= 0) return W_GIVE_JUMP_PENALTY;
        if (raceGap == 1) return W_GIVE_JUMP_PENALTY * 0.7;
        if (raceGap == 2) return W_GIVE_JUMP_PENALTY * 0.4;
        return W_GIVE_JUMP_PENALTY * 0.2;
    }

    // bonus if after this move we could jump over the opponent on the next turn
    private static double evaluateJumpSetup(GameState simState, Player simBot, Player simOpponent) {
        Position newPos = simBot.getPosition();
        Position oppPos = simOpponent.getPosition();
        int botGoalRow = simBot.getGoalRow();

        int rowDiff = Math.abs(newPos.getRow() - oppPos.getRow());
        int colDiff = Math.abs(newPos.getCol() - oppPos.getCol());

        if (rowDiff + colDiff != 1) return 0.0; // not adjacent

        if (rowDiff != 1 || colDiff != 0) return 0.0;

        int botDirection = (botGoalRow > newPos.getRow()) ? 1 : -1;
        int oppRelativePos = (oppPos.getRow() - newPos.getRow()) * botDirection;

        if (oppRelativePos <= 0) return 0.0; // opponent isn't in our way

        int jumpRow = oppPos.getRow() + botDirection;
        if (jumpRow >= 0 && jumpRow <= 8) return W_JUMP_SETUP;

        return 0.0;
    }

    // how many rows closer to goal: +1 forward, 0 sideways, -1 back
    private static int computeRowAdvancement(Position from, Position to, int goalRow) {
        if (goalRow > from.getRow()) {
            return to.getRow() - from.getRow(); // goal is below
        } else {
            return from.getRow() - to.getRow(); // goal is above
        }
    }

    // how much the opponent gains from their best response move
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
