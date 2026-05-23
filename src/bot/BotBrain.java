// v2.0 — refactored and cleaned, May 2026
package bot;

import logic.MoveValidator;
import logic.PathFinder;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

// Main decision-maker. Picks between moving and placing walls based on race state and heuristics.
public class BotBrain {

    private static final int MAX_CONSECUTIVE_WALLS = 3;
    private static final int MAX_CUMULATIVE_SELF_HARM = 5;

    private int consecutiveWalls = 0;
    private int cumulativeSelfHarm = 0;

    private final int openingRandomMoves;
    private final Random rng;
    private int turnCount = 0;
    private boolean silent = false;

    public BotBrain() {
        this(0);
    }

    public void setSilent(boolean silent) {
        this.silent = silent;
    }

    // openingRandomMoves: how many random moves at the start (3-6 is good for bot-vs-bot variety)
    public BotBrain(int openingRandomMoves) {
        this.openingRandomMoves = openingRandomMoves;
        this.rng = new Random();
    }

    public static class BotAction {
        public enum Type { MOVE, WALL }

        public final Type type;
        public final Position moveTarget;
        public final Wall wallToPlace;
        public final double score;

        private BotAction(Type type, Position moveTarget, Wall wallToPlace, double score) {
            this.type = type;
            this.moveTarget = moveTarget;
            this.wallToPlace = wallToPlace;
            this.score = score;
        }

        public static BotAction move(Position target, double score) {
            return new BotAction(Type.MOVE, target, null, score);
        }

        public static BotAction wall(Wall wall, double score) {
            return new BotAction(Type.WALL, null, wall, score);
        }

        @Override
        public String toString() {
            if (type == Type.MOVE) {
                return "BOT MOVES to " + moveTarget + " (score: " + String.format("%.2f", score) + ")";
            }
            return "BOT PLACES WALL " + wallToPlace + " (score: " + String.format("%.2f", score) + ")";
        }
    }

    public BotAction computeBestAction(GameState state) {
        Player bot = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();

        int botDist = PathFinder.aStarShortestPath(state, bot);
        int oppDist = PathFinder.aStarShortestPath(state, opponent);
        int wallsLeft = bot.getWallsRemaining();

        boolean hasForwardMove = hasForwardMoveAvailable(state, bot);
        turnCount++;

        if (!silent) System.out.println("dist=" + botDist + "/" + oppDist + " walls=" + wallsLeft);

        // random opening moves for variety in bot-vs-bot
        if (turnCount <= openingRandomMoves) {
            BotAction randomAction = pickRandomOpeningMove(state, bot);
            if (randomAction != null) return randomAction;
        }

        // instant win — no need to think further
        if (botDist == 1) {
            if (!silent) System.out.println("instant win!");
            return doMove(forceMove(state));
        }

        int raceGap = oppDist - botDist; // positive = we're ahead
        boolean weAreWinning = raceGap > 0;
        boolean weAreLosing  = raceGap < 0;
        boolean closeRace    = Math.abs(raceGap) <= 2;

        // way ahead — just run for the goal
        if (weAreWinning && raceGap >= 4 && botDist <= 4 && hasForwardMove) {
            if (!silent) System.out.println("rushing, ahead by " + raceGap);
            BoardGraph rushGraph = new BoardGraph();
            rushGraph.buildFromState(state, opponent.getPosition());
            MoveEvaluator.ScoredMove rushMove = MoveEvaluator.findBestMove(state, rushGraph);
            if (rushMove != null) return doMove(BotAction.move(rushMove.target, rushMove.score + 50));
        }

        // emergency blocking — opponent is about to win
        if (oppDist <= 2 && wallsLeft > 0) {
            WallEvaluator.ScoredWall emergencyWall = WallEvaluator.findBestWall(state);
            if (emergencyWall != null && emergencyWall.score > 0) {
                boolean shouldBlock = (oppDist == 1) || (oppDist == 2 && botDist > 2);
                if (shouldBlock) {
                    int oppAfter = PathFinder.aStarWithWall(state, opponent, emergencyWall.wall);
                    int emDamage = (oppAfter >= 0) ? oppAfter - oppDist : 0;
                    int botAfter = PathFinder.aStarWithWall(state, bot, emergencyWall.wall);
                    int emSelfHarm = (botAfter >= 0) ? botAfter - botDist : 999;
                    if (emDamage > 0 && emSelfHarm <= 2) {
                        if (!silent) System.out.println("emergency block, dmg=" + emDamage);
                        return doWall(BotAction.wall(emergencyWall.wall, emergencyWall.score));
                    }
                }
            }
        }

        // losing badly — try an aggressive wall
        if (weAreLosing && raceGap <= -3 && wallsLeft > 0 && oppDist > 2) {
            WallEvaluator.ScoredWall aggressiveWall = WallEvaluator.findBestWall(state);
            if (aggressiveWall != null && aggressiveWall.score > 10) {
                int oppAfter = PathFinder.aStarWithWall(state, opponent, aggressiveWall.wall);
                int pathDmg  = (oppAfter >= 0) ? oppAfter - oppDist : 0;
                int botAfter = PathFinder.aStarWithWall(state, bot, aggressiveWall.wall);
                int selfHarm = (botAfter >= 0) ? botAfter - botDist : 999;
                if (pathDmg >= 2 && selfHarm <= pathDmg) {
                    if (!silent) System.out.println("aggressive wall, losing by " + (-raceGap));
                    return doWall(BotAction.wall(aggressiveWall.wall, aggressiveWall.score + 30));
                }
            }
        }

        // endgame blocking — opponent is close but not immediate
        if (oppDist <= 4 && oppDist > 2 && wallsLeft > 0) {
            WallEvaluator.ScoredWall blockWall = WallEvaluator.findBestWall(state);
            if (blockWall != null) {
                int oppAfter = PathFinder.aStarWithWall(state, opponent, blockWall.wall);
                int pathDmg  = (oppAfter >= 0) ? oppAfter - oppDist : 0;
                int botAfter = PathFinder.aStarWithWall(state, bot, blockWall.wall);
                int selfHarm = (botAfter >= 0) ? botAfter - botDist : 999;
                int netTempo = pathDmg - selfHarm;

                boolean goodEndgameWall = pathDmg >= 3 && netTempo >= 2 && selfHarm <= 1;
                boolean mustWall = weAreLosing && oppDist < botDist && pathDmg >= 2;

                if (goodEndgameWall || mustWall) {
                    if (!silent) System.out.println("endgame block, dmg=" + pathDmg + " net=" + netTempo);
                    return doWall(BotAction.wall(blockWall.wall, blockWall.score + 20));
                }
            }
        }

        // evaluate best move
        BoardGraph graph = new BoardGraph();
        graph.buildFromState(state, opponent.getPosition());
        MoveEvaluator.ScoredMove bestMove = MoveEvaluator.findBestMove(state, graph);

        double moveValue = (bestMove != null) ? bestMove.score : 0;

        boolean bestMoveIsForward = false;
        if (bestMove != null) {
            int botRow = bot.getPosition().getRow();
            int goalRow = bot.getGoalRow();
            int direction = (goalRow > botRow) ? 1 : -1;
            bestMoveIsForward = (bestMove.target.getRow() - botRow) * direction > 0;
        }

        // evaluate best wall
        WallEvaluator.ScoredWall bestWall = null;
        double wallValue = -999;

        boolean canWall = wallsLeft > 0
                       && consecutiveWalls < MAX_CONSECUTIVE_WALLS
                       && cumulativeSelfHarm <= MAX_CUMULATIVE_SELF_HARM;

        if (canWall && oppDist >= 0 && botDist >= 0) {
            bestWall = WallEvaluator.findBestWall(state);

            if (bestWall != null && bestWall.score > 0) {
                int botPathAfterWall = PathFinder.aStarWithWall(state, bot, bestWall.wall);
                int selfHarmActual   = (botPathAfterWall >= 0) ? botPathAfterWall - botDist : 999;
                int oppPathAfter     = PathFinder.aStarWithWall(state, opponent, bestWall.wall);
                int pathDamage       = (oppPathAfter >= 0) ? oppPathAfter - oppDist : 0;
                boolean isEmergency  = oppDist <= 2;

                bestWall = applyWallFilters(bestWall, isEmergency, selfHarmActual, pathDamage,
                                            botPathAfterWall, wallsLeft);

                if (bestWall != null) {
                    wallValue = computeWallValue(bestWall, pathDamage, selfHarmActual,
                                                 moveValue, hasForwardMove, bestMoveIsForward,
                                                 weAreWinning, weAreLosing, closeRace,
                                                 raceGap, oppDist);
                    if (!silent) System.out.println("wall eval: dmg=" + pathDamage + " harm=" + selfHarmActual
                            + " wallVal=" + (int) wallValue + " moveVal=" + (int) moveValue);
                }
            }
        } else if (!canWall && wallsLeft > 0) {
            if (!silent) System.out.println("wall suppressed, too many consecutive");
        }

        if (bestMove == null && bestWall == null) return null;

        boolean preferWall = bestWall != null && wallValue > moveValue && bestWall.score > 0;
        if (preferWall) {
            int botPathAfter = PathFinder.aStarWithWall(state, bot, bestWall.wall);
            int selfHarm = (botPathAfter >= 0) ? botPathAfter - botDist : 0;
            cumulativeSelfHarm += Math.max(0, selfHarm);
            if (!silent) System.out.println("placing wall: " + bestWall.wall);
            return doWall(BotAction.wall(bestWall.wall, bestWall.score));
        }

        if (bestMove != null) {
            if (!silent) System.out.println("moving to: " + bestMove.target);
            return doMove(BotAction.move(bestMove.target, bestMove.score));
        }

        return doWall(BotAction.wall(bestWall.wall, bestWall.score));
    }

    // filters out walls that hurt us too much; returns null to signal rejection
    private WallEvaluator.ScoredWall applyWallFilters(WallEvaluator.ScoredWall wall,
                                                        boolean isEmergency,
                                                        int selfHarmActual, int pathDamage,
                                                        int botPathAfterWall, int wallsLeft) {
        if (!isEmergency) {
            if (selfHarmActual >= 3) return null;
            if (selfHarmActual == 2 && pathDamage < 4) return null;
            if (selfHarmActual == 1 && pathDamage < 2) return null;
            if (wallsLeft <= 3 && pathDamage < 2) {
                if (!silent) System.out.println("conserving walls, only " + wallsLeft + " left");
                return null;
            }
        }

        if (isEmergency && (selfHarmActual >= 4 || botPathAfterWall >= 16)) {
            if (!silent) System.out.println("emergency reject, too much self harm");
            return null;
        }

        return wall;
    }

    // computes the final value to compare against moveValue
    private double computeWallValue(WallEvaluator.ScoredWall bestWall,
                                     int pathDamage, int selfHarmActual,
                                     double moveValue, boolean hasForwardMove, boolean bestMoveIsForward,
                                     boolean weAreWinning, boolean weAreLosing, boolean closeRace,
                                     int raceGap, int oppDist) {
        double netTempo = pathDamage - selfHarmActual - 1.0;
        double wallValue = netTempo * 15.0 + bestWall.score * 0.5;

        if (!hasForwardMove && pathDamage >= 2) wallValue += 25.0;
        if (!bestMoveIsForward && pathDamage >= 2) wallValue += 10.0;

        if (weAreLosing && -raceGap > 2 && pathDamage >= 2) {
            wallValue += Math.min(-raceGap * 2.0, 20.0);
        }

        if (oppDist <= 3) wallValue += (4 - oppDist) * 10.0;
        if (bestMoveIsForward && netTempo >= 1.0) wallValue += 10.0;
        if (weAreWinning && raceGap >= 3 && oppDist >= 6) wallValue -= 20.0;

        if (weAreLosing && pathDamage >= 2) {
            wallValue += Math.min(-raceGap * 5.0, 25.0);
        }

        if (closeRace && netTempo >= 1.0) wallValue += 15.0;

        return wallValue;
    }

    private BotAction pickRandomOpeningMove(GameState state, Player bot) {
        List<Position> validMoves = MoveValidator.getValidMoves(state, bot);
        if (validMoves.isEmpty()) return null;

        int botRow = bot.getPosition().getRow();
        int goalRow = bot.getGoalRow();
        int direction = (goalRow > botRow) ? 1 : -1;

        List<Position> forwardMoves = new ArrayList<>();
        List<Position> sidewaysMoves = new ArrayList<>();
        for (Position m : validMoves) {
            int rowDelta = (m.getRow() - botRow) * direction;
            if (rowDelta > 0) forwardMoves.add(m);
            else if (rowDelta == 0) sidewaysMoves.add(m);
        }

        Position randomMove;
        if (turnCount == 1) {
            // first move: always go forward
            randomMove = forwardMoves.isEmpty() ? null : forwardMoves.get(rng.nextInt(forwardMoves.size()));
        } else {
            List<Position> pool = new ArrayList<>(forwardMoves);
            pool.addAll(sidewaysMoves);
            randomMove = pool.isEmpty() ? null : pool.get(rng.nextInt(pool.size()));
        }

        if (randomMove == null) {
            randomMove = validMoves.get(rng.nextInt(validMoves.size()));
        }

        if (!silent) System.out.println("random opening move: " + randomMove);
        return doMove(BotAction.move(randomMove, 50.0));
    }

    private boolean hasForwardMoveAvailable(GameState state, Player bot) {
        List<Position> validMoves = MoveValidator.getValidMoves(state, bot);
        int botRow = bot.getPosition().getRow();
        int goalRow = bot.getGoalRow();
        int direction = (goalRow > botRow) ? 1 : -1;

        for (Position move : validMoves) {
            if ((move.getRow() - botRow) * direction > 0) return true;
        }
        return false;
    }

    // track consecutive walls so we don't over-wall

    private BotAction doMove(BotAction action) {
        if (action != null) consecutiveWalls = 0;
        return action;
    }

    private BotAction doWall(BotAction action) {
        if (action != null) consecutiveWalls++;
        return action;
    }

    // for instant wins — always pick the best move regardless of everything else
    private BotAction forceMove(GameState state) {
        Player opponent = state.getOtherPlayer();
        BoardGraph graph = new BoardGraph();
        graph.buildFromState(state, opponent.getPosition());
        MoveEvaluator.ScoredMove bestMove = MoveEvaluator.findBestMove(state, graph);
        if (bestMove != null) return BotAction.move(bestMove.target, bestMove.score);
        return null;
    }
}
