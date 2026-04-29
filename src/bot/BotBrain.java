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

/**
 * Main decision-making class for the bot. Picks between moving and placing walls
 * based on race state, path distances, and a bunch of heuristics.
 */
public class BotBrain {

    private static final int MAX_CONSECUTIVE_WALLS = 3;
    private static final int MAX_CUMULATIVE_SELF_HARM = 5;

    private int consecutiveWalls = 0;
    private int cumulativeSelfHarm = 0;

    private final int openingRandomMoves;
    private final Random rng;
    private int turnCount = 0;
    private boolean silent = false;

    /** Default constructor - no randomness. */
    public BotBrain() {
        this(0);
    }

    /** Turn off debug prints for batch runs. */
    public void setSilent(boolean silent) {
        this.silent = silent;
    }

    /**
     * Constructor with random opening moves for bot vs bot games.
     * @param openingRandomMoves how many random moves at the start (3-6 is good)
     */
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
            } else {
                return "BOT PLACES WALL " + wallToPlace + " (score: " + String.format("%.2f", score) + ")";
            }
        }
    }

    public BotAction computeBestAction(GameState state) {
        Player bot = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();

        int botDist = PathFinder.aStarShortestPath(state, bot);
        int oppDist = PathFinder.aStarShortestPath(state, opponent);
        int wallsLeft = bot.getWallsRemaining();

        // Check forward move availability
        boolean hasForwardMove = hasForwardMoveAvailable(state, bot);

        turnCount++;

        if (!silent) System.out.println("dist=" + botDist + "/" + oppDist + " walls=" + wallsLeft);

        // random opening moves for variety
        if (turnCount <= openingRandomMoves) {
            List<Position> validMoves = MoveValidator.getValidMoves(state, bot);
            if (!validMoves.isEmpty()) {
                int botRow = bot.getPosition().getRow();
                int goalRow = bot.getGoalRow();
                int direction = (goalRow > botRow) ? 1 : -1;

                // split into forward and sideways (no backward)
                List<Position> forwardMoves = new java.util.ArrayList<>();
                List<Position> sidewaysMoves = new java.util.ArrayList<>();
                for (Position m : validMoves) {
                    int rowDelta = (m.getRow() - botRow) * direction;
                    if (rowDelta > 0) forwardMoves.add(m);
                    else if (rowDelta == 0) sidewaysMoves.add(m);
                }

                Position randomMove = null;
                if (turnCount == 1) {
                    // first move: just go forward
                    if (!forwardMoves.isEmpty()) {
                        randomMove = forwardMoves.get(rng.nextInt(forwardMoves.size()));
                    }
                } else {
                    // later turns: pick from forward + sideways
                    java.util.List<Position> pool = new java.util.ArrayList<>(forwardMoves);
                    pool.addAll(sidewaysMoves);
                    if (!pool.isEmpty()) {
                        randomMove = pool.get(rng.nextInt(pool.size()));
                    }
                }
                if (randomMove == null) {
                    randomMove = validMoves.get(rng.nextInt(validMoves.size()));
                }

                if (!silent) System.out.println("random opening move: " + randomMove);
                return doMove(BotAction.move(randomMove, 50.0));
            }
        }

        // instant win check
        if (botDist == 1) {
            if (!silent) System.out.println("instant win!");
            return doMove(forceMove(state));
        }

        // race calculation
        int raceGap = oppDist - botDist;  // positive = we're ahead
        boolean weAreWinning = raceGap > 0;
        boolean weAreLosing = raceGap < 0;
        boolean closeRace = Math.abs(raceGap) <= 2;

        // rush mode - if we're way ahead, just run for it
        if (weAreWinning && raceGap >= 4 && botDist <= 4 && hasForwardMove) {
            if (!silent) System.out.println("rushing, ahead by " + raceGap);
            BoardGraph rushGraph = new BoardGraph();
            rushGraph.buildFromState(state, opponent.getPosition());
            MoveEvaluator.ScoredMove rushMove = MoveEvaluator.findBestMove(state, rushGraph, this);
            if (rushMove != null) {
                return doMove(BotAction.move(rushMove.target, rushMove.score + 50));
            }
        }

        // emergency blocking if opponent is about to win
        if (oppDist <= 2 && wallsLeft > 0) {
            WallEvaluator.ScoredWall emergencyWall = WallEvaluator.findBestWall(state);
            if (emergencyWall != null && emergencyWall.score > 0) {
                // For oppDist=1: ALWAYS block
                // For oppDist=2: block if wall does damage AND bot isn't about to win
                boolean shouldBlock = (oppDist == 1)
                    || (oppDist == 2 && botDist > 2);
                if (shouldBlock) {
                    int oppAfter = PathFinder.aStarWithWall(state, opponent, emergencyWall.wall);
                    int emDamage = (oppAfter >= 0) ? oppAfter - oppDist : 0;
                    int botAfter = PathFinder.aStarWithWall(state, bot, emergencyWall.wall);
                    int emSelfHarm = (botAfter >= 0) ? botAfter - botDist : 999;
                    // Only block if the wall actually helps and doesn't hurt us badly
                    if (emDamage > 0 && emSelfHarm <= 2) {
                        if (!silent) System.out.println("emergency block, dmg=" + emDamage);
                        return doWall(BotAction.wall(emergencyWall.wall, emergencyWall.score));
                    }
                }
            }
        }

        // aggressive walling when losing badly
        if (weAreLosing && raceGap <= -3 && wallsLeft > 0 && oppDist > 2) {
            WallEvaluator.ScoredWall aggressiveWall = WallEvaluator.findBestWall(state);
            if (aggressiveWall != null && aggressiveWall.score > 10) {
                int oppAfter = PathFinder.aStarWithWall(state, opponent, aggressiveWall.wall);
                int pathDmg = (oppAfter >= 0) ? oppAfter - oppDist : 0;
                int botAfter = PathFinder.aStarWithWall(state, bot, aggressiveWall.wall);
                int selfHarm = (botAfter >= 0) ? botAfter - botDist : 999;

                // accept walls with decent damage even with some self-harm
                if (pathDmg >= 2 && selfHarm <= pathDmg) {
                    if (!silent) System.out.println("aggressive wall, losing by " + (-raceGap));
                    return doWall(BotAction.wall(aggressiveWall.wall, aggressiveWall.score + 30));
                }
            }
        }

        // endgame blocking when opponent is close
        if (oppDist <= 4 && oppDist > 2 && wallsLeft > 0) {
            WallEvaluator.ScoredWall blockWall = WallEvaluator.findBestWall(state);
            if (blockWall != null) {
                int oppAfter = PathFinder.aStarWithWall(state, opponent, blockWall.wall);
                int pathDmg = (oppAfter >= 0) ? oppAfter - oppDist : 0;
                int botAfter = PathFinder.aStarWithWall(state, bot, blockWall.wall);
                int selfHarm = (botAfter >= 0) ? botAfter - botDist : 999;

                // in endgame, a wall that gives us 2+ tempo is usually worth it
                int netTempo = pathDmg - selfHarm;
                boolean goodEndgameWall = pathDmg >= 3 && netTempo >= 2 && selfHarm <= 1;

                // if we can't catch up, wall is mandatory
                boolean mustWall = weAreLosing && oppDist < botDist && pathDmg >= 2;

                if (goodEndgameWall || mustWall) {
                    if (!silent) System.out.println("endgame block, dmg=" + pathDmg + " net=" + netTempo);
                    return doWall(BotAction.wall(blockWall.wall, blockWall.score + 20));
                }
            }
        }

        // find best move
        BoardGraph graph = new BoardGraph();
        graph.buildFromState(state, opponent.getPosition());
        MoveEvaluator.ScoredMove bestMove = MoveEvaluator.findBestMove(state, graph, this);

        double moveValue = 0;
        if (bestMove != null) {
            moveValue = bestMove.score;
        }

        // check if best move goes forward
        boolean bestMoveIsForward = false;
        if (bestMove != null) {
            int botRow = bot.getPosition().getRow();
            int goalRow = bot.getGoalRow();
            int direction = (goalRow > botRow) ? 1 : -1;
            int moveRow = bestMove.target.getRow();
            bestMoveIsForward = (moveRow - botRow) * direction > 0;
        }

        // find best wall
        WallEvaluator.ScoredWall bestWall = null;
        double wallValue = -999;

        boolean canWall = wallsLeft > 0
                       && consecutiveWalls < MAX_CONSECUTIVE_WALLS
                       && cumulativeSelfHarm <= MAX_CUMULATIVE_SELF_HARM;

        if (canWall && oppDist >= 0 && botDist >= 0) {
            bestWall = WallEvaluator.findBestWall(state);

            if (bestWall != null && bestWall.score > 0) {
                int botPathAfterWall = PathFinder.aStarWithWall(state, bot, bestWall.wall);
                int selfHarmActual = (botPathAfterWall >= 0) ? botPathAfterWall - botDist : 999;

                int oppPathAfter = PathFinder.aStarWithWall(state, opponent, bestWall.wall);
                int pathDamage = (oppPathAfter >= 0) ? oppPathAfter - oppDist : 0;

                boolean isEmergency = oppDist <= 2;

                // reject walls that hurt us too much
                boolean rejectWall = false;
                if (!isEmergency) {
                    if (selfHarmActual >= 3) {
                        rejectWall = true;
                    } else if (selfHarmActual == 2) {
                        if (pathDamage < 4) rejectWall = true;
                    } else if (selfHarmActual == 1) {
                        if (pathDamage < 2) rejectWall = true;
                    }
                }

                // conserve walls when running low
                if (!isEmergency && wallsLeft <= 3 && pathDamage < 2) {
                    rejectWall = true;
                    if (!silent) System.out.println("conserving walls, only " + wallsLeft + " left");
                }

                if (rejectWall) {
                    if (!silent) System.out.println("rejected wall, selfHarm=" + selfHarmActual);
                    bestWall = null;
                } else if (isEmergency && (selfHarmActual >= 4 || botPathAfterWall >= 16)) {
                    if (!silent) System.out.println("emergency reject, too much self harm");
                    bestWall = null;
                } else if (bestWall != null) {
                    // wall scoring - net tempo is how many turns the wall gains
                    double netTempo = pathDamage - selfHarmActual - 1.0;

                    // base value from net tempo
                    wallValue = netTempo * 15.0;

                    // add evaluator score for synergy/position
                    wallValue += bestWall.score * 0.5;

                    // extra points if we can't go forward
                    if (!hasForwardMove && pathDamage >= 2) {
                        wallValue += 25.0;
                    }

                    // bonus if our best move is sideways anyway
                    if (!bestMoveIsForward && pathDamage >= 2) {
                        wallValue += 10.0;
                    }

                    // bonus when we're losing the race
                    if (weAreLosing && -raceGap > 2 && pathDamage >= 2) {
                        double racingBonus = Math.min(-raceGap * 2.0, 20.0);
                        wallValue += racingBonus;
                    }

                    // emergency proximity bonus
                    if (oppDist <= 3) {
                        wallValue += (4 - oppDist) * 10.0;
                    }

                    // proactive walling - good walls are worth placing even with forward moves
                    if (bestMoveIsForward && netTempo >= 1.0) {
                        wallValue += 10.0;
                    }

                    // don't over-wall when winning comfortably
                    if (weAreWinning && raceGap >= 3 && oppDist >= 6) {
                        wallValue -= 20.0;
                    }

                    // be aggressive with walls when losing
                    if (weAreLosing && pathDamage >= 2) {
                        double losingBonus = Math.min(-raceGap * 5.0, 25.0);
                        wallValue += losingBonus;
                    }

                    // close race - tempo walls are critical
                    if (closeRace && netTempo >= 1.0) {
                        wallValue += 15.0;
                    }

                    if (!silent) System.out.println("wall eval: dmg=" + pathDamage + " harm=" + selfHarmActual
                            + " wallVal=" + (int) wallValue + " moveVal=" + (int) moveValue);
                }
            }
        } else if (!canWall && wallsLeft > 0) {
            if (!silent) System.out.println("wall suppressed, too many consecutive");
        }

        // decide between wall and move
        if (bestMove == null && bestWall == null) return null;

        boolean preferWall = (bestWall != null && wallValue > moveValue && bestWall.score > 0);

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

        if (bestWall != null) {
            return doWall(BotAction.wall(bestWall.wall, bestWall.score));
        }

        return null;
    }

    /** Checks if there's any move that goes toward the goal. */
    private boolean hasForwardMoveAvailable(GameState state, Player bot) {
        List<Position> validMoves = MoveValidator.getValidMoves(state, bot);
        int botRow = bot.getPosition().getRow();
        int goalRow = bot.getGoalRow();
        int direction = (goalRow > botRow) ? 1 : -1;

        for (Position move : validMoves) {
            int moveRow = move.getRow();
            if ((moveRow - botRow) * direction > 0) {
                return true;
            }
        }
        return false;
    }

    // action wrappers - track consecutive walls

    private BotAction doMove(BotAction action) {
        if (action != null) {
            consecutiveWalls = 0;
        }
        return action;
    }

    private BotAction doWall(BotAction action) {
        if (action != null) {
            consecutiveWalls++;
        }
        return action;
    }

    private BotAction forceMove(GameState state) {
        Player bot = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();

        BoardGraph graph = new BoardGraph();
        graph.buildFromState(state, opponent.getPosition());

        // for instant wins, always pick the best move
        MoveEvaluator.ScoredMove bestMove = MoveEvaluator.findBestMove(state, graph);
        if (bestMove != null) {
            return BotAction.move(bestMove.target, bestMove.score);
        }

        return null;
    }

}
