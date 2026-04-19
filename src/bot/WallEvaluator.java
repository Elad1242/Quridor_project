package bot;

import logic.FlowCalculator;
import logic.MoveValidator;
import logic.PathFinder;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.*;

// Scores wall placements. Best wall = highest (pathDamage - selfHarm).
public class WallEvaluator {

    public static boolean silent = false;

    public static class ScoredWall {
        public final Wall wall;
        public final double score;

        public ScoredWall(Wall wall, double score) {
            this.wall = wall;
            this.score = score;
        }

        @Override
        public String toString() {
            return "Wall at " + wall + " (score: " + String.format("%.2f", score) + ")";
        }
    }

    public static ScoredWall findBestWall(GameState state) {
        return findBestWall(state, -1);
    }

    public static ScoredWall findBestWall(GameState state, int lockedRow) {
        Player bot = state.getCurrentPlayer();
        Player opponent = state.getOtherPlayer();

        if (!bot.hasWalls()) return null;

        int opponentPath = PathFinder.aStarShortestPath(state, opponent);
        if (opponentPath < 0) return null;

        int botPath = PathFinder.aStarShortestPath(state, bot);
        if (botPath < 0) return null;

        boolean emergency = opponentPath <= 2;

        List<Position> opponentRoute = PathFinder.getAStarPath(state, opponent);
        Set<String> onOpponentPath = new HashSet<>();
        if (opponentRoute != null) {
            for (Position p : opponentRoute) {
                onOpponentPath.add(p.getRow() + "," + p.getCol());
            }
        }

        ScoredWall bestWall = null;

        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                for (Wall.Orientation orient : Wall.Orientation.values()) {
                    Wall candidate = new Wall(row, col, orient);

                    if (WallValidator.isValidWallPlacement(state, candidate)) {
                        double score = scoreWall(state, candidate, bot, opponent,
                                                 botPath, opponentPath, onOpponentPath,
                                                 emergency);

                        if (score > 0 && (bestWall == null || score > bestWall.score)) {
                            bestWall = new ScoredWall(candidate, score);
                        }
                    }
                }
            }
        }

        if (!silent) {
            if (bestWall != null) {
                System.out.println("best wall: " + bestWall.wall + " score=" + String.format("%.1f", bestWall.score));
            } else {
                System.out.println("no wall with positive score found");
            }
        }

        return bestWall;
    }

    private static double scoreWall(GameState state, Wall candidate,
                                     Player bot, Player opponent,
                                     int botPathBefore, int opponentPathBefore,
                                     Set<String> onOpponentPath,
                                     boolean emergency) {

        // compute path impact
        int opponentPathAfter = PathFinder.aStarWithWall(state, opponent, candidate);
        if (opponentPathAfter < 0) return -1; // would block someone off

        int botPathAfter = PathFinder.aStarWithWall(state, bot, candidate);
        if (botPathAfter < 0) return -1; // would block us off

        int pathDamage = opponentPathAfter - opponentPathBefore;
        int selfHarm = botPathAfter - botPathBefore;

        if (pathDamage <= 0) return -1; // wall doesn't slow opponent

        // Emergency: accept more self-harm
        if (emergency) {
            if (selfHarm >= 6) return -1;
            if (selfHarm > pathDamage + 3) return -1;
            if (botPathAfter >= 20) return -1;
        } else {
            // Normal: reject walls that hurt us as much or more than opponent
            if (selfHarm >= pathDamage && selfHarm >= 2) return -1;
            if (selfHarm >= 4) return -1;
            if (botPathAfter >= 16 && selfHarm >= 2) return -1;
        }

        // flow safety check
        // Reject walls that drastically reduce bot's flexibility (unless emergency)
        if (!emergency && selfHarm >= 1) {
            int botFlowBefore = FlowCalculator.calculateMaxFlow(state, bot);
            if (botFlowBefore > 1) {
                int botFlowAfter = FlowCalculator.calculateMaxFlowWithWall(state, bot, candidate);
                if (botFlowAfter <= 1 && pathDamage < 4) {
                    return -1; // makes us too vulnerable
                }
            }
        }

        // net advantage is the main signal
        double netAdvantage = pathDamage - selfHarm;
        double score = netAdvantage * 3.0;

        // raw path damage (quadratic scaling)
        score += pathDamage * pathDamage * 0.5;

        // Emergency bonus
        if (emergency) {
            score += pathDamage * 5.0;
        }

        // step damage bonuses
        if (pathDamage >= 5) score += 8.0;
        else if (pathDamage >= 4) score += 5.0;
        else if (pathDamage >= 3) score += 3.0;
        else if (pathDamage >= 2) score += 1.5;

        // Zero self-harm bonus
        if (selfHarm == 0) score += 2.0;

        // Self-harm penalty
        if (selfHarm > 0) {
            score -= selfHarm * (emergency ? 1.0 : 2.0);
        }

        // wall synergy
        int synergy = countAdjacentWalls(state, candidate);
        score += synergy * 2.5;

        // extending existing barriers is extra good
        if (synergy >= 2) {
            score += 5.0;
        }

        // funnel bonus - forces opponent through a narrow path
        int funnelBonus = evaluateFunnelCreation(state, candidate, opponent, opponentPathBefore);
        score += funnelBonus;

        // walls near opponent's goal are extra valuable
        double killZoneBonus = evaluateKillZone(state, candidate, opponent);
        score += killZoneBonus;

        // positional scoring
        int oppRow = opponent.getPosition().getRow();
        int oppGoalRow = opponent.getGoalRow();
        int wallRow = candidate.getRow();
        int wallCol = candidate.getCol();
        int oppCol = opponent.getPosition().getCol();
        int oppDirection = (oppGoalRow > oppRow) ? 1 : -1;
        int distanceAheadOfOpp = (wallRow - oppRow) * oppDirection;
        int absDistFromOpp = Math.abs(wallRow - oppRow);

        // bonus if wall is on opponent's path
        boolean nearPath = onOpponentPath.contains(wallRow + "," + wallCol)
                        || onOpponentPath.contains(wallRow + "," + (wallCol + 1))
                        || onOpponentPath.contains((wallRow + 1) + "," + wallCol);
        if (nearPath) score += 3.0;

        // walls between opponent and their goal are most effective
        if (distanceAheadOfOpp >= 0 && distanceAheadOfOpp <= 2) {
            score += 3.0;
        } else if (distanceAheadOfOpp == 3) {
            score += 2.0;
        } else if (distanceAheadOfOpp == 4) {
            score += 1.0;
        } else if (distanceAheadOfOpp < 0 && distanceAheadOfOpp >= -2) {
            score += 0.5; // just behind opponent, can still help
        }

        // walls far from opponent are less useful
        if (emergency) {
            if (absDistFromOpp >= 7) score *= 0.3;
            else if (absDistFromOpp >= 6) score *= 0.5;
            else if (absDistFromOpp >= 5) score *= 0.7;
        } else {
            if (absDistFromOpp >= 7) score *= 0.15;
            else if (absDistFromOpp >= 6) score *= 0.25;
            else if (absDistFromOpp >= 5) score *= 0.5;
            else if (absDistFromOpp >= 4) score *= 0.75;
        }

        // Column proximity to opponent
        int colDist = Math.abs(wallCol - oppCol);
        if (colDist >= 6) score *= 0.4;
        else if (colDist >= 5) score *= 0.5;
        else if (colDist >= 4) score *= 0.7;

        // soft penalty for walls near our own path (BotBrain does the real safety check)
        int botRow = bot.getPosition().getRow();
        int botGoalRow = bot.getGoalRow();
        int botDirection = (botGoalRow > botRow) ? 1 : -1;
        int distanceBehindBot = (botRow - wallRow) * botDirection;

        if (!emergency) {
            if (candidate.isHorizontal()) {
                if (distanceBehindBot < 0 && -distanceBehindBot <= 2) {
                    score *= 0.6; // horizontal wall ahead of us, risky
                } else if (distanceBehindBot == 0) {
                    score *= 0.4; // at our current row
                }
            }
            if (candidate.isVertical() && distanceBehindBot < 0 && -distanceBehindBot <= 2) {
                score *= 0.8; // lighter penalty for vertical walls
            }
        }

        return score;
    }

    private static int countAdjacentWalls(GameState state, Wall candidate) {
        int count = 0;
        int cr = candidate.getRow();
        int cc = candidate.getCol();

        for (Wall existing : state.getWalls()) {
            int er = existing.getRow();
            int ec = existing.getCol();

            if (candidate.getOrientation() == existing.getOrientation()) {
                if (candidate.isHorizontal()) {
                    if (cr == er && Math.abs(cc - ec) == 2) count++;
                    if (Math.abs(cr - er) == 1 && Math.abs(cc - ec) <= 1) count++;
                } else {
                    if (cc == ec && Math.abs(cr - er) == 2) count++;
                    if (Math.abs(cc - ec) == 1 && Math.abs(cr - er) <= 1) count++;
                }
            } else {
                if (Math.abs(cr - er) <= 1 && Math.abs(cc - ec) <= 1) count++;
            }
        }

        return count;
    }

    // Bonus for walls that funnel opponent through bottlenecks
    private static int evaluateFunnelCreation(GameState state, Wall candidate,
                                               Player opponent, int oppPathBefore) {
        // simulate placing the wall
        GameState sim = state.deepCopy();
        sim.addWall(candidate);

        // get opponent's new path
        List<Position> newPath = PathFinder.getAStarPath(sim, opponent);
        if (newPath == null || newPath.size() < 2) return 0;

        // count bottleneck positions (where opponent has few options)
        int bottleneckCount = 0;
        Player simOpp = sim.getOtherPlayer();

        for (int i = 1; i < Math.min(newPath.size(), 5); i++) {
            Position pos = newPath.get(i);
            Position origPos = simOpp.getPosition();
            simOpp.setPosition(pos);
            List<Position> movesFromPos = MoveValidator.getValidMoves(sim, simOpp);
            simOpp.setPosition(origPos);

            if (movesFromPos.size() <= 2) {
                bottleneckCount++;
            }
        }

        // more bottlenecks = better
        return bottleneckCount * 2;
    }

    // Bonus for walls near opponent's goal row
    public static double evaluateKillZone(GameState state, Wall candidate, Player opponent) {
        int oppGoalRow = opponent.getGoalRow();
        int wallRow = candidate.getRow();

        int distToGoal = Math.abs(wallRow - oppGoalRow);
        if (distToGoal <= 1) {
            return 8.0;
        } else if (distToGoal == 2) {
            return 4.0;
        } else if (distToGoal == 3) {
            return 2.0;
        }
        return 0.0;
    }
}
