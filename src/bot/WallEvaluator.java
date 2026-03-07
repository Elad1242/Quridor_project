package bot;

import logic.FlowCalculator;
import logic.PathFinder;
import logic.WallValidator;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.*;

/**
 * WALL EVALUATOR v10 — Complete redesign.
 *
 * PREVIOUS VERSIONS FAILED because:
 *   1. "Positional Safety Zone" hard-rejected walls ahead of bot within 4 rows
 *      → In endgame bot couldn't place ANY walls (all rejected)
 *   2. Scoring was too low — walls rarely beat sideways moves (score ~15)
 *   3. "No wall with positive score found" appeared constantly
 *
 * v10 CHANGES:
 *   1. REMOVED hard-rejection of walls ahead of bot. Instead, use SOFT PENALTY
 *      based on how much the wall actually hurts the bot (selfHarm check).
 *   2. SIMPLIFIED scoring: focus on pathDamage, selfHarm, and proximity to opponent.
 *   3. Walls that are ON the opponent's shortest path get big bonuses.
 *   4. The bot's own safety is checked by selfHarm in BotBrain, not here.
 *   5. Emergency mode is more aggressive.
 *
 * KEY INSIGHT: The best wall is the one that maximizes (pathDamage - selfHarm).
 * Everything else (synergy, position, proximity) is a tiebreaker.
 */
public class WallEvaluator {

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

                    if (!WallValidator.isValidWallPlacement(state, candidate)) continue;

                    double score = scoreWall(state, candidate, bot, opponent,
                                             botPath, opponentPath, onOpponentPath,
                                             emergency);

                    if (score > 0 && (bestWall == null || score > bestWall.score)) {
                        bestWall = new ScoredWall(candidate, score);
                    }
                }
            }
        }

        if (bestWall != null) {
            System.out.println("[WALL-EVAL v10] Best wall: " + bestWall.wall
                    + " score=" + String.format("%.1f", bestWall.score)
                    + (emergency ? " [EMERGENCY]" : ""));
        } else {
            System.out.println("[WALL-EVAL v10] No wall with positive score found."
                    + (emergency ? " [EMERGENCY MODE ACTIVE]" : ""));
        }

        return bestWall;
    }

    private static double scoreWall(GameState state, Wall candidate,
                                     Player bot, Player opponent,
                                     int botPathBefore, int opponentPathBefore,
                                     Set<String> onOpponentPath,
                                     boolean emergency) {

        // === Compute path impact ===
        int opponentPathAfter = PathFinder.aStarWithWall(state, opponent, candidate);
        if (opponentPathAfter < 0) return -1; // Would trap opponent (illegal)

        int botPathAfter = PathFinder.aStarWithWall(state, bot, candidate);
        if (botPathAfter < 0) return -1; // Would trap bot (illegal)

        int pathDamage = opponentPathAfter - opponentPathBefore;
        int selfHarm = botPathAfter - botPathBefore;

        // === HARD REJECTIONS ===
        if (pathDamage <= 0) return -1; // Wall doesn't slow opponent at all

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

        // === FLOW SAFETY CHECK ===
        // Reject walls that drastically reduce bot's flexibility (unless emergency)
        if (!emergency && selfHarm >= 1) {
            int botFlowBefore = FlowCalculator.calculateMaxFlow(state, bot);
            if (botFlowBefore > 1) {
                int botFlowAfter = FlowCalculator.calculateMaxFlowWithWall(state, bot, candidate);
                if (botFlowAfter <= 1 && pathDamage < 4) {
                    return -1; // Wall makes bot very vulnerable
                }
            }
        }

        // === SCORING ===
        // PRIMARY: net advantage (pathDamage - selfHarm)
        double netAdvantage = pathDamage - selfHarm;
        double score = netAdvantage * 3.0;

        // SECONDARY: raw path damage (quadratic — big damage is disproportionately good)
        score += pathDamage * pathDamage * 0.5;

        // Emergency bonus
        if (emergency) {
            score += pathDamage * 5.0;
        }

        // === STEP DAMAGE BONUSES ===
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

        // === WALL SYNERGY ===
        int synergy = countAdjacentWalls(state, candidate);
        score += synergy * 2.5;

        // === POSITIONAL SCORING ===
        int oppRow = opponent.getPosition().getRow();
        int oppGoalRow = opponent.getGoalRow();
        int wallRow = candidate.getRow();
        int wallCol = candidate.getCol();
        int oppCol = opponent.getPosition().getCol();
        int oppDirection = (oppGoalRow > oppRow) ? 1 : -1;
        int distanceAheadOfOpp = (wallRow - oppRow) * oppDirection;
        int absDistFromOpp = Math.abs(wallRow - oppRow);

        // === ON OPPONENT'S PATH BONUS ===
        boolean nearPath = onOpponentPath.contains(wallRow + "," + wallCol)
                        || onOpponentPath.contains(wallRow + "," + (wallCol + 1))
                        || onOpponentPath.contains((wallRow + 1) + "," + wallCol);
        if (nearPath) score += 3.0;

        // === AHEAD OF OPPONENT BONUS ===
        // Walls ahead of the opponent (between them and their goal) are most effective
        if (distanceAheadOfOpp >= 0 && distanceAheadOfOpp <= 2) {
            score += 3.0;
        } else if (distanceAheadOfOpp == 3) {
            score += 2.0;
        } else if (distanceAheadOfOpp == 4) {
            score += 1.0;
        } else if (distanceAheadOfOpp < 0 && distanceAheadOfOpp >= -2) {
            score += 0.5; // Just behind opponent — can still be useful
        }

        // === DISTANCE PENALTY ===
        // Walls far from the opponent are less useful
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

        // === SOFT POSITIONAL PENALTY (replaces hard rejection in v8) ===
        // Walls ahead of BOT between bot and its goal are somewhat risky.
        // But this is a SOFT penalty, not hard rejection. The selfHarm check
        // in BotBrain handles the real safety check.
        int botRow = bot.getPosition().getRow();
        int botGoalRow = bot.getGoalRow();
        int botDirection = (botGoalRow > botRow) ? 1 : -1;
        int distanceBehindBot = (botRow - wallRow) * botDirection;

        if (!emergency) {
            if (candidate.isHorizontal()) {
                if (distanceBehindBot < 0 && -distanceBehindBot <= 2) {
                    // Horizontal wall 1-2 rows ahead of bot — somewhat risky
                    score *= 0.6;
                } else if (distanceBehindBot == 0) {
                    // At bot's current row
                    score *= 0.4;
                }
            }
            if (candidate.isVertical() && distanceBehindBot < 0 && -distanceBehindBot <= 2) {
                score *= 0.8; // Lighter penalty for vertical walls ahead
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

}
