package bot;

import logic.MoveValidator;
import logic.PathFinder;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.List;
import java.util.Random;

/**
 * The Bot's central decision engine — v10b.
 *
 * KEY FIX (v10b): The bot was placing pathDmg=1 walls when trapped instead of
 * moving to escape. The NO-FORWARD BONUS inflated terrible wall scores.
 *
 * NEW STRATEGY:
 *   - Only give NO-FORWARD BONUS to walls with pathDmg >= 2 (meaningful walls)
 *   - pathDmg=1 walls are only placed when the bot has a forward move AND the
 *     wall beats the move on raw net tempo merit
 *   - When trapped and walls are weak: MOVE to escape (reduce botDist)
 *   - When trapped and a STRONG wall exists (pathDmg>=2): WALL to punish opponent
 *   - PROACTIVE walling: when bot has a forward move but wall has netTempo >= 1,
 *     consider the wall even early in the game
 *
 * EXPLORATION (for Bot vs Bot / ML training):
 *   - Near-best random selection: picks randomly among moves within 8pts of best
 *   - Wall-vs-move coin flip: when values are close, randomly picks strategy
 *   - openingRandomMoves: first N moves are uniformly random (forward+sideways)
 *   - 0 openingRandomMoves = fully deterministic (default, for human vs bot)
 */
public class BotBrain {

    private static final int MAX_CONSECUTIVE_WALLS = 3;
    private static final int MAX_CUMULATIVE_SELF_HARM = 5;

    // --- Tracking ---
    private int consecutiveWalls = 0;
    private int cumulativeSelfHarm = 0;

    // --- Exploration / Randomness (for Bot vs Bot and ML training) ---
    private final double explorationNoise;   // 0.0 = deterministic, 8.0 = moderate additive noise
    private final int openingRandomMoves;     // Number of random opening moves (from ALL valid moves)
    private final Random rng;
    private int turnCount = 0;
    private boolean silent = false;

    /** Default constructor: deterministic bot (for human vs bot). */
    public BotBrain() {
        this(0.0, 0);
    }

    /** Enable silent mode (no debug prints). Used during batch simulation. */
    public void setSilent(boolean silent) {
        this.silent = silent;
    }

    /**
     * Constructor with exploration parameters (for Bot vs Bot / ML training).
     *
     * Exploration creates diverse but HIGH-QUALITY games for ML training:
     *   - Opening moves are uniformly random from forward+sideways options
     *   - Near-best selection: picks randomly among moves within 8pts of best
     *   - Wall-vs-move coin flip when values are close (within 10 pts)
     *   - Endgame tightens threshold to 2pts for precision
     *   - Emergency rules (instant win, must-block) are NEVER affected
     *
     * @param explorationNoise (legacy, currently unused — set to 0.0)
     * @param openingRandomMoves number of initial random moves (3-4 recommended)
     */
    public BotBrain(double explorationNoise, int openingRandomMoves) {
        this.explorationNoise = explorationNoise;
        this.openingRandomMoves = openingRandomMoves;
        // Use unique seed per instance to prevent two bots created in quick
        // succession from getting the same Random seed (System.nanoTime collision)
        this.rng = new Random(System.nanoTime() ^ Thread.currentThread().getId() ^ (long)(Math.random() * Long.MAX_VALUE));
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

        if (!silent) System.out.println("[BOT v10] botDist=" + botDist + " oppDist=" + oppDist
                + " wallsLeft=" + wallsLeft + " consecutive=" + consecutiveWalls
                + " selfHarmTotal=" + cumulativeSelfHarm
                + " fwd=" + hasForwardMove + " turn=" + turnCount);

        // === RULE 0: Random opening (for diverse Bot vs Bot games) ===
        // Turn 1: ALWAYS move forward (maintain strong opening position)
        // Turns 2-N: pick uniformly from forward + sideways moves
        // This creates column diversity while preserving opening quality.
        if (turnCount <= openingRandomMoves) {
            List<Position> validMoves = MoveValidator.getValidMoves(state, bot);
            if (!validMoves.isEmpty()) {
                int botRow = bot.getPosition().getRow();
                int goalRow = bot.getGoalRow();
                int direction = (goalRow > botRow) ? 1 : -1;

                // Separate forward and sideways moves
                List<Position> forwardMoves = new java.util.ArrayList<>();
                List<Position> sidewaysMoves = new java.util.ArrayList<>();
                for (Position m : validMoves) {
                    int rowDelta = (m.getRow() - botRow) * direction;
                    if (rowDelta > 0) forwardMoves.add(m);
                    else if (rowDelta == 0) sidewaysMoves.add(m);
                }

                Position randomMove;
                if (turnCount == 1) {
                    // Turn 1: always forward (don't waste a turn sideways from start)
                    if (!forwardMoves.isEmpty()) {
                        randomMove = forwardMoves.get(rng.nextInt(forwardMoves.size()));
                    } else {
                        randomMove = validMoves.get(rng.nextInt(validMoves.size()));
                    }
                } else {
                    // Turns 2+: uniform random from forward + sideways
                    List<Position> goodMoves = new java.util.ArrayList<>();
                    goodMoves.addAll(forwardMoves);
                    goodMoves.addAll(sidewaysMoves);
                    if (!goodMoves.isEmpty()) {
                        randomMove = goodMoves.get(rng.nextInt(goodMoves.size()));
                    } else {
                        randomMove = validMoves.get(rng.nextInt(validMoves.size()));
                    }
                }

                if (!silent) System.out.println("[BOT v10] >> RANDOM OPENING: " + randomMove + " (turn " + turnCount + "/" + openingRandomMoves + ")");
                return doMove(BotAction.move(randomMove, 50.0));
            }
        }

        // === RULE 1: Instant win ===
        if (botDist == 1) {
            if (!silent) System.out.println("[BOT v10] >> INSTANT WIN!");
            return doMove(forceMove(state));
        }

        // === RULE 2: Emergency blocking (opponent about to win) ===
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
                        if (!silent) System.out.println("[BOT v10] >> EMERGENCY BLOCK: " + emergencyWall.wall
                                + " (score=" + String.format("%.1f", emergencyWall.score)
                                + " oppDist=" + oppDist + " dmg=" + emDamage + ")");
                        return doWall(BotAction.wall(emergencyWall.wall, emergencyWall.score));
                    }
                }
            }
        }

        // === EVALUATE BEST MOVE ===
        BoardGraph graph = new BoardGraph();
        graph.buildFromState(state, opponent.getPosition());
        MoveEvaluator.ScoredMove bestMove = MoveEvaluator.findBestMove(state, graph, this);

        double moveValue = 0;
        if (bestMove != null) {
            moveValue = bestMove.score;
        }

        // === CLASSIFY MOVE TYPE ===
        boolean bestMoveIsForward = false;
        if (bestMove != null) {
            int botRow = bot.getPosition().getRow();
            int goalRow = bot.getGoalRow();
            int direction = (goalRow > botRow) ? 1 : -1;
            int moveRow = bestMove.target.getRow();
            bestMoveIsForward = (moveRow - botRow) * direction > 0;
        }

        // === EVALUATE BEST WALL ===
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

                // === SELF-HARM REJECTION ===
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

                // === WALL CONSERVATION ===
                // When walls are scarce (≤3), only place walls with significant damage
                if (!isEmergency && wallsLeft <= 3 && pathDamage < 2) {
                    rejectWall = true;
                    if (!silent) System.out.println("[BOT v10] CONSERVE WALLS: only " + wallsLeft + " left, pathDmg=" + pathDamage);
                }

                if (rejectWall) {
                    if (!silent) System.out.println("[BOT v10] HARM REJECT wall " + bestWall.wall
                            + " selfHarm=" + selfHarmActual + " pathDmg=" + pathDamage
                            + " botPathAfter=" + botPathAfterWall);
                    bestWall = null;
                } else if (isEmergency && (selfHarmActual >= 4 || botPathAfterWall >= 16)) {
                    if (!silent) System.out.println("[BOT v10] EMERGENCY REJECT wall " + bestWall.wall
                            + " selfHarm=" + selfHarmActual);
                    bestWall = null;
                } else if (bestWall != null) {
                    // === v10b WALL SCORING ===
                    // Net tempo: how many turns does this wall gain in the race?
                    double netTempo = pathDamage - selfHarmActual - 1.0;

                    // Base wall value from net tempo
                    wallValue = netTempo * 15.0;

                    // Add raw evaluator score (synergy, position bonuses)
                    wallValue += bestWall.score * 0.5;

                    // === CONTEXT BONUSES (v10b: GATED by pathDamage) ===

                    // BONUS 1: No forward move — but ONLY for strong walls (pathDmg >= 2)
                    // pathDmg=1 walls waste wall tokens when trapped; better to move
                    if (!hasForwardMove && pathDamage >= 2) {
                        wallValue += 25.0;
                        if (!silent) System.out.println("[BOT v10] NO-FORWARD BONUS +25 (pathDmg=" + pathDamage + ")");
                    }

                    // BONUS 2: Best move is sideways AND wall has good damage
                    if (!bestMoveIsForward && pathDamage >= 2) {
                        wallValue += 10.0;
                        if (!silent) System.out.println("[BOT v10] SIDEWAYS-MOVE BONUS +10");
                    }

                    // BONUS 3: Bot is losing the race — but only for strong walls
                    double raceGap = botDist - oppDist;
                    if (raceGap > 2 && pathDamage >= 2) {
                        double racingBonus = Math.min(raceGap * 2.0, 20.0);
                        wallValue += racingBonus;
                        if (!silent) System.out.println("[BOT v10] LOSING-RACE BONUS +" + String.format("%.0f", racingBonus));
                    }

                    // BONUS 4: Emergency proximity
                    if (oppDist <= 3) {
                        wallValue += (4 - oppDist) * 10.0;
                    }

                    // BONUS 5: PROACTIVE WALLING — even with forward move, good walls are worth placing
                    // When the bot has a forward move but the wall has netTempo >= 1,
                    // the wall is objectively better than just moving forward
                    if (bestMoveIsForward && netTempo >= 1.0) {
                        wallValue += 10.0; // Make it competitive with forward moves (~65)
                        if (!silent) System.out.println("[BOT v10] PROACTIVE-WALL BONUS +10 (netTempo=" + String.format("%.1f", netTempo) + ")");
                    }

                    // PENALTY: Don't over-wall when winning comfortably
                    if (raceGap < -3 && oppDist >= 6) {
                        wallValue -= 15.0;
                    }

                    if (!silent) System.out.println("[BOT v10] wallRaw=" + String.format("%.1f", bestWall.score)
                            + " pathDmg=" + pathDamage + " selfHarm=" + selfHarmActual
                            + " netTempo=" + String.format("%.1f", netTempo)
                            + " wallValue=" + String.format("%.1f", wallValue)
                            + " moveValue=" + String.format("%.1f", moveValue)
                            + " moveIsFwd=" + bestMoveIsForward);
                }
            }
        } else if (!canWall && wallsLeft > 0) {
            if (!silent) System.out.println("[BOT v10] WALL SUPPRESSED: consecutive=" + consecutiveWalls
                    + " selfHarmTotal=" + cumulativeSelfHarm);
        }

        // === DIVERSITY NOTE ===
        // Game diversity for ML training comes from random opening moves (Rule 0).
        // Each BotBrain instance has a unique Random seed, so two bots created
        // simultaneously will make different random opening choices.
        // After the opening, play is fully deterministic for maximum quality.

        // === DYNAMIC DECISION ===
        if (bestMove == null && bestWall == null) return null;

        // Determine if wall or move wins
        boolean preferWall = (bestWall != null && wallValue > moveValue && bestWall.score > 0);

        // Note: wall-vs-move decision is kept deterministic for maximum quality.
        // Diversity comes only from random opening moves (Rule 0) with different seeds.

        if (preferWall) {
            int botPathAfter = PathFinder.aStarWithWall(state, bot, bestWall.wall);
            int selfHarm = (botPathAfter >= 0) ? botPathAfter - botDist : 0;
            cumulativeSelfHarm += Math.max(0, selfHarm);

            if (!silent) System.out.println("[BOT v10] >> WALL: " + bestWall.wall
                    + " (wallVal=" + String.format("%.1f", wallValue)
                    + " > moveVal=" + String.format("%.1f", moveValue)
                    + " selfHarmTotal=" + cumulativeSelfHarm + ")");
            return doWall(BotAction.wall(bestWall.wall, bestWall.score));
        }

        if (bestMove != null) {
            if (!silent) System.out.println("[BOT v10] >> MOVE: " + bestMove.target
                    + " (moveVal=" + String.format("%.1f", moveValue)
                    + ", wallVal=" + String.format("%.1f", wallValue) + ")");
            return doMove(BotAction.move(bestMove.target, bestMove.score));
        }

        if (bestWall != null) {
            return doWall(BotAction.wall(bestWall.wall, bestWall.score));
        }

        return null;
    }

    /**
     * Checks if the bot has any move that advances toward its goal row.
     */
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

    // ==================== ACTION WRAPPERS ====================

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

        // forceMove is for instant wins — NO noise (always pick the best winning move)
        MoveEvaluator.ScoredMove bestMove = MoveEvaluator.findBestMove(state, graph);
        if (bestMove != null) {
            return BotAction.move(bestMove.target, bestMove.score);
        }

        return null;
    }
}
