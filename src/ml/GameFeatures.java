// v2.0 — refactored and cleaned, May 2026
package ml;

import bot.BoardGraph;
import logic.FlowCalculator;
import logic.MoveValidator;
import logic.PathFinder;
import model.GameState;
import model.Player;
import model.Position;
import model.Wall;

import java.util.List;

// Extracts 27 features from the board for the neural network (22 global + 5 per-action).
public class GameFeatures {

    public static final int NUM_FEATURES = 22;
    public static final int NUM_ACTION_FEATURES = 5;
    public static final int TOTAL_FEATURES = NUM_FEATURES + NUM_ACTION_FEATURES; // 27

    // 22 global features from the current player's perspective
    public static double[] extract(GameState state) {
        double[] f = new double[NUM_FEATURES];

        Player me  = state.getCurrentPlayer();
        Player opp = state.getOtherPlayer();
        Position myPos  = me.getPosition();
        Position oppPos = opp.getPosition();

        int myAstar  = PathFinder.aStarShortestPath(state, me);
        int oppAstar = PathFinder.aStarShortestPath(state, opp);
        int myBfs    = PathFinder.shortestPath(state, me);
        int oppBfs   = PathFinder.shortestPath(state, opp);

        // clamp unreachable (-1) to a large value
        if (myAstar  < 0) myAstar  = 20;
        if (oppAstar < 0) oppAstar = 20;
        if (myBfs    < 0) myBfs    = 20;
        if (oppBfs   < 0) oppBfs   = 20;

        // race features
        f[0] = myAstar  / 16.0;
        f[1] = oppAstar / 16.0;
        f[2] = (oppAstar - myAstar) / 16.0; // positive = I'm ahead
        f[3] = myBfs    / 16.0;
        f[4] = oppBfs   / 16.0;
        f[5] = (myAstar < oppAstar) ? 1.0 : 0.0;

        // position features
        f[6]  = myPos.getRow()  / 8.0;
        f[7]  = oppPos.getRow() / 8.0;
        f[8]  = myPos.getCol()  / 8.0;
        f[9]  = oppPos.getCol() / 8.0;
        f[10] = Math.abs(myPos.getRow()  - oppPos.getRow()) / 8.0;
        f[11] = Math.abs(myPos.getCol()  - oppPos.getCol()) / 8.0;

        // resource features
        f[12] = me.getWallsRemaining()  / 10.0;
        f[13] = opp.getWallsRemaining() / 10.0;
        f[14] = (me.getWallsRemaining() - opp.getWallsRemaining()) / 10.0;
        f[15] = state.getWalls().size() / 20.0;

        // mobility features
        List<Position> myMoves = MoveValidator.getValidMoves(state, me);
        int goalRow    = me.getGoalRow();
        int forwardDir = (goalRow < myPos.getRow()) ? -1 : 1;
        int forwardMoves = 0;
        for (Position p : myMoves) {
            if ((p.getRow() - myPos.getRow()) * forwardDir > 0) forwardMoves++;
        }
        f[16] = forwardMoves   / 4.0;
        f[17] = myMoves.size() / 8.0;

        // flow = number of independent paths to goal
        f[18] = FlowCalculator.calculateMaxFlow(state, me)  / 6.0;
        f[19] = FlowCalculator.calculateMaxFlow(state, opp) / 6.0;

        // Dijkstra safety cost (weighted path length)
        BoardGraph myGraph = new BoardGraph();
        myGraph.buildFromState(state, oppPos);
        f[20] = PathFinder.dijkstraSafestPath(myGraph, myPos, goalRow) / 20.0;

        // opponent's safety from their own perspective
        GameState oppState = state.deepCopy();
        oppState.nextTurn();
        BoardGraph oppGraph = new BoardGraph();
        oppGraph.buildFromState(oppState, myPos);
        f[21] = PathFinder.dijkstraSafestPath(oppGraph, oppPos, opp.getGoalRow()) / 20.0;

        return f;
    }

    // 27 features for a MOVE action: 22 global + 5 action-specific
    public static double[] extractForMove(GameState state, Position move) {
        double[] global = extract(state);
        double[] full = new double[TOTAL_FEATURES];
        System.arraycopy(global, 0, full, 0, NUM_FEATURES);

        Player me = state.getCurrentPlayer();
        int myDistBefore = PathFinder.aStarShortestPath(state, me);
        if (myDistBefore < 0) myDistBefore = 20;

        GameState sim = state.deepCopy();
        sim.getCurrentPlayer().setPosition(move);
        int myDistAfter = PathFinder.aStarShortestPath(sim, sim.getCurrentPlayer());
        if (myDistAfter < 0) myDistAfter = 20;

        int currentRow = me.getPosition().getRow();
        int goalRow    = me.getGoalRow();
        int forwardDir = (goalRow < currentRow) ? -1 : 1;
        int rowChange  = (move.getRow() - currentRow) * forwardDir;

        List<Position> astarPath = PathFinder.getAStarPath(state, me);
        boolean onAstar = astarPath != null && astarPath.size() >= 2 && astarPath.get(1).equals(move);

        full[22] = (myDistBefore - myDistAfter) / 4.0; // path gain (positive = got closer)
        full[23] = rowChange / 2.0;                     // row advance (positive = toward goal)
        full[24] = onAstar ? 1.0 : 0.0;                 // follows A* path
        full[25] = myDistAfter / 16.0;                  // absolute distance after move
        full[26] = 0.0;                                  // actionType = MOVE

        return full;
    }

    // 27 features for a WALL action: 22 global + 5 action-specific
    public static double[] extractForWall(GameState state, Wall wall) {
        double[] global = extract(state);
        double[] full = new double[TOTAL_FEATURES];
        System.arraycopy(global, 0, full, 0, NUM_FEATURES);

        Player me  = state.getCurrentPlayer();
        Player opp = state.getOtherPlayer();

        int myDistBefore  = PathFinder.aStarShortestPath(state, me);
        int oppDistBefore = PathFinder.aStarShortestPath(state, opp);
        if (myDistBefore  < 0) myDistBefore  = 20;
        if (oppDistBefore < 0) oppDistBefore = 20;

        int myDistAfter  = PathFinder.aStarWithWall(state, me,  wall);
        int oppDistAfter = PathFinder.aStarWithWall(state, opp, wall);
        if (myDistAfter  < 0) myDistAfter  = 20;
        if (oppDistAfter < 0) oppDistAfter = 20;

        int pathDamage = oppDistAfter - oppDistBefore;
        int selfHarm   = myDistAfter  - myDistBefore;
        int netDamage  = pathDamage   - selfHarm;

        full[22] = pathDamage  / 6.0;    // how much opponent path increased
        full[23] = selfHarm    / 6.0;    // how much my path increased (bad)
        full[24] = netDamage   / 6.0;    // net advantage
        full[25] = oppDistAfter / 16.0;  // opponent's new distance
        full[26] = 1.0;                  // actionType = WALL

        return full;
    }
}
