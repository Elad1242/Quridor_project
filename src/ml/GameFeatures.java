package ml;

import bot.BoardGraph;
import logic.FlowCalculator;
import logic.PathFinder;
import model.GameState;
import model.Player;
import model.Position;

/**
 * Extracts 12 numerical features from a GameState for neural network input.
 *
 * Features are always computed from the perspective of the CURRENT player
 * (the one whose turn it is). This ensures consistent orientation regardless
 * of which player index is active.
 *
 * Feature list:
 *  0. myDist         - A* shortest path distance to my goal
 *  1. oppDist        - A* shortest path distance to opponent's goal
 *  2. myWallsLeft    - How many walls I have remaining
 *  3. oppWallsLeft   - How many walls opponent has remaining
 *  4. myMaxFlow      - Number of independent paths I have to goal
 *  5. oppMaxFlow     - Number of independent paths opponent has to goal
 *  6. mySafety       - Dijkstra weighted (safe) path cost to my goal
 *  7. oppSafety      - Dijkstra weighted (safe) path cost to opponent's goal
 *  8. raceGap        - myDist - oppDist (negative = I'm closer = good)
 *  9. turnNumber     - Current turn count in the game
 * 10. totalWalls     - Total walls placed on the board
 * 11. distToOpponent - Manhattan distance between the two pawns
 */
public class GameFeatures {

    public static final int FEATURE_COUNT = 12;

    /**
     * Extracts all 12 features from the given game state.
     * Features are from the perspective of state.getCurrentPlayer().
     *
     * @param state the game state to analyze
     * @return array of 12 doubles representing the features
     */
    public static double[] extract(GameState state) {
        Player me = state.getCurrentPlayer();
        Player opp = state.getOtherPlayer();

        // Distances via A*
        int myDist = PathFinder.aStarShortestPath(state, me);
        int oppDist = PathFinder.aStarShortestPath(state, opp);

        // Walls remaining
        int myWallsLeft = me.getWallsRemaining();
        int oppWallsLeft = opp.getWallsRemaining();

        // Max flow (independent paths)
        int myMaxFlow = FlowCalculator.calculateMaxFlow(state, me);
        int oppMaxFlow = FlowCalculator.calculateMaxFlow(state, opp);

        // Safety (weighted Dijkstra path cost)
        // Build graph from each player's perspective for correct wall ownership
        int myIndex = state.getCurrentPlayerIndex();
        int oppIndex = 1 - myIndex;

        BoardGraph myGraph = new BoardGraph();
        myGraph.buildFromStateForPlayer(state, opp.getPosition(), myIndex);
        double mySafety = PathFinder.dijkstraSafestPath(myGraph, me.getPosition(), me.getGoalRow());

        BoardGraph oppGraph = new BoardGraph();
        oppGraph.buildFromStateForPlayer(state, me.getPosition(), oppIndex);
        double oppSafety = PathFinder.dijkstraSafestPath(oppGraph, opp.getPosition(), opp.getGoalRow());

        // Derived features
        double raceGap = myDist - oppDist;
        int turnNumber = state.getTurnCount();
        int totalWalls = state.getWalls().size();
        int distToOpponent = manhattan(me.getPosition(), opp.getPosition());

        return new double[] {
            myDist, oppDist,
            myWallsLeft, oppWallsLeft,
            myMaxFlow, oppMaxFlow,
            mySafety, oppSafety,
            raceGap, turnNumber,
            totalWalls, distToOpponent
        };
    }

    /**
     * Returns human-readable names for each feature index.
     */
    public static String[] featureNames() {
        return new String[] {
            "myDist", "oppDist",
            "myWallsLeft", "oppWallsLeft",
            "myMaxFlow", "oppMaxFlow",
            "mySafety", "oppSafety",
            "raceGap", "turnNumber",
            "totalWalls", "distToOpponent"
        };
    }

    private static int manhattan(Position a, Position b) {
        return Math.abs(a.getRow() - b.getRow()) + Math.abs(a.getCol() - b.getCol());
    }
}
