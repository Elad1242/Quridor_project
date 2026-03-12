package bot;

import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import org.bson.Document;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * Exports game records to MongoDB Atlas for neural network training.
 *
 * Each game is stored as a single document in the "games" collection:
 * {
 *   gameId: 1,
 *   winner: 0,
 *   totalTurns: 28,
 *   timestamp: ISODate,
 *   turns: [
 *     {
 *       turnNumber: 0,
 *       currentPlayer: 0,
 *       features: { myDist: 8, oppDist: 8, ... },
 *       action: { type: "move", row: 7, col: 4, orientation: null }
 *     },
 *     ...
 *   ]
 * }
 */
public class MongoExporter {

    private final MongoClient client;
    private final MongoCollection<Document> collection;

    /**
     * Connects to MongoDB Atlas using the provided connection string.
     *
     * @param connectionString MongoDB Atlas URI (e.g., "mongodb+srv://user:pass@cluster.mongodb.net/")
     * @param databaseName     database name (e.g., "quoridor")
     */
    public MongoExporter(String connectionString, String databaseName) {
        this.client = MongoClients.create(connectionString);
        MongoDatabase db = client.getDatabase(databaseName);
        this.collection = db.getCollection("games");
    }

    /**
     * Exports a single game record to MongoDB.
     *
     * @param gameId the sequential game identifier
     * @param record the complete game record from GameSimulator
     */
    public void exportGame(int gameId, GameSimulator.GameRecord record) {
        Document doc = new Document();
        doc.append("gameId", gameId);
        doc.append("winner", record.winnerIndex);
        doc.append("totalTurns", record.totalTurns);
        doc.append("timestamp", new Date());

        // Build turns array
        List<Document> turnDocs = new ArrayList<>();
        String[] featureNames = GameFeatures.featureNames();

        for (int i = 0; i < record.turns.size(); i++) {
            GameSimulator.TurnRecord turn = record.turns.get(i);

            // Features as named fields for readability in MongoDB
            Document featureDoc = new Document();
            for (int f = 0; f < turn.features.length; f++) {
                featureDoc.append(featureNames[f], turn.features[f]);
            }

            // Action details
            Document actionDoc = new Document();
            actionDoc.append("type", turn.action.type);
            actionDoc.append("row", turn.action.row);
            actionDoc.append("col", turn.action.col);
            actionDoc.append("orientation", turn.action.orientation);

            Document turnDoc = new Document();
            turnDoc.append("turnNumber", i);
            turnDoc.append("currentPlayer", turn.currentPlayer);
            turnDoc.append("features", featureDoc);
            turnDoc.append("action", actionDoc);

            turnDocs.add(turnDoc);
        }

        doc.append("turns", turnDocs);
        collection.insertOne(doc);
    }

    /**
     * Exports multiple game records in a batch for better performance.
     *
     * @param games  list of game records
     * @param startId the starting game ID
     */
    public void exportBatch(List<GameSimulator.GameRecord> games, int startId) {
        List<Document> docs = new ArrayList<>();
        String[] featureNames = GameFeatures.featureNames();

        for (int g = 0; g < games.size(); g++) {
            GameSimulator.GameRecord record = games.get(g);
            Document doc = new Document();
            doc.append("gameId", startId + g);
            doc.append("winner", record.winnerIndex);
            doc.append("totalTurns", record.totalTurns);
            doc.append("timestamp", new Date());

            List<Document> turnDocs = new ArrayList<>();
            for (int i = 0; i < record.turns.size(); i++) {
                GameSimulator.TurnRecord turn = record.turns.get(i);

                Document featureDoc = new Document();
                for (int f = 0; f < turn.features.length; f++) {
                    featureDoc.append(featureNames[f], turn.features[f]);
                }

                Document actionDoc = new Document();
                actionDoc.append("type", turn.action.type);
                actionDoc.append("row", turn.action.row);
                actionDoc.append("col", turn.action.col);
                actionDoc.append("orientation", turn.action.orientation);

                Document turnDoc = new Document();
                turnDoc.append("turnNumber", i);
                turnDoc.append("currentPlayer", turn.currentPlayer);
                turnDoc.append("features", featureDoc);
                turnDoc.append("action", actionDoc);

                turnDocs.add(turnDoc);
            }

            doc.append("turns", turnDocs);
            docs.add(doc);
        }

        if (!docs.isEmpty()) {
            collection.insertMany(docs);
        }
    }

    /**
     * Returns the number of games already stored in the collection.
     */
    public long getGameCount() {
        return collection.countDocuments();
    }

    /**
     * Closes the MongoDB connection.
     */
    public void close() {
        if (client != null) {
            client.close();
        }
    }
}
