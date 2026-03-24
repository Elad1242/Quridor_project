package ml;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Efficient binary I/O for training data.
 *
 * Format: each record = 648 floats (8x9x9 board) + 1 float (label) = 2596 bytes.
 * File header: 4 bytes magic "QDAT" + 4 bytes record count.
 */
public class TrainingDataWriter implements Closeable {

    private static final byte[] MAGIC = {'Q', 'D', 'A', 'T'};
    private static final int BOARD_FLOATS = BoardEncoder.CHANNELS * BoardEncoder.BOARD_SIZE * BoardEncoder.BOARD_SIZE; // 648
    private static final int RECORD_BYTES = (BOARD_FLOATS + 1) * 4; // 2596

    private final DataOutputStream out;
    private int recordCount = 0;
    private final String filePath;

    public TrainingDataWriter(String filePath) throws IOException {
        this.filePath = filePath;
        this.out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filePath), 1 << 20));
        // Write header placeholder (will be updated on close)
        out.write(MAGIC);
        out.writeInt(0); // placeholder for record count
    }

    /**
     * Writes one training sample (board state + label).
     */
    public synchronized void write(float[][][] board, float label) throws IOException {
        for (int ch = 0; ch < BoardEncoder.CHANNELS; ch++) {
            for (int r = 0; r < BoardEncoder.BOARD_SIZE; r++) {
                for (int c = 0; c < BoardEncoder.BOARD_SIZE; c++) {
                    out.writeFloat(board[ch][r][c]);
                }
            }
        }
        out.writeFloat(label);
        recordCount++;
    }

    @Override
    public void close() throws IOException {
        out.close();
        // Update record count in header
        try (RandomAccessFile raf = new RandomAccessFile(filePath, "rw")) {
            raf.seek(4); // skip magic
            raf.writeInt(recordCount);
        }
    }

    public int getRecordCount() {
        return recordCount;
    }

    // ===== READER =====

    /**
     * Reads all training data from a binary file.
     */
    public static TrainingData read(String filePath) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath), 1 << 20))) {
            byte[] magic = new byte[4];
            in.readFully(magic);
            if (magic[0] != 'Q' || magic[1] != 'D' || magic[2] != 'A' || magic[3] != 'T') {
                throw new IOException("Invalid file format: bad magic bytes");
            }
            int count = in.readInt();

            float[][][][] boards = new float[count][BoardEncoder.CHANNELS][BoardEncoder.BOARD_SIZE][BoardEncoder.BOARD_SIZE];
            float[] labels = new float[count];

            for (int i = 0; i < count; i++) {
                for (int ch = 0; ch < BoardEncoder.CHANNELS; ch++) {
                    for (int r = 0; r < BoardEncoder.BOARD_SIZE; r++) {
                        for (int c = 0; c < BoardEncoder.BOARD_SIZE; c++) {
                            boards[i][ch][r][c] = in.readFloat();
                        }
                    }
                }
                labels[i] = in.readFloat();
            }

            return new TrainingData(boards, labels);
        }
    }

    /**
     * Reads training data from multiple files and merges them.
     */
    public static TrainingData readAll(String... filePaths) throws IOException {
        List<float[][][]> allBoards = new ArrayList<>();
        List<Float> allLabels = new ArrayList<>();

        for (String path : filePaths) {
            File f = new File(path);
            if (!f.exists()) continue;
            TrainingData td = read(path);
            for (int i = 0; i < td.labels.length; i++) {
                allBoards.add(td.boards[i]);
                allLabels.add(td.labels[i]);
            }
        }

        float[][][][] boards = new float[allBoards.size()][][][];
        float[] labels = new float[allLabels.size()];
        for (int i = 0; i < allBoards.size(); i++) {
            boards[i] = allBoards.get(i);
            labels[i] = allLabels.get(i);
        }

        return new TrainingData(boards, labels);
    }

    /**
     * Container for training data.
     */
    public static class TrainingData {
        public final float[][][][] boards; // [samples][channels][rows][cols]
        public final float[] labels;       // [samples]

        public TrainingData(float[][][][] boards, float[] labels) {
            this.boards = boards;
            this.labels = labels;
        }

        public int size() {
            return labels.length;
        }

        /**
         * Shuffles the data in-place.
         */
        public void shuffle(Random rng) {
            for (int i = labels.length - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                // Swap boards
                float[][][] tmpBoard = boards[i];
                boards[i] = boards[j];
                boards[j] = tmpBoard;
                // Swap labels
                float tmpLabel = labels[i];
                labels[i] = labels[j];
                labels[j] = tmpLabel;
            }
        }

        /**
         * Splits into train/validation sets.
         */
        public TrainingData[] split(double trainRatio, Random rng) {
            shuffle(rng);
            int trainSize = (int) (labels.length * trainRatio);
            int valSize = labels.length - trainSize;

            float[][][][] trainBoards = new float[trainSize][][][];
            float[] trainLabels = new float[trainSize];
            System.arraycopy(boards, 0, trainBoards, 0, trainSize);
            System.arraycopy(labels, 0, trainLabels, 0, trainSize);

            float[][][][] valBoards = new float[valSize][][][];
            float[] valLabels = new float[valSize];
            System.arraycopy(boards, trainSize, valBoards, 0, valSize);
            System.arraycopy(labels, trainSize, valLabels, 0, valSize);

            return new TrainingData[]{
                new TrainingData(trainBoards, trainLabels),
                new TrainingData(valBoards, valLabels)
            };
        }
    }
}
