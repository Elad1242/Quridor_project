package ml;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * CNN Value Network for Quoridor position evaluation.
 *
 * Architecture: ResNet-style with 4 residual blocks.
 *   Input:  8 x 9 x 9
 *   Conv(8→128, 3x3, pad=1) + BN + ReLU
 *   ResBlock(128) x 4
 *   Conv(128→1, 1x1) + BN + ReLU    (value head)
 *   Flatten(81) → Dense(256) + ReLU → Dropout(0.3) → Dense(1) + Sigmoid
 *
 * ~1.2M parameters.
 */
public class ValueNetwork {

    public static final int CHANNELS = 8;
    public static final int BOARD_SIZE = 9;
    private static final int FILTERS = 128;
    private static final int NUM_RES_BLOCKS = 4;

    private Model model;
    private NDManager manager;

    public ValueNetwork() {
        this.model = Model.newInstance("quoridor-value-net");
        this.model.setBlock(buildNetwork());
        this.manager = NDManager.newBaseManager();
    }

    /**
     * Builds the CNN block architecture.
     */
    public static Block buildNetwork() {
        SequentialBlock net = new SequentialBlock();

        // Initial convolution: 8 -> 128
        net.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .setFilters(FILTERS)
                .build());
        net.add(BatchNorm.builder().build());
        net.add(Activation::relu);

        // 4 Residual blocks
        for (int i = 0; i < NUM_RES_BLOCKS; i++) {
            net.add(new ResidualBlock(FILTERS));
        }

        // Value head: 128 -> 1 channel
        net.add(Conv2d.builder()
                .setKernelShape(new Shape(1, 1))
                .setFilters(1)
                .build());
        net.add(BatchNorm.builder().build());
        net.add(Activation::relu);

        // Flatten 1x9x9 = 81
        net.add(Blocks.batchFlattenBlock());

        // Dense layers
        net.add(Linear.builder().setUnits(256).build());
        net.add(Activation::relu);
        net.add(Dropout.builder().optRate(0.3f).build());
        net.add(Linear.builder().setUnits(1).build());
        net.add(Activation::sigmoid);

        return net;
    }

    /**
     * Evaluates a single board state.
     * @return value in [0, 1] — how good this position is for the current player
     */
    public float evaluate(float[][][] board) {
        try (NDManager subManager = manager.newSubManager()) {
            NDArray input = subManager.create(board).reshape(1, CHANNELS, BOARD_SIZE, BOARD_SIZE);
            NDArray output = predict(input);
            return output.getFloat(0);
        }
    }

    /**
     * Evaluates a batch of board states.
     */
    public float[] evaluateBatch(float[][][][] boards) {
        try (NDManager subManager = manager.newSubManager()) {
            NDArray input = subManager.create(boards);
            NDArray output = predict(input);
            return output.toFloatArray();
        }
    }

    private NDArray predict(NDArray input) {
        ParameterStore ps = new ParameterStore(input.getManager(), false);
        return model.getBlock().forward(ps, new NDList(input), false).singletonOrThrow();
    }

    /**
     * Gets the DJL Model for training.
     */
    public Model getModel() {
        return model;
    }

    /**
     * Saves the model to disk.
     */
    public void save(String dirPath, String name) throws IOException {
        Path dir = Paths.get(dirPath);
        dir.toFile().mkdirs();
        model.setProperty("Epoch", "0");
        model.save(dir, name);
        System.out.println("Model saved to " + dirPath + "/" + name);
    }

    /**
     * Loads a model from disk.
     */
    public void load(String dirPath, String name) throws Exception {
        Path dir = Paths.get(dirPath);
        model.setBlock(buildNetwork());
        model.load(dir, name);
        System.out.println("Model loaded from " + dirPath + "/" + name);
    }

    public void close() {
        if (manager != null) manager.close();
        if (model != null) model.close();
    }

    /**
     * Residual Block: conv + bn + relu + conv + bn + skip connection + relu
     */
    public static class ResidualBlock extends ai.djl.nn.AbstractBlock {

        private final SequentialBlock convPath;

        public ResidualBlock(int filters) {
            super();
            convPath = new SequentialBlock();
            convPath.add(Conv2d.builder()
                    .setKernelShape(new Shape(3, 3))
                    .optPadding(new Shape(1, 1))
                    .setFilters(filters)
                    .build());
            convPath.add(BatchNorm.builder().build());
            convPath.add(Activation::relu);
            convPath.add(Conv2d.builder()
                    .setKernelShape(new Shape(3, 3))
                    .optPadding(new Shape(1, 1))
                    .setFilters(filters)
                    .build());
            convPath.add(BatchNorm.builder().build());
            addChildBlock("convPath", convPath);
        }

        @Override
        protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                          ai.djl.util.PairList<String, Object> params) {
            NDArray x = inputs.singletonOrThrow();
            NDArray residual = convPath.forward(ps, inputs, training, params).singletonOrThrow();
            return new NDList(Activation.relu(residual.add(x)));
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            return inputShapes;
        }
    }
}
