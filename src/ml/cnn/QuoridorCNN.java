package ml.cnn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

/**
 * Convolutional Neural Network for Quoridor action evaluation.
 *
 * Architecture:
 * - Input: 8 channels x 9x9 board representation
 * - 3 convolutional layers with batch normalization
 * - Global average pooling
 * - 2 dense layers
 * - Output: single value (sigmoid) predicting action quality
 *
 * The CNN learns spatial patterns in the board state to evaluate
 * how good a position is for the current player.
 */
public class QuoridorCNN {

    private MultiLayerNetwork model;

    private static final int CHANNELS = BoardEncoder.CHANNELS;
    private static final int BOARD_SIZE = BoardEncoder.BOARD_SIZE;

    /**
     * Creates a new CNN with randomly initialized weights.
     */
    public QuoridorCNN() {
        this.model = createModel();
        this.model.init();
    }

    /**
     * Loads a pre-trained CNN from a file.
     *
     * @param modelPath Path to saved model file
     */
    public QuoridorCNN(String modelPath) {
        try {
            this.model = ModelSerializer.restoreMultiLayerNetwork(new File(modelPath));
            System.out.println("Loaded CNN model from: " + modelPath);
        } catch (IOException e) {
            System.err.println("Failed to load model, creating new one: " + e.getMessage());
            this.model = createModel();
            this.model.init();
        }
    }

    /**
     * Creates a lightweight CNN architecture (fewer parameters for memory efficiency).
     */
    private MultiLayerNetwork createModel() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .l2(0.0001)
                .list()

                // Conv Block 1: 8 -> 32 channels
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(CHANNELS)
                        .nOut(32)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new BatchNormalization())

                // Conv Block 2: 32 -> 64 channels
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nOut(64)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new BatchNormalization())

                // Max pooling to reduce spatial dimensions
                .layer(4, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())

                // Conv Block 3: 64 -> 64 channels (on 4x4 after pooling)
                .layer(5, new ConvolutionLayer.Builder(3, 3)
                        .nOut(64)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(6, new BatchNormalization())

                // Global Average Pooling (reduces to 64 values)
                .layer(7, new GlobalPoolingLayer.Builder(PoolingType.AVG).build())

                // Dense layer
                .layer(8, new DenseLayer.Builder()
                        .nOut(64)
                        .activation(Activation.RELU)
                        .dropOut(0.3)
                        .build())

                // Output layer
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .build())

                .setInputType(InputType.convolutional(BOARD_SIZE, BOARD_SIZE, CHANNELS))
                .build();

        return new MultiLayerNetwork(conf);
    }

    /**
     * Predicts the quality of a board state.
     *
     * @param boardState Encoded board state [channels, height, width]
     * @return Value between 0 and 1 indicating position quality
     */
    public double predict(INDArray boardState) {
        // Add batch dimension if needed
        INDArray input;
        if (boardState.rank() == 3) {
            input = boardState.reshape(1, CHANNELS, BOARD_SIZE, BOARD_SIZE);
        } else {
            input = boardState;
        }

        INDArray output = model.output(input);
        return output.getDouble(0);
    }

    /**
     * Predicts quality for multiple board states at once.
     *
     * @param batchStates Batch of encoded states [batch, channels, height, width]
     * @return Array of quality predictions
     */
    public double[] predictBatch(INDArray batchStates) {
        INDArray output = model.output(batchStates);
        double[] results = new double[(int) batchStates.size(0)];
        for (int i = 0; i < results.length; i++) {
            results[i] = output.getDouble(i, 0);
        }
        return results;
    }

    /**
     * Trains the model on a batch of data.
     *
     * @param inputs Batch of board states [batch, channels, height, width]
     * @param labels Target values [batch, 1]
     */
    public void fit(INDArray inputs, INDArray labels) {
        model.fit(inputs, labels);
    }

    /**
     * Gets the underlying network for advanced operations.
     */
    public MultiLayerNetwork getNetwork() {
        return model;
    }

    /**
     * Saves the model to a file.
     *
     * @param path File path to save to
     */
    public void save(String path) {
        try {
            ModelSerializer.writeModel(model, new File(path), true);
            System.out.println("Model saved to: " + path);
        } catch (IOException e) {
            System.err.println("Failed to save model: " + e.getMessage());
        }
    }

    /**
     * Gets the total number of parameters in the network.
     */
    public long getParameterCount() {
        return model.numParams();
    }

    /**
     * Prints a summary of the network architecture.
     */
    public void printSummary() {
        System.out.println("=== Quoridor CNN Architecture ===");
        System.out.println("Input: " + CHANNELS + " channels x " + BOARD_SIZE + "x" + BOARD_SIZE);
        System.out.println("Total parameters: " + getParameterCount());
        System.out.println(model.summary());
    }
}
