package ml;

import java.io.*;
import java.util.Random;

/**
 * From-scratch feedforward neural network.
 * No external libraries — pure Java arrays and math.
 *
 * Architecture: configurable layers with ReLU hidden activations and Sigmoid output.
 * Training: mini-batch SGD with momentum.
 * Loss: Mean Squared Error (MSE).
 *
 * Default for Quoridor: 22 → 64 → 32 → 16 → 1
 */
public class NeuralNetwork {

    private final int[] layerSizes;
    private final int numLayers; // number of weight layers (= layerSizes.length - 1)

    // Weights: weights[L][j][i] = weight from neuron i in layer L to neuron j in layer L+1
    private double[][][] weights;
    // Biases: biases[L][j] = bias for neuron j in layer L+1
    private double[][] biases;

    // Momentum velocities
    private double[][][] velocityW;
    private double[][] velocityB;

    // === Constructor ===

    public NeuralNetwork(int... layerSizes) {
        this.layerSizes = layerSizes;
        this.numLayers = layerSizes.length - 1;
        this.weights = new double[numLayers][][];
        this.biases = new double[numLayers][];
        this.velocityW = new double[numLayers][][];
        this.velocityB = new double[numLayers][];

        Random rng = new Random(42);
        for (int L = 0; L < numLayers; L++) {
            int fanIn = layerSizes[L];
            int fanOut = layerSizes[L + 1];
            weights[L] = new double[fanOut][fanIn];
            biases[L] = new double[fanOut];
            velocityW[L] = new double[fanOut][fanIn];
            velocityB[L] = new double[fanOut];

            // He initialization: stddev = sqrt(2 / fanIn)
            double stddev = Math.sqrt(2.0 / fanIn);
            for (int j = 0; j < fanOut; j++) {
                for (int i = 0; i < fanIn; i++) {
                    weights[L][j][i] = rng.nextGaussian() * stddev;
                }
                biases[L][j] = 0.01; // small positive bias for ReLU
            }
        }
    }

    // === Forward Pass ===

    /**
     * Forward pass. Returns the output (single value for value network).
     * Also stores intermediate values for backpropagation.
     */
    public ForwardResult forward(double[] input) {
        double[][] activations = new double[numLayers + 1][];
        double[][] preActivations = new double[numLayers][];

        activations[0] = input;

        for (int L = 0; L < numLayers; L++) {
            int outSize = layerSizes[L + 1];
            int inSize = layerSizes[L];
            preActivations[L] = new double[outSize];
            activations[L + 1] = new double[outSize];

            for (int j = 0; j < outSize; j++) {
                double sum = biases[L][j];
                for (int i = 0; i < inSize; i++) {
                    sum += weights[L][j][i] * activations[L][i];
                }
                preActivations[L][j] = sum;

                // Activation: ReLU for hidden layers, Sigmoid for output
                if (L < numLayers - 1) {
                    activations[L + 1][j] = Math.max(0, sum); // ReLU
                } else {
                    activations[L + 1][j] = 1.0 / (1.0 + Math.exp(-sum)); // Sigmoid
                }
            }
        }

        return new ForwardResult(activations, preActivations);
    }

    /**
     * Simple predict — just returns the output value.
     */
    public double predict(double[] input) {
        return forward(input).getOutput()[0];
    }

    // === Backpropagation ===

    /**
     * Computes gradients for one sample using backpropagation.
     * Returns the gradients (dW, dB) without applying them.
     */
    public Gradients backprop(double[] input, double target) {
        ForwardResult fwd = forward(input);
        double[][] activations = fwd.activations;
        double[][] preActivations = fwd.preActivations;

        double[][][] dW = new double[numLayers][][];
        double[][] dB = new double[numLayers][];
        double[][] deltas = new double[numLayers][];

        // Output layer delta: (predicted - target) * sigmoid'(z)
        int lastL = numLayers - 1;
        double predicted = activations[numLayers][0];
        double error = predicted - target;
        double sigmoidDeriv = predicted * (1 - predicted);
        deltas[lastL] = new double[]{error * sigmoidDeriv};

        // Hidden layer deltas (backward)
        for (int L = lastL - 1; L >= 0; L--) {
            int size = layerSizes[L + 1];
            int nextSize = layerSizes[L + 2];
            deltas[L] = new double[size];

            for (int j = 0; j < size; j++) {
                double sum = 0;
                for (int k = 0; k < nextSize; k++) {
                    sum += weights[L + 1][k][j] * deltas[L + 1][k];
                }
                // ReLU derivative: 1 if z > 0, else 0
                deltas[L][j] = sum * (preActivations[L][j] > 0 ? 1.0 : 0.0);
            }
        }

        // Compute gradients
        for (int L = 0; L < numLayers; L++) {
            int outSize = layerSizes[L + 1];
            int inSize = layerSizes[L];
            dW[L] = new double[outSize][inSize];
            dB[L] = new double[outSize];

            for (int j = 0; j < outSize; j++) {
                dB[L][j] = deltas[L][j];
                for (int i = 0; i < inSize; i++) {
                    dW[L][j][i] = deltas[L][j] * activations[L][i];
                }
            }
        }

        double loss = 0.5 * error * error; // MSE for single sample
        return new Gradients(dW, dB, loss);
    }

    // === Training Step ===

    /**
     * Updates weights using accumulated gradients from a mini-batch.
     */
    public void updateWeights(double[][][] gradW, double[][] gradB, int batchSize,
                               double learningRate, double momentum, double weightDecay) {
        for (int L = 0; L < numLayers; L++) {
            int outSize = layerSizes[L + 1];
            int inSize = layerSizes[L];

            for (int j = 0; j < outSize; j++) {
                // Bias update
                velocityB[L][j] = momentum * velocityB[L][j]
                        - learningRate * (gradB[L][j] / batchSize);
                biases[L][j] += velocityB[L][j];

                // Weight update with L2 regularization
                for (int i = 0; i < inSize; i++) {
                    velocityW[L][j][i] = momentum * velocityW[L][j][i]
                            - learningRate * (gradW[L][j][i] / batchSize + weightDecay * weights[L][j][i]);
                    weights[L][j][i] += velocityW[L][j][i];
                }
            }
        }
    }

    // === Save / Load ===

    public void save(String filePath) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filePath)))) {
            // Write layer sizes
            out.writeInt(layerSizes.length);
            for (int s : layerSizes) out.writeInt(s);

            // Write weights and biases
            for (int L = 0; L < numLayers; L++) {
                for (int j = 0; j < layerSizes[L + 1]; j++) {
                    for (int i = 0; i < layerSizes[L]; i++) {
                        out.writeDouble(weights[L][j][i]);
                    }
                    out.writeDouble(biases[L][j]);
                }
            }
        }
    }

    public static NeuralNetwork load(String filePath) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            int numSizes = in.readInt();
            int[] sizes = new int[numSizes];
            for (int i = 0; i < numSizes; i++) sizes[i] = in.readInt();

            NeuralNetwork nn = new NeuralNetwork(sizes);
            for (int L = 0; L < nn.numLayers; L++) {
                for (int j = 0; j < sizes[L + 1]; j++) {
                    for (int i = 0; i < sizes[L]; i++) {
                        nn.weights[L][j][i] = in.readDouble();
                    }
                    nn.biases[L][j] = in.readDouble();
                }
            }
            return nn;
        }
    }

    public int getInputSize() { return layerSizes[0]; }
    public int getOutputSize() { return layerSizes[layerSizes.length - 1]; }
    public int getParamCount() {
        int count = 0;
        for (int L = 0; L < numLayers; L++) {
            count += layerSizes[L + 1] * layerSizes[L]; // weights
            count += layerSizes[L + 1]; // biases
        }
        return count;
    }

    // === Inner classes ===

    public static class ForwardResult {
        public final double[][] activations;
        public final double[][] preActivations;

        ForwardResult(double[][] activations, double[][] preActivations) {
            this.activations = activations;
            this.preActivations = preActivations;
        }

        public double[] getOutput() {
            return activations[activations.length - 1];
        }
    }

    public static class Gradients {
        public final double[][][] dW;
        public final double[][] dB;
        public final double loss;

        Gradients(double[][][] dW, double[][] dB, double loss) {
            this.dW = dW;
            this.dB = dB;
            this.loss = loss;
        }
    }
}
