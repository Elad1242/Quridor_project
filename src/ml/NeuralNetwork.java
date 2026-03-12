package ml;

import java.io.*;
import java.util.Random;

/**
 * A simple feedforward neural network for position evaluation.
 *
 * Architecture: 12 inputs → 32 (ReLU) → 16 (ReLU) → 1 (Sigmoid)
 *
 * The network takes 12 game features and outputs a win probability (0.0 to 1.0).
 * Trained via backpropagation with MSE loss.
 *
 * No external libraries — pure Java implementation.
 */
public class NeuralNetwork {

    // Layer sizes
    private final int inputSize;
    private final int hidden1Size;
    private final int hidden2Size;
    private final int outputSize = 1;

    // Weights and biases
    private double[][] w1; // inputSize x hidden1Size
    private double[] b1;   // hidden1Size
    private double[][] w2; // hidden1Size x hidden2Size
    private double[] b2;   // hidden2Size
    private double[][] w3; // hidden2Size x outputSize
    private double[] b3;   // outputSize

    // Cached activations for backpropagation
    private double[] z1, a1; // hidden layer 1
    private double[] z2, a2; // hidden layer 2
    private double[] z3, a3; // output layer

    private double learningRate;

    /**
     * Creates a neural network with default architecture (12 → 32 → 16 → 1).
     */
    public NeuralNetwork() {
        this(GameFeatures.FEATURE_COUNT, 32, 16, 0.001);
    }

    /**
     * Creates a neural network with custom architecture.
     *
     * @param inputSize   number of input features
     * @param hidden1Size number of neurons in first hidden layer
     * @param hidden2Size number of neurons in second hidden layer
     * @param learningRate learning rate for training
     */
    public NeuralNetwork(int inputSize, int hidden1Size, int hidden2Size, double learningRate) {
        this.inputSize = inputSize;
        this.hidden1Size = hidden1Size;
        this.hidden2Size = hidden2Size;
        this.learningRate = learningRate;

        // Initialize weights with He initialization (good for ReLU)
        Random rng = new Random(42);
        w1 = heInit(inputSize, hidden1Size, rng);
        b1 = new double[hidden1Size];
        w2 = heInit(hidden1Size, hidden2Size, rng);
        b2 = new double[hidden2Size];
        w3 = heInit(hidden2Size, outputSize, rng);
        b3 = new double[outputSize];
    }

    /**
     * He initialization: weights ~ N(0, sqrt(2/fanIn))
     * Optimal for ReLU activations.
     */
    private double[][] heInit(int fanIn, int fanOut, Random rng) {
        double[][] weights = new double[fanIn][fanOut];
        double stddev = Math.sqrt(2.0 / fanIn);
        for (int i = 0; i < fanIn; i++) {
            for (int j = 0; j < fanOut; j++) {
                weights[i][j] = rng.nextGaussian() * stddev;
            }
        }
        return weights;
    }

    // ==================== FORWARD PASS ====================

    /**
     * Forward pass: computes win probability for given features.
     *
     * @param input 12 game features (already normalized)
     * @return win probability between 0.0 and 1.0
     */
    public double predict(double[] input) {
        // Layer 1: input → hidden1 (ReLU)
        z1 = new double[hidden1Size];
        a1 = new double[hidden1Size];
        for (int j = 0; j < hidden1Size; j++) {
            double sum = b1[j];
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * w1[i][j];
            }
            z1[j] = sum;
            a1[j] = relu(sum);
        }

        // Layer 2: hidden1 → hidden2 (ReLU)
        z2 = new double[hidden2Size];
        a2 = new double[hidden2Size];
        for (int j = 0; j < hidden2Size; j++) {
            double sum = b2[j];
            for (int i = 0; i < hidden1Size; i++) {
                sum += a1[i] * w2[i][j];
            }
            z2[j] = sum;
            a2[j] = relu(sum);
        }

        // Layer 3: hidden2 → output (Sigmoid)
        z3 = new double[outputSize];
        a3 = new double[outputSize];
        double sum = b3[0];
        for (int i = 0; i < hidden2Size; i++) {
            sum += a2[i] * w3[i][0];
        }
        z3[0] = sum;
        a3[0] = sigmoid(sum);

        return a3[0];
    }

    // ==================== BACKPROPAGATION ====================

    /**
     * Trains on a single sample using backpropagation with MSE loss.
     *
     * @param input  12 game features
     * @param target expected output (1.0 = win, 0.0 = loss)
     * @return the loss for this sample
     */
    public double train(double[] input, double target) {
        // Forward pass (caches activations)
        double output = predict(input);

        // MSE loss: L = (output - target)^2
        double loss = (output - target) * (output - target);

        // dL/dOutput = 2 * (output - target)
        double dOutput = 2.0 * (output - target);

        // === Output layer gradients ===
        // dL/dz3 = dL/dOutput * sigmoid'(z3)
        double dz3 = dOutput * sigmoidDerivative(z3[0]);

        // Gradients for w3 and b3
        for (int i = 0; i < hidden2Size; i++) {
            w3[i][0] -= learningRate * dz3 * a2[i];
        }
        b3[0] -= learningRate * dz3;

        // === Hidden layer 2 gradients ===
        double[] dz2 = new double[hidden2Size];
        for (int j = 0; j < hidden2Size; j++) {
            double da2 = dz3 * w3[j][0];
            dz2[j] = da2 * reluDerivative(z2[j]);
        }

        for (int i = 0; i < hidden1Size; i++) {
            for (int j = 0; j < hidden2Size; j++) {
                w2[i][j] -= learningRate * dz2[j] * a1[i];
            }
        }
        for (int j = 0; j < hidden2Size; j++) {
            b2[j] -= learningRate * dz2[j];
        }

        // === Hidden layer 1 gradients ===
        double[] dz1 = new double[hidden1Size];
        for (int j = 0; j < hidden1Size; j++) {
            double da1 = 0;
            for (int k = 0; k < hidden2Size; k++) {
                da1 += dz2[k] * w2[j][k];
            }
            dz1[j] = da1 * reluDerivative(z1[j]);
        }

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hidden1Size; j++) {
                w1[i][j] -= learningRate * dz1[j] * input[i];
            }
        }
        for (int j = 0; j < hidden1Size; j++) {
            b1[j] -= learningRate * dz1[j];
        }

        return loss;
    }

    // ==================== ACTIVATION FUNCTIONS ====================

    private double relu(double x) {
        return Math.max(0, x);
    }

    private double reluDerivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    private double sigmoid(double x) {
        // Clamp to prevent overflow
        if (x > 500) return 1.0;
        if (x < -500) return 0.0;
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    // ==================== SAVE / LOAD ====================

    /**
     * Saves the network weights to a file.
     * Format: all weights and biases as raw doubles.
     */
    public void save(String filePath) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filePath))) {
            // Header
            dos.writeInt(inputSize);
            dos.writeInt(hidden1Size);
            dos.writeInt(hidden2Size);

            // w1, b1
            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < hidden1Size; j++)
                    dos.writeDouble(w1[i][j]);
            for (int j = 0; j < hidden1Size; j++)
                dos.writeDouble(b1[j]);

            // w2, b2
            for (int i = 0; i < hidden1Size; i++)
                for (int j = 0; j < hidden2Size; j++)
                    dos.writeDouble(w2[i][j]);
            for (int j = 0; j < hidden2Size; j++)
                dos.writeDouble(b2[j]);

            // w3, b3
            for (int i = 0; i < hidden2Size; i++)
                dos.writeDouble(w3[i][0]);
            dos.writeDouble(b3[0]);
        }
        System.out.println("Network saved to " + filePath);
    }

    /**
     * Loads network weights from a file.
     */
    public static NeuralNetwork load(String filePath) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filePath))) {
            int inputSize = dis.readInt();
            int hidden1Size = dis.readInt();
            int hidden2Size = dis.readInt();

            NeuralNetwork nn = new NeuralNetwork(inputSize, hidden1Size, hidden2Size, 0.001);

            // w1, b1
            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < hidden1Size; j++)
                    nn.w1[i][j] = dis.readDouble();
            for (int j = 0; j < hidden1Size; j++)
                nn.b1[j] = dis.readDouble();

            // w2, b2
            for (int i = 0; i < hidden1Size; i++)
                for (int j = 0; j < hidden2Size; j++)
                    nn.w2[i][j] = dis.readDouble();
            for (int j = 0; j < hidden2Size; j++)
                nn.b2[j] = dis.readDouble();

            // w3, b3
            for (int i = 0; i < hidden2Size; i++)
                nn.w3[i][0] = dis.readDouble();
            nn.b3[0] = dis.readDouble();

            System.out.println("Network loaded from " + filePath);
            return nn;
        }
    }

    // ==================== UTILITY ====================

    public void setLearningRate(double lr) {
        this.learningRate = lr;
    }

    public double getLearningRate() {
        return learningRate;
    }

    @Override
    public String toString() {
        int totalParams = (inputSize * hidden1Size + hidden1Size)
                        + (hidden1Size * hidden2Size + hidden2Size)
                        + (hidden2Size * outputSize + outputSize);
        return "NeuralNetwork[" + inputSize + " → " + hidden1Size
             + " → " + hidden2Size + " → " + outputSize
             + "] (" + totalParams + " parameters)";
    }
}
