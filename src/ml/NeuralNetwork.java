package ml;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.persist.EncogDirectoryPersistence;

import java.io.File;

/**
 * Wrapper around Encog's BasicNetwork for position evaluation.
 *
 * Architecture: 18 inputs → 32 (tanh) → 16 (tanh) → 1 (Sigmoid)
 *
 * The network takes 18 features (12 raw + 6 derived) and outputs a win probability (0.0 to 1.0).
 * Uses the Encog ML library for network creation, training, and persistence.
 *
 * Tanh activation is used for hidden layers because it works well with RPROP
 * training and avoids the "dying neuron" problem that ReLU can cause.
 */
public class NeuralNetwork {

    private BasicNetwork network;

    /**
     * Creates a new neural network with the default architecture.
     * 18 inputs → 32 hidden (tanh) → 16 hidden (tanh) → 1 output (Sigmoid)
     */
    public NeuralNetwork() {
        network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, NNTrainer.INPUT_SIZE)); // 18 inputs (12 raw + 6 derived)
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 32));          // hidden 1
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 16));          // hidden 2
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));       // output
        network.getStructure().finalizeStructure();
        network.reset(); // random initial weights
    }

    /**
     * Creates a wrapper around an existing Encog network (used when loading from file).
     */
    private NeuralNetwork(BasicNetwork network) {
        this.network = network;
    }

    /**
     * Forward pass: computes win probability for given features.
     *
     * @param input 18 normalized game features
     * @return win probability between 0.0 and 1.0
     */
    public double predict(double[] input) {
        MLData inputData = new BasicMLData(input);
        MLData output = network.compute(inputData);
        return output.getData(0);
    }

    /**
     * Returns the underlying Encog network (used by NNTrainer for training).
     */
    public BasicNetwork getNetwork() {
        return network;
    }

    /**
     * Saves the network weights to a file.
     */
    public void save(String filePath) {
        EncogDirectoryPersistence.saveObject(new File(filePath), network);
        System.out.println("Network saved to " + filePath);
    }

    /**
     * Loads a trained network from a file.
     */
    public static NeuralNetwork load(String filePath) {
        BasicNetwork loaded = (BasicNetwork) EncogDirectoryPersistence.loadObject(new File(filePath));
        System.out.println("Network loaded from " + filePath);
        return new NeuralNetwork(loaded);
    }

    @Override
    public String toString() {
        return "NeuralNetwork[18 → 32(tanh) → 16(tanh) → 1(Sigmoid)]";
    }
}
