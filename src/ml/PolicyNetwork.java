package ml;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.persist.EncogDirectoryPersistence;

import java.io.File;

/**
 * Policy Network for action selection in Quoridor.
 *
 * Architecture (Improved):
 *   Input: 28 features (18 state + 10 action)
 *   Hidden 1: 128 neurons (tanh) - wider for better feature extraction
 *   Hidden 2: 64 neurons (tanh)
 *   Hidden 3: 32 neurons (tanh)
 *   Hidden 4: 16 neurons (tanh) - additional depth for complex patterns
 *   Output: 1 (sigmoid) - probability this is a good action
 *
 * This network is deeper and wider to better learn complex action patterns.
 * The network learns: P(this action leads to winning | state, action)
 */
public class PolicyNetwork {

    // Input size: 18 state features + 10 action features
    public static final int INPUT_SIZE = 28;

    private BasicNetwork network;

    /**
     * Creates a new policy network with improved architecture.
     */
    public PolicyNetwork() {
        this(false);  // Use new architecture by default
    }

    /**
     * Creates a policy network.
     * @param useLegacyArchitecture if true, uses smaller 64-32-16 architecture for compatibility
     */
    public PolicyNetwork(boolean useLegacyArchitecture) {
        network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, INPUT_SIZE));

        if (useLegacyArchitecture) {
            // Original architecture (for loading old models)
            network.addLayer(new BasicLayer(new ActivationTANH(), true, 64));
            network.addLayer(new BasicLayer(new ActivationTANH(), true, 32));
            network.addLayer(new BasicLayer(new ActivationTANH(), true, 16));
        } else {
            // Improved wider/deeper architecture
            network.addLayer(new BasicLayer(new ActivationTANH(), true, 128));
            network.addLayer(new BasicLayer(new ActivationTANH(), true, 64));
            network.addLayer(new BasicLayer(new ActivationTANH(), true, 32));
            network.addLayer(new BasicLayer(new ActivationTANH(), true, 16));
        }

        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
        network.getStructure().finalizeStructure();
        network.reset();
    }

    /**
     * Creates a policy network from an existing Encog network.
     */
    public PolicyNetwork(BasicNetwork network) {
        this.network = network;
    }

    /**
     * Predicts the probability that this action is good.
     *
     * @param stateFeatures Normalized state features (18)
     * @param actionFeatures Normalized action features (10)
     * @return Probability in [0, 1]
     */
    public double predict(double[] stateFeatures, double[] actionFeatures) {
        double[] combined = new double[INPUT_SIZE];
        System.arraycopy(stateFeatures, 0, combined, 0, 18);
        System.arraycopy(actionFeatures, 0, combined, 18, 10);

        double[] output = new double[1];
        network.compute(combined, output);
        return output[0];
    }

    /**
     * Predicts using combined input array.
     */
    public double predict(double[] combined) {
        double[] output = new double[1];
        network.compute(combined, output);
        return output[0];
    }

    /**
     * Returns the underlying Encog network for training.
     */
    public BasicNetwork getNetwork() {
        return network;
    }

    /**
     * Saves the network to a file.
     */
    public void save(String path) {
        EncogDirectoryPersistence.saveObject(new File(path), network);
        System.out.println("Policy network saved to " + path);
    }

    /**
     * Loads a network from a file.
     */
    public static PolicyNetwork load(String path) {
        File file = new File(path);
        if (!file.exists()) {
            System.out.println("No policy network found at " + path + ", creating new one");
            return new PolicyNetwork();
        }
        BasicNetwork loaded = (BasicNetwork) EncogDirectoryPersistence.loadObject(file);
        System.out.println("Policy network loaded from " + path);
        return new PolicyNetwork(loaded);
    }
}
