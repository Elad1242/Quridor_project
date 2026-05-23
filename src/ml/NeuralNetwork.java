// v2.0 — refactored and cleaned, May 2026
package ml;

import java.io.*;
import java.util.Random;

// רשת נוירונים feedforward שנכתבה מאפס.
// ארכיטקטורה: 27 → 64 → 32 → 16 → 1
// שכבות נסתרות עם ReLU, שכבת פלט עם sigmoid, פונקציית שגיאה MSE.
public class NeuralNetwork {

    private final int[] layerSizes;
    private final int numLayers; // מספר שכבות המשקלים = מספר השכבות פחות אחת

    // weights[L][j][i] — משקל מנוירון i בשכבה L לנוירון j בשכבה L+1
    private double[][][] weights;
    // biases[L][j] — הטיה של נוירון j בשכבה L+1
    private double[][] biases;

    // וקטורי המהירות לאלגוריתם המומנטום (זוכרים את כיוון העדכון הקודם)
    private double[][][] velocityW;
    private double[][] velocityB;

    public NeuralNetwork(int... layerSizes) {
        this.layerSizes = layerSizes;
        this.numLayers  = layerSizes.length - 1;
        this.weights    = new double[numLayers][][];
        this.biases     = new double[numLayers][];
        this.velocityW  = new double[numLayers][][];
        this.velocityB  = new double[numLayers][];

        Random rng = new Random(42);
        for (int L = 0; L < numLayers; L++) {
            int fanIn  = layerSizes[L];
            int fanOut = layerSizes[L + 1];
            weights[L]   = new double[fanOut][fanIn];
            biases[L]    = new double[fanOut];
            velocityW[L] = new double[fanOut][fanIn];
            velocityB[L] = new double[fanOut];

            // אתחול He: סטיית תקן של sqrt(2/fanIn) — מתאים במיוחד לReLU
            // מונע שהגרדיאנטים יתאפסו או יתפוצצו בשכבות העמוקות
            double stddev = Math.sqrt(2.0 / fanIn);
            for (int j = 0; j < fanOut; j++) {
                for (int i = 0; i < fanIn; i++) {
                    weights[L][j][i] = rng.nextGaussian() * stddev;
                }
                biases[L][j] = 0.01; // הטיה חיובית קטנה כדי שנוירוני ReLU יתחילו פעילים
            }
        }
    }

    // --- העברה קדימה ---

    // מחשב את הפלט של הרשת ושומר ערכי ביניים הנחוצים לbackprop
    public ForwardResult forward(double[] input) {
        double[][] activations    = new double[numLayers + 1][];
        double[][] preActivations = new double[numLayers][];

        activations[0] = input; // שכבת הכניסה היא הפיצ'רים עצמם

        for (int L = 0; L < numLayers; L++) {
            int outSize = layerSizes[L + 1];
            int inSize  = layerSizes[L];
            preActivations[L]    = new double[outSize];
            activations[L + 1]   = new double[outSize];

            for (int j = 0; j < outSize; j++) {
                // סכום משוקלל: bias + סיגמא(weight * activation)
                double sum = biases[L][j];
                for (int i = 0; i < inSize; i++) {
                    sum += weights[L][j][i] * activations[L][i];
                }
                preActivations[L][j] = sum; // שומרים לפני הפעלה — נחוץ לגזירה בbackprop

                // שכבות נסתרות: ReLU — מאפס ערכים שליליים
                // שכבת פלט: sigmoid — מוציאה הסתברות בין 0 ל-1
                if (L < numLayers - 1) {
                    activations[L + 1][j] = Math.max(0, sum);
                } else {
                    activations[L + 1][j] = 1.0 / (1.0 + Math.exp(-sum));
                }
            }
        }

        return new ForwardResult(activations, preActivations);
    }

    // חיזוי בלבד — משמש בזמן משחק (לא שומר ערכי ביניים)
    public double predict(double[] input) {
        return forward(input).getOutput()[0];
    }

    // --- backpropagation ---

    // מחשב גרדיאנטים עבור דגימה אחת — לא מעדכן משקלים (הקורא צובר ומעדכן בסוף הbatch)
    public Gradients backprop(double[] input, double target) {
        ForwardResult fwd         = forward(input);
        double[][] activations    = fwd.activations;
        double[][] preActivations = fwd.preActivations;

        double[][][] dW   = new double[numLayers][][];
        double[][] dB     = new double[numLayers][];
        double[][] deltas = new double[numLayers][];

        // שכבת הפלט: delta = שגיאה × נגזרת sigmoid
        // נגזרת sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        int lastL        = numLayers - 1;
        double predicted = activations[numLayers][0];
        double error     = predicted - target;
        deltas[lastL]    = new double[]{error * predicted * (1 - predicted)};

        // מעבר אחורה בשכבות הנסתרות: כל delta מחושב מהשכבה שאחריו
        for (int L = lastL - 1; L >= 0; L--) {
            int size     = layerSizes[L + 1];
            int nextSize = layerSizes[L + 2];
            deltas[L]    = new double[size];

            for (int j = 0; j < size; j++) {
                // חיבור הדלתות מהשכבה הבאה, משוקלל לפי המשקלים
                double sum = 0;
                for (int k = 0; k < nextSize; k++) {
                    sum += weights[L + 1][k][j] * deltas[L + 1][k];
                }
                // נגזרת ReLU: 1 אם הקלט היה חיובי, 0 אחרת
                deltas[L][j] = sum * (preActivations[L][j] > 0 ? 1.0 : 0.0);
            }
        }

        // בניית מערכי הגרדיאנטים: dW = delta × activation, dB = delta
        for (int L = 0; L < numLayers; L++) {
            int outSize = layerSizes[L + 1];
            int inSize  = layerSizes[L];
            dW[L] = new double[outSize][inSize];
            dB[L] = new double[outSize];

            for (int j = 0; j < outSize; j++) {
                dB[L][j] = deltas[L][j];
                for (int i = 0; i < inSize; i++) {
                    dW[L][j][i] = deltas[L][j] * activations[L][i];
                }
            }
        }

        double loss = 0.5 * error * error; // שגיאה ריבועית לדגימה אחת (MSE)
        return new Gradients(dW, dB, loss);
    }

    // עדכון המשקלים בסוף כל batch — ממצע את הגרדיאנטים ומפעיל מומנטום
    public void updateWeights(double[][][] gradW, double[][] gradB, int batchSize,
                               double learningRate, double momentum, double weightDecay) {
        for (int L = 0; L < numLayers; L++) {
            int outSize = layerSizes[L + 1];
            int inSize  = layerSizes[L];

            for (int j = 0; j < outSize; j++) {
                // מומנטום: שמור חלק מהכיוון הקודם כדי להאיץ ולייצב את הלמידה
                velocityB[L][j] = momentum * velocityB[L][j]
                        - learningRate * (gradB[L][j] / batchSize);
                biases[L][j] += velocityB[L][j];

                for (int i = 0; i < inSize; i++) {
                    // weightDecay מונע overfitting על ידי עונש על משקלים גדולים
                    velocityW[L][j][i] = momentum * velocityW[L][j][i]
                            - learningRate * (gradW[L][j][i] / batchSize + weightDecay * weights[L][j][i]);
                    weights[L][j][i] += velocityW[L][j][i];
                }
            }
        }
    }

    // --- save / load ---

    public void save(String filePath) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filePath)))) {
            out.writeInt(layerSizes.length);
            for (int s : layerSizes) out.writeInt(s);

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
            int[] sizes  = new int[numSizes];
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

    public int getParamCount() {
        int count = 0;
        for (int L = 0; L < numLayers; L++) {
            count += layerSizes[L + 1] * layerSizes[L]; // weights
            count += layerSizes[L + 1];                  // biases
        }
        return count;
    }

    public static class ForwardResult {
        public final double[][] activations;
        public final double[][] preActivations;

        ForwardResult(double[][] activations, double[][] preActivations) {
            this.activations    = activations;
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
            this.dW   = dW;
            this.dB   = dB;
            this.loss = loss;
        }
    }
}
