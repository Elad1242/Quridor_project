# The Neural Network Side of the Quoridor ML Project

## A guide for understanding every concept, every decision, and every detail

---

## 1. What this document is

This is a complete technical explanation of the neural network component of my Quoridor project. It covers only the ML side — the algorithmic bot (BotBrain) and the graph algorithms (A\*, BFS, Dijkstra, Edmonds-Karp) are mentioned only where they interact with the ML side, and are covered in a separate document. The goal here is to explain, in order: what the network does, how I built it, why I made each design decision, and what ML concepts you need to understand to follow the code.

## 2. The big picture: what the network does

The neural network in my project does **one thing**: it takes a vector of 27 numbers describing a candidate action in a Quoridor game, and returns a single number between 0 and 1 that represents how good that action looks. Nothing more, nothing less.

The bot itself, `FeatureBot`, uses the network inside a loop:
1. Enumerate every legal action the bot could take this turn (pawn moves + walls)
2. For each candidate action, convert the board state + action into 27 features
3. Run those 27 features through the neural network, get a score back
4. Pick the action with the highest score

The network does NOT decide *which moves are legal* — that's the job of the algorithmic layer (`MoveValidator`, `WallValidator`). The network does NOT compute distances or paths — that's done by algorithms (`A*`, `BFS`, `Dijkstra`, `Edmonds-Karp Max-Flow`) that run inside the feature extractor. The network's job is purely to **rank candidate actions** that have already been prepared for it.

## 3. The architecture

The network is a **fully-connected feedforward neural network** — sometimes called a "vanilla NN" or "multi-layer perceptron". It is NOT a convolutional network (CNN) and NOT a recurrent network (RNN). Those are suited for images and sequences; we have a fixed-size vector of engineered features, so a simple feedforward structure is the right choice.

The structure is:

```
Input layer:   27 neurons   (feature vector)
    ↓
Hidden layer 1: 64 neurons   + ReLU activation
    ↓
Hidden layer 2: 32 neurons   + ReLU activation
    ↓
Hidden layer 3: 16 neurons   + ReLU activation
    ↓
Output layer:  1 neuron     + Sigmoid activation  (range: 0 to 1)
```

Total trainable parameters:
- Weights: 27×64 + 64×32 + 32×16 + 16×1 = 1728 + 2048 + 512 + 16 = **4,304 weights**
- Biases: 64 + 32 + 16 + 1 = **113 biases**
- **Total: 4,417 parameters**

This is considered a tiny network by modern standards (GPT-3 has 175 **billion** parameters, for comparison), but for tabular data with 27 engineered features, this size is more than sufficient. A larger network would overfit on 2 million training samples; a smaller one would underfit.

## 4. The 27 input features

The input to the network is a carefully engineered 27-dimensional vector. These features replace the raw board pixels that a CNN would use. The split is 22 features describing the overall board state + 5 features describing the specific candidate action being scored.

### 22 global features (describe the whole board state)

- **Race features (6):** my A\* distance to goal, opponent's A\* distance, the gap between them, my BFS distance, opponent's BFS distance, a binary flag for who's closer.
- **Position features (6):** my row, opponent's row, my column, opponent's column, row distance between us, column distance between us.
- **Resource features (4):** my walls remaining, opponent's walls remaining, the wall advantage, total walls placed so far.
- **Tactical features (4):** how many forward moves I have available, total number of legal moves, my Max-Flow value (number of independent paths), opponent's Max-Flow.
- **Safety features (2):** my Dijkstra-weighted safe-path cost, opponent's Dijkstra-weighted safe-path cost.

### 5 per-action features (describe the specific candidate)

For a MOVE candidate: path gain (how much the move reduces my A\* distance), row advance (how many rows closer to the goal), on-A\*-path flag, distance after the move, and an action-type marker set to 0.

For a WALL candidate: path damage (how much this wall increases the opponent's distance), self-harm (how much it increases mine), net damage (damage minus self-harm), opponent's distance after the wall, and an action-type marker set to 1.

Every feature is normalized to roughly the range [0, 1] by dividing by a sensible maximum value (for example, `myAstar / 16.0` since the longest reasonable A\* distance on a 9x9 board is around 16). Normalization matters because neural networks train much more stably when all inputs are on a similar scale — if one feature ranges 0-1 and another ranges 0-1000, the second would dominate the gradients and prevent the first from learning.

### Why engineered features instead of raw pixels?

Earlier in the project I tried a CNN that took the raw 8-channel 9x9 board as input. That CNN had about 300,000 parameters and needed to learn pathfinding and distance estimation from scratch. It failed with 0% win rate — there simply wasn't enough data or compute to learn those concepts end-to-end. Engineered features work because they encode the hard problem (pathfinding) explicitly, letting the network focus on the easier problem of combining pre-computed signals.

## 5. How a forward pass works

A forward pass is what happens when you give the network an input and get an output. It's just a sequence of matrix multiplications and activation functions.

For each layer, the operation is:
1. Take the input vector (the previous layer's output)
2. Multiply it by the weight matrix of this layer
3. Add the bias vector
4. Apply the activation function

In code, for a single neuron `j` in a layer:
```
preActivation[j] = bias[j] + sum over i of (weight[j][i] * input[i])
activation[j] = activationFunction(preActivation[j])
```

For the whole network with my architecture, a single forward pass performs:
- Layer 1: 64 × 27 = 1,728 multiply-adds, then 64 ReLU operations
- Layer 2: 32 × 64 = 2,048 multiply-adds, then 32 ReLU operations
- Layer 3: 16 × 32 = 512 multiply-adds, then 16 ReLU operations
- Output:  1 × 16 = 16 multiply-adds, then 1 Sigmoid operation

Total: ~4,304 multiplications and additions plus 113 activation-function calls. On a modern CPU this takes roughly 5-10 microseconds per forward pass.

## 6. Activation functions: ReLU and Sigmoid

Activation functions are the nonlinearity that gives a neural network its power. Without them, a network of any depth would mathematically collapse to a single linear function (multiple matrix multiplications with no nonlinearity in between equal one matrix multiplication). Nonlinearity lets the network learn curved, complex decision boundaries.

### ReLU (Rectified Linear Unit)

The formula is trivially simple:
```
ReLU(x) = max(0, x)
```
- If the input is negative, output 0
- If the input is positive, output the input unchanged

**Why ReLU for hidden layers:**
- Computationally cheap (no expensive exponentials)
- Doesn't suffer from "vanishing gradients" for positive inputs — its derivative is exactly 1 where x > 0
- Naturally creates sparsity: many neurons output exactly 0, which tends to improve generalization

**The main downside:** "dead neurons" can occur when a neuron's pre-activation is always negative. Its output stays at 0 and its gradient stays at 0, so it never updates and never learns. I mitigate this by initializing biases to a small positive value (0.01) so neurons start in the active range.

### Sigmoid (at the output)

The formula:
```
Sigmoid(x) = 1 / (1 + e^(-x))
```

Sigmoid squashes any real input into the range (0, 1). Very negative inputs produce values close to 0; very positive inputs produce values close to 1; zero produces 0.5.

**Why Sigmoid for the output:**
- My training labels are 0.9 and 0.1 (both in [0, 1]), so the output needs to live in the same range
- Sigmoid combined with MSE loss is a standard pairing for this kind of regression-style task
- It provides natural bounding: the network can't output nonsensical values like 1.5 or -0.3

## 7. Weight initialization: He initialization

Before any training happens, all 4,304 weights need starting values. If they all start at zero, every neuron in a layer will compute identical outputs and receive identical gradients, so the network will never break symmetry and won't learn. If they start too large, the forward pass will produce exploding values. If they start too small, the forward pass will vanish into numerical noise.

**He initialization** (named after Kaiming He, who introduced it in 2015) is the standard choice for networks using ReLU activations. The formula for each weight is:

```
weight = random Gaussian sample × sqrt(2 / fanIn)
```

where `fanIn` is the number of inputs to the neuron. For my first layer, fanIn = 27, so stddev = sqrt(2/27) ≈ 0.272. For the second layer, fanIn = 64, so stddev ≈ 0.177. And so on.

**Why this specific formula:** He showed mathematically that this scaling keeps the variance of activations and gradients roughly constant across layers when ReLU is used, avoiding both vanishing and exploding problems. The `2` in the numerator specifically compensates for the fact that ReLU kills half the neurons on average.

The biases are initialized to 0.01 (a small positive number) rather than 0, so ReLU neurons start in the "active" zone and don't immediately become dead.

## 8. The loss function: Mean Squared Error

To train the network, we need a way to measure how wrong its predictions are. The loss function I chose is **Mean Squared Error (MSE)**:

```
loss = 0.5 × (prediction - target)^2
```

For example, if the target is 0.9 and the network predicts 0.7, the loss is `0.5 × 0.04 = 0.02`. If the network predicts 0.9 exactly, the loss is 0.

**Why MSE:**
- It's mathematically simple and its derivative is clean: `dLoss/dPrediction = prediction - target`
- It pairs well with Sigmoid output
- For a regression-style problem with continuous targets in (0, 1), MSE is a standard choice

**Alternative I did not use:** cross-entropy loss is more common for classification problems (where labels are 0 or 1 exactly). My labels are 0.9 and 0.1, not 1 and 0, so they're technically soft targets — MSE handles these naturally whereas cross-entropy would need a different formulation.

## 9. Backpropagation: how the network learns

Backpropagation is the algorithm that computes the gradient of the loss with respect to every weight in the network — that is, for each weight, how much would the loss change if we nudged that weight slightly? Once we know the gradient of every weight, we can update each weight to reduce the loss.

The algorithm is essentially the chain rule from calculus applied systematically through the network.

### The intuition

If you have a function `f(g(x))` and you want to know how sensitive `f` is to changes in `x`, calculus tells you `df/dx = df/dg × dg/dx`. The chain rule lets you multiply local sensitivities along the chain from output to input. Backprop does exactly this, but through a neural network with many layers and many neurons per layer.

### The algorithm, step by step

1. **Forward pass** — give the network an input, compute every layer's activation, save them. These are called the "cached activations".

2. **Output layer gradient** — compute `delta = (prediction - target) × sigmoid'(output pre-activation)`. Sigmoid's derivative is conveniently `sigmoid(x) × (1 - sigmoid(x))`, which we already have from the forward pass.

3. **Propagate backward** — for each hidden layer, going from output toward input:
   ```
   delta[L][j] = (sum over k of weight[L+1][k][j] * delta[L+1][k]) × ReLU'(preActivation[L][j])
   ```
   where ReLU's derivative is 1 for positive pre-activations and 0 otherwise.

4. **Compute weight gradients** — once every layer's delta is known, the gradient for each weight is:
   ```
   dLoss/dWeight[L][j][i] = delta[L][j] × activation[L][i]
   dLoss/dBias[L][j]      = delta[L][j]
   ```

5. **Update weights** — apply the gradient with an optimization rule (described in section 10).

Everything here is in the `NeuralNetwork.backprop` method in my code, written from scratch without any autograd library.

## 10. Mini-batch Stochastic Gradient Descent with Momentum

There are three flavors of gradient descent:

| Method | How often to update weights | Pros | Cons |
|---|---|---|---|
| Full-batch GD | After seeing ALL training data | Smooth, exact direction | Unusably slow for 2M samples |
| Pure SGD | After every single sample | Fast, lots of updates | Noisy, unstable |
| Mini-batch SGD | After every N samples (batch) | Fast AND stable | The modern standard |

I use mini-batch SGD with a batch size of 64. The flow is:

1. Shuffle the training data (so each epoch sees samples in a different order — prevents the network from memorizing order-based patterns).
2. Take the first 64 samples.
3. Run each through forward + backward to get gradients.
4. Average the gradients across those 64 samples.
5. Apply a single weight update using this averaged gradient.
6. Take the next 64 samples.
7. Repeat until the whole dataset is consumed — that's one epoch.

### Momentum

Vanilla SGD can be slow to converge because each update only knows about the current batch. Momentum adds a "velocity" term that remembers the direction of recent updates. The formula for each weight is:

```
velocity = momentum × velocity - learningRate × gradient
weight   = weight + velocity
```

where momentum = 0.9 in my project.

**The intuition:** think of a ball rolling down a hill. Without momentum, the ball moves in whatever direction the ground tilts *right now*, which can be jittery on a bumpy surface. With momentum, the ball has inertia — it keeps moving in the direction it was already going, which smooths out the jitter and accelerates progress in consistent directions.

In practice, momentum makes training converge roughly 2-5x faster than plain SGD.

### Weight decay (L2 regularization)

A tiny extra term in the update rule:
```
velocity = momentum × velocity - learningRate × (gradient + weightDecay × weight)
```

This gently shrinks every weight toward zero on every update. My weightDecay value is 1e-5 — very small, but it adds up over thousands of updates. The effect is to prevent any single weight from growing very large, which is a form of regularization that improves generalization on unseen data.

## 11. Training data: pairwise preference learning

The training data is where a lot of ML projects go wrong, and it's where my project went through the most iteration. Here's how I arrived at the final approach.

### What didn't work (and why it matters)

My first attempt used **regression on a continuous score**. I had BotBrain (the algorithmic bot) play games; for each position, I computed BotBrain's internal score for its chosen move and used that score as the training label. So the network learned "given position X, predict the numeric score BotBrain gave to this position".

The network reached 99.9% accuracy on this regression task — it could match BotBrain's scores almost perfectly. And it lost 100% of games against BotBrain. The reason was **1-ply wall bias**: at play time, the network evaluated candidate actions by scoring the resulting positions. A wall that adds 3 steps to the opponent's path produces a much larger score change than a move that advances me 1 step — so the network always preferred walls, even when moving was better. Structurally, the scoring function was set up so that walls ALWAYS looked better than moves.

### The fix: pairwise preference learning

Instead of trying to predict numeric scores, I switched to learning **preferences between actions**. The setup is:

- Whenever BotBrain takes its turn, record the feature vector of its chosen action with label **0.9** (positive example).
- Also record the feature vectors of **two random legal actions BotBrain didn't choose** with label **0.1** each (negative examples).

So each BotBrain turn produces 3 training samples: 1 positive and 2 negatives. The network learns "given these 27 features, what's the probability that this is the kind of action BotBrain would choose?"

**Why this works:**
- The network can no longer exploit the wall-vs-move magnitude difference, because both positives and negatives come from all kinds of actions. The labels 0.9 and 0.1 are attached to actions based on BotBrain's preference, not based on the magnitude of state change.
- At inference time, I rank all candidate actions by their scores and pick the highest. The ranking is what matters, not the absolute score.

### Opponent diversity

BotBrain doesn't play only against itself while generating training data. It cycles through 5 different opponents — UniformRandom, ForwardRandom, RandomWaller, SemiSmart, and a copy of BotBrain with random openings. Every 5th game switches opponent type. This creates **positional diversity**: BotBrain encounters a wide variety of board states, including weird ones that would never come up in self-play but can come up when playing the final trained MLBot.

Without this diversity, the network would be great at handling positions that look like BotBrain-vs-BotBrain games but fail on unusual positions — this is called **distribution shift**, and it killed one of my earlier attempts that trained only on self-play.

### Final dataset

- 50,000 games total, split equally across the 5 opponent types
- Each game produces roughly 40 training samples (20 BotBrain turns × 3 samples each)
- Total: approximately **2 million samples**
- Stored in a binary file `training_features.dat` (~400 MB) in a simple format: number of samples, features per sample, then raw doubles

## 12. Hyperparameters I chose, and why

These are the "knobs" I set before training starts. The network learns the weights automatically, but hyperparameters are chosen by me.

| Hyperparameter | Value | Why |
|---|---|---|
| Architecture | 27-64-32-16-1 | Wide enough to capture feature interactions; deep enough for nonlinearity; small enough to avoid overfitting |
| Activation (hidden) | ReLU | Standard for modern networks; cheap; doesn't vanish |
| Activation (output) | Sigmoid | Matches [0, 1] label range |
| Loss | MSE | Clean gradient; pairs with sigmoid; works for soft labels |
| Initialization | He | Correct scaling for ReLU networks |
| Batch size | 64 | Industry standard; small enough for frequent updates, large enough to be stable |
| Learning rate | 0.001 | Proposal suggested 0.1; that was too large and failed to converge. 0.001 is the standard default for SGD-with-momentum |
| Momentum | 0.9 | Industry standard; improves convergence speed |
| Weight decay | 1e-5 | Very light regularization; network is small so overfitting is already limited |
| Max epochs | 200 | Upper bound; early stopping usually ends training around epoch 80-120 |
| Early stopping patience | 50 | Stop if validation loss hasn't improved for 50 epochs |

## 13. The training loop

Here's what happens from start to finish when I run `FeatureTrainer`:

1. Load `training_features.dat` — read the 2M samples into memory.
2. Shuffle the data and split 90/10: 1.8M samples for training, 0.2M for validation.
3. Initialize a fresh network: `new NeuralNetwork(27, 64, 32, 16, 1)`. This creates the architecture with He-initialized weights.
4. Start the training loop. For each epoch (up to 200):
   - Shuffle the training data
   - Walk through it in mini-batches of 64. For each mini-batch:
     - Run forward + backward on every sample to get per-sample gradients
     - Sum the gradients across the batch
     - Call `updateWeights` which applies the momentum+weight-decay update rule
   - Every 5 epochs, compute validation loss on the 0.2M held-out samples
   - If validation loss improved, save the model to `feature_model.bin`
   - If 50 epochs pass without improvement, halve the learning rate
   - If halving doesn't help either, stop training (early stopping)
5. At the end, reload the best saved model (not the final epoch's weights — the best one seen during training).

Total training time on CPU: **roughly 30-60 minutes**. Most of that time is spent in nested for-loops doing the matrix math for forward and backward passes.

## 14. Inference: how FeatureBot uses the trained network to play

At play time (during an actual game), the network is frozen — no more learning happens. The bot uses it like this:

1. **Instant-win check (algorithmic)** — if any legal move lands on the goal row, take it. Don't even consult the network.
2. **Emergency-block check (algorithmic)** — if the opponent is 1-2 steps from their goal, force a wall placement. Consult the network only to pick *which* wall, not whether to wall.
3. **Normal scoring (network-driven):**
   - Enumerate every legal pawn move (the algorithmic `MoveValidator` does this)
   - Enumerate every legal wall (the algorithmic `WallValidator` does this)
   - Pre-filter bad walls (the algorithmic filter removes walls that do zero damage or self-harm > 2)
   - For each surviving candidate action, compute the 27 features, run the network once, record the score
4. **Top-k sampling (algorithmic, described below)** — pick the action to play.

### Top-k sampling

After the network has scored all candidates, naively you'd pick the one with the highest score. This is called **argmax**. But when two bots are both deterministic and play identically, the games become 100% repeatable — every MLBot-vs-BotBrain game plays out exactly the same way.

To fix this, I use **top-k sampling with a score threshold of 0.01**: find the maximum score, keep all candidates whose score is within 0.01 of it, then pick one of them uniformly at random. If one candidate clearly dominates (say, 0.87 vs 0.42 next-best), only that one passes the filter and the behavior is identical to argmax. If multiple candidates are tied or very close, the bot picks among them randomly, which breaks the deterministic loop without sacrificing playing strength.

A controlled experiment confirmed this design doesn't hurt win rate: argmax alone gave 61% wins in 200 games, top-k with 0.01 threshold gave 56-63% across multiple runs. The difference is within statistical noise, but the visible game variety goes from "every game identical" to "every game different".

## 15. Key concepts to understand

- **Neuron**: a single unit that computes a weighted sum of its inputs plus a bias, then applies an activation function. Networks have many neurons per layer.
- **Layer**: a group of neurons that all receive the same inputs (from the previous layer) and produce outputs that feed the next layer.
- **Weights**: the learnable numbers that multiply the inputs. The network learns weights during training.
- **Biases**: additive constants per neuron, also learned. Allow the neuron to shift its activation threshold.
- **Activation function**: a nonlinear function applied to each neuron's output. Without it, the whole network collapses to a linear function.
- **Forward pass**: computing outputs from inputs, layer by layer.
- **Backward pass (backprop)**: computing gradients of the loss with respect to each weight by applying the chain rule, working from the output layer back to the input.
- **Loss function**: a scalar measuring how wrong the network's prediction is. We minimize it during training.
- **Gradient**: a vector of partial derivatives telling us how much the loss changes if we nudge each weight. The gradient's negative direction is the direction of steepest descent of the loss.
- **Epoch**: one complete pass through the entire training dataset.
- **Mini-batch**: a small subset of training data (64 samples in my case) processed together before updating weights.
- **Learning rate**: the step size for weight updates. Too large = training diverges; too small = training takes forever.
- **Momentum**: a trick that adds inertia to weight updates, accelerating convergence.
- **Weight decay (L2 regularization)**: a small penalty on large weights, added to the loss. Prevents overfitting.
- **Overfitting**: when the network memorizes the training data instead of learning general patterns. Detected by validation loss going up while training loss keeps going down.
- **Validation set**: held-out data not used for training, used to detect overfitting and decide when to stop.
- **Early stopping**: ending training when validation loss stops improving.
- **Sigmoid**: activation function that squashes inputs to (0, 1). Used at the output here.
- **ReLU**: activation function that clips negatives to zero. Used in hidden layers here.
- **He initialization**: weight initialization scheme appropriate for ReLU networks.
- **Pairwise preference learning**: training the network to prefer one action over others, rather than predict a numeric value.
- **Distribution shift**: when the positions seen at training time differ systematically from the positions seen at play time; causes the network to perform worse than its training accuracy would suggest.
- **Argmax**: picking the option with the highest score.
- **Top-k sampling**: picking one option randomly from the top-scoring few options.

## 16. Why a neural network and not something else

I considered several alternatives before settling on a neural network:

- **Random Forest**: ensemble of decision trees. Pros: easy to implement, fast to train. Cons: large memory footprint, struggles to generalize to positions not seen during training. Less well-suited to regression tasks with complex feature interactions.
- **Logistic Regression**: just a weighted sum with a sigmoid. Pros: dead simple. Cons: purely linear (apart from the sigmoid), can't capture feature interactions like "my path is short AND I have walls remaining AND the opponent has a wall advantage". Underfits the Quoridor policy.
- **K-Nearest Neighbors**: look up most similar positions in the training set. Pros: no training. Cons: needs to store all 2M samples at inference time, slow per-query, terrible at generalization.
- **Gradient Boosted Trees (XGBoost-style)**: like Random Forest but with sequential tree-fitting. Pros: strong on tabular data. Cons: would violate Rule 8 (no external libraries) unless implemented from scratch, which is significantly harder than a simple NN.

The neural network won on these criteria: small memory footprint (35 KB trained model), good generalization to unseen positions, simple to implement from scratch in Java, fast inference (5-10 microseconds per score).

## 17. Common questions to anticipate

**Q: Is this a CNN?**
A: No. CNNs use convolutional filters over 2D data like images. I use a feedforward (fully-connected) network over a 1D feature vector. My earlier CNN attempt failed; feature engineering + feedforward is what works here.

**Q: Does the network "understand" the game?**
A: It doesn't understand anything in a human sense. It has learned a statistical mapping from 27-dimensional feature vectors to preference scores. The "understanding" of concepts like paths and walls lives in the algorithms that produce the features, not in the network itself.

**Q: The output is a number between 0 and 1. Does that mean probability of winning?**
A: No. It's a similarity score to BotBrain's chosen actions during training. Values near 0.9 look like actions BotBrain would pick; values near 0.1 look like actions it would reject. It's not a probability of anything — it's only meaningful for ranking candidate actions against each other.

**Q: How do you know the network actually learned something useful?**
A: Two measurements. First, against weaker opponents (GreedyPathBot, SemiSmart), the network wins 100% on both sides of the board in controlled tests. Second, against the algorithmic bot BotBrain itself, the network wins 63.7% of 300 games with a 95% confidence interval excluding 50%. Both results show genuine learning beyond the tempo advantage.

**Q: Why didn't you train with reinforcement learning instead?**
A: I tried. Attempts 7-9 of the project were pure RL (REINFORCE with various curriculum and DAgger tricks). All failed with 0% win rate. RL needs massive self-play and compute; on a single CPU with 2M samples, supervised learning from BotBrain is much more tractable.

**Q: How long does training take?**
A: About 30-60 minutes on CPU. Inference is instant (microseconds per score, low milliseconds per turn including feature extraction).

**Q: What would you improve if you had more time?**
A: Train the network against weakened BotBrain variants to see if it can exceed any single teacher. Add a 2nd order feature: "how good would my next move be after this one?" — this adds a small amount of lookahead without violating the no-Minimax rule. Explore temperature-scaled softmax for the top-k sampling instead of hard threshold.

## 18. Summary in one paragraph

The ML component of my Quoridor project is a small feedforward neural network (27 → 64 → 32 → 16 → 1, about 4400 parameters, written from scratch in Java with no external libraries). It takes a 27-dimensional feature vector (22 global features plus 5 per-action features, computed by graph algorithms like A\*, Dijkstra, and Max-Flow) and outputs a single score in (0, 1) representing how similar a candidate action looks to what the algorithmic teacher BotBrain would choose. The network was trained on 2 million samples from 50,000 games using pairwise preference learning — BotBrain's chosen action labeled 0.9, two random alternatives labeled 0.1 — with mini-batch SGD, momentum 0.9, learning rate 0.001, batch size 64, and early stopping. At play time, the bot enumerates legal actions, filters out obviously bad walls, feeds each candidate through the network, and picks the highest-scoring one (with top-k sampling to break ties). The trained network beats its own teacher 63.7% of the time in 300 games — a statistically significant result indicating the network has learned a useful policy, not just a copy of BotBrain.
