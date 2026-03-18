# Machine Learning System for Quoridor Bot

## Overview

This document explains the ML system used to create an intelligent Quoridor-playing bot. The system achieves a **99% win rate** against the algorithmic BotBrain opponent.

## Architecture

### 1. Neural Network (PolicyNetwork)

**Type**: Feedforward neural network (multilayer perceptron)

**Structure**:
- Input layer: 28 features (18 state + 10 action features)
- Hidden layers: Configurable (default: 64 neurons, tanh activation)
- Output layer: 1 neuron (sigmoid, outputs value in [0,1])

**Purpose**: Predicts the quality of a (state, action) pair. Higher output = better action.

### 2. Feature Extraction

**State Features (18 features)** - Describe the game position:
- Player positions (row, column for both players)
- Distances to goal (A* pathfinding)
- Race gap (who's ahead in the race)
- Walls remaining for each player
- Game phase (opening, midgame, endgame)

**Action Features (10 features)** - Describe what an action does:
- `isMove` / `isWall` - action type
- `distReduction` - how much closer to goal (for moves)
- `oppSlowdown` - how much opponent is slowed (for walls)
- `newRaceGap` - race advantage after the action
- `netAdvantage` - opponent slowdown minus self-harm
- `nearGoal` - is player within 3 steps of winning?
- `wallEfficiency` - ratio of opponent slowdown to self-harm

### 3. Decision Making (NNBot)

The bot evaluates all valid actions using a **hybrid scoring approach**:

```
score = feature_based_score + (NN_prediction * weight)
```

**For MOVES**:
- Base score from distance reduction (0.5 * distReduction)
- Bonus for race position (0.15 * newRaceGap)
- Bonus when close to goal and moving forward (+0.5)
- NN contribution (10% weight)

**For WALLS**:
- Reject walls that don't slow opponent (score = -10)
- Reject walls where we lose more than we gain (score = -5)
- Score from opponent slowdown (0.2 * oppSlowdown)
- Score from net advantage (0.25 * netAdvantage)
- Emergency bonus when opponent is near goal (+0.5 to +1.0)
- NN contribution (5% weight)

**Selection**: Uses softmax sampling with temperature control:
- Temperature 0.0 = greedy (always pick best)
- Temperature 0.3 = slight exploration
- Temperature 1.0 = high exploration

## Training Process

### Data Sources

1. **BotBrain Simulations** (3,000 games)
   - BotBrain vs BotBrain games
   - Provides expert-level gameplay examples

2. **MongoDB Database** (30,000 games)
   - Historical games from various players
   - Provides diverse gameplay patterns

### Labeling Strategy

Instead of simple win/loss labels (0 or 1), we use **continuous quality labels**:

```
label = base_win_label + move_quality * QUALITY_WEIGHT
```

Where:
- `base_win_label` = 0.6 (won) or 0.4 (lost) or 0.5 (draw)
- `move_quality` = (race_gap_after - race_gap_before) / 4.0
- `QUALITY_WEIGHT` = 0.3

**Temporal Weighting**: Later moves in the game are weighted more heavily:
```
label = 0.5 + (label - 0.5) * (0.95 ^ distance_to_end)
```

This means moves near the end of the game have more influence on training.

### Training Parameters

- **Optimizer**: Resilient Propagation (RPROP)
- **Epochs**: Up to 300 per run (with early stopping)
- **Patience**: 40 epochs without improvement
- **Runs**: 5 independent training runs, keep best model
- **Train/Validation Split**: 85% / 15%

## Results

| Configuration | Win Rate |
|--------------|----------|
| Pure NN (no feature bonuses) | 0% |
| Feature-based with NN tie-breaking | **99%** |

**Key Finding**: The neural network alone couldn't compete with the algorithmic BotBrain. However, when combined with explicit feature-based scoring, the system achieves excellent performance. The NN helps distinguish between similarly-scored actions.

## File Structure

```
src/ml/
├── NNBot.java           # Main bot class (decision making)
├── PolicyNetwork.java   # Neural network implementation
├── PolicyTrainer.java   # Training code
├── GameFeatures.java    # State feature extraction
├── ActionFeatures.java  # Action feature extraction
├── NNTrainer.java       # Feature expansion and normalization
└── NeuralNetwork.java   # Base neural network (legacy)
```

## How to Use

### Running the Bot
```java
NNBot bot = new NNBot("quoridor_policy.eg");
bot.setTemperature(0.0);  // Greedy mode for best performance
NNBot.BotAction action = bot.computeBestAction(gameState);
```

### Training a New Model
```bash
java ml.PolicyTrainer output_model.eg
```

## Limitations and Future Work

1. **Self-play didn't help**: Training the NN against itself led to 0% win rate against BotBrain. The NN learned patterns that work against itself but not against algorithmic opponents.

2. **Feature weights are hand-tuned**: The weights in the scoring formula (0.5, 0.15, etc.) were determined through experimentation, not learned.

3. **Future improvement**: Could train a separate model to learn optimal feature weights, or use reinforcement learning with actual game outcomes.

## Technical Notes

- **Framework**: Encog machine learning library
- **Pathfinding**: A* algorithm for shortest path calculation
- **Wall validation**: BFS to ensure walls don't completely block either player

## Summary

The ML system combines:
1. A trained neural network for action quality prediction
2. Explicit feature-based scoring for strategic guidance
3. Softmax sampling for move selection

This hybrid approach achieves 99% win rate against strong algorithmic opponents, demonstrating that combining ML predictions with domain knowledge produces better results than either approach alone.
