#!/bin/bash
# Quoridor ML Training Pipeline
# Run inside Docker container on RunPod GPU

set -e

echo "========================================"
echo "Quoridor ML Training Pipeline"
echo "========================================"
echo ""

# Check GPU availability
nvidia-smi 2>/dev/null && echo "GPU detected!" || echo "WARNING: No GPU detected, will use CPU"

GAMES=${NUM_GAMES:-100000}
DATA_DIR=${DATA_DIR:-/data/training}
MODEL_DIR=${MODEL_DIR:-/data/models}
EVAL_GAMES=${EVAL_GAMES:-500}

echo "Config: ${GAMES} games, data=${DATA_DIR}, models=${MODEL_DIR}"
echo ""

# Step 1: Generate training data
echo "========================================"
echo "STEP 1: Generating training data..."
echo "========================================"
mvn -f pom.xml exec:java \
    -Dexec.mainClass=ml.DataGenerator \
    -Dexec.args="${GAMES} ${DATA_DIR}" \
    -B

echo ""
echo "Data files:"
ls -lh ${DATA_DIR}/*.qdat 2>/dev/null || echo "No data files found!"
echo ""

# Step 2: Train the model
echo "========================================"
echo "STEP 2: Training value network..."
echo "========================================"
mvn -f pom.xml exec:java \
    -Dexec.mainClass=ml.MLTrainer \
    -Dexec.args="${DATA_DIR} ${MODEL_DIR}" \
    -B

echo ""
echo "Model files:"
ls -lh ${MODEL_DIR}/ 2>/dev/null || echo "No model files found!"
echo ""

# Step 3: Evaluate
echo "========================================"
echo "STEP 3: Evaluating MLBot vs BotBrain..."
echo "========================================"
mvn -f pom.xml exec:java \
    -Dexec.mainClass=ml.EvalHarness \
    -Dexec.args="${MODEL_DIR} best-model ${EVAL_GAMES}" \
    -B

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
