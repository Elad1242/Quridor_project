#!/bin/bash
#
# Quoridor CNN Cloud Deployment Script
# Works on: RunPod, Vast.ai, Lambda Labs, AWS, Google Cloud
#

set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║     QUORIDOR CNN - CLOUD DEPLOYMENT SCRIPT           ║"
echo "╚══════════════════════════════════════════════════════╝"

# ============== CONFIGURATION ==============
# Adjust these for your needs:

REPO_URL="https://github.com/YOUR_USERNAME/Quridor_project.git"  # UPDATE THIS
BRANCH="main"
USE_GPU=true

# ============== INSTALLATION ==============

echo ""
echo "▶ Step 1: Installing dependencies..."

# Update system
apt-get update -qq

# Install Java 17
if ! command -v java &> /dev/null; then
    echo "  Installing Java 17..."
    apt-get install -y openjdk-17-jdk -qq
fi

# Install Maven
if ! command -v mvn &> /dev/null; then
    echo "  Installing Maven..."
    apt-get install -y maven -qq
fi

# Install git if needed
if ! command -v git &> /dev/null; then
    echo "  Installing git..."
    apt-get install -y git -qq
fi

echo "  ✓ Dependencies installed"

# ============== PROJECT SETUP ==============

echo ""
echo "▶ Step 2: Setting up project..."

# Clone or update repo
if [ -d "Quridor_project" ]; then
    echo "  Updating existing project..."
    cd Quridor_project
    git pull origin $BRANCH
else
    echo "  Cloning project..."
    git clone -b $BRANCH $REPO_URL
    cd Quridor_project
fi

echo "  ✓ Project ready"

# ============== GPU CONFIGURATION ==============

echo ""
echo "▶ Step 3: Configuring for GPU..."

if [ "$USE_GPU" = true ]; then
    # Check if NVIDIA GPU is available
    if command -v nvidia-smi &> /dev/null; then
        echo "  GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

        # Update pom.xml to use CUDA
        echo "  Switching to CUDA backend..."
        sed -i 's/nd4j-native-platform/nd4j-cuda-11.8-platform/g' pom.xml

        echo "  ✓ GPU enabled"
    else
        echo "  ⚠ No GPU detected, using CPU"
    fi
else
    echo "  Using CPU backend"
fi

# ============== BUILD ==============

echo ""
echo "▶ Step 4: Building project..."

# Set memory options
export JAVA_TOOL_OPTIONS="-Xmx24G -XX:+UseG1GC"

# Download dependencies and compile
mvn dependency:go-offline -B -q
mvn compile -B -q

echo "  ✓ Build complete"

# ============== TRAINING ==============

echo ""
echo "▶ Step 5: Starting training..."
echo ""
echo "  This will take a few hours depending on GPU."
echo "  Models will be saved to:"
echo "    - quoridor_cnn_imitation.zip (after imitation learning)"
echo "    - quoridor_cnn_best.zip (best performing)"
echo "    - quoridor_cnn_final.zip (final model)"
echo ""

# Run training
mvn exec:java -Dexec.mainClass="ml.cnn.CloudTrainer"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║              TRAINING COMPLETE!                       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Models saved in current directory."
echo "Copy them with: scp *.zip your-computer:destination/"
