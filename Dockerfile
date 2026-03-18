# Dockerfile for Quoridor CNN Training on GPU Cloud
# Compatible with RunPod, Vast.ai, Lambda Labs, AWS

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Java 17 and Maven
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    maven \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Java home
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Create working directory
WORKDIR /app

# Copy project files
COPY pom.xml .
COPY src/ src/
COPY CLAUDE.md .
COPY CNN_NOTES.txt .

# Download dependencies (cache layer)
RUN mvn dependency:go-offline -B

# Build the project
RUN mvn compile -B

# Set memory options for training
ENV JAVA_TOOL_OPTIONS="-Xmx24G -XX:+UseG1GC"

# Default command: run cloud trainer
CMD ["mvn", "exec:java", "-Dexec.mainClass=ml.cnn.CloudTrainer"]
