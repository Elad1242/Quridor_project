# Dockerfile for Quoridor ML Training on GPU Cloud (RunPod)
# Uses DJL + PyTorch with auto CUDA detection

# H100 requires CUDA 12.1+. Using 12.1 for maximum compatibility.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Java 17 and Maven
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    maven \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /app

# Copy pom first for dependency caching
COPY pom-ml.xml pom.xml

# Download dependencies (cached Docker layer)
RUN mvn dependency:go-offline -B

# Copy source code
COPY src/ src/

# Build (excludes UI and old ML code via pom-ml.xml)
RUN mvn compile -B

# Memory and GPU settings
ENV JAVA_TOOL_OPTIONS="-Xmx24G -XX:+UseG1GC"

# Create data directories
RUN mkdir -p /data/training /data/models

# Default: generate data then train
COPY scripts/run_training.sh /app/run_training.sh
RUN chmod +x /app/run_training.sh

CMD ["/app/run_training.sh"]
