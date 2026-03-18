# Cloud Training Guide for Quoridor CNN

## Quick Start (RunPod - Recommended)

### 1. Create RunPod Account
Go to [runpod.io](https://runpod.io) and sign up.

### 2. Launch GPU Instance
- Click "Deploy" → "GPU Cloud"
- Select: **RTX A5000** ($0.29/hr) or **A100** ($1.09/hr)
- Template: **RunPod Pytorch 2.0** (has CUDA pre-installed)
- Storage: 20GB
- Click "Deploy"

### 3. Connect via SSH or Web Terminal

### 4. Run Training
```bash
# One-liner setup and train:
apt-get update && apt-get install -y openjdk-17-jdk maven git && \
git clone https://github.com/YOUR_USERNAME/Quridor_project.git && \
cd Quridor_project && \
cp pom-cuda.xml pom.xml && \
export JAVA_TOOL_OPTIONS="-Xmx24G" && \
mvn compile exec:java -Dexec.mainClass="ml.cnn.CloudTrainer"
```

### 5. Download Results
```bash
# From your local machine:
scp root@YOUR_POD_IP:~/Quridor_project/quoridor_cnn_best.zip .
```

---

## Alternative Platforms

### Vast.ai (Cheapest)
```bash
# 1. Go to vast.ai and rent A100 (~$0.80/hr)
# 2. SSH into instance
# 3. Run the same commands as above
```

### Lambda Labs
```bash
# 1. Go to lambdalabs.com
# 2. Launch A100 instance (~$1.10/hr)
# 3. Run the same commands
```

### AWS EC2
```bash
# 1. Launch g4dn.xlarge (T4 GPU, ~$0.52/hr)
# 2. Use Deep Learning AMI (Ubuntu)
# 3. Run:
sudo apt install openjdk-17-jdk maven
git clone https://github.com/YOUR_USERNAME/Quridor_project.git
cd Quridor_project
cp pom-cuda.xml pom.xml
export JAVA_TOOL_OPTIONS="-Xmx24G"
mvn compile exec:java -Dexec.mainClass="ml.cnn.CloudTrainer"
```

### Google Colab (Free but Limited)
```python
# In a Colab notebook:
!apt-get install openjdk-17-jdk maven -qq
!git clone https://github.com/YOUR_USERNAME/Quridor_project.git
%cd Quridor_project
!cp pom-cuda.xml pom.xml
!mvn compile exec:java -Dexec.mainClass="ml.cnn.CloudTrainer"
```

---

## Training Parameters

Edit `CloudTrainer.java` to adjust:

```java
IMITATION_GAMES = 50000;    // Increase for better imitation
SELF_PLAY_ROUNDS = 20;      // More rounds = better self-improvement
GAMES_PER_ROUND = 5000;     // Games per self-play round
BATCH_SIZE = 256;           // Larger = faster on GPU
```

### Recommended Settings by GPU:

| GPU | IMITATION_GAMES | GAMES_PER_ROUND | Time to 75% |
|-----|-----------------|-----------------|-------------|
| T4 (free/cheap) | 20,000 | 2,000 | ~4-6 hours |
| A5000 | 50,000 | 5,000 | ~2-3 hours |
| A100 | 100,000 | 10,000 | ~1-2 hours |
| H100 | 200,000 | 20,000 | ~30-60 min |

---

## Estimated Costs

| Platform | GPU | Est. Time | Est. Cost |
|----------|-----|-----------|-----------|
| Google Colab | T4 | 4-6 hours | FREE |
| Vast.ai | A100 | 2 hours | ~$2 |
| RunPod | A5000 | 3 hours | ~$1 |
| Lambda Labs | A100 | 2 hours | ~$2 |
| AWS | T4 | 5 hours | ~$3 |

---

## Troubleshooting

### "CUDA not found"
```bash
# Check CUDA installation:
nvidia-smi
nvcc --version

# If missing, install:
apt-get install nvidia-cuda-toolkit
```

### "Out of memory"
```bash
# Reduce batch size in CloudTrainer.java:
BATCH_SIZE = 128;  # or 64

# Or reduce games:
IMITATION_GAMES = 20000;
```

### "Maven not found"
```bash
apt-get update && apt-get install maven
```

### "Java version error"
```bash
apt-get install openjdk-17-jdk
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
```

---

## Files Produced

After training, you'll have:
- `quoridor_cnn_imitation.zip` - Model after imitation learning
- `quoridor_cnn_best.zip` - Best performing model
- `quoridor_cnn_final.zip` - Final model
- `quoridor_cnn_checkpoint_r*.zip` - Checkpoints every 5 rounds

Copy the best model to your local project:
```bash
scp user@cloud-ip:~/Quridor_project/quoridor_cnn_best.zip ./src/ml/cnn/
```

Then use it in your game!
