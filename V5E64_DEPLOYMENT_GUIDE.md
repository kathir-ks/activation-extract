# TPU v5e-64 Deployment Guide

Complete guide for running activation extraction on Google Cloud TPU v5e-64 with Docker support.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
4. [Docker Setup](#docker-setup)
5. [TPU v5e-64 Setup](#tpu-v5e-64-setup)
6. [Deployment](#deployment)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

## Overview

### TPU v5e-64 Specifications
- **Architecture**: 4 hosts × 8 chips/host = 32 total chips
- **Memory**: 16 GB HBM per chip = 512 GB total
- **Network**: Multi-host coordination via JAX distributed
- **Best for**: Large models (7B+) with FSDP-style sharding

### What This Does
1. Converts HuggingFace dataset to ARC format (JSONL)
2. Extracts activations from Qwen models using JAX
3. Shards model across TPU chips for efficient inference
4. Stores activations in GCS with automatic sharding
5. Supports multi-machine deployment for massive datasets

## Prerequisites

### 1. Google Cloud Setup
```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Login and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable compute.googleapis.com
gcloud services enable tpu.googleapis.com
gcloud services enable storage.googleapis.com
```

### 2. GCS Bucket
```bash
# Create bucket for activations
export GCS_BUCKET="your-activation-bucket"
gsutil mb gs://$GCS_BUCKET

# Verify access
gsutil ls gs://$GCS_BUCKET
```

### 3. Docker (Optional but Recommended)
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
```

## Dataset Preparation

### Step 1: Convert HuggingFace Dataset to ARC Format

```bash
# Run conversion script
python convert_hf_to_arc_format.py \
    --dataset_name "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems" \
    --column_name "examples" \
    --output_file "arc_formatted_challenges.jsonl" \
    --max_tasks 10000 \
    --max_train_examples None \
    --verbose

# Expected output:
# ======================================================================
# Converting HuggingFace dataset to ARC format
# ======================================================================
#   Dataset: barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems
#   Column: examples
#   Output: arc_formatted_challenges.jsonl
#   Max tasks: 10000
#   Max train examples: unlimited (no slicing!)
# ======================================================================
# Converting: 100%|████████████████████████| 10000/10000 [00:30<00:00, 330.12it/s]
#
# ======================================================================
# ✅ Conversion complete!
# ======================================================================
#   Tasks converted: 10000
#   Invalid tasks skipped: 123
#   Output file: arc_formatted_challenges.jsonl
#   File size: 45.32 MB
# ======================================================================
```

### Step 2: Upload Dataset to GCS (for TPU access)

```bash
# Upload dataset
gsutil cp arc_formatted_challenges.jsonl gs://$GCS_BUCKET/datasets/

# Verify
gsutil ls -lh gs://$GCS_BUCKET/datasets/
```

### Dataset Format

The converted dataset is in JSONL format (one task per line):

```json
{
  "task_id": "task_00000001",
  "train": [
    {"input": [[0,1],[2,3]], "output": [[1,0],[3,2]]},
    {"input": [[4,5],[6,7]], "output": [[5,4],[7,6]]}
  ],
  "test": [
    {"input": [[8,9],[10,11]]}
  ]
}
```

**Key difference from your original code**: We keep ALL training examples (no slicing to 5 pairs).

## Docker Setup

### Option 1: Build Docker Image Locally

```bash
# Build image
docker build -t activation-extraction:latest .

# Test image
docker run --rm activation-extraction:latest \
    python convert_hf_to_arc_format.py --help
```

### Option 2: Use Google Container Registry

```bash
# Set registry
export PROJECT_ID=$(gcloud config get-value project)
export IMAGE_NAME="gcr.io/$PROJECT_ID/activation-extraction:latest"

# Build and push
docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME

# Verify
gcloud container images list --repository=gcr.io/$PROJECT_ID
```

### Test Docker Image Locally

```bash
# Test dataset conversion
docker run --rm \
    -v $(pwd):/workspace/data \
    -e GCS_BUCKET=$GCS_BUCKET \
    activation-extraction:latest \
    python convert_hf_to_arc_format.py \
        --max_tasks 10 \
        --output_file /workspace/data/test.jsonl \
        --verbose

# Verify output
ls -lh test.jsonl
```

## TPU v5e-64 Setup

### Step 1: Create TPU v5e-64

```bash
# Set zone (choose one with v5e availability)
export ZONE="us-central2-b"  # or us-west1-a, us-east5-c

# Create TPU v5e-64
gcloud compute tpus tpu-vm create tpu-v5e-64 \
    --zone=$ZONE \
    --accelerator-type=v5litepod-64 \
    --version=tpu-vm-v4-base \
    --metadata="startup-script=#!/bin/bash
        apt-get update
        apt-get install -y docker.io git
        systemctl enable docker
        systemctl start docker
    "

# Wait for creation (takes ~5 minutes)
gcloud compute tpus tpu-vm describe tpu-v5e-64 --zone=$ZONE
```

### Step 2: SSH into TPU (Host 0)

```bash
# SSH to host 0
gcloud compute tpus tpu-vm ssh tpu-v5e-64 \
    --zone=$ZONE \
    --worker=0

# You're now on host 0 of the TPU v5e-64
```

### Step 3: Setup on Each Host

You need to run these commands on ALL 4 hosts (worker=0,1,2,3):

```bash
# On each host, run:
for WORKER in 0 1 2 3; do
    gcloud compute tpus tpu-vm ssh tpu-v5e-64 \
        --zone=$ZONE \
        --worker=$WORKER \
        --command="
            # Install dependencies
            pip3 install --upgrade pip
            pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
            pip3 install torch transformers datasets fsspec gcsfs tqdm

            # Clone repo (or copy files)
            mkdir -p ~/activation-extraction

            # Setup GCS credentials
            gcloud auth application-default login
        "
done
```

### Step 4: Copy Code to All Hosts

```bash
# Copy files to all hosts
for WORKER in 0 1 2 3; do
    gcloud compute tpus tpu-vm scp \
        --zone=$ZONE \
        --worker=$WORKER \
        --recurse \
        qwen2_jax.py \
        qwen2_jax_with_hooks.py \
        extract_activations_arc_v5e64.py \
        extract_activations_fineweb_multihost.py \
        convert_hf_to_arc_format.py \
        launch_v5e64.sh \
        arc24/ \
        tpu-v5e-64:~/activation-extraction/
done
```

### Step 5: Copy or Download Dataset

```bash
# Option 1: Download from GCS (recommended)
for WORKER in 0 1 2 3; do
    gcloud compute tpus tpu-vm ssh tpu-v5e-64 \
        --zone=$ZONE \
        --worker=$WORKER \
        --command="
            cd ~/activation-extraction
            gsutil cp gs://$GCS_BUCKET/datasets/arc_formatted_challenges.jsonl .
        "
done

# Option 2: Copy from local
for WORKER in 0 1 2 3; do
    gcloud compute tpus tpu-vm scp \
        --zone=$ZONE \
        --worker=$WORKER \
        arc_formatted_challenges.jsonl \
        tpu-v5e-64:~/activation-extraction/
done
```

## Deployment

### Single v5e-64 Machine Deployment

This is the most common scenario: running extraction on one v5e-64 (4 hosts, 32 chips).

#### Step 1: Get Coordinator Address (Host 0 IP)

```bash
# On host 0, get internal IP
gcloud compute tpus tpu-vm ssh tpu-v5e-64 \
    --zone=$ZONE \
    --worker=0 \
    --command="hostname -i"

# Example output: 10.128.0.2
export COORDINATOR_IP="10.128.0.2"
```

#### Step 2: Launch on All Hosts

Create a launch script `launch_all_hosts.sh`:

```bash
#!/bin/bash

export ZONE="us-central2-b"
export COORDINATOR_IP="10.128.0.2"  # Replace with actual IP
export GCS_BUCKET="your-activation-bucket"

# Launch on each host
for WORKER in 0 1 2 3; do
    echo "Launching on host $WORKER..."

    gcloud compute tpus tpu-vm ssh tpu-v5e-64 \
        --zone=$ZONE \
        --worker=$WORKER \
        --command="
            cd ~/activation-extraction

            # Set environment variables
            export HOST_ID=$WORKER
            export NUM_HOSTS=4
            export COORDINATOR_ADDRESS=\"${COORDINATOR_IP}:8476\"
            export GCS_BUCKET=\"$GCS_BUCKET\"
            export MODEL_PATH=\"KathirKs/qwen-2.5-7b\"
            export DATASET_PATH=\"arc_formatted_challenges.jsonl\"
            export BATCH_SIZE=16
            export MESH_TYPE=\"2d\"
            export MACHINE_ID=0
            export TOTAL_MACHINES=1

            # Launch in background
            nohup bash launch_v5e64.sh > launch_host${WORKER}.log 2>&1 &

            echo \"Launched on host $WORKER (PID: \$!)\"
        " &
done

wait
echo "All hosts launched!"
```

#### Step 3: Monitor Progress

```bash
# Check logs on each host
for WORKER in 0 1 2 3; do
    echo "=== Host $WORKER ==="
    gcloud compute tpus tpu-vm ssh tpu-v5e-64 \
        --zone=$ZONE \
        --worker=$WORKER \
        --command="tail -20 ~/activation-extraction/logs/extraction_machine0_host${WORKER}_*.log"
done
```

### Multi-Machine Deployment

If you have multiple v5e-64 machines (e.g., 4 machines = 128 chips total):

```bash
# On Machine 0, Host 0:
export MACHINE_ID=0
export TOTAL_MACHINES=4

# On Machine 1, Host 0:
export MACHINE_ID=1
export TOTAL_MACHINES=4

# ... etc for machines 2 and 3

# Each machine processes different shard of dataset
```

## Monitoring

### Real-time Progress Monitoring

```bash
# Monitor all hosts
watch -n 5 'for WORKER in 0 1 2 3; do \
    echo "=== Host $WORKER ==="; \
    gcloud compute tpus tpu-vm ssh tpu-v5e-64 \
        --zone=$ZONE \
        --worker=$WORKER \
        --command="tail -5 ~/activation-extraction/logs/extraction_*.log 2>/dev/null" \
    done'
```

### Check GCS Upload Progress

```bash
# Check uploaded shards
gsutil ls -lh gs://$GCS_BUCKET/activations_arc_v5e64/

# Count shards per host
for WORKER in 0 1 2 3; do
    echo "Host $WORKER:"
    gsutil ls gs://$GCS_BUCKET/activations_arc_v5e64/machine_000_host_0${WORKER}/ | wc -l
done

# Check total size
gsutil du -sh gs://$GCS_BUCKET/activations_arc_v5e64/
```

### Performance Metrics

Expected performance for Qwen 2.5 7B on v5e-64:
- **Compilation time**: ~60-120s (first batch only)
- **Inference time**: ~0.3-0.5s per batch (after compilation)
- **Throughput**: ~30-50 samples/sec (across all hosts)
- **Memory usage**: ~8-10 GB per chip (model) + 4-6 GB (activations)

## Troubleshooting

### Issue 1: JAX Distributed Initialization Fails

```bash
# Error: Failed to initialize JAX distributed
# Solution: Check coordinator address and network connectivity

# On host 0, verify IP
hostname -i

# On other hosts, verify connectivity
ping -c 3 <COORDINATOR_IP>

# Check firewall rules
gcloud compute firewall-rules list | grep allow-internal
```

### Issue 2: Out of Memory (OOM)

```bash
# Error: Out of memory on TPU
# Solutions:
# 1. Reduce batch size
export BATCH_SIZE=8  # or 4

# 2. Reduce max sequence length
export MAX_SEQ_LENGTH=1024

# 3. Use 1D mesh instead of 2D
export MESH_TYPE="1d"

# 4. Extract fewer layers
# Edit extract_activations_arc_v5e64.py line ~302:
# cfg.layers_to_extract = list(range(20, 28))  # Only last 8 layers
```

### Issue 3: Slow Upload to GCS

```bash
# Check network bandwidth
gcloud compute tpus tpu-vm ssh tpu-v5e-64 \
    --zone=$ZONE \
    --worker=0 \
    --command="gsutil perfdiag gs://$GCS_BUCKET"

# Solution: Increase shard size to reduce upload frequency
export SHARD_SIZE_GB=5.0  # Larger shards = fewer uploads
```

### Issue 4: Dataset Not Found

```bash
# Verify dataset exists on each host
for WORKER in 0 1 2 3; do
    gcloud compute tpus tpu-vm ssh tpu-v5e-64 \
        --zone=$ZONE \
        --worker=$WORKER \
        --command="ls -lh ~/activation-extraction/arc_formatted_challenges.jsonl"
done

# If missing, re-copy
# ... (see Step 5 in TPU v5e-64 Setup)
```

### Issue 5: Docker Issues

```bash
# Error: Docker daemon not running
gcloud compute tpus tpu-vm ssh tpu-v5e-64 \
    --zone=$ZONE \
    --worker=0 \
    --command="sudo systemctl start docker"

# Error: Permission denied
gcloud compute tpus tpu-vm ssh tpu-v5e-64 \
    --zone=$ZONE \
    --worker=0 \
    --command="sudo usermod -aG docker $USER"
# Then logout and login again
```

## Cost Estimation

### TPU v5e-64 Pricing (as of 2024)
- **On-demand**: ~$12/hour
- **Preemptible**: ~$3/hour (can be interrupted)
- **Spot**: ~$2/hour (lowest price, can be interrupted)

### Example Cost Calculation

Processing 10,000 tasks with 8 predictions per task = 80,000 samples:
- **Throughput**: ~40 samples/sec
- **Time**: 80,000 / 40 = 2,000 seconds ≈ 33 minutes
- **Cost**: 0.55 hours × $12/hour = **$6.60** (on-demand)
- **Cost**: 0.55 hours × $2/hour = **$1.10** (spot)

### Storage Cost
- **GCS Standard**: $0.020 per GB/month
- **Activation size**: ~1 GB per 1,000 samples
- **10,000 tasks**: ~80 GB
- **Monthly cost**: 80 GB × $0.020 = **$1.60/month**

## Best Practices

### 1. Use Spot/Preemptible TPUs
```bash
# Create preemptible TPU (70% cheaper!)
gcloud compute tpus tpu-vm create tpu-v5e-64 \
    --zone=$ZONE \
    --accelerator-type=v5litepod-64 \
    --version=tpu-vm-v4-base \
    --preemptible
```

### 2. Enable Checkpointing
Add checkpointing to resume if preempted:
```python
# TODO: Implement checkpointing in extraction script
# Save progress every N batches
# Resume from last checkpoint on restart
```

### 3. Optimize Batch Size
```bash
# Test different batch sizes
for BS in 8 16 32; do
    export BATCH_SIZE=$BS
    # Run and measure throughput
done
```

### 4. Delete TPU When Done
```bash
# IMPORTANT: Don't forget to delete TPU!
gcloud compute tpus tpu-vm delete tpu-v5e-64 --zone=$ZONE

# Verify deletion
gcloud compute tpus tpu-vm list --zone=$ZONE
```

## Quick Start Commands

```bash
# 1. Convert dataset
python convert_hf_to_arc_format.py \
    --max_tasks 10000 \
    --verbose

# 2. Create TPU
gcloud compute tpus tpu-vm create tpu-v5e-64 \
    --zone=us-central2-b \
    --accelerator-type=v5litepod-64 \
    --version=tpu-vm-v4-base \
    --preemptible

# 3. Setup (run on all hosts)
# ... see "TPU v5e-64 Setup" section

# 4. Launch extraction
bash launch_all_hosts.sh

# 5. Monitor
watch -n 5 'gsutil du -sh gs://$GCS_BUCKET/activations_arc_v5e64/'

# 6. Cleanup
gcloud compute tpus tpu-vm delete tpu-v5e-64 --zone=us-central2-b
```

## Summary

You now have a complete pipeline for:
1. ✅ Converting HuggingFace datasets to ARC format (no slicing!)
2. ✅ Running activation extraction on TPU v5e-64
3. ✅ Using Docker for portability
4. ✅ Uploading to GCS automatically
5. ✅ Multi-host coordination with JAX distributed
6. ✅ Monitoring and troubleshooting

**Next steps**: Run the pipeline and monitor the results in GCS!
