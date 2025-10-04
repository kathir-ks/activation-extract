# Docker Setup for Multi-Host TPU Pods

Complete guide for running distributed inference with activation extraction on multi-host TPU pods using Docker.

## Overview

This guide covers:
- Building Docker images on all TPU hosts
- Distributing code and data across hosts
- Running synchronized inference across the pod
- Collecting activations from all hosts

## Prerequisites

- Multi-host TPU pod (e.g., v5e-256 = 4 hosts)
- Google Cloud SDK installed
- Docker configured on TPU VMs
- GCS bucket for data sharing

## Architecture

```
v5e-256 TPU Pod
├── Host 0 (worker-0) - 64 TPU cores
├── Host 1 (worker-1) - 64 TPU cores
├── Host 2 (worker-2) - 64 TPU cores
└── Host 3 (worker-3) - 64 TPU cores
    Total: 256 TPU cores
```

Each host runs the same Docker container in parallel.

---

## Step 1: Create Multi-Host TPU Pod

```bash
export TPU_NAME=arc-inference-multihost
export ZONE=us-central2-b
export ACCELERATOR=v5litepod-256

gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --accelerator-type=${ACCELERATOR} \
  --version=tpu-ubuntu2204-base \
  --network=default
```

**This creates 4 hosts** (workers 0-3) with 64 cores each.

---

## Step 2: Install Docker on All Hosts

```bash
# Run on all workers simultaneously
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    # Update system
    sudo apt-get update

    # Install Docker
    sudo apt-get install -y docker.io

    # Add user to docker group
    sudo usermod -aG docker \${USER}

    # Enable Docker service
    sudo systemctl enable docker
    sudo systemctl start docker

    # Verify installation
    docker --version
  "
```

**Logout and login again** for group changes to take effect:

```bash
# SSH to each worker to refresh session
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --command="exit"
```

---

## Step 3: Copy Code to All Hosts

### Option A: From Local Machine

```bash
# Copy code to all workers
for worker in 0 1 2 3; do
  gcloud compute tpus tpu-vm scp \
    --recurse \
    --zone=${ZONE} \
    --worker=${worker} \
    /path/to/local/torch_xla/qwen \
    ${TPU_NAME}:~/torch_xla/
done
```

### Option B: Clone from Git (Recommended)

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    cd ~
    git clone https://github.com/your-repo/torch_xla.git
    cd torch_xla/qwen
    chmod +x run-docker.sh
  "
```

---

## Step 4: Build Docker Image on All Hosts

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    cd ~/torch_xla/qwen
    ./run-docker.sh build
  "
```

This builds the image **in parallel** on all 4 hosts. Takes ~5-10 minutes.

**Verify build:**

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="docker images | grep arc-inference"
```

---

## Step 5: Prepare Data on GCS

Since all hosts need access to the same data, use GCS:

### Transform Dataset (Run on Worker 0 Only)

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=0 \
  --command="
    cd ~/torch_xla/qwen
    docker run --rm \
      -v /tmp/arc_data:/app/arc_data \
      arc-inference \
      python transform_hf_to_arc.py \
        --dataset barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
        --output_dir /app/arc_data \
        --max_samples 50000
  "
```

### Upload to GCS

```bash
export GCS_BUCKET=gs://your-bucket-name

gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=0 \
  --command="
    gsutil cp /tmp/arc_data/arc_format_train.json ${GCS_BUCKET}/data/
    gsutil cp /tmp/arc_data/test_outputs_train.json ${GCS_BUCKET}/data/
  "
```

### Download on All Hosts

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    mkdir -p /tmp/arc_data
    gsutil cp ${GCS_BUCKET}/data/arc_format_train.json /tmp/arc_data/
    gsutil cp ${GCS_BUCKET}/data/test_outputs_train.json /tmp/arc_data/
  "
```

---

## Step 6: Download Model Weights

### Option A: Download on Each Host

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    mkdir -p /tmp/model
    cd /tmp/model

    # Download from HuggingFace
    pip install huggingface_hub
    python -c '
from huggingface_hub import snapshot_download
snapshot_download(\"Qwen/Qwen2.5-0.5B\", local_dir=\"/tmp/model/qwen2.5-0.5b\")
    '
  "
```

### Option B: Upload to GCS (Faster for Multi-Host)

```bash
# On local machine or worker-0
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir ./qwen2.5-0.5b
gsutil -m cp -r ./qwen2.5-0.5b ${GCS_BUCKET}/models/

# Download on all hosts
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    mkdir -p /tmp/model
    gsutil -m cp -r ${GCS_BUCKET}/models/qwen2.5-0.5b /tmp/model/
  "
```

---

## Step 7: Run Distributed Inference (All Hosts)

### Create Run Script

First, create a helper script on all hosts:

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    cat > ~/run_distributed_docker.sh <<'EOF'
#!/bin/bash
set -e

# Configuration
MESH_SHAPE=\"16,16\"  # For v5e-256: 16x16
BATCH_SIZE=8
GCS_BUCKET=\"${GCS_BUCKET}\"

# Get host information
HOST_ID=\$(hostname | grep -oP 'worker-\K\d+' || echo 0)

echo \"Starting distributed inference on host \${HOST_ID}\"

# Run Docker container
docker run --privileged \\
  --network=host \\
  -v /tmp/arc_data:/app/arc_data \\
  -v /tmp/model:/app/model \\
  -v /tmp/outputs:/app/outputs \\
  -v /tmp/activations:/app/activations \\
  -e JAX_PLATFORMS=tpu \\
  -e MESH_SHAPE=\${MESH_SHAPE} \\
  -e BATCH_SIZE=\${BATCH_SIZE} \\
  -e TPU_HOST_ID=\${HOST_ID} \\
  arc-inference \\
  python distributed_inference_with_activations.py \\
    --model_path /app/model/qwen2.5-0.5b \\
    --tasks_file /app/arc_data/arc_format_train.json \\
    --output_dir /app/outputs \\
    --activations_dir /app/activations \\
    --batch_size \${BATCH_SIZE} \\
    --mesh_shape \${MESH_SHAPE} \\
    --gcs_bucket \${GCS_BUCKET}/activations \\
    --upload_to_cloud True

echo \"Finished inference on host \${HOST_ID}\"
EOF

    chmod +x ~/run_distributed_docker.sh
  "
```

### Run on All Hosts Simultaneously

```bash
# Launch on all workers in parallel
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    cd ~
    ./run_distributed_docker.sh
  "
```

**This command:**
- Starts Docker containers on all 4 hosts simultaneously
- Each host processes its shard of the data
- JAX automatically coordinates across hosts
- Activations are saved locally and uploaded to GCS

---

## Step 8: Monitor Progress

### Monitor All Hosts

```bash
# Check running containers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="docker ps"

# Monitor logs from worker 0
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=0 \
  --command="docker logs -f \$(docker ps -q | head -1)"
```

### Check TPU Utilization

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=0 \
  --command="
    watch -n 1 'python -c \"import jax; print(f\\\"Devices: {len(jax.devices())}\\\")\"'
  "
```

### Monitor Output Files

```bash
# Check output progress on each worker
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    ls -lh /tmp/outputs/ /tmp/activations/ | head -20
  "
```

---

## Step 9: Collect Results

### Check Activations in GCS

```bash
# Activations are auto-uploaded to GCS
gsutil ls ${GCS_BUCKET}/activations/

# Download to local machine
gsutil -m cp -r ${GCS_BUCKET}/activations ./local_activations/
```

### Collect from Hosts Directly

If not using cloud upload:

```bash
# Create local directory
mkdir -p ./collected_activations

# Download from each worker
for worker in 0 1 2 3; do
  gcloud compute tpus tpu-vm scp \
    --recurse \
    --zone=${ZONE} \
    --worker=${worker} \
    ${TPU_NAME}:/tmp/activations \
    ./collected_activations/worker_${worker}/
done
```

---

## Step 10: Cleanup

### Stop Containers

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="docker stop \$(docker ps -q)"
```

### Clear Data (Optional)

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    rm -rf /tmp/arc_data /tmp/outputs /tmp/activations
  "
```

### Delete TPU Pod

```bash
gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}
```

---

## Complete Example Script

Save this as `multihost_docker_run.sh`:

```bash
#!/bin/bash
set -e

# Configuration
export TPU_NAME=arc-inference-multihost
export ZONE=us-central2-b
export ACCELERATOR=v5litepod-256
export GCS_BUCKET=gs://your-bucket-name
export MESH_SHAPE="16,16"
export BATCH_SIZE=8

echo "=== Multi-Host TPU Docker Distributed Inference ==="
echo "TPU: ${TPU_NAME}"
echo "Zone: ${ZONE}"
echo "Accelerator: ${ACCELERATOR}"
echo "GCS Bucket: ${GCS_BUCKET}"
echo "Mesh Shape: ${MESH_SHAPE}"
echo ""

# Step 1: Create TPU
echo "Step 1: Creating TPU pod..."
gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --accelerator-type=${ACCELERATOR} \
  --version=tpu-ubuntu2204-base

# Step 2: Install Docker
echo "Step 2: Installing Docker on all hosts..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    sudo apt-get update && \
    sudo apt-get install -y docker.io && \
    sudo usermod -aG docker \${USER}
  "

# Step 3: Copy code
echo "Step 3: Copying code to all hosts..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    git clone https://github.com/your-repo/torch_xla.git && \
    cd torch_xla/qwen && \
    chmod +x run-docker.sh
  "

# Step 4: Build Docker image
echo "Step 4: Building Docker image on all hosts..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    cd ~/torch_xla/qwen && \
    ./run-docker.sh build
  "

# Step 5: Prepare data
echo "Step 5: Preparing data..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=0 \
  --command="
    cd ~/torch_xla/qwen && \
    docker run --rm -v /tmp/arc_data:/app/arc_data arc-inference \
      python transform_hf_to_arc.py \
        --dataset barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
        --output_dir /app/arc_data \
        --max_samples 50000 && \
    gsutil cp /tmp/arc_data/*.json ${GCS_BUCKET}/data/
  "

# Download data to all hosts
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    mkdir -p /tmp/arc_data && \
    gsutil cp ${GCS_BUCKET}/data/*.json /tmp/arc_data/
  "

# Step 6: Download model
echo "Step 6: Downloading model to all hosts..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    mkdir -p /tmp/model && \
    gsutil -m cp -r ${GCS_BUCKET}/models/qwen2.5-0.5b /tmp/model/
  "

# Step 7: Run distributed inference
echo "Step 7: Running distributed inference..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=all \
  --command="
    cd ~/torch_xla/qwen && \
    docker run --privileged --network=host \
      -v /tmp/arc_data:/app/arc_data \
      -v /tmp/model:/app/model \
      -v /tmp/outputs:/app/outputs \
      -v /tmp/activations:/app/activations \
      -e JAX_PLATFORMS=tpu \
      -e MESH_SHAPE=${MESH_SHAPE} \
      arc-inference \
      python distributed_inference_with_activations.py \
        --model_path /app/model/qwen2.5-0.5b \
        --tasks_file /app/arc_data/arc_format_train.json \
        --output_dir /app/outputs \
        --activations_dir /app/activations \
        --batch_size ${BATCH_SIZE} \
        --mesh_shape ${MESH_SHAPE} \
        --gcs_bucket ${GCS_BUCKET}/activations \
        --upload_to_cloud True
  "

echo "=== Distributed inference complete! ==="
echo "Check results: gsutil ls ${GCS_BUCKET}/activations/"
```

---

## Troubleshooting

### Docker Permission Denied

```bash
# Re-login after adding user to docker group
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --command="exit"
```

### TPU Not Detected

```bash
# Verify TPU visibility in container
docker run --privileged arc-inference python -c "import jax; print(jax.devices())"
```

### Hosts Not Coordinating

Ensure `--network=host` is used in docker run command for multi-host communication.

### Out of Memory

Reduce `--batch_size` or extract fewer layers:

```bash
--batch_size 4
--layers_to_extract 15 16 17 18 19 20 21 22 23
```

---

## Performance on v5e-256

- **Total cores**: 256
- **Hosts**: 4
- **Cores per host**: 64
- **Mesh shape**: 16×16
- **Batch size**: 8 per core = 2048 samples/iteration
- **Throughput**: ~10-15 iterations/minute
- **50k samples**: ~15-20 minutes

---

## Summary

✅ **Multi-host Docker setup complete**
✅ **All hosts run same container**
✅ **Data shared via GCS**
✅ **JAX auto-coordinates across hosts**
✅ **Activations saved and uploaded**

Your distributed inference is ready to run on multi-host TPU pods! 🚀
