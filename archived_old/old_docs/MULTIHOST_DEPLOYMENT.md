# Multihost TPU v5e-64 Deployment Guide

This guide explains how to deploy the activation extraction pipeline across multiple TPU v5e-64 machines (4 hosts).

## Table of Contents
1. [Docker Registry Setup](#docker-registry-setup)
2. [Building and Pushing Docker Image](#building-and-pushing-docker-image)
3. [TPU Pod Setup](#tpu-pod-setup)
4. [Running on Multihost](#running-on-multihost)
5. [Troubleshooting](#troubleshooting)

---

## Docker Registry Setup

**Recommended Approach:** Build once, push to Google Container Registry (GCR), pull on each host.

### Why use a registry?
- ✅ **Faster deployment:** Build once (2-3 min), pull on all hosts (30 sec each)
- ✅ **Consistency:** All hosts use identical image
- ✅ **Bandwidth efficient:** No need to clone repo on each machine
- ❌ **Without registry:** Clone + build on each host = 8-12 minutes × 4 hosts

### Alternative: Build on each host
- Only if you don't have GCR access or need to iterate rapidly
- Each host must clone repo and build independently
- Takes 8-12 minutes per host

---

## Building and Pushing Docker Image

### Option A: Push to Google Container Registry (Recommended)

```bash
# 1. Set your GCP project
export PROJECT_ID="your-gcp-project-id"

# 2. Tag the image for GCR
docker tag activation-extraction:latest gcr.io/${PROJECT_ID}/activation-extraction:latest

# 3. Configure Docker for GCR
gcloud auth configure-docker

# 4. Push to GCR
docker push gcr.io/${PROJECT_ID}/activation-extraction:latest
```

**Image will be available at:** `gcr.io/${PROJECT_ID}/activation-extraction:latest`

### Option B: Push to Artifact Registry (Alternative)

```bash
# 1. Set variables
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export REPO_NAME="tpu-images"

# 2. Create repository (if not exists)
gcloud artifacts repositories create ${REPO_NAME} \
  --repository-format=docker \
  --location=${REGION} \
  --description="TPU activation extraction images"

# 3. Tag and push
docker tag activation-extraction:latest ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/activation-extraction:latest
gcloud auth configure-docker ${REGION}-docker.pkg.dev
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/activation-extraction:latest
```

**Image will be available at:** `${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/activation-extraction:latest`

### Option C: Build on Each Host (Not Recommended)

```bash
# On each host, run:
cd /path/to/torch_xla/qwen
sudo docker build -t activation-extraction:latest .
```

---

## TPU Pod Setup

### TPU v5e-64 Pod Architecture

A v5e-64 pod consists of:
- **4 hosts** (worker-0, worker-1, worker-2, worker-3)
- **8 TPU chips per host** = 32 total TPU chips
- **Each host** needs the Docker image and can access GCS

### Prerequisites on Each Host

1. **Docker installed**
2. **GCS credentials configured** (`~/.config/gcloud`)
3. **Sufficient disk space** for HuggingFace cache (~10 GB)

---

## Running on Multihost

### Architecture

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  worker-0   │  │  worker-1   │  │  worker-2   │  │  worker-3   │
│  (host 0)   │  │  (host 1)   │  │  (host 2)   │  │  (host 3)   │
│  8 TPU chips│  │  8 TPU chips│  │  8 TPU chips│  │  8 TPU chips│
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                        │
                   JAX multihost
                   coordination
                        │
                        ▼
              gs://fineweb-data-us-central1-a/
```

### Step 1: Pull Docker Image on All Hosts

**If using GCR:**

```bash
# On each host (worker-0, worker-1, worker-2, worker-3), run:
export PROJECT_ID="your-gcp-project-id"
sudo docker pull gcr.io/${PROJECT_ID}/activation-extraction:latest
```

**If using Artifact Registry:**

```bash
# On each host, run:
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export REPO_NAME="tpu-images"
sudo docker pull ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/activation-extraction:latest
```

### Step 2: Prepare Dataset and Cache

**On worker-0 (coordinator):**

```bash
# Create shared dataset (already exists in your case)
# The dataset file should be accessible from all hosts via GCS or shared filesystem

# Ensure HuggingFace cache directory exists on each host
mkdir -p ~/.cache/huggingface
```

### Step 3: Get Coordinator Address

**On worker-0:**

```bash
# Get internal IP of worker-0
export COORDINATOR_IP=$(hostname -I | awk '{print $1}')
echo "Coordinator IP: ${COORDINATOR_IP}"
# Example output: 10.128.0.5
```

Share this IP with all other hosts.

### Step 4: Run on All Hosts Simultaneously

**IMPORTANT:** Start all hosts within ~30 seconds of each other to avoid timeout.

#### On worker-0 (coordinator):

```bash
export COORDINATOR_IP="10.128.0.5"  # Replace with actual IP
export PROJECT_ID="your-gcp-project-id"

sudo docker run --rm --net=host --privileged \
  -v ~/torch_xla/qwen:/workspace/data \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  -v ~/.cache/huggingface:/cache/huggingface \
  gcr.io/${PROJECT_ID}/activation-extraction:latest \
  -c "cd /workspace/data && python /workspace/extract_activations_arc_v5e64.py \
    --machine_id 0 \
    --total_machines 1 \
    --host_id 0 \
    --num_hosts 4 \
    --multihost \
    --coordinator_address ${COORDINATOR_IP}:8476 \
    --model_path KathirKs/qwen-2.5-0.5b \
    --dataset_path gs://your-bucket/dataset.jsonl \
    --batch_size 4 \
    --max_seq_length 512 \
    --output_dir /workspace/data/output \
    --upload_to_gcs \
    --gcs_bucket fineweb-data-us-central1-a \
    --gcs_prefix activations_arc_v5e64 \
    --shard_size_gb 1.0 \
    --compress_shards \
    --verbose"
```

#### On worker-1:

```bash
export COORDINATOR_IP="10.128.0.5"  # Same as worker-0
export PROJECT_ID="your-gcp-project-id"

sudo docker run --rm --net=host --privileged \
  -v ~/torch_xla/qwen:/workspace/data \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  -v ~/.cache/huggingface:/cache/huggingface \
  gcr.io/${PROJECT_ID}/activation-extraction:latest \
  -c "cd /workspace/data && python /workspace/extract_activations_arc_v5e64.py \
    --machine_id 0 \
    --total_machines 1 \
    --host_id 1 \
    --num_hosts 4 \
    --multihost \
    --coordinator_address ${COORDINATOR_IP}:8476 \
    --model_path KathirKs/qwen-2.5-0.5b \
    --dataset_path gs://your-bucket/dataset.jsonl \
    --batch_size 4 \
    --max_seq_length 512 \
    --output_dir /workspace/data/output \
    --upload_to_gcs \
    --gcs_bucket fineweb-data-us-central1-a \
    --gcs_prefix activations_arc_v5e64 \
    --shard_size_gb 1.0 \
    --compress_shards \
    --verbose"
```

#### On worker-2:

```bash
export COORDINATOR_IP="10.128.0.5"  # Same as worker-0
export PROJECT_ID="your-gcp-project-id"

sudo docker run --rm --net=host --privileged \
  -v ~/torch_xla/qwen:/workspace/data \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  -v ~/.cache/huggingface:/cache/huggingface \
  gcr.io/${PROJECT_ID}/activation-extraction:latest \
  -c "cd /workspace/data && python /workspace/extract_activations_arc_v5e64.py \
    --machine_id 0 \
    --total_machines 1 \
    --host_id 2 \
    --num_hosts 4 \
    --multihost \
    --coordinator_address ${COORDINATOR_IP}:8476 \
    --model_path KathirKs/qwen-2.5-0.5b \
    --dataset_path gs://your-bucket/dataset.jsonl \
    --batch_size 4 \
    --max_seq_length 512 \
    --output_dir /workspace/data/output \
    --upload_to_gcs \
    --gcs_bucket fineweb-data-us-central1-a \
    --gcs_prefix activations_arc_v5e64 \
    --shard_size_gb 1.0 \
    --compress_shards \
    --verbose"
```

#### On worker-3:

```bash
export COORDINATOR_IP="10.128.0.5"  # Same as worker-0
export PROJECT_ID="your-gcp-project-id"

sudo docker run --rm --net=host --privileged \
  -v ~/torch_xla/qwen:/workspace/data \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  -v ~/.cache/huggingface:/cache/huggingface \
  gcr.io/${PROJECT_ID}/activation-extraction:latest \
  -c "cd /workspace/data && python /workspace/extract_activations_arc_v5e64.py \
    --machine_id 0 \
    --total_machines 1 \
    --host_id 3 \
    --num_hosts 4 \
    --multihost \
    --coordinator_address ${COORDINATOR_IP}:8476 \
    --model_path KathirKs/qwen-2.5-0.5b \
    --dataset_path gs://your-bucket/dataset.jsonl \
    --batch_size 4 \
    --max_seq_length 512 \
    --output_dir /workspace/data/output \
    --upload_to_gcs \
    --gcs_bucket fineweb-data-us-central1-a \
    --gcs_prefix activations_arc_v5e64 \
    --shard_size_gb 1.0 \
    --compress_shards \
    --verbose"
```

### Key Parameters for Multihost

| Parameter | Description | Value |
|-----------|-------------|-------|
| `--machine_id` | Which machine in multi-machine setup | 0 (single machine setup) |
| `--total_machines` | Total machines | 1 (single machine with multiple hosts) |
| `--host_id` | Which host (0-3) | 0, 1, 2, or 3 |
| `--num_hosts` | Total hosts in pod | 4 |
| `--multihost` | Enable multihost mode | (flag, no value) |
| `--coordinator_address` | IP:port of worker-0 | `10.128.0.5:8476` |

### Using the Launch Script

Alternatively, you can use the `launch_v5e64.sh` script which sets these automatically:

```bash
# On each host, set these environment variables BEFORE running:
export COORDINATOR_ADDRESS="10.128.0.5:8476"
export HOST_ID=0  # Change to 0, 1, 2, 3 for each host
export NUM_HOSTS=4

# Then run:
sudo docker run --rm --net=host --privileged \
  -v ~/torch_xla/qwen:/workspace/data \
  -v ~/.config/gcloud:/root/.config/gcloud:ro \
  -v ~/.cache/huggingface:/cache/huggingface \
  -e COORDINATOR_ADDRESS=${COORDINATOR_ADDRESS} \
  -e HOST_ID=${HOST_ID} \
  -e NUM_HOSTS=${NUM_HOSTS} \
  gcr.io/${PROJECT_ID}/activation-extraction:latest \
  -c "/workspace/launch_v5e64.sh"
```

---

## Automated Deployment with tmux/GNU Parallel

For easier deployment, you can use a script to launch on all hosts:

### Create deployment script on worker-0:

```bash
cat > deploy_multihost.sh << 'EOF'
#!/bin/bash

# Configuration
export PROJECT_ID="your-gcp-project-id"
export COORDINATOR_IP=$(hostname -I | awk '{print $1}')
export IMAGE="gcr.io/${PROJECT_ID}/activation-extraction:latest"

# Host IPs
HOSTS=(
  "worker-0-ip"  # worker-0 (this machine)
  "worker-1-ip"  # worker-1
  "worker-2-ip"  # worker-2
  "worker-3-ip"  # worker-3
)

# Function to run on each host
run_on_host() {
  local host_id=$1
  local host_ip=$2

  echo "Starting host ${host_id} on ${host_ip}..."

  if [ "${host_ip}" == "worker-0-ip" ]; then
    # Run locally on worker-0
    sudo docker run --rm --net=host --privileged \
      -v ~/torch_xla/qwen:/workspace/data \
      -v ~/.config/gcloud:/root/.config/gcloud:ro \
      -v ~/.cache/huggingface:/cache/huggingface \
      ${IMAGE} \
      -c "cd /workspace/data && python /workspace/extract_activations_arc_v5e64.py \
        --machine_id 0 --total_machines 1 --host_id ${host_id} --num_hosts 4 \
        --multihost --coordinator_address ${COORDINATOR_IP}:8476 \
        --model_path KathirKs/qwen-2.5-0.5b \
        --dataset_path gs://your-bucket/dataset.jsonl \
        --batch_size 4 --max_seq_length 512 \
        --output_dir /workspace/data/output \
        --upload_to_gcs --gcs_bucket fineweb-data-us-central1-a \
        --gcs_prefix activations_arc_v5e64 --shard_size_gb 1.0 \
        --compress_shards --verbose" &
  else
    # Run remotely via SSH
    ssh ${host_ip} "sudo docker run --rm --net=host --privileged \
      -v ~/torch_xla/qwen:/workspace/data \
      -v ~/.config/gcloud:/root/.config/gcloud:ro \
      -v ~/.cache/huggingface:/cache/huggingface \
      ${IMAGE} \
      -c \"cd /workspace/data && python /workspace/extract_activations_arc_v5e64.py \
        --machine_id 0 --total_machines 1 --host_id ${host_id} --num_hosts 4 \
        --multihost --coordinator_address ${COORDINATOR_IP}:8476 \
        --model_path KathirKs/qwen-2.5-0.5b \
        --dataset_path gs://your-bucket/dataset.jsonl \
        --batch_size 4 --max_seq_length 512 \
        --output_dir /workspace/data/output \
        --upload_to_gcs --gcs_bucket fineweb-data-us-central1-a \
        --gcs_prefix activations_arc_v5e64 --shard_size_gb 1.0 \
        --compress_shards --verbose\"" &
  fi
}

# Launch on all hosts in parallel
for i in ${!HOSTS[@]}; do
  run_on_host $i ${HOSTS[$i]}
done

# Wait for all to complete
wait

echo "All hosts completed!"
EOF

chmod +x deploy_multihost.sh
```

---

## Troubleshooting

### Issue: "Connection timeout" during multihost initialization

**Cause:** Hosts didn't start within ~30 seconds of each other

**Solution:**
- Use tmux or the deployment script above
- Ensure all hosts can ping each other
- Check firewall rules allow port 8476

### Issue: "No space left on device"

**Cause:** HuggingFace cache directory full

**Solution:**
```bash
# Clean cache on each host
rm -rf ~/.cache/huggingface/*

# Or mount a larger disk
sudo mkdir -p /mnt/disks/cache
sudo mount /dev/sdb /mnt/disks/cache  # Your attached disk
# Then use: -v /mnt/disks/cache:/cache/huggingface
```

### Issue: Different hosts have different Docker images

**Cause:** Built separately or pulled at different times

**Solution:**
```bash
# On all hosts, pull the same image with specific tag
sudo docker pull gcr.io/${PROJECT_ID}/activation-extraction:v1.0
```

### Issue: Host can't access GCS

**Cause:** GCS credentials not mounted or expired

**Solution:**
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login

# Verify credentials work
gsutil ls gs://fineweb-data-us-central1-a/
```

---

## Performance Tips

1. **Pre-download models:** Download models to cache before running to avoid download during execution
2. **Use persistent disks:** Mount /cache/huggingface to a persistent disk (not tmpfs)
3. **Monitor progress:** Use `--verbose` flag to see detailed progress on each host
4. **Shard size:** 1 GB shards are optimal for GCS (fewer objects, good parallelism)

---

## Summary

**Recommended Workflow:**

1. Build Docker image once on local machine or CI/CD
2. Push to GCR: `docker push gcr.io/${PROJECT_ID}/activation-extraction:latest`
3. Pull on all 4 hosts: `docker pull gcr.io/${PROJECT_ID}/activation-extraction:latest`
4. Start all hosts within 30 seconds using tmux or deployment script
5. Monitor progress on worker-0 (coordinator)
6. Verify GCS uploads: `gsutil ls gs://fineweb-data-us-central1-a/activations_arc_v5e64/`

**Time estimate:**
- Build + push: 5 minutes (once)
- Pull on 4 hosts: 2 minutes (concurrent)
- Total setup: ~7 minutes

**vs building on each host:**
- Build on 4 hosts: 8-12 minutes × 4 = 32-48 minutes
