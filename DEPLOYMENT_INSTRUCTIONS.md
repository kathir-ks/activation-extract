# Deployment Instructions for New Google Cloud Account

## Prerequisites

On your new machine/account, you need:
- Google Cloud SDK installed (`gcloud`)
- Docker installed
- Access to a GCP project with:
  - TPU v5e-64 quota (or v4 for testing)
  - Google Container Registry enabled
  - Cloud Storage bucket

---

## Step 1: Transfer Files to New Machine

### Files to Copy

**Required Files:**
```
Dockerfile
kvcache_utils.py
qwen2_jax.py
qwen2_jax_with_hooks.py
extract_activations_arc_v5e64.py
extract_activations_fineweb_multihost.py
convert_hf_to_arc_format.py
launch_v5e64.sh
arc24/                          # Entire directory
```

**Transfer Method:**
```bash
# Option A: Use rsync
rsync -avz /home/kathirks_gc/torch_xla/qwen/ user@new-machine:/path/to/qwen/

# Option B: Create tarball and copy
cd /home/kathirks_gc/torch_xla
tar czf qwen_deployment.tar.gz qwen/
scp qwen_deployment.tar.gz user@new-machine:/path/to/

# On new machine:
tar xzf qwen_deployment.tar.gz
```

---

## Step 2: Setup Google Cloud on New Machine

```bash
# Login to Google Cloud
gcloud auth login

# Set your project
export PROJECT_ID="your-new-project-id"
gcloud config set project $PROJECT_ID

# Configure Docker to use GCR
gcloud auth configure-docker
```

---

## Step 3: Build and Push Docker Image

```bash
cd /path/to/qwen

# Build the image
docker build -t activation-extraction:latest .

# Tag for GCR
docker tag activation-extraction:latest \
  gcr.io/${PROJECT_ID}/activation-extraction:latest

# Push to Google Container Registry
docker push gcr.io/${PROJECT_ID}/activation-extraction:latest
```

**Expected time:** Build ~10 min, Push ~5-8 min

---

## Step 4: Create TPU VMs and Deploy

### For TPU v5e-64 (4 hosts):

```bash
export ZONE="us-central1-a"
export TPU_NAME="arc-extraction"
export BUCKET_NAME="your-bucket-name"

# Create TPU
gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --accelerator-type=v5litepod-64 \
  --version=tpu-ubuntu2204-base

# Get coordinator IP
COORDINATOR_IP=$(gcloud compute tpus tpu-vm describe ${TPU_NAME} \
  --zone=${ZONE} \
  --format='value(networkEndpoints[0].ipAddress)')

# Deploy to each worker
for WORKER_ID in 0 1 2 3; do
  gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=${WORKER_ID} \
    --command="
      # Setup Docker
      sudo apt-get update && sudo apt-get install -y docker.io
      sudo usermod -aG docker \$USER
      gcloud auth configure-docker

      # Pull image
      sudo docker pull gcr.io/${PROJECT_ID}/activation-extraction:latest

      # Run extraction
      mkdir -p ~/data ~/.cache/huggingface
      sudo docker run -d --name extraction \
        --net=host --privileged \
        -v ~/data:/workspace/data \
        -v ~/.config/gcloud:/root/.config/gcloud:ro \
        -v ~/.cache/huggingface:/cache/huggingface \
        gcr.io/${PROJECT_ID}/activation-extraction:latest \
        -c \"python /workspace/extract_activations_arc_v5e64.py \
          --host_id ${WORKER_ID} \
          --num_hosts 4 \
          --multihost \
          --coordinator_address ${COORDINATOR_IP}:8476 \
          --dataset_path gs://${BUCKET_NAME}/dataset.jsonl \
          --model_path KathirKs/qwen-2.5-0.5b \
          --batch_size 4 \
          --gcs_bucket ${BUCKET_NAME} \
          --upload_to_gcs \
          --compress_shards \
          --verbose\"
    " &
done
```

---

## Step 5: Monitor Progress

```bash
# Check logs (worker 0)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0 \
  --command="sudo docker logs -f extraction"

# Check GCS output
gsutil ls -lh gs://${BUCKET_NAME}/activations/
```

---

## Performance Expectations

- **Warmup:** ~21 seconds (first batch, JIT compilation)
- **Throughput:** ~5.7 seconds/batch (after warmup)
- **4-host run:** ~47 minutes for 8,000 samples

---

## Cleanup

```bash
# Stop containers
for i in 0 1 2 3; do
  gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=$i \
    --command="sudo docker stop extraction"
done

# Delete TPU
gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}
```
