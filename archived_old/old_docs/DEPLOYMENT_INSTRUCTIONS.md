# Deployment Instructions for New Google Cloud Account

## Prerequisites

On your new machine/account, you need:
- Google Cloud SDK installed (`gcloud`)
- Docker installed
- Access to a GCP project with:
  - TPU v5e-64 quota (or v4 for testing)
  - Artifact Registry enabled
  - Cloud Storage bucket

---

## Quick Reference: Build and Push to Artifact Registry

> **IMPORTANT:** Do NOT use `gcr.io` (Google Container Registry) as it is deprecated.
> Always use Artifact Registry: `${AR_REGION}-docker.pkg.dev/...`

```bash
# 1. Set your project and region variables
export PROJECT_ID="absolute-axis-470415-g6"
export AR_REGION="us-central1"
export AR_REPO="arc-agi-us-central1"

# 2. Create Artifact Registry repository (if not exists)
gcloud artifacts repositories create ${AR_REPO} \
  --repository-format=docker \
  --location=${AR_REGION} \
  --project=${PROJECT_ID} \
  --description="Docker repository for ARC-AGI activation extraction"

# 3. Configure Docker authentication for Artifact Registry
gcloud auth configure-docker ${AR_REGION}-docker.pkg.dev

# 4. Build the image
docker build -t ${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/activation-extraction .

# 5. Push to Artifact Registry
docker push ${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/activation-extraction
```

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

# Set your region (choose based on your TPU location)
export AR_REGION="us-central1"

# Create Artifact Registry repository if it doesn't exist
gcloud artifacts repositories create arc-agi-${AR_REGION} \
  --repository-format=docker \
  --location=${AR_REGION} \
  --description="Docker repository for ARC-AGI activation extraction"

# Configure Docker to use Artifact Registry
gcloud auth configure-docker ${AR_REGION}-docker.pkg.dev
```

---

## Step 3: Build and Push Docker Image

```bash
cd /path/to/qwen

# Set repository name and image path
export AR_REPO="arc-agi-${AR_REGION}"
export IMAGE_TAG="activation-extraction"
export IMAGE_PATH="${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/activation-extraction:${IMAGE_TAG}"

# Build the image
docker build -t activation-extraction:latest .

# Tag for Artifact Registry
docker tag activation-extraction:latest ${IMAGE_PATH}

# Push to Artifact Registry
docker push ${IMAGE_PATH}
```

**Expected time:** Build ~10 min, Push ~5-8 min

**Note:** If the repository doesn't exist, the push will fail. Make sure you created it in Step 2.

---

## Step 4: Create TPU VMs and Deploy

### For TPU v5e-64 (16 hosts):

```bash
export ZONE="us-central1-a"
export TPU_NAME="arc-extraction"
export BUCKET_NAME="your-bucket-name"
export NUM_HOSTS=16  # v5e-64 has 16 hosts

# Create TPU
gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --accelerator-type=v5litepod-64 \
  --version=tpu-ubuntu2204-base

# Get coordinator IP
COORDINATOR_IP=$(gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --worker=0 \
  --command="hostname -I | awk '{print \$1}'" 2>/dev/null | tr -d '\r\n ')

export COORDINATOR_ADDRESS="${COORDINATOR_IP}:8476"

# Deploy to each worker
for WORKER_ID in {0..15}; do
  gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
    --zone=${ZONE} \
    --worker=${WORKER_ID} \
    --command="
      # Setup Docker
      sudo apt-get update && sudo apt-get install -y docker.io
      sudo usermod -aG docker \$USER
      gcloud auth configure-docker ${AR_REGION}-docker.pkg.dev

      # Pull image from Artifact Registry
      sudo docker pull ${IMAGE_PATH}

      # Run extraction
      mkdir -p ~/data ~/.cache/huggingface
      sudo docker run -d --name extraction \
        --net=host --privileged \
        -v ~/data:/workspace/data \
        -v ~/.config/gcloud:/root/.config/gcloud:ro \
        -v ~/.cache/huggingface:/cache/huggingface \
        ${IMAGE_PATH} \
        -c \"python /workspace/extract_activations_arc_v5e64.py \
          --host_id ${WORKER_ID} \
          --num_hosts ${NUM_HOSTS} \
          --multihost \
          --coordinator_address ${COORDINATOR_ADDRESS} \
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

**Alternative: Use the Automated Deployment Script**

For easier deployment, use the provided deployment script:

```bash
# Edit deploy/deploy_8hosts_parallel.sh to set your configuration
# Then run:
cd /path/to/qwen
./deploy/deploy_8hosts_parallel.sh deploy
```

---

## Step 5: Monitor Progress

```bash
# Check logs (worker 0)
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0 \
  --command="sudo docker logs -f extraction"

# Check GCS output
gsutil ls -lh gs://${BUCKET_NAME}/activations/

# Or use the deployment script
./deploy/deploy_8hosts_parallel.sh monitor
./deploy/deploy_8hosts_parallel.sh status
./deploy/deploy_8hosts_parallel.sh logs 100  # View last 100 lines from all hosts
```

---

## Performance Expectations

- **Warmup:** ~21 seconds (first batch, JIT compilation)
- **Throughput:** ~5.7 seconds/batch (after warmup)
- **8-host run:** ~23-25 minutes for 8,000 samples (approximately 2x faster than 4-host)

---

## Cleanup

```bash
# Stop containers
for i in {0..15}; do
  gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=$i \
    --command="sudo docker stop extraction && sudo docker rm extraction"
done

# Or use the deployment script
./deploy/deploy_8hosts_parallel.sh stop

# Delete TPU
gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}
```

---

## Troubleshooting

### If Artifact Registry repository doesn't exist:
```bash
# List existing repositories
gcloud artifacts repositories list --location=${AR_REGION}

# Create if needed
gcloud artifacts repositories create arc-agi-${AR_REGION} \
  --repository-format=docker \
  --location=${AR_REGION}
```

### If Docker authentication fails:
```bash
# Re-authenticate
gcloud auth configure-docker ${AR_REGION}-docker.pkg.dev --quiet

# Verify you can access the repository
gcloud artifacts docker images list ${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/arc-agi-${AR_REGION}
```

### If containers fail to start:
```bash
# Check container logs
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0 \
  --command="sudo docker logs extraction"

# Check if container exists
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0 \
  --command="sudo docker ps -a"
```
