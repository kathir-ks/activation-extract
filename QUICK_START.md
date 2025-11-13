# Quick Start - Build and Deploy on New Machine

## On Your New Machine

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd torch_xla/qwen
```

### 2. Setup GCP
```bash
# Login and set project
gcloud auth login
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID
gcloud auth configure-docker
```

### 3. Build and Push Docker Image
```bash
# Build
docker build -t activation-extraction:latest .

# Tag for GCR
docker tag activation-extraction:latest gcr.io/${PROJECT_ID}/activation-extraction:latest

# Push to registry
docker push gcr.io/${PROJECT_ID}/activation-extraction:latest
```

Build time: ~10 minutes
Push time: ~5-8 minutes

### 4. Create TPU and Deploy

**Single command deployment (v5e-64, 4 hosts):**

```bash
export ZONE="us-central1-a"
export TPU_NAME="arc-extraction"
export BUCKET="your-bucket-name"
export DATASET="gs://${BUCKET}/dataset.jsonl"
export MODEL="KathirKs/qwen-2.5-0.5b"

# Create TPU
gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --accelerator-type=v5litepod-64 \
  --version=tpu-ubuntu2204-base

# Get coordinator IP
COORDINATOR_IP=$(gcloud compute tpus tpu-vm describe ${TPU_NAME} \
  --zone=${ZONE} --format='value(networkEndpoints[0].ipAddress)')

# Deploy to all 4 workers
for WORKER in 0 1 2 3; do
  gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=${WORKER} --command="
    # Install Docker
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker \$USER

    # Pull image
    gcloud auth configure-docker
    sudo docker pull gcr.io/${PROJECT_ID}/activation-extraction:latest

    # Run extraction
    mkdir -p ~/data ~/.cache/huggingface
    sudo docker run -d --name extraction --net=host --privileged \
      -v ~/data:/workspace/data \
      -v ~/.config/gcloud:/root/.config/gcloud:ro \
      -v ~/.cache/huggingface:/cache/huggingface \
      gcr.io/${PROJECT_ID}/activation-extraction:latest \
      -c 'python /workspace/extract_activations_arc_v5e64.py \
        --host_id ${WORKER} --num_hosts 4 --multihost \
        --coordinator_address ${COORDINATOR_IP}:8476 \
        --dataset_path ${DATASET} \
        --model_path ${MODEL} \
        --batch_size 4 --max_seq_length 512 \
        --gcs_bucket ${BUCKET} --upload_to_gcs \
        --shard_size_gb 1.0 --compress_shards --verbose'
  " &
done

wait
echo "All workers started!"
```

### 5. Monitor
```bash
# Check worker 0 logs
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0 \
  --command="sudo docker logs -f extraction"

# Check GCS output
gsutil ls -lh gs://${BUCKET}/activations/
```

### 6. Cleanup
```bash
# Stop all workers
for i in 0 1 2 3; do
  gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=$i \
    --command="sudo docker stop extraction"
done

# Delete TPU
gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}
```

---

## Performance

With JIT optimizations:
- **First batch:** 21s (JIT compilation warmup)
- **Subsequent batches:** 5.7s per batch
- **8,000 samples:** ~47 minutes on 4 hosts

---

## Key Features Enabled

✅ JIT compilation (5-10x faster forward pass)
✅ Fixed batch sizes (no recompilation)
✅ Async device→host transfers (2-3x faster)
✅ Model sharding across TPU devices
✅ Automatic GCS upload with compression

---

## Troubleshooting

**TPU not detected:**
```bash
sudo docker run --rm --net=host gcr.io/${PROJECT_ID}/activation-extraction:latest \
  -c "python -c 'import jax; print(jax.devices())'"
```

**Out of memory:**
- Reduce `--batch_size` to 2
- Reduce `--max_seq_length` to 256

**Model download slow:**
- Pre-download model to GCS and mount volume
- Or increase timeout: `export HF_HUB_DOWNLOAD_TIMEOUT=600`
