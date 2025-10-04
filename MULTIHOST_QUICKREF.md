# Multi-Host TPU Docker Commands - Quick Reference

## Setup Variables

```bash
export TPU_NAME=arc-inference-multihost
export ZONE=us-central2-b
export ACCELERATOR=v5litepod-256  # or v5litepod-128, v4-256, etc.
export GCS_BUCKET=gs://your-bucket-name
```

## 1. Create TPU Pod

```bash
gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --accelerator-type=${ACCELERATOR} \
  --version=tpu-ubuntu2204-base
```

## 2. Install Docker (All Hosts)

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command="
    sudo apt-get update && \
    sudo apt-get install -y docker.io && \
    sudo usermod -aG docker \${USER}
  "
```

## 3. Copy Code (All Hosts)

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command="
    git clone https://github.com/your-repo/torch_xla.git && \
    cd torch_xla/qwen && \
    chmod +x run-docker.sh
  "
```

## 4. Build Docker Image (All Hosts)

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command="cd ~/torch_xla/qwen && ./run-docker.sh build"
```

## 5. Transform Dataset (Worker 0)

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0 \
  --command="
    cd ~/torch_xla/qwen && \
    docker run --rm -v /tmp/arc_data:/app/arc_data arc-agi-inference:latest \
      python transform_hf_to_arc.py \
        --dataset barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
        --output_dir /app/arc_data \
        --max_samples 50000
  "
```

## 6. Upload Data to GCS (Worker 0)

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0 \
  --command="gsutil cp /tmp/arc_data/*.json ${GCS_BUCKET}/data/"
```

## 7. Download Data (All Hosts)

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command="
    mkdir -p /tmp/arc_data && \
    gsutil cp ${GCS_BUCKET}/data/*.json /tmp/arc_data/
  "
```

## 8. Download Model (All Hosts)

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command="
    mkdir -p /tmp/model && \
    gsutil -m cp -r ${GCS_BUCKET}/models/qwen2.5-0.5b /tmp/model/
  "
```

## 9. Run Distributed Inference (All Hosts)

### For v5e-256 (4 hosts × 64 cores = 256 cores)

```bash
export MESH_SHAPE="16,16"

gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command="
    cd ~/torch_xla/qwen && \
    docker run --privileged --network=host \
      -v /tmp/arc_data:/app/arc_data \
      -v /tmp/model:/app/model \
      -v /tmp/outputs:/app/outputs \
      -v /tmp/activations:/app/activations \
      -e JAX_PLATFORMS=tpu \
      -e MESH_SHAPE=${MESH_SHAPE} \
      arc-agi-inference:latest \
      python distributed_inference_with_activations.py \
        --model_path /app/model/qwen2.5-0.5b \
        --tasks_file /app/arc_data/arc_format_train.json \
        --output_dir /app/outputs \
        --activations_dir /app/activations \
        --batch_size 8 \
        --mesh_shape ${MESH_SHAPE} \
        --gcs_bucket ${GCS_BUCKET}/activations \
        --upload_to_cloud True
  "
```

### For v5e-128 (2 hosts × 64 cores = 128 cores)

```bash
export MESH_SHAPE="8,16"  # Adjust based on setup
```

### For v4-512 (8 hosts × 64 cores = 512 cores)

```bash
export MESH_SHAPE="16,32"  # Adjust based on setup
```

## 10. Monitor Progress

### Check Running Containers

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command="docker ps"
```

### View Logs (Worker 0)

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0 \
  --command="docker logs -f \$(docker ps -q | head -1)"
```

### Check TPU Devices

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0 \
  --command="python -c 'import jax; print(f\"Devices: {len(jax.devices())}\")'"
```

### Check Output Files

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command="ls -lh /tmp/outputs/ /tmp/activations/ | head -10"
```

## 11. Check GCS Results

```bash
# List activation files
gsutil ls ${GCS_BUCKET}/activations/

# Download to local machine
gsutil -m cp -r ${GCS_BUCKET}/activations ./local_activations/
```

## 12. Cleanup

### Stop Containers

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command="docker stop \$(docker ps -q)"
```

### Clear Data

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command="rm -rf /tmp/arc_data /tmp/outputs /tmp/activations /tmp/model"
```

### Delete TPU

```bash
gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE}
```

## Mesh Shapes by TPU Size

| TPU Type | Cores | Hosts | Mesh Shape | Batch Size |
|----------|-------|-------|------------|------------|
| v5e-64   | 64    | 1     | 8,8        | 8          |
| v5e-128  | 128   | 2     | 8,16       | 8          |
| v5e-256  | 256   | 4     | 16,16      | 8          |
| v4-128   | 128   | 2     | 8,16       | 8          |
| v4-256   | 256   | 4     | 16,16      | 8          |
| v4-512   | 512   | 8     | 16,32      | 8          |

## Common Issues

### Docker Permission Denied

```bash
# Re-login after adding to docker group
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --command="exit"
```

### TPU Not Detected

```bash
# Use --privileged flag
docker run --privileged ...
```

### Multi-Host Coordination Issues

```bash
# Use --network=host
docker run --network=host ...
```

### Out of Memory

```bash
# Reduce batch size
--batch_size 4

# Extract fewer layers
--layers_to_extract 15 16 17 18 19 20 21 22 23
```

## One-Liner for Complete Run

```bash
# Set variables and run everything
export TPU_NAME=arc-multihost ZONE=us-central2-b ACCELERATOR=v5litepod-256 GCS_BUCKET=gs://your-bucket MESH_SHAPE="16,16" && \
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --command="cd ~/torch_xla/qwen && docker run --privileged --network=host -v /tmp/arc_data:/app/arc_data -v /tmp/model:/app/model -v /tmp/outputs:/app/outputs -v /tmp/activations:/app/activations -e JAX_PLATFORMS=tpu -e MESH_SHAPE=${MESH_SHAPE} arc-agi-inference:latest python distributed_inference_with_activations.py --model_path /app/model/qwen2.5-0.5b --tasks_file /app/arc_data/arc_format_train.json --output_dir /app/outputs --activations_dir /app/activations --batch_size 8 --mesh_shape ${MESH_SHAPE} --gcs_bucket ${GCS_BUCKET}/activations --upload_to_cloud True"
```

## SSH to Specific Worker

```bash
# Worker 0
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0

# Worker 1
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=1

# All workers
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all
```

## Copy Files Between Workers

```bash
# From local to worker 0
gcloud compute tpus tpu-vm scp --zone=${ZONE} --worker=0 \
  local_file.txt ${TPU_NAME}:/tmp/

# From worker 0 to local
gcloud compute tpus tpu-vm scp --zone=${ZONE} --worker=0 \
  ${TPU_NAME}:/tmp/file.txt ./
```

---

**For detailed explanations, see:** `DOCKER_MULTIHOST.md`
