#!/bin/bash
# Multihost TPU startup script - to be run on each worker independently
# This script auto-detects the worker environment and coordinates via JAX distributed

set -e

echo "========================================"
echo "  Multihost TPU Worker Init"
echo "========================================"

# Auto-detect environment
export HOST_ID="${CLOUD_TPU_TASK_ID:-0}"
export NUM_HOSTS="${TPU_WORKER_COUNT:-1}"

echo "Host ID: $HOST_ID"
echo "Num Hosts: $NUM_HOSTS"

# Get coordinator from worker hostnames
if [ -n "$TPU_WORKER_HOSTNAMES" ]; then
    FIRST_HOST=$(echo "$TPU_WORKER_HOSTNAMES" | cut -d',' -f1)
    export COORDINATOR_ADDRESS="${FIRST_HOST}:8476"
    echo "Coordinator: $COORDINATOR_ADDRESS"
fi

# Configuration from environment or defaults
DATASET_PATH="${DATASET_PATH:-gs://fineweb-data-us-central1/datasets/stream_000.jsonl}"
GCS_BUCKET="${GCS_BUCKET:-fineweb-data-us-central1}"
TOPOLOGY="${TOPOLOGY:-v5litepod-64}"
MAX_TASKS="${MAX_TASKS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "Dataset: $DATASET_PATH"
echo "Topology: $TOPOLOGY"
echo "Max Tasks: $MAX_TASKS"
echo "Batch Size: $BATCH_SIZE"
echo "========================================"

# Change to project directory
cd ~/activation-extract-multihost

# Run extraction
python3 multihost_extract.py \
    --topology "$TOPOLOGY" \
    --dataset_path "$DATASET_PATH" \
    --max_tasks "$MAX_TASKS" \
    --gcs_bucket "$GCS_BUCKET" \
    --batch_size "$BATCH_SIZE" \
    --upload_to_gcs \
    --verbose

echo "========================================"
echo "  Worker $HOST_ID Complete"
echo "========================================"
