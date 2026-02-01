#!/bin/bash
# Test multihost TPU extraction on v5litepod-64

set -e

echo "=========================================="
echo "  Multihost TPU Extraction Test"
echo "=========================================="

# Configuration
TPU_NAME="node-v5e-64-europe-west4-b"
ZONE="europe-west4-b"
DATASET_PATH="gs://fineweb-data-europe-west4/datasets/stream_000.jsonl"
GCS_BUCKET="fineweb-data-europe-west4"
TOPOLOGY="v5litepod-64"

# Test with small number of tasks first
MAX_TASKS=100

echo "Running test extraction on $TPU_NAME"
echo "Dataset: $DATASET_PATH"
echo "Max tasks: $MAX_TASKS"
echo ""

# Run on all workers
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE \
    --worker=all \
    --command="cd ~/activation-extract-multihost && \
        export DATASET_PATH=$DATASET_PATH && \
        export GCS_BUCKET=$GCS_BUCKET && \
        export TOPOLOGY=$TOPOLOGY && \
        python3 multihost_extract.py \
            --topology v5litepod-64 \
            --dataset_path $DATASET_PATH \
            --max_tasks $MAX_TASKS \
            --gcs_bucket $GCS_BUCKET \
            --batch_size 64 \
            --upload_to_gcs \
            --verbose"

echo ""
echo "=========================================="
echo "  Test Complete!"
echo "=========================================="
