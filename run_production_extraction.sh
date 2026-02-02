#!/bin/bash
# Production multihost TPU extraction for europe-west4
# Uses first 16 data streams on v5litepod-64

set -e

echo "=========================================="
echo "  Production Multihost TPU Extraction"
echo "=========================================="

# Configuration - ALL IN EUROPE-WEST4 (same region = free data transfer)
TPU_NAME="node-v5e-64-europe-west4-b"
ZONE="europe-west4-b"
GCS_BUCKET="fineweb-data-europe-west4"  # europe-west4 bucket
TOPOLOGY="v5litepod-64"
BATCH_SIZE=64
MAX_TASKS=1000  # Per stream

# First 16 streams (streams 0-15)
STREAMS=(
    "gs://fineweb-data-europe-west4/datasets/stream_000.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_001.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_002.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_003.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_004.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_005.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_006.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_007.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_008.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_009.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_010.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_011.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_012.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_013.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_014.jsonl"
    "gs://fineweb-data-europe-west4/datasets/stream_015.jsonl"
)

echo "TPU: $TPU_NAME (zone: $ZONE)"
echo "GCS Bucket: $GCS_BUCKET (europe-west4)"
echo "Streams: ${#STREAMS[@]}"
echo "Max tasks per stream: $MAX_TASKS"
echo ""

# Process each stream
for i in "${!STREAMS[@]}"; do
    STREAM="${STREAMS[$i]}"
    STREAM_NUM=$(printf "%03d" $i)
    
    echo "=========================================="
    echo "Processing stream $STREAM_NUM: $STREAM"
    echo "=========================================="
    
    # Run on all workers
    gcloud compute tpus tpu-vm ssh $TPU_NAME \
        --zone=$ZONE \
        --worker=all \
        --command="cd ~/activation-extract-multihost && \
            python3 multihost_extract.py \
                --topology $TOPOLOGY \
                --dataset_path $STREAM \
                --max_tasks $MAX_TASKS \
                --gcs_bucket $GCS_BUCKET \
                --gcs_prefix activations/stream_${STREAM_NUM} \
                --batch_size $BATCH_SIZE \
                --upload_to_gcs \
                --verbose"
    
    echo "✓ Stream $STREAM_NUM complete"
    echo ""
done

echo "=========================================="
echo "  All 16 Streams Complete!"
echo "=========================================="
echo "Output: gs://$GCS_BUCKET/activations/"
