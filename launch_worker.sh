#!/bin/bash
# Launch script for parallel independent TPU workers
#
# This script launches a single worker for activation extraction.
# It reads the TPU_WORKER_ID from environment and processes the corresponding stream.
#
# Usage:
#   # Set worker ID and launch
#   export TPU_WORKER_ID=5
#   ./launch_worker.sh
#
#   # Or pass as argument
#   ./launch_worker.sh 5
#
#   # With custom parameters
#   export TPU_WORKER_ID=5
#   export GCS_BUCKET=my-bucket
#   export MODEL_PATH=Qwen/Qwen2.5-0.5B
#   ./launch_worker.sh

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Worker ID (from environment or argument)
if [ -n "$1" ]; then
    export TPU_WORKER_ID=$1
fi

if [ -z "$TPU_WORKER_ID" ]; then
    echo "Error: TPU_WORKER_ID not set"
    echo "Usage: export TPU_WORKER_ID=5 && ./launch_worker.sh"
    echo "   or: ./launch_worker.sh 5"
    exit 1
fi

# Dataset configuration
DATASET_DIR="${DATASET_DIR:-./data/streams}"
DATASET_PATH="${DATASET_DIR}/stream_$(printf '%03d' $TPU_WORKER_ID).jsonl"

# Model configuration
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B}"

# Output configuration
OUTPUT_DIR="${OUTPUT_DIR:-./activations/worker_${TPU_WORKER_ID}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

# GCS configuration
GCS_BUCKET="${GCS_BUCKET:-}"
UPLOAD_TO_GCS="${UPLOAD_TO_GCS:-false}"

# Extraction configuration
BATCH_SIZE="${BATCH_SIZE:-4}"
SHARD_SIZE_GB="${SHARD_SIZE_GB:-1.0}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"

# ============================================================================
# Validation
# ============================================================================

echo "=========================================="
echo "LAUNCHING WORKER ${TPU_WORKER_ID}"
echo "=========================================="
echo "Dataset:     ${DATASET_PATH}"
echo "Model:       ${MODEL_PATH}"
echo "Output:      ${OUTPUT_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Batch size:  ${BATCH_SIZE}"

if [ ! -f "$DATASET_PATH" ]; then
    echo ""
    echo "Error: Dataset file not found: ${DATASET_PATH}"
    echo ""
    echo "Please create dataset streams first:"
    echo "  python create_dataset_streams.py --num_streams 32 --output_dir ${DATASET_DIR}"
    exit 1
fi

# ============================================================================
# Build command
# ============================================================================

CMD="python extract_activations.py \
    --dataset_path ${DATASET_PATH} \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --shard_size_gb ${SHARD_SIZE_GB} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --enable_checkpointing \
    --verbose"

# Add GCS upload if configured
if [ "$UPLOAD_TO_GCS" = "true" ] && [ -n "$GCS_BUCKET" ]; then
    echo "GCS Upload:  Enabled (gs://${GCS_BUCKET}/activations/tpu_${TPU_WORKER_ID}/)"
    CMD="${CMD} --upload_to_gcs --gcs_bucket ${GCS_BUCKET}"
else
    echo "GCS Upload:  Disabled"
fi

echo "=========================================="
echo ""

# ============================================================================
# Launch
# ============================================================================

echo "Starting extraction..."
echo ""

exec $CMD
