#!/bin/bash
#
# End-to-end activation extraction script for single TPU worker
# 1. Sets up environment (clones repo, installs dependencies)
# 2. Auto-detects worker ID from environment
# 3. Runs extraction with checkpointing and GCS upload
#
# Usage (on TPU VM):
#   bash <(curl -s https://raw.githubusercontent.com/kathir-ks/activation-extract/main/scripts/run_extraction_worker.sh) \
#     --gcs_bucket BUCKET --dataset_stream STREAM_PATH [options]
#
# Example:
#   ./run_extraction_worker.sh \
#     --gcs_bucket fineweb-data-us-central1 \
#     --dataset_stream gs://fineweb-data-us-central1/datasets/stream_000.jsonl \
#     --model Qwen/Qwen2.5-0.5B
#

set -e

# Default configuration
MODEL="Qwen/Qwen2.5-0.5B"
GCS_BUCKET=""
DATASET_STREAM=""
GCS_PREFIX="activations"
BATCH_SIZE=4
SHARD_SIZE_GB=1.0
MAX_SEQ_LENGTH=2048
LAYERS_TO_EXTRACT="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
ACTIVATION_TYPE="residual"  # 'residual', 'mlp', or 'attn'
OUTPUT_DIR="./activations"
CHECKPOINT_DIR="./checkpoints"
MAX_TASKS=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --gcs_bucket) GCS_BUCKET="$2"; shift ;;
        --dataset_stream) DATASET_STREAM="$2"; shift ;;
        --gcs_prefix) GCS_PREFIX="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --shard_size_gb) SHARD_SIZE_GB="$2"; shift ;;
        --max_seq_length) MAX_SEQ_LENGTH="$2"; shift ;;
        --layers) LAYERS_TO_EXTRACT="$2"; shift ;;
        --activation_type) ACTIVATION_TYPE="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --checkpoint_dir) CHECKPOINT_DIR="$2"; shift ;;
        --max_tasks) MAX_TASKS="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$GCS_BUCKET" ] || [ -z "$DATASET_STREAM" ]; then
    echo "Error: --gcs_bucket and --dataset_stream are required"
    echo ""
    echo "Usage: $0 --gcs_bucket BUCKET --dataset_stream STREAM_PATH [options]"
    echo ""
    echo "Required:"
    echo "  --gcs_bucket BUCKET          GCS bucket name for output"
    echo "  --dataset_stream STREAM      Path to dataset stream (local or gs://)"
    echo ""
    echo "Optional:"
    echo "  --model MODEL                Model path (default: Qwen/Qwen2.5-0.5B)"
    echo "  --gcs_prefix PREFIX          GCS prefix (default: activations)"
    echo "  --batch_size N               Batch size (default: 4)"
    echo "  --shard_size_gb SIZE         Shard size in GB (default: 1.0)"
    echo "  --max_seq_length LEN         Max sequence length (default: 2048)"
    echo "  --layers \"L1 L2 ...\"        Space-separated layer indices (default: all 24)"
    echo "  --activation_type TYPE       Activation type: 'residual', 'mlp', 'attn' (default: residual)"
    echo "  --output_dir DIR             Local output directory (default: ./activations)"
    echo "  --checkpoint_dir DIR         Checkpoint directory (default: ./checkpoints)"
    echo "  --max_tasks N                Max tasks to process (optional)"
    exit 1
fi

# Auto-detect worker ID from TPU_WORKER_ID environment variable
WORKER_ID="${TPU_WORKER_ID:-0}"

# Step 0: Setup environment (if not already done)
REPO_DIR="$HOME/activation-extract"
if [ ! -d "$REPO_DIR" ]; then
    echo "=========================================="
    echo "FIRST-TIME SETUP"
    echo "=========================================="
    echo "Setting up TPU worker environment..."
    echo ""

    # Download and run setup script
    curl -s https://raw.githubusercontent.com/kathir-ks/activation-extract/main/scripts/setup_tpu_worker.sh | bash

    if [ $? -ne 0 ]; then
        echo "✗ Setup failed!"
        exit 1
    fi

    # Change to repo directory
    cd "$REPO_DIR"
    echo ""
else
    echo "✓ Environment already set up at $REPO_DIR"
    cd "$REPO_DIR"

    # Pull latest changes
    echo "Updating repository..."
    git pull --quiet || true
    echo ""
fi

echo "=========================================="
echo "ACTIVATION EXTRACTION WORKER"
echo "=========================================="
echo "Worker ID: $WORKER_ID (detected from TPU_WORKER_ID=${TPU_WORKER_ID:-not set})"
echo "Model: $MODEL"
echo "Dataset stream: $DATASET_STREAM"
echo "GCS output: gs://$GCS_BUCKET/$GCS_PREFIX/tpu_$WORKER_ID/"
echo "Batch size: $BATCH_SIZE"
echo "Shard size: ${SHARD_SIZE_GB}GB"
echo "Max sequence length: $MAX_SEQ_LENGTH"
echo "Layers to extract: $LAYERS_TO_EXTRACT"
echo "Activation type: $ACTIVATION_TYPE"
echo "Output dir: $OUTPUT_DIR"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Download dataset stream if it's a GCS path
LOCAL_DATASET="$DATASET_STREAM"
if [[ "$DATASET_STREAM" == gs://* ]]; then
    echo "Downloading dataset stream from GCS..."
    LOCAL_DATASET="./dataset_stream_$WORKER_ID.jsonl"
    gsutil cp "$DATASET_STREAM" "$LOCAL_DATASET"
    echo "✓ Downloaded to $LOCAL_DATASET"
    echo ""
fi

# Check if checkpoint exists (resume support)
CHECKPOINT_FILE="$CHECKPOINT_DIR/checkpoint_worker_$WORKER_ID.json"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "✓ Found checkpoint file: $CHECKPOINT_FILE"
    echo "  Extraction will resume from last position"
    echo ""
else
    echo "No checkpoint found - starting fresh extraction"
    echo ""
fi

# Build python command
PYTHON_CMD="python extract_activations.py \
  --worker_id $WORKER_ID \
  --dataset_path \"$LOCAL_DATASET\" \
  --model_path \"$MODEL\" \
  --output_dir \"$OUTPUT_DIR\" \
  --batch_size $BATCH_SIZE \
  --shard_size_gb $SHARD_SIZE_GB \
  --max_seq_length $MAX_SEQ_LENGTH \
  --layers_to_extract $LAYERS_TO_EXTRACT \
  --activation_type $ACTIVATION_TYPE \
  --enable_checkpointing \
  --checkpoint_dir \"$CHECKPOINT_DIR\" \
  --upload_to_gcs \
  --gcs_bucket \"$GCS_BUCKET\" \
  --gcs_prefix \"$GCS_PREFIX\" \
  --compress_shards \
  --delete_local_after_upload \
  --verbose"

# Add max_tasks if specified
if [ -n "$MAX_TASKS" ]; then
    PYTHON_CMD="$PYTHON_CMD --max_tasks $MAX_TASKS"
fi

# Log command
echo "Running command:"
echo "$PYTHON_CMD"
echo ""
echo "=========================================="
echo "STARTING EXTRACTION"
echo "=========================================="
echo ""

# Run extraction
eval $PYTHON_CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ EXTRACTION COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo ""
    echo "Output location: gs://$GCS_BUCKET/$GCS_PREFIX/tpu_$WORKER_ID/"
    echo ""
    echo "To list uploaded files:"
    echo "  gsutil ls gs://$GCS_BUCKET/$GCS_PREFIX/tpu_$WORKER_ID/"
    echo ""
    echo "To check all workers:"
    echo "  gsutil ls gs://$GCS_BUCKET/$GCS_PREFIX/"
else
    echo "✗ EXTRACTION FAILED (exit code: $EXIT_CODE)"
    echo "=========================================="
    echo ""
    echo "Check logs above for error details"
    echo "Checkpoint saved at: $CHECKPOINT_FILE"
    echo "Re-run this script to resume from checkpoint"
fi

exit $EXIT_CODE
