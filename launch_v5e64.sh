#!/bin/bash
#
# Launch activation extraction on TPU v5e-64
#
# TPU v5e-64 has 4 hosts, 8 chips per host = 32 total chips
# This script should be run on EACH host with appropriate HOST_ID
#

set -e

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

# TPU v5e-64 Configuration
export NUM_HOSTS=4                          # v5e-64 has 4 hosts
export HOST_ID=${HOST_ID:-0}                # Set via environment or default to 0
export COORDINATOR_ADDRESS=${COORDINATOR_ADDRESS:-"$(hostname -i):8476"}  # Auto-detect or set manually

# Model Configuration
export MODEL_PATH="KathirKs/qwen-2.5-7b"    # Model to use

# Dataset Configuration
export DATASET_PATH="arc_formatted_challenges.jsonl"  # Path to converted dataset
export MAX_TASKS=10000                      # Max tasks per machine (None for all)

# Extraction Configuration
export BATCH_SIZE=16                        # Batch size (16 works well for 7B model)
export MAX_SEQ_LENGTH=2048                  # Max sequence length
export MESH_TYPE="2d"                       # Mesh type: 1d, 2d, or 3d

# GCS Configuration (REQUIRED for v5e-64)
export GCS_BUCKET="fineweb-data-us-central1-a"  # Your GCS bucket
export GCS_PREFIX="activations_arc_v5e64"       # Prefix in bucket
export SHARD_SIZE_GB=1.0                        # Shard size in GB (1000 MB)
export DELETE_LOCAL=true                        # Delete local files after upload

# Multi-Machine Configuration (if running on multiple v5e-64 machines)
export MACHINE_ID=${MACHINE_ID:-0}          # This machine's ID
export TOTAL_MACHINES=${TOTAL_MACHINES:-1}  # Total number of v5e-64 machines

# Other Configuration
export RANDOM_SEED=42
export VERBOSE=true

# ============================================================================
# SCRIPT START - DO NOT EDIT BELOW
# ============================================================================

echo "========================================================================"
echo "TPU v5e-64 Activation Extraction"
echo "========================================================================"
echo "Configuration:"
echo "  Machine ID: $MACHINE_ID / $((TOTAL_MACHINES-1))"
echo "  Host ID: $HOST_ID / $((NUM_HOSTS-1))"
echo "  Coordinator: $COORDINATOR_ADDRESS"
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET_PATH"
echo "  Batch size: $BATCH_SIZE"
echo "  Mesh type: $MESH_TYPE"
echo "  GCS bucket: gs://$GCS_BUCKET/$GCS_PREFIX/"
echo "========================================================================"

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset file not found: $DATASET_PATH"
    echo "Please run the dataset conversion first:"
    echo "  python convert_hf_to_arc_format.py --output_file $DATASET_PATH"
    exit 1
fi

# Check if model path is valid
if [[ ! "$MODEL_PATH" =~ ^[A-Za-z0-9_/-]+$ ]]; then
    echo "❌ Error: Invalid model path: $MODEL_PATH"
    exit 1
fi

# Create output directory
OUTPUT_DIR="./activations_arc_v5e64_machine${MACHINE_ID}_host${HOST_ID}"
mkdir -p "$OUTPUT_DIR"

# Create log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/extraction_machine${MACHINE_ID}_host${HOST_ID}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Starting extraction..."
echo "  Output directory: $OUTPUT_DIR"
echo "  Log file: $LOG_FILE"
echo ""

# Build command
CMD="python extract_activations_arc_v5e64.py \
    --machine_id $MACHINE_ID \
    --total_machines $TOTAL_MACHINES \
    --multihost \
    --coordinator_address $COORDINATOR_ADDRESS \
    --host_id $HOST_ID \
    --num_hosts $NUM_HOSTS \
    --mesh_type $MESH_TYPE \
    --model_path $MODEL_PATH \
    --dataset_path $DATASET_PATH \
    --batch_size $BATCH_SIZE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir $OUTPUT_DIR \
    --upload_to_gcs \
    --gcs_bucket $GCS_BUCKET \
    --gcs_prefix $GCS_PREFIX \
    --shard_size_gb $SHARD_SIZE_GB \
    --compress_shards \
    --random_seed $RANDOM_SEED \
    --verbose"

# Add max_tasks if set
if [ ! -z "$MAX_TASKS" ] && [ "$MAX_TASKS" != "None" ]; then
    CMD="$CMD --max_tasks $MAX_TASKS"
fi

# Add delete local if enabled
if [ "$DELETE_LOCAL" = true ]; then
    CMD="$CMD --delete_local_after_upload"
fi

# Run extraction
echo "Command:"
echo "$CMD"
echo ""

# Run with logging
$CMD 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ Extraction complete!"
    echo "========================================================================"
    echo "  Host $HOST_ID on Machine $MACHINE_ID finished successfully"
    echo "  GCS path: gs://$GCS_BUCKET/$GCS_PREFIX/"
    echo "  Log file: $LOG_FILE"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "❌ Extraction failed with exit code: $EXIT_CODE"
    echo "========================================================================"
    echo "  Check log file: $LOG_FILE"
    echo "========================================================================"
    exit $EXIT_CODE
fi
