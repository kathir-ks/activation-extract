#!/bin/bash
# =============================================================================
# Multihost TPU Extraction Launch Script
# =============================================================================
#
# This script launches activation extraction on a TPU pod slice.
# It auto-detects the multihost environment from TPU runtime variables.
#
# Usage:
#   # On TPU pod (via gcloud ssh --worker=all)
#   gcloud compute tpus tpu-vm ssh MY_TPU_NAME --worker=all \
#       --command="cd ~/activation-extract && ./scripts/launch_multihost.sh"
#
#   # With custom parameters
#   DATASET_PATH=gs://bucket/data.jsonl GCS_BUCKET=my-bucket \
#       ./scripts/launch_multihost.sh
#
# Environment Variables:
#   Required:
#     DATASET_PATH   - Path to dataset (GCS or local)
#     GCS_BUCKET     - GCS bucket for uploads
#
#   Optional:
#     TOPOLOGY       - TPU topology (default: auto-detect)
#     MODEL_PATH     - Model to use (default: Qwen/Qwen2.5-0.5B)
#     BATCH_SIZE     - Global batch size (default: 32)
#     MAX_TASKS      - Maximum tasks to process (default: all)
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Multihost TPU Extraction Launcher    ${NC}"
echo -e "${GREEN}========================================${NC}"

# =============================================================================
# Auto-detect multihost environment
# =============================================================================

# Coordinator address (use first worker as coordinator)
if [ -n "$TPU_WORKER_HOSTNAMES" ]; then
    COORDINATOR_HOST=$(echo "$TPU_WORKER_HOSTNAMES" | cut -d',' -f1)
    export COORDINATOR_ADDRESS="${COORDINATOR_HOST}:8476"
    echo -e "Coordinator: ${COORDINATOR_ADDRESS}"
fi

# Host ID from TPU runtime
export HOST_ID="${TPU_WORKER_ID:-${CLOUD_TPU_TASK_ID:-0}}"
echo -e "Host ID: ${HOST_ID}"

# Number of hosts
export NUM_HOSTS="${TPU_WORKER_COUNT:-1}"
echo -e "Num Hosts: ${NUM_HOSTS}"

# =============================================================================
# Configuration
# =============================================================================

# Required
DATASET_PATH="${DATASET_PATH:?DATASET_PATH environment variable is required}"
GCS_BUCKET="${GCS_BUCKET:?GCS_BUCKET environment variable is required}"

# Optional with defaults
TOPOLOGY="${TOPOLOGY:-v5e-64}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-./activations}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"

echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Topology:      ${TOPOLOGY}"
echo -e "  Dataset:       ${DATASET_PATH}"
echo -e "  Model:         ${MODEL_PATH}"
echo -e "  GCS Bucket:    ${GCS_BUCKET}"
echo -e "  Batch Size:    ${BATCH_SIZE}"
echo -e "  Max Seq Len:   ${MAX_SEQ_LENGTH}"
echo ""

# =============================================================================
# Pre-flight checks
# =============================================================================

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Check if script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$PROJECT_DIR/multihost_extract.py" ]; then
    echo -e "${RED}Error: multihost_extract.py not found in $PROJECT_DIR${NC}"
    exit 1
fi

# Check GCS authentication
if [ -n "$GCS_BUCKET" ]; then
    echo -e "Checking GCS access..."
    if ! gsutil ls "gs://${GCS_BUCKET}/" &> /dev/null; then
        echo -e "${YELLOW}Warning: Cannot access gs://${GCS_BUCKET}. Check authentication.${NC}"
    else
        echo -e "${GREEN}GCS access OK${NC}"
    fi
fi

# =============================================================================
# Build command
# =============================================================================

CMD="python3 $PROJECT_DIR/multihost_extract.py"
CMD="$CMD --topology $TOPOLOGY"
CMD="$CMD --dataset_path $DATASET_PATH"
CMD="$CMD --gcs_bucket $GCS_BUCKET"
CMD="$CMD --model_path $MODEL_PATH"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --max_seq_length $MAX_SEQ_LENGTH"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --checkpoint_dir $CHECKPOINT_DIR"
CMD="$CMD --upload_to_gcs"
CMD="$CMD --compress_shards"
CMD="$CMD --verbose"

# Optional: max tasks
if [ -n "$MAX_TASKS" ]; then
    CMD="$CMD --max_tasks $MAX_TASKS"
fi

# Optional: explicit coordinator (for manual setups)
if [ -n "$COORDINATOR_ADDRESS" ] && [ "$NUM_HOSTS" -gt 1 ]; then
    CMD="$CMD --coordinator_address $COORDINATOR_ADDRESS"
    CMD="$CMD --host_id $HOST_ID"
    CMD="$CMD --num_hosts $NUM_HOSTS"
fi

# =============================================================================
# Run extraction
# =============================================================================

echo ""
echo -e "${GREEN}Starting extraction...${NC}"
echo -e "Command: $CMD"
echo ""

cd "$PROJECT_DIR"
exec $CMD
