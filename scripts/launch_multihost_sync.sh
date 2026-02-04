#!/bin/bash
# ============================================================================
# Launch Multihost Extraction with Socket Barrier Synchronization
# ============================================================================
#
# This script launches the multihost extraction on a TPU pod with socket-based
# barrier synchronization to ensure all workers are synchronized before critical
# operations.
#
# Usage:
#   ./scripts/launch_multihost_sync.sh --topology v5e-64 \
#       --dataset gs://bucket/data.jsonl --gcs_bucket my-bucket
#
# The script:
# 1. Detects Worker 0's internal IP to use as barrier controller
# 2. Launches extraction on all workers with barrier sync enabled
# 3. Worker 0 runs both server and client, others run only client
# ============================================================================

set -e

# Default values
TPU_NAME="${TPU_NAME:-node-v5e-64-europe-west4-b}"
ZONE="${ZONE:-europe-west4-b}"
TOPOLOGY="${TOPOLOGY:-v5litepod-64}"
DATASET_PATH=""
GCS_BUCKET=""
GCS_PREFIX="activations"
BATCH_SIZE=32
MAX_TASKS=""
BARRIER_PORT=5555
VERBOSE="--verbose"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tpu-name)
            TPU_NAME="$2"
            shift 2
            ;;
        --zone)
            ZONE="$2"
            shift 2
            ;;
        --topology)
            TOPOLOGY="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --gcs_bucket|--gcs-bucket)
            GCS_BUCKET="$2"
            shift 2
            ;;
        --gcs_prefix|--gcs-prefix)
            GCS_PREFIX="$2"
            shift 2
            ;;
        --batch_size|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_tasks|--max-tasks)
            MAX_TASKS="$2"
            shift 2
            ;;
        --barrier-port)
            BARRIER_PORT="$2"
            shift 2
            ;;
        --quiet)
            VERBOSE=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATASET_PATH" ]]; then
    echo "Error: --dataset is required"
    exit 1
fi

if [[ -z "$GCS_BUCKET" ]]; then
    echo "Error: --gcs_bucket is required"
    exit 1
fi

echo "============================================================================"
echo "MULTIHOST EXTRACTION WITH SOCKET BARRIER SYNC"
echo "============================================================================"
echo "TPU Name: $TPU_NAME"
echo "Zone: $ZONE"
echo "Topology: $TOPOLOGY"
echo "Dataset: $DATASET_PATH"
echo "GCS Bucket: $GCS_BUCKET"
echo "GCS Prefix: $GCS_PREFIX"
echo "Batch Size: $BATCH_SIZE"
echo "Barrier Port: $BARRIER_PORT"
echo "============================================================================"

# Step 1: Get Worker 0's internal IP for barrier controller
echo ""
echo "Step 1: Getting Worker 0 internal IP..."

WORKER0_IP=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --worker=0 \
    --command="hostname -I | awk '{print \$1}'" 2>/dev/null | tr -d '\n')

if [[ -z "$WORKER0_IP" ]]; then
    echo "Error: Could not get Worker 0 IP address"
    exit 1
fi

echo "  Worker 0 IP: $WORKER0_IP"

# Step 2: Build the extraction command
echo ""
echo "Step 2: Building extraction command..."

CMD="cd ~/activation-extract-multihost && "
CMD+="python3 multihost_extract.py "
CMD+="--topology $TOPOLOGY "
CMD+="--dataset_path $DATASET_PATH "
CMD+="--gcs_bucket $GCS_BUCKET "
CMD+="--gcs_prefix $GCS_PREFIX "
CMD+="--batch_size $BATCH_SIZE "
CMD+="--enable_barrier_sync "
CMD+="--barrier_port $BARRIER_PORT "
CMD+="--barrier_controller_host $WORKER0_IP "
CMD+="--upload_to_gcs "

if [[ -n "$MAX_TASKS" ]]; then
    CMD+="--max_tasks $MAX_TASKS "
fi

if [[ -n "$VERBOSE" ]]; then
    CMD+="$VERBOSE "
fi

echo "  Command: $CMD"

# Step 3: Execute on all workers
echo ""
echo "Step 3: Launching extraction on all workers..."
echo "============================================================================"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" \
    --worker=all \
    --command="$CMD 2>&1"

echo ""
echo "============================================================================"
echo "Extraction complete!"
echo "============================================================================"
