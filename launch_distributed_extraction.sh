#!/bin/bash
# Launch distributed activation extraction across 32 TPU machines
#
# Prerequisites:
# 1. All 32 machines should have the same code and environment
# 2. SSH access to all machines (or use your TPU orchestration system)
# 3. GCS bucket configured for storing results
#
# Usage:
#   ./launch_distributed_extraction.sh

# ============================================================================
# CONFIGURATION
# ============================================================================

# Total number of machines
TOTAL_MACHINES=32

# GCS bucket for results
GCS_BUCKET="your-bucket-name"  # CHANGE THIS

# Model and dataset config
MODEL_PATH="KathirKs/qwen-2.5-0.5b"
DATASET_NAME="HuggingFaceFW/fineweb-edu"
DATASET_CONFIG="sample-10BT"  # Use "default" for full dataset
DATASET_SPLIT="train"

# Extraction config
BATCH_SIZE=8
MAX_SEQ_LENGTH=2048
LAYERS_TO_EXTRACT="10 11 12 13 14 15 16 17 18 19 20 21 22 23"

# Shard config
SHARD_SIZE_GB=1.0

# Optional: limit samples per machine (for testing)
# MAX_SAMPLES=1000  # Uncomment to limit

# ============================================================================
# FUNCTION TO LAUNCH ON SINGLE MACHINE
# ============================================================================

launch_on_machine() {
    local MACHINE_ID=$1
    local MACHINE_HOST=$2

    echo "Launching on machine ${MACHINE_ID} (${MACHINE_HOST})..."

    # Build the command
    CMD="cd /home/kathirks_gc/torch_xla/qwen && \
         python extract_activations_fineweb.py \
         --machine_id ${MACHINE_ID} \
         --total_machines ${TOTAL_MACHINES} \
         --model_path ${MODEL_PATH} \
         --dataset_name ${DATASET_NAME} \
         --dataset_config ${DATASET_CONFIG} \
         --dataset_split ${DATASET_SPLIT} \
         --batch_size ${BATCH_SIZE} \
         --max_seq_length ${MAX_SEQ_LENGTH} \
         --layers_to_extract ${LAYERS_TO_EXTRACT} \
         --shard_size_gb ${SHARD_SIZE_GB} \
         --upload_to_gcs \
         --gcs_bucket ${GCS_BUCKET} \
         --compress_shards \
         --delete_local_after_upload \
         --verbose"

    # Add max_samples if defined
    if [ ! -z "${MAX_SAMPLES}" ]; then
        CMD="${CMD} --max_samples ${MAX_SAMPLES}"
    fi

    # Execute on remote machine (adjust based on your infrastructure)
    # Option 1: SSH (if you have SSH access)
    ssh "${MACHINE_HOST}" "${CMD}" > "logs/machine_${MACHINE_ID}.log" 2>&1 &

    # Option 2: Use gcloud TPU (if using GCP TPU VMs)
    # gcloud compute tpus tpu-vm ssh "${MACHINE_HOST}" --zone=us-central1-a --command="${CMD}" > "logs/machine_${MACHINE_ID}.log" 2>&1 &

    # Option 3: Use your custom orchestration system
    # your_orchestration_tool run --machine "${MACHINE_HOST}" --command "${CMD}"

    echo "  ✓ Started on machine ${MACHINE_ID}"
}

# ============================================================================
# MAIN LAUNCH LOGIC
# ============================================================================

echo "========================================================================"
echo "LAUNCHING DISTRIBUTED ACTIVATION EXTRACTION"
echo "========================================================================"
echo "Total machines: ${TOTAL_MACHINES}"
echo "GCS bucket: gs://${GCS_BUCKET}/activations_fineweb/"
echo "Dataset: ${DATASET_NAME} (${DATASET_CONFIG})"
echo "========================================================================"

# Create logs directory
mkdir -p logs

# Define your machine hostnames/IPs
# Adjust this based on your infrastructure
MACHINES=(
    "tpu-vm-0"
    "tpu-vm-1"
    "tpu-vm-2"
    "tpu-vm-3"
    "tpu-vm-4"
    "tpu-vm-5"
    "tpu-vm-6"
    "tpu-vm-7"
    "tpu-vm-8"
    "tpu-vm-9"
    "tpu-vm-10"
    "tpu-vm-11"
    "tpu-vm-12"
    "tpu-vm-13"
    "tpu-vm-14"
    "tpu-vm-15"
    "tpu-vm-16"
    "tpu-vm-17"
    "tpu-vm-18"
    "tpu-vm-19"
    "tpu-vm-20"
    "tpu-vm-21"
    "tpu-vm-22"
    "tpu-vm-23"
    "tpu-vm-24"
    "tpu-vm-25"
    "tpu-vm-26"
    "tpu-vm-27"
    "tpu-vm-28"
    "tpu-vm-29"
    "tpu-vm-30"
    "tpu-vm-31"
)

# Launch on all machines
for MACHINE_ID in $(seq 0 $((TOTAL_MACHINES - 1))); do
    MACHINE_HOST="${MACHINES[$MACHINE_ID]}"
    launch_on_machine ${MACHINE_ID} ${MACHINE_HOST}

    # Small delay to avoid overwhelming the system
    sleep 2
done

echo "========================================================================"
echo "All jobs launched!"
echo "Monitor progress with: tail -f logs/machine_*.log"
echo "Check GCS: gsutil ls gs://${GCS_BUCKET}/activations_fineweb/"
echo "========================================================================"

# Optional: Wait for all background jobs to complete
# wait
# echo "All jobs completed!"
