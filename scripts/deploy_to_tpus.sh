#!/bin/bash
#
# Complete deployment script for activation extraction across multiple TPU workers
#
# Features:
#   - Multi-zone TPU deployment
#   - Automatic TPU creation/recreation
#   - Dataset stream preparation and upload
#   - Remote setup (clone repo + install deps)
#   - Launch extraction with checkpointing
#
# Usage: See help message below
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
GCS_BUCKET=""
GCS_DATASET_PREFIX="datasets"
GCS_ACTIVATION_PREFIX="activations"
ZONES="us-central1-a"
WORKERS_PER_ZONE=1
DATASET_NAME="barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems"
MODEL="Qwen/Qwen2.5-0.5B"
TPU_TYPE="v3-8"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DATASET_DIR="./dataset_streams"
MAX_SAMPLES_PER_STREAM=""
STREAM_OFFSET=0  # Starting stream index (for multi-region coordination)
ACTIVATION_TYPE="residual"  # 'residual', 'mlp', or 'attn'
CREATE_TPUS=false
SKIP_DATASET=false
SKIP_LAUNCH=false
MONITOR=false
MONITOR_INTERVAL=60  # Check interval in seconds
NOHUP=false  # Run monitoring in background with nohup

# Global tracking
declare -A TPU_TO_STREAM  # Map TPU name -> dataset stream path
declare -A TPU_TO_ZONE    # Map TPU name -> zone

function print_usage() {
    cat <<EOF
Usage: $0 --gcs_bucket BUCKET --zones ZONES --workers_per_zone N [OPTIONS]

Required:
  --gcs_bucket BUCKET        GCS bucket name for datasets and outputs
  --zones ZONES              Comma-separated list of zones
                            Example: us-central1-a,us-central1-b,europe-west4-a
  --workers_per_zone N       Number of workers per zone

Optional:
  --dataset DATASET          HuggingFace dataset name
                            (default: $DATASET_NAME)
  --model MODEL             Model path (default: $MODEL)
  --tpu_type TYPE           TPU type (default: $TPU_TYPE)
  --max_samples N           Max samples per stream (for testing)
  --stream_offset N         Starting stream index for multi-region coordination
                            (default: 0). Use to avoid duplicate processing across regions.
  --activation_type TYPE    Activation type to extract: 'residual', 'mlp', 'attn'
                            (default: residual)
  --create_tpus             Create TPUs before deployment
  --skip_dataset            Skip dataset preparation (use existing)
  --skip_launch             Skip launching extraction (just prepare)
  --monitor                 Enable continuous monitoring with auto-recovery
  --monitor_interval N      Monitoring check interval in seconds (default: $MONITOR_INTERVAL)
  --nohup                   Run monitoring in background with nohup (requires --monitor)

Examples:
  # Full deployment with background monitoring (recommended for long-running jobs)
  $0 --gcs_bucket my-bucket \\
     --zones us-central1-a,us-central1-b \\
     --workers_per_zone 4 \\
     --create_tpus \\
     --monitor \\
     --nohup

  # Full deployment with foreground monitoring
  $0 --gcs_bucket my-bucket \\
     --zones us-central1-a,us-central1-b \\
     --workers_per_zone 4 \\
     --create_tpus \\
     --monitor

  # Test with limited data and monitoring
  $0 --gcs_bucket my-bucket \\
     --zones us-central1-a \\
     --workers_per_zone 2 \\
     --max_samples 100 \\
     --create_tpus \\
     --monitor

  # Just launch extraction (TPUs already exist) with background monitoring
  $0 --gcs_bucket my-bucket \\
     --zones us-central1-a \\
     --workers_per_zone 4 \\
     --skip_dataset \\
     --monitor \\
     --nohup

  # Without monitoring (manual management)
  $0 --gcs_bucket my-bucket \\
     --zones us-central1-a \\
     --workers_per_zone 4 \\
     --create_tpus

Multi-Region Deployment:
  When deploying across multiple regions, use --stream_offset to ensure unique data:

  # Region 1: US (8 workers) - processes streams 0-7
  $0 --gcs_bucket fineweb-data-us-central1 \\
     --zones us-central1-a \\
     --workers_per_zone 8 \\
     --stream_offset 0 \\
     --skip_dataset \\
     --create_tpus --monitor

  # Region 2: Europe (8 workers) - processes streams 8-15
  $0 --gcs_bucket fineweb-data-europe-west4 \\
     --zones europe-west4-b \\
     --workers_per_zone 8 \\
     --stream_offset 8 \\
     --skip_dataset \\
     --create_tpus --monitor

  Note: First create ALL streams once (num_streams = total workers across all regions)

Naming Convention:
  TPUs are named: tpu-{region}-{zone_letter}-{worker_id}
  Examples: tpu-us-central1-a-0, tpu-us-central1-b-3, tpu-europe-west4-a-1

EOF
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gcs_bucket) GCS_BUCKET="$2"; shift ;;
        --zones) ZONES="$2"; shift ;;
        --workers_per_zone) WORKERS_PER_ZONE="$2"; shift ;;
        --dataset) DATASET_NAME="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --tpu_type) TPU_TYPE="$2"; shift ;;
        --max_samples) MAX_SAMPLES_PER_STREAM="$2"; shift ;;
        --stream_offset) STREAM_OFFSET="$2"; shift ;;
        --activation_type) ACTIVATION_TYPE="$2"; shift ;;
        --monitor_interval) MONITOR_INTERVAL="$2"; shift ;;
        --create_tpus) CREATE_TPUS=true ;;
        --skip_dataset) SKIP_DATASET=true ;;
        --skip_launch) SKIP_LAUNCH=true ;;
        --monitor) MONITOR=true ;;
        --nohup) NOHUP=true ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo -e "${RED}Unknown parameter: $1${NC}"; print_usage; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$GCS_BUCKET" ] || [ -z "$ZONES" ] || [ -z "$WORKERS_PER_ZONE" ]; then
    echo -e "${RED}Error: --gcs_bucket, --zones, and --workers_per_zone are required${NC}"
    print_usage
    exit 1
fi

# Validate nohup flag
if [ "$NOHUP" = true ] && [ "$MONITOR" = false ]; then
    echo -e "${RED}Error: --nohup requires --monitor to be enabled${NC}"
    print_usage
    exit 1
fi

# Calculate total workers
IFS=',' read -ra ZONE_ARRAY <<< "$ZONES"
TOTAL_WORKERS=$((${#ZONE_ARRAY[@]} * WORKERS_PER_ZONE))

echo -e "${GREEN}=========================================="
echo "ACTIVATION EXTRACTION DEPLOYMENT"
echo -e "==========================================${NC}"
echo "GCS bucket: gs://$GCS_BUCKET"
echo "Zones: $ZONES"
echo "Workers per zone: $WORKERS_PER_ZONE"
echo "Total workers: $TOTAL_WORKERS"
echo "Dataset: $DATASET_NAME"
echo "Model: $MODEL"
echo "TPU type: $TPU_TYPE"
echo "Activation type: $ACTIVATION_TYPE"
echo ""

# Helper function to get TPU name
function get_tpu_name() {
    local zone=$1
    local worker_id=$2
    local region=$(echo "$zone" | rev | cut -d'-' -f2- | rev)
    local zone_letter=$(echo "$zone" | rev | cut -d'-' -f1 | rev)
    echo "tpu-${region}-${zone_letter}-${worker_id}"
}

# Get worker progress from checkpoint
function get_worker_progress() {
    local zone=$1
    local tpu_name=$2

    # Try to read checkpoint file from TPU
    local checkpoint=$(gcloud compute tpus tpu-vm ssh "$tpu_name" \
        --zone="$zone" \
        --command="cat ~/activation-extract/checkpoints/checkpoint_worker_*.json 2>/dev/null" 2>/dev/null || echo "{}")

    # Extract samples processed
    local samples=$(echo "$checkpoint" | grep -o '"total_samples_processed": *[0-9]*' | grep -o '[0-9]*' || echo "0")
    local shards=$(echo "$checkpoint" | grep -o '"total_shards": *[0-9]*' | grep -o '[0-9]*' || echo "0")
    local status=$(echo "$checkpoint" | grep -o '"status": *"[^"]*"' | cut -d'"' -f4 || echo "unknown")

    echo "${samples}|${shards}|${status}"
}

# Get GCS shard count for worker
function get_gcs_shard_count() {
    local worker_id=$1

    local count=$(gsutil ls "gs://$GCS_BUCKET/$GCS_ACTIVATION_PREFIX/tpu_${worker_id}/*.pkl.gz" 2>/dev/null | wc -l || echo "0")
    echo "$count"
}

# Launch extraction on TPU
function launch_extraction() {
    local zone=$1
    local tpu_name=$2
    local dataset_stream=$3

    local max_tasks_arg=""
    if [ -n "$MAX_SAMPLES_PER_STREAM" ]; then
        max_tasks_arg="--max_tasks $MAX_SAMPLES_PER_STREAM"
    fi

    gcloud compute tpus tpu-vm ssh "$tpu_name" \
        --zone="$zone" \
        --command="bash <(curl -s https://raw.githubusercontent.com/kathir-ks/activation-extract/main/scripts/run_extraction_worker.sh) \
            --gcs_bucket \"$GCS_BUCKET\" \
            --dataset_stream \"$dataset_stream\" \
            --model \"$MODEL\" \
            --gcs_prefix \"$GCS_ACTIVATION_PREFIX\" \
            --activation_type \"$ACTIVATION_TYPE\" \
            $max_tasks_arg \
            > ~/extraction.log 2>&1 &" &>/dev/null

    return $?
}

# Create single TPU with infinite retry logic
function create_tpu() {
    local zone=$1
    local tpu_name=$2

    local retry_count=0
    local max_retries=1000000  # Essentially infinite
    local retry_delay=30

    while [ $retry_count -lt $max_retries ]; do
        if gcloud compute tpus tpu-vm create "$tpu_name" \
            --zone="$zone" \
            --accelerator-type="$TPU_TYPE" \
            --version="tpu-ubuntu2204-base" \
            --preemptible \
            --quiet 2>&1; then
            return 0
        fi

        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            echo -e "${YELLOW}[TPU Lifecycle]   Retry $retry_count for $tpu_name (waiting ${retry_delay}s)...${NC}"
            sleep $retry_delay
        fi
    done

    return 1
}

# Delete TPU
function delete_tpu() {
    local zone=$1
    local tpu_name=$2

    gcloud compute tpus tpu-vm delete "$tpu_name" \
        --zone="$zone" \
        --quiet 2>/dev/null

    return $?
}

# Get TPU status
function get_tpu_status() {
    local zone=$1
    local tpu_name=$2

    local status=$(gcloud compute tpus tpu-vm describe "$tpu_name" \
        --zone="$zone" \
        --format="value(state)" 2>/dev/null || echo "NOT_FOUND")

    echo "$status"
}

# Display progress dashboard
function display_progress() {
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         ACTIVATION EXTRACTION - LIVE MONITORING               ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}GCS Bucket:${NC} gs://$GCS_BUCKET"
    echo -e "${BLUE}Total Workers:${NC} $TOTAL_WORKERS across ${#ZONE_ARRAY[@]} zone(s)"
    echo -e "${BLUE}Check Interval:${NC} ${MONITOR_INTERVAL}s"
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  TPU NAME                 STATUS      SAMPLES    SHARDS   GCS  ║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════╣${NC}"

    local total_samples=0
    local total_shards=0
    local total_gcs=0
    local healthy_count=0
    local preempted_count=0
    local working_count=0

    for zone in "${ZONE_ARRAY[@]}"; do
        for worker_id in $(seq 0 $((WORKERS_PER_ZONE - 1))); do
            local tpu_name=$(get_tpu_name "$zone" "$worker_id")
            local status=$(get_tpu_status "$zone" "$tpu_name")

            # Get progress
            local progress=$(get_worker_progress "$zone" "$tpu_name")
            IFS='|' read -r samples shards work_status <<< "$progress"

            # Get GCS count
            local gcs_count=$(get_gcs_shard_count "$worker_id")

            # Color code status
            local status_display=""
            case "$status" in
                READY)
                    status_display="${GREEN}READY  ${NC}"
                    healthy_count=$((healthy_count + 1))
                    working_count=$((working_count + 1))
                    ;;
                PREEMPTED)
                    status_display="${RED}PREEMPT${NC}"
                    preempted_count=$((preempted_count + 1))
                    ;;
                NOT_FOUND)
                    status_display="${RED}MISSING${NC}"
                    preempted_count=$((preempted_count + 1))
                    ;;
                CREATING|STARTING)
                    status_display="${YELLOW}STARTING${NC}"
                    working_count=$((working_count + 1))
                    ;;
                *)
                    status_display="${YELLOW}${status:0:7}${NC}"
                    ;;
            esac

            # Accumulate totals
            total_samples=$((total_samples + samples))
            total_shards=$((total_shards + shards))
            total_gcs=$((total_gcs + gcs_count))

            # Print row
            printf "${CYAN}║${NC}  %-25s %b  %7s  %7s  %5s ${CYAN}║${NC}\n" \
                "$tpu_name" "$status_display" "$samples" "$shards" "$gcs_count"
        done
    done

    echo -e "${CYAN}╠════════════════════════════════════════════════════════════════╣${NC}"
    printf "${CYAN}║${NC}  ${GREEN}TOTALS${NC}                              %7s  %7s  %5s ${CYAN}║${NC}\n" \
        "$total_samples" "$total_shards" "$total_gcs"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Status Summary:${NC}"
    echo -e "  ${GREEN}✓ Healthy:${NC} $healthy_count"
    echo -e "  ${YELLOW}⟳ Working:${NC} $working_count"
    echo -e "  ${RED}✗ Needs Recovery:${NC} $preempted_count"
    echo ""
    echo -e "${BLUE}Last update:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${BLUE}Press Ctrl+C to stop monitoring${NC}"
    echo ""
}

# Loop 1: TPU Lifecycle Management
# Continuously ensures all TPUs exist and recreates them on preemption
function tpu_lifecycle_loop() {
    echo -e "${GREEN}[TPU Lifecycle] Starting continuous TPU lifecycle management${NC}"

    local cycle=0
    set +e  # Don't crash on errors

    while true; do
        cycle=$((cycle + 1))
        echo -e "${BLUE}[TPU Lifecycle] Cycle $cycle - Checking TPU health...${NC}"

        for zone in "${ZONE_ARRAY[@]}"; do
            for worker_id in $(seq 0 $((WORKERS_PER_ZONE - 1))); do
                local tpu_name=$(get_tpu_name "$zone" "$worker_id")
                local status=$(get_tpu_status "$zone" "$tpu_name")

                # If TPU doesn't exist or is preempted, recreate it
                if [ "$status" = "PREEMPTED" ] || [ "$status" = "NOT_FOUND" ] || [ "$status" = "TERMINATED" ]; then
                    echo -e "${YELLOW}[TPU Lifecycle] $tpu_name is $status - recreating...${NC}"

                    # Delete if exists
                    delete_tpu "$zone" "$tpu_name" 2>/dev/null || true
                    sleep 5

                    # Recreate with infinite retries
                    create_tpu "$zone" "$tpu_name" && \
                        echo -e "${GREEN}[TPU Lifecycle] ✓ $tpu_name recreated${NC}" || \
                        echo -e "${YELLOW}[TPU Lifecycle] Will retry $tpu_name next cycle${NC}"
                fi
            done
        done

        sleep "$MONITOR_INTERVAL"
    done
}

# Loop 2: Extraction Management
# Continuously ensures extraction is running on all READY TPUs
function extraction_management_loop() {
    echo -e "${GREEN}[Extraction] Starting continuous extraction management${NC}"

    local cycle=0
    set +e  # Don't crash on errors

    # Track which TPUs have extraction running
    declare -A EXTRACTION_RUNNING

    while true; do
        cycle=$((cycle + 1))
        echo -e "${BLUE}[Extraction] Cycle $cycle - Checking extraction status...${NC}"

        for zone in "${ZONE_ARRAY[@]}"; do
            for worker_id in $(seq 0 $((WORKERS_PER_ZONE - 1))); do
                local tpu_name=$(get_tpu_name "$zone" "$worker_id")
                local status=$(get_tpu_status "$zone" "$tpu_name")
                local dataset_stream="${TPU_TO_STREAM[$tpu_name]}"

                # If TPU is READY and extraction not running, launch it
                if [ "$status" = "READY" ]; then
                    # Check if extraction process is running on the TPU
                    local has_extraction=$(gcloud compute tpus tpu-vm ssh "$tpu_name" \
                        --zone="$zone" \
                        --command="pgrep -f 'python.*extract' > /dev/null && echo 'yes' || echo 'no'" 2>/dev/null || echo "no")

                    if [ "$has_extraction" = "no" ]; then
                        echo -e "${YELLOW}[Extraction] Launching extraction on $tpu_name${NC}"

                        if launch_extraction "$zone" "$tpu_name" "$dataset_stream"; then
                            EXTRACTION_RUNNING[$tpu_name]=1
                            echo -e "${GREEN}[Extraction] ✓ Extraction started on $tpu_name${NC}"
                        else
                            echo -e "${YELLOW}[Extraction] Failed to launch on $tpu_name, will retry${NC}"
                        fi
                    fi
                else
                    # TPU not ready, mark extraction as not running
                    unset EXTRACTION_RUNNING[$tpu_name]
                fi
            done
        done

        sleep "$MONITOR_INTERVAL"
    done
}

# Combined monitoring display loop
function display_loop() {
    echo -e "${GREEN}[Display] Starting progress display${NC}"

    set +e
    while true; do
        display_progress
        sleep "$MONITOR_INTERVAL"
    done
}

# Main orchestrator - runs all loops in parallel
function monitor_and_manage() {
    echo -e "${GREEN}=========================================="
    echo "STARTING DUAL-LOOP MANAGEMENT SYSTEM"
    echo -e "==========================================${NC}"
    echo "Monitor interval: ${MONITOR_INTERVAL}s"
    echo "Loop 1: TPU Lifecycle (creation/recreation)"
    echo "Loop 2: Extraction Management (launch/relaunch)"
    echo ""
    echo "Press Ctrl+C to stop all loops"
    sleep 3

    # Disable set -e for the entire management system
    set +e

    # Start TPU lifecycle loop in background
    tpu_lifecycle_loop &
    local tpu_loop_pid=$!
    echo -e "${GREEN}[Main] TPU Lifecycle Loop started (PID: $tpu_loop_pid)${NC}"

    # Start extraction management loop in background
    extraction_management_loop &
    local extraction_loop_pid=$!
    echo -e "${GREEN}[Main] Extraction Management Loop started (PID: $extraction_loop_pid)${NC}"

    # Start display loop in foreground (or background if in nohup mode)
    display_loop &
    local display_loop_pid=$!
    echo -e "${GREEN}[Main] Display Loop started (PID: $display_loop_pid)${NC}"

    # Wait for any loop to exit (they shouldn't, but handle gracefully)
    wait $tpu_loop_pid $extraction_loop_pid $display_loop_pid
}

# Step 0: Create TPUs if requested (skipped when using --monitor, as the lifecycle loop handles it)
if [ "$CREATE_TPUS" = true ] && [ "$MONITOR" = false ]; then
    echo -e "${YELLOW}Step 0: Creating TPUs...${NC}"
    echo ""

    # Disable set -e for TPU creation to handle failures gracefully
    set +e
    bash "$CODE_DIR/scripts/manage_tpus.sh" create \
        --zones "$ZONES" \
        --workers_per_zone "$WORKERS_PER_ZONE" \
        --tpu_type "$TPU_TYPE"
    TPU_CREATE_EXIT_CODE=$?
    set -e

    if [ $TPU_CREATE_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ TPUs created${NC}"
    else
        echo -e "${YELLOW}⚠ Some TPUs may have failed to create (exit code: $TPU_CREATE_EXIT_CODE)${NC}"
        echo -e "${YELLOW}  Or manually check TPU status with: bash scripts/manage_tpus.sh status --zones $ZONES${NC}"
    fi

    # Wait for TPUs to be ready
    echo ""
    echo "Waiting 30 seconds for TPUs to initialize..."
    sleep 30
    echo ""
elif [ "$CREATE_TPUS" = true ] && [ "$MONITOR" = true ]; then
    echo -e "${YELLOW}Step 0: Skipping initial TPU creation - lifecycle loop will handle it${NC}"
    echo -e "${CYAN}  The continuous TPU lifecycle loop will create and manage all TPUs${NC}"
    echo ""
else
    echo -e "${YELLOW}Skipping TPU creation (use --create_tpus to create)${NC}"
    echo ""
fi

# Step 1: Prepare dataset streams
if [ "$SKIP_DATASET" = false ]; then
    echo -e "${YELLOW}Step 1: Preparing dataset streams...${NC}"
    echo ""

    mkdir -p "$LOCAL_DATASET_DIR"

    DATASET_CMD="python create_dataset_streams.py \
      --dataset_name \"$DATASET_NAME\" \
      --num_streams $TOTAL_WORKERS \
      --output_dir \"$LOCAL_DATASET_DIR\" \
      --verbose"

    if [ -n "$MAX_SAMPLES_PER_STREAM" ]; then
        DATASET_CMD="$DATASET_CMD --max_samples $MAX_SAMPLES_PER_STREAM"
    fi

    set +e
    eval $DATASET_CMD
    DATASET_EXIT_CODE=$?
    set -e

    if [ $DATASET_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ Dataset streams created${NC}"
    else
        echo -e "${RED}✗ Dataset preparation failed${NC}"
        exit 1
    fi

    # Upload to GCS
    echo ""
    echo -e "${YELLOW}Uploading dataset streams to GCS...${NC}"
    set +e
    bash "$CODE_DIR/scripts/upload_dataset_streams_to_gcs.sh" \
        --bucket "$GCS_BUCKET" \
        --prefix "$GCS_DATASET_PREFIX" \
        --local_dir "$LOCAL_DATASET_DIR"
    UPLOAD_EXIT_CODE=$?
    set -e

    if [ $UPLOAD_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ Dataset streams uploaded${NC}"
    else
        echo -e "${RED}✗ Upload failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Step 1: Skipping dataset preparation (--skip_dataset)${NC}"
fi

echo ""

# Step 2: Launch extraction on all workers
if [ "$SKIP_LAUNCH" = false ]; then
    echo -e "${YELLOW}Step 2: Launching extraction on all workers...${NC}"
    echo ""

    STREAM_IDX=$STREAM_OFFSET  # Start from offset for multi-region coordination

    for zone in "${ZONE_ARRAY[@]}"; do
        echo -e "${YELLOW}--- Zone: $zone ---${NC}"

        for worker_id in $(seq 0 $((WORKERS_PER_ZONE - 1))); do
            TPU_NAME=$(get_tpu_name "$zone" "$worker_id")
            DATASET_STREAM="gs://$GCS_BUCKET/$GCS_DATASET_PREFIX/stream_$(printf "%03d" $STREAM_IDX).jsonl"

            # Store mapping for monitoring
            TPU_TO_STREAM[$TPU_NAME]="$DATASET_STREAM"
            TPU_TO_ZONE[$TPU_NAME]="$zone"

            echo "Launching on $TPU_NAME (stream $STREAM_IDX from bucket gs://$GCS_BUCKET)..."

            MAX_TASKS_ARG=""
            if [ -n "$MAX_SAMPLES_PER_STREAM" ]; then
                MAX_TASKS_ARG="--max_tasks $MAX_SAMPLES_PER_STREAM"
            fi

            # Run setup and extraction script remotely via curl
            # Use || true to prevent set -e from crashing if TPU not ready
            gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
                --zone="$zone" \
                --command="bash <(curl -s https://raw.githubusercontent.com/kathir-ks/activation-extract/main/scripts/run_extraction_worker.sh) \
                    --gcs_bucket \"$GCS_BUCKET\" \
                    --dataset_stream \"$DATASET_STREAM\" \
                    --model \"$MODEL\" \
                    --gcs_prefix \"$GCS_ACTIVATION_PREFIX\" \
                    --activation_type \"$ACTIVATION_TYPE\" \
                    $MAX_TASKS_ARG \
                    > ~/extraction.log 2>&1 &" &>/dev/null &

            STREAM_IDX=$((STREAM_IDX + 1))
            sleep 2  # Stagger starts
        done

        echo ""
    done

    # Wait for background SSH commands (ignore failures - monitoring will recover)
    set +e
    wait
    set -e

    echo -e "${GREEN}✓ Extraction launched on all workers${NC}"
else
    echo -e "${YELLOW}Step 2: Skipping extraction launch (--skip_launch)${NC}"

    # Still need to populate mappings for monitoring
    STREAM_IDX=$STREAM_OFFSET  # Start from offset for multi-region coordination
    for zone in "${ZONE_ARRAY[@]}"; do
        for worker_id in $(seq 0 $((WORKERS_PER_ZONE - 1))); do
            TPU_NAME=$(get_tpu_name "$zone" "$worker_id")
            DATASET_STREAM="gs://$GCS_BUCKET/$GCS_DATASET_PREFIX/stream_$(printf "%03d" $STREAM_IDX).jsonl"

            TPU_TO_STREAM[$TPU_NAME]="$DATASET_STREAM"
            TPU_TO_ZONE[$TPU_NAME]="$zone"

            STREAM_IDX=$((STREAM_IDX + 1))
        done
    done
fi

echo ""
echo -e "${GREEN}=========================================="
echo "DEPLOYMENT COMPLETE"
echo -e "==========================================${NC}"
echo ""
echo "Total workers: $TOTAL_WORKERS across ${#ZONE_ARRAY[@]} zone(s)"
echo "Dataset streams: gs://$GCS_BUCKET/$GCS_DATASET_PREFIX/"
echo "Activations output: gs://$GCS_BUCKET/$GCS_ACTIVATION_PREFIX/"
echo ""

# Step 3: Enter monitoring mode if requested
if [ "$MONITOR" = true ]; then
    if [ "$NOHUP" = true ]; then
        # Run monitoring in background with nohup
        NOHUP_LOG="$CODE_DIR/monitoring.log"
        echo -e "${CYAN}Starting monitoring in background with nohup...${NC}"
        echo "Monitoring log: $NOHUP_LOG"
        echo ""

        # Export all necessary variables and functions for the subshell
        export GCS_BUCKET GCS_DATASET_PREFIX GCS_ACTIVATION_PREFIX
        export ZONES WORKERS_PER_ZONE TPU_TYPE MODEL ACTIVATION_TYPE MAX_SAMPLES_PER_STREAM
        export MONITOR_INTERVAL CODE_DIR
        export RED GREEN YELLOW BLUE CYAN NC
        export -f monitor_and_manage get_tpu_name get_tpu_status delete_tpu create_tpu launch_extraction
        export -f tpu_lifecycle_loop extraction_management_loop display_loop display_progress get_worker_progress get_gcs_shard_count

        # Export associative arrays (bash-specific)
        declare -p TPU_TO_STREAM > /tmp/tpu_to_stream_$$
        declare -p TPU_TO_ZONE > /tmp/tpu_to_zone_$$

        # Start monitoring in background
        nohup bash -c "
            source /tmp/tpu_to_stream_$$
            source /tmp/tpu_to_zone_$$
            monitor_and_manage
        " > "$NOHUP_LOG" 2>&1 &

        MONITOR_PID=$!

        # Clean up temp files
        rm -f /tmp/tpu_to_stream_$$ /tmp/tpu_to_zone_$$

        echo -e "${GREEN}✓ Monitoring started in background (PID: $MONITOR_PID)${NC}"
        echo ""
        echo "Monitor the log with:"
        echo "  tail -f $NOHUP_LOG"
        echo ""
        echo "Stop monitoring with:"
        echo "  kill $MONITOR_PID"
        echo ""
    else
        # Run monitoring in foreground
        echo -e "${CYAN}Auto-monitoring enabled - entering continuous monitoring mode...${NC}"
        echo ""
        sleep 2

        # Start monitoring loop
        monitor_and_manage
    fi
else
    # Show manual monitoring commands
    echo "Monitor extraction:"
    echo ""
    echo "  # Check worker log (example: zone ${ZONE_ARRAY[0]}, worker 0)"
    TPU_EXAMPLE=$(get_tpu_name "${ZONE_ARRAY[0]}" 0)
    echo "  gcloud compute tpus tpu-vm ssh $TPU_EXAMPLE --zone=${ZONE_ARRAY[0]} --command='tail -f ~/activation-extract/extraction.log'"
    echo ""
    echo "  # Check TPU status across all zones"
    echo "  bash scripts/manage_tpus.sh status --zones $ZONES"
    echo ""
    echo "  # Check GCS uploads"
    echo "  gsutil ls gs://$GCS_BUCKET/$GCS_ACTIVATION_PREFIX/"
    echo ""
    echo "  # Recreate preempted TPUs manually"
    echo "  bash scripts/manage_tpus.sh recreate-preempted --zones $ZONES --workers_per_zone $WORKERS_PER_ZONE"
    echo ""
    echo -e "${YELLOW}TIP: Use --monitor flag to enable automatic preemption handling${NC}"
    echo ""
fi