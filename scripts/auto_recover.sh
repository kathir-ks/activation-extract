#!/bin/bash
# ============================================================================
# Auto Recovery — Run extraction with automatic preemption recovery
# ============================================================================
#
# This script runs FROM YOUR LOCAL MACHINE (not on the TPU). It:
#   1. Sets up all TPU workers (clone repo, install deps)
#   2. Launches extraction on all workers via SSH
#   3. If preempted (SSH dies), waits for TPU to come back READY
#   4. Re-sets up workers and re-launches (checkpoint/resume handles the rest)
#   5. Repeats until extraction completes or max retries exceeded
#
# The extraction script uses GCS-backed checkpoints, so even if local disk
# is wiped on preemption, it can resume from the last saved checkpoint.
#
# Usage:
#   ./scripts/auto_recover.sh \
#       --tpu_name node-v5e-64-us-central1-a \
#       --zone us-central1-a \
#       --dataset_path gs://bucket/datasets/stream_000.jsonl \
#       --gcs_bucket my-bucket \
#       --gcs_prefix activations/run_001 \
#       --batch_size 32 \
#       --layers "0 5 10 15 20 23"
#
# ============================================================================

set -uo pipefail
# Note: not using set -e because we handle SSH exit codes manually

# ── Defaults ────────────────────────────────────────────────────────────────

TPU_NAME=""
ZONE=""
TOPOLOGY="v5litepod-64"
DATASET_PATH=""
GCS_BUCKET=""
GCS_PREFIX="activations"
BATCH_SIZE=32
MAX_TASKS=""
LAYERS=""
MODEL_PATH="Qwen/Qwen2.5-0.5B"
ACTIVATION_TYPE="residual"
BARRIER_PORT=5555
MAX_RETRIES=50
POLL_INTERVAL=60        # seconds between TPU status checks
SETUP_TIMEOUT=600       # seconds to wait for worker setup
BRANCH="main"
EXTRA_ARGS=""           # extra args passed through to multihost_extract.py
LOG_DIR="./recovery_logs"

# ── Parse arguments ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --tpu_name)          TPU_NAME="$2";          shift 2 ;;
        --zone)              ZONE="$2";              shift 2 ;;
        --topology)          TOPOLOGY="$2";          shift 2 ;;
        --dataset_path)      DATASET_PATH="$2";      shift 2 ;;
        --gcs_bucket)        GCS_BUCKET="$2";        shift 2 ;;
        --gcs_prefix)        GCS_PREFIX="$2";        shift 2 ;;
        --batch_size)        BATCH_SIZE="$2";        shift 2 ;;
        --max_tasks)         MAX_TASKS="$2";         shift 2 ;;
        --layers)            LAYERS="$2";            shift 2 ;;
        --model_path)        MODEL_PATH="$2";        shift 2 ;;
        --activation_type)   ACTIVATION_TYPE="$2";   shift 2 ;;
        --barrier_port)      BARRIER_PORT="$2";      shift 2 ;;
        --max_retries)       MAX_RETRIES="$2";       shift 2 ;;
        --poll_interval)     POLL_INTERVAL="$2";     shift 2 ;;
        --branch)            BRANCH="$2";            shift 2 ;;
        --extra_args)        EXTRA_ARGS="$2";        shift 2 ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validate ────────────────────────────────────────────────────────────────

if [[ -z "$TPU_NAME" || -z "$ZONE" || -z "$GCS_BUCKET" ]]; then
    echo "Error: --tpu_name, --zone, and --gcs_bucket are required"
    exit 1
fi

if [[ -z "$DATASET_PATH" ]]; then
    echo "Error: --dataset_path is required"
    exit 1
fi

# ── Setup ───────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/recovery_$(date '+%Y%m%d_%H%M%S').log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOGFILE"
}

# ── Functions ───────────────────────────────────────────────────────────────

get_tpu_state() {
    gcloud compute tpus tpu-vm describe "$TPU_NAME" \
        --zone="$ZONE" \
        --format="value(state)" 2>/dev/null || echo "NOT_FOUND"
}

wait_for_tpu_ready() {
    local attempt=0
    while true; do
        local state
        state=$(get_tpu_state)
        if [[ "$state" == "READY" ]]; then
            log "TPU is READY"
            return 0
        fi
        attempt=$((attempt + 1))
        if [[ $attempt -ge $((MAX_RETRIES * 60 / POLL_INTERVAL)) ]]; then
            log "ERROR: Timed out waiting for TPU to become READY (state: $state)"
            return 1
        fi
        log "  TPU state: $state — waiting ${POLL_INTERVAL}s (check $attempt)..."
        sleep "$POLL_INTERVAL"
    done
}

setup_workers() {
    log "Setting up all workers..."
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" \
        --worker=all \
        --command="BRANCH=$BRANCH bash -s" \
        < "$(dirname "$0")/setup_worker.sh" \
        2>&1 | tee -a "$LOGFILE"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        log "WARNING: Worker setup exited with code $rc"
        return $rc
    fi
    log "All workers set up successfully"
    return 0
}

build_extraction_cmd() {
    local cmd="cd ~/activation-extract && python3 multihost_extract.py"
    cmd+=" --topology $TOPOLOGY"
    cmd+=" --dataset_path $DATASET_PATH"
    cmd+=" --gcs_bucket $GCS_BUCKET"
    cmd+=" --gcs_prefix $GCS_PREFIX"
    cmd+=" --batch_size $BATCH_SIZE"
    cmd+=" --model_path $MODEL_PATH"
    cmd+=" --activation_type $ACTIVATION_TYPE"
    cmd+=" --enable_barrier_sync"
    cmd+=" --barrier_port $BARRIER_PORT"
    cmd+=" --upload_to_gcs"
    cmd+=" --enable_checkpointing"
    cmd+=" --checkpoint_gcs_bucket $GCS_BUCKET"
    cmd+=" --checkpoint_gcs_prefix ${GCS_PREFIX}/checkpoints"
    cmd+=" --verbose"

    if [[ -n "$LAYERS" ]]; then
        cmd+=" --layers_to_extract $LAYERS"
    fi
    if [[ -n "$MAX_TASKS" ]]; then
        cmd+=" --max_tasks $MAX_TASKS"
    fi
    if [[ -n "$EXTRA_ARGS" ]]; then
        cmd+=" $EXTRA_ARGS"
    fi

    echo "$cmd"
}

run_extraction() {
    local cmd
    cmd=$(build_extraction_cmd)
    log "Launching extraction on all workers..."
    log "  Command: $cmd"

    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" \
        --worker=all \
        --command="$cmd" \
        2>&1 | tee -a "$LOGFILE"
    return ${PIPESTATUS[0]}
}

# ── Main loop ───────────────────────────────────────────────────────────────

log "============================================================================"
log "AUTO RECOVERY — FSDP Extraction Pipeline"
log "============================================================================"
log "TPU:          $TPU_NAME"
log "Zone:         $ZONE"
log "Topology:     $TOPOLOGY"
log "Dataset:      $DATASET_PATH"
log "GCS:          gs://$GCS_BUCKET/$GCS_PREFIX"
log "Batch size:   $BATCH_SIZE"
log "Layers:       ${LAYERS:-all}"
log "Model:        $MODEL_PATH"
log "Max retries:  $MAX_RETRIES"
log "Log file:     $LOGFILE"
log "============================================================================"

attempt=0
while [[ $attempt -lt $MAX_RETRIES ]]; do
    attempt=$((attempt + 1))
    log ""
    log "━━━ Attempt $attempt / $MAX_RETRIES ━━━"

    # Step 1: Wait for TPU to be READY
    log "Checking TPU state..."
    if ! wait_for_tpu_ready; then
        log "FATAL: Could not get TPU to READY state. Exiting."
        exit 1
    fi

    # Step 2: Setup workers (clone/update repo, install deps)
    # Retry setup a few times in case SSH isn't ready immediately after READY
    setup_ok=false
    for setup_try in 1 2 3; do
        if setup_workers; then
            setup_ok=true
            break
        fi
        log "  Setup attempt $setup_try failed, retrying in 30s..."
        sleep 30
    done

    if [[ "$setup_ok" != "true" ]]; then
        log "ERROR: Failed to setup workers after 3 attempts"
        log "  Waiting ${POLL_INTERVAL}s before retrying..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Step 3: Run extraction
    run_extraction
    exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log ""
        log "============================================================================"
        log "EXTRACTION COMPLETED SUCCESSFULLY"
        log "============================================================================"
        log "Output: gs://$GCS_BUCKET/$GCS_PREFIX"
        log "Total attempts: $attempt"
        log "Log: $LOGFILE"
        exit 0
    fi

    # Diagnose the failure
    tpu_state=$(get_tpu_state)
    log "Extraction exited with code $exit_code (TPU state: $tpu_state)"

    if [[ "$tpu_state" == "PREEMPTED" || "$tpu_state" == "NOT_FOUND" ]]; then
        log "TPU was preempted. Will wait for it to come back..."
        # For preemptible TPUs that auto-restart, just wait.
        # For spot TPUs that need recreation, the operator should use
        # manage_tpus.sh in another terminal, or set up a separate
        # monitoring script.
        continue
    fi

    if [[ $exit_code -eq 255 ]]; then
        # SSH connection lost — likely preemption in progress
        log "SSH connection lost (exit 255). Likely preemption. Waiting..."
        sleep 30
        continue
    fi

    # Non-preemption failure
    log "WARNING: Non-preemption failure (exit code $exit_code)"
    log "  Check logs for errors. Retrying in ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done

log ""
log "============================================================================"
log "FATAL: Max retries ($MAX_RETRIES) exceeded"
log "============================================================================"
log "Last TPU state: $(get_tpu_state)"
log "Log: $LOGFILE"
exit 1
