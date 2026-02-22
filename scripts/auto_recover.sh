#!/bin/bash
# ============================================================================
# Auto Recovery Script for Preemptible TPU Extraction
# ============================================================================
#
# Runs from your LOCAL MACHINE (not on the TPU). Launches extraction on the
# TPU pod, monitors for preemption, waits for the TPU to come back, then
# re-clones and re-launches. GCS-backed checkpoints handle resume.
#
# Usage:
#   ./scripts/auto_recover.sh \
#       --tpu_name node-v5e-64-us-central1-a \
#       --zone us-central1-a \
#       --dataset_path gs://bucket/datasets/stream_000.jsonl \
#       --gcs_bucket fineweb-data-us-central1 \
#       --gcs_prefix activations/prod_run_001 \
#       --batch_size 32 \
#       --layers "0 5 10 15 20 23" \
#       --max_retries 50
#
# Exit codes:
#   0  — Extraction completed successfully
#   1  — Permanent failure (non-preemption error)
#   2  — Max retries exceeded
# ============================================================================

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────

TPU_NAME=""
ZONE=""
DATASET_PATH=""
GCS_BUCKET=""
GCS_PREFIX="activations/prod_run"
BATCH_SIZE=32
LAYERS=""
MAX_TASKS=""
MAX_RETRIES=50
POLL_INTERVAL=60        # seconds between TPU status checks
TOPOLOGY="v5litepod-64"
MODEL_PATH="Qwen/Qwen2.5-0.5B"
BARRIER_PORT=5555
BRANCH="main"
CHECKPOINT_GCS_BUCKET=""  # defaults to GCS_BUCKET if empty
MAX_SEQ_LENGTH=2048

# ── Parse arguments ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --tpu_name)           TPU_NAME="$2";           shift 2 ;;
        --zone)               ZONE="$2";               shift 2 ;;
        --dataset_path)       DATASET_PATH="$2";       shift 2 ;;
        --gcs_bucket)         GCS_BUCKET="$2";         shift 2 ;;
        --gcs_prefix)         GCS_PREFIX="$2";         shift 2 ;;
        --batch_size)         BATCH_SIZE="$2";         shift 2 ;;
        --layers)             LAYERS="$2";             shift 2 ;;
        --max_tasks)          MAX_TASKS="$2";          shift 2 ;;
        --max_retries)        MAX_RETRIES="$2";        shift 2 ;;
        --poll_interval)      POLL_INTERVAL="$2";      shift 2 ;;
        --topology)           TOPOLOGY="$2";           shift 2 ;;
        --model_path)         MODEL_PATH="$2";         shift 2 ;;
        --barrier_port)       BARRIER_PORT="$2";       shift 2 ;;
        --branch)             BRANCH="$2";             shift 2 ;;
        --checkpoint_gcs_bucket) CHECKPOINT_GCS_BUCKET="$2"; shift 2 ;;
        --max_seq_length)     MAX_SEQ_LENGTH="$2";     shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validate required args ───────────────────────────────────────────────────

if [[ -z "$TPU_NAME" || -z "$ZONE" || -z "$DATASET_PATH" || -z "$GCS_BUCKET" ]]; then
    echo "Error: Required arguments: --tpu_name, --zone, --dataset_path, --gcs_bucket"
    exit 1
fi

# Default checkpoint bucket to GCS bucket
CHECKPOINT_GCS_BUCKET="${CHECKPOINT_GCS_BUCKET:-$GCS_BUCKET}"

# ── Helper functions ─────────────────────────────────────────────────────────

log() {
    echo "[auto_recover] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

get_tpu_state() {
    # Returns the TPU state (READY, PREEMPTED, etc.) or "NOT_FOUND"
    local state
    state=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
        --zone="$ZONE" \
        --format='get(state)' 2>/dev/null) || echo "NOT_FOUND"
    echo "${state:-NOT_FOUND}"
}

wait_for_tpu_ready() {
    log "Waiting for TPU '$TPU_NAME' to become READY..."
    local attempts=0
    local max_wait_attempts=$(( 60 * 60 / POLL_INTERVAL ))  # 1 hour max wait

    while true; do
        local state
        state=$(get_tpu_state)

        case "$state" in
            READY)
                log "TPU is READY"
                return 0
                ;;
            PREEMPTED)
                log "TPU is PREEMPTED — waiting for auto-restart... (attempt $attempts)"
                ;;
            CREATING|STARTING|RESTARTING)
                log "TPU is $state — waiting... (attempt $attempts)"
                ;;
            NOT_FOUND)
                log "TPU not found — it may need manual recreation"
                log "  Run: gcloud compute tpus tpu-vm create $TPU_NAME --zone=$ZONE ..."
                return 1
                ;;
            *)
                log "TPU state: $state (attempt $attempts)"
                ;;
        esac

        attempts=$((attempts + 1))
        if [[ $attempts -ge $max_wait_attempts ]]; then
            log "ERROR: TPU not ready after $attempts attempts ($((attempts * POLL_INTERVAL))s)"
            return 1
        fi

        sleep "$POLL_INTERVAL"
    done
}

setup_all_workers() {
    log "Setting up all workers (clone, install deps)..."
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" \
        --worker=all \
        --command="bash -s" < "$(dirname "$0")/setup_worker.sh" 2>&1 | \
        while IFS= read -r line; do echo "  [setup] $line"; done

    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        log "WARNING: setup_worker.sh exited with code $rc on some workers"
    fi
    return $rc
}

build_extraction_cmd() {
    local cmd="cd ~/activation-extract && python3 multihost_extract.py"
    cmd+=" --topology $TOPOLOGY"
    cmd+=" --dataset_path $DATASET_PATH"
    cmd+=" --gcs_bucket $GCS_BUCKET"
    cmd+=" --gcs_prefix $GCS_PREFIX"
    cmd+=" --batch_size $BATCH_SIZE"
    cmd+=" --max_seq_length $MAX_SEQ_LENGTH"
    cmd+=" --model_path $MODEL_PATH"
    cmd+=" --enable_barrier_sync"
    cmd+=" --barrier_port $BARRIER_PORT"
    cmd+=" --upload_to_gcs"
    cmd+=" --enable_checkpointing"
    cmd+=" --checkpoint_gcs_bucket $CHECKPOINT_GCS_BUCKET"

    if [[ -n "$LAYERS" ]]; then
        cmd+=" --layers_to_extract $LAYERS"
    fi
    if [[ -n "$MAX_TASKS" ]]; then
        cmd+=" --max_tasks $MAX_TASKS"
    fi

    echo "$cmd"
}

# ── Main loop ────────────────────────────────────────────────────────────────

log "============================================================================"
log "AUTO RECOVERY for PREEMPTIBLE TPU EXTRACTION"
log "============================================================================"
log "TPU:        $TPU_NAME"
log "Zone:       $ZONE"
log "Topology:   $TOPOLOGY"
log "Dataset:    $DATASET_PATH"
log "GCS:        gs://$GCS_BUCKET/$GCS_PREFIX"
log "Checkpoint: gs://$CHECKPOINT_GCS_BUCKET/checkpoints/"
log "Layers:     ${LAYERS:-all}"
log "Max retries: $MAX_RETRIES"
log "============================================================================"

attempt=0

while [[ $attempt -lt $MAX_RETRIES ]]; do
    attempt=$((attempt + 1))
    log ""
    log "========== ATTEMPT $attempt / $MAX_RETRIES =========="

    # Step 1: Ensure TPU is ready
    if ! wait_for_tpu_ready; then
        log "ERROR: TPU not available. Exiting."
        exit 1
    fi

    # Step 2: Setup workers (clone/install)
    # Give TPU a moment to fully initialize after becoming READY
    sleep 10
    if ! setup_all_workers; then
        log "WARNING: Worker setup had issues, attempting extraction anyway..."
    fi

    # Step 3: Build and run extraction
    EXTRACT_CMD=$(build_extraction_cmd)
    log "Launching extraction..."
    log "  CMD: $EXTRACT_CMD"

    set +e
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" \
        --worker=all \
        --command="$EXTRACT_CMD" 2>&1 | \
        while IFS= read -r line; do echo "  $line"; done
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    log "Extraction exited with code: $EXIT_CODE"

    # Step 4: Interpret exit code
    case $EXIT_CODE in
        0)
            log "============================================================================"
            log "EXTRACTION COMPLETED SUCCESSFULLY"
            log "============================================================================"
            log "Output: gs://$GCS_BUCKET/$GCS_PREFIX/"
            log "Total attempts: $attempt"
            exit 0
            ;;
        255)
            # SSH connection lost — likely preemption
            log "SSH connection lost (exit 255) — likely preemption"
            log "Will wait for TPU and retry..."
            # Small delay before checking state
            sleep 10
            ;;
        *)
            # Check if TPU was preempted
            state=$(get_tpu_state)
            if [[ "$state" == "PREEMPTED" || "$state" == "NOT_FOUND" ]]; then
                log "TPU state is $state — treating as preemption"
                log "Will wait for TPU and retry..."
                sleep 10
            else
                log "ERROR: Non-preemption failure (exit code $EXIT_CODE, TPU state: $state)"
                log "This may be a code bug. Check logs above."
                log "Retrying in case it's transient..."
                sleep 30
            fi
            ;;
    esac
done

log "============================================================================"
log "ERROR: Max retries ($MAX_RETRIES) exceeded"
log "============================================================================"
exit 2
