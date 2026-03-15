#!/bin/bash
# =============================================================================
# Resilient extraction launcher with preemption recovery
#
# Run with nohup:
#   nohup bash scripts/launch_extraction.sh > launch.log 2>&1 &
#
# This script:
# 1. Sets up all TPU workers (code + deps)
# 2. Launches multihost extraction
# 3. Monitors for preemption
# 4. On preemption: waits for TPU recreation, then re-setup + relaunch
#    (extraction resumes from GCS checkpoint automatically)
# =============================================================================

set -uo pipefail

TPU_NAME="node-v5e-64-europe-west4-b"
ZONE="europe-west4-b"
NUM_WORKERS=16
TARBALL="/tmp/activation-extract.tar.gz"
REPO_DIR="/home/kathirks_gc/activation-extract"

# Extraction config
LAYER=19
GCS_PREFIX="activations/layer19_gridchunk_50k"
CHECKPOINT_PREFIX="checkpoints/gridchunk_layer19"

POLL_INTERVAL=300  # Check every 5 minutes
MAX_RECOVERY_WAIT=1800  # Wait up to 30 min for TPU recreation

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_tpu_status() {
    gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" \
        --format="value(state)" 2>/dev/null
}

get_worker0_ip() {
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0 \
        --command="hostname -I | awk '{print \$1}'" 2>/dev/null | tail -1
}

check_extraction_running() {
    local result
    result=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0 \
        --command="pgrep -af 'python3.*multihost_extract' 2>/dev/null | grep -v pgrep | head -1 || echo ''" 2>/dev/null | tail -1)
    [[ -n "$result" ]]
}

check_extraction_completed() {
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=1 \
        --command="grep -q 'EXTRACTION COMPLETE' ~/activation-extract/extraction.log 2>/dev/null && echo 'DONE' || echo 'NOT_DONE'" \
        2>/dev/null | tail -1
}

create_tarball() {
    log "Creating tarball..."
    cd /home/kathirks_gc
    tar czf "$TARBALL" \
        --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='activations' --exclude='checkpoints' --exclude='nohup.out' \
        --exclude='extraction.log' --exclude='launch.log' \
        activation-extract/
    cd "$REPO_DIR"
    log "Tarball created: $(ls -lh $TARBALL | awk '{print $5}')"
}

setup_worker() {
    local w=$1
    gcloud compute tpus tpu-vm scp "$TARBALL" \
        "$TPU_NAME":~/activation-extract.tar.gz \
        --zone="$ZONE" --worker="$w" 2>/dev/null

    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker="$w" --command="
        cd ~ && rm -rf activation-extract && tar xzf activation-extract.tar.gz &&
        pip install -q 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>/dev/null &&
        pip install -q flax 'transformers>=4.38.0' 'jinja2>=3.1.0' torch tqdm gcsfs 'google-cloud-storage>=2.14.0' numpy 2>/dev/null &&
        echo 'Worker $w ready'
    " 2>/dev/null | tail -1

    log "  Worker $w set up"
}

setup_all_workers() {
    log "Setting up all $NUM_WORKERS workers..."
    create_tarball

    for batch_start in 0 4 8 12; do
        local batch_end=$((batch_start + 3))
        if [ $batch_end -ge $NUM_WORKERS ]; then
            batch_end=$((NUM_WORKERS - 1))
        fi
        log "  Batch: workers $batch_start-$batch_end"
        for w in $(seq $batch_start $batch_end); do
            setup_worker "$w" &
        done
        wait
    done
    log "All workers set up"
}

launch_extraction() {
    # Detect worker 0 IP for barrier sync
    local barrier_host
    barrier_host=$(get_worker0_ip)
    log "Worker 0 IP (barrier host): $barrier_host"

    log "Launching extraction on all workers..."
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all --command="
        cd ~/activation-extract &&
        { [ -f extraction.log ] && mv extraction.log extraction.log.prev || true; } &&
        rm -f checkpoints/*.json &&
        nohup python3 -u multihost_extract.py \
            --topology v5litepod-64 \
            --model_path KathirKs/qwen-2.5-0.5b \
            --dataset_path gs://arc-data-europe-west4/dataset_streams/combined_50k.jsonl \
            --max_tasks 50000 \
            --pipeline grid_chunking \
            --predictions_per_task 8 \
            --layers_to_extract $LAYER \
            --activation_type residual \
            --batch_size 16 \
            --max_seq_length 5120 \
            --upload_to_gcs \
            --gcs_bucket arc-data-europe-west4 \
            --gcs_prefix $GCS_PREFIX \
            --shard_size_gb 1.0 \
            --delete_local_after_upload \
            --enable_barrier_sync \
            --barrier_controller_host $barrier_host \
            --barrier_port 5555 \
            --checkpoint_gcs_prefix $CHECKPOINT_PREFIX \
            > extraction.log 2>&1 &
        echo 'Launched'
    " 2>/dev/null | tail -20

    log "Extraction launched"
}

wait_for_tpu_ready() {
    log "Waiting for TPU to become READY..."
    local elapsed=0
    while [ $elapsed -lt $MAX_RECOVERY_WAIT ]; do
        local status
        status=$(check_tpu_status)
        if [ "$status" = "READY" ]; then
            log "TPU is READY"
            sleep 30
            return 0
        fi
        log "  TPU status: ${status:-UNKNOWN} (waited ${elapsed}s)"
        sleep 30
        elapsed=$((elapsed + 30))
    done
    log "ERROR: TPU did not become READY within ${MAX_RECOVERY_WAIT}s"
    return 1
}

# =============================================================================
# Main
# =============================================================================

log "=========================================="
log "Resilient Extraction Launcher"
log "=========================================="
log "TPU: $TPU_NAME ($ZONE)"
log "Layer: $LAYER (residual)"
log "Pipeline: grid_chunking (chunk_size=5120)"
log "GCS output: gs://arc-data-europe-west4/$GCS_PREFIX/"
log "Checkpoints: gs://arc-data-europe-west4/$CHECKPOINT_PREFIX/"
log "Poll interval: ${POLL_INTERVAL}s"
log "=========================================="

# Check current state
tpu_status=$(check_tpu_status)
log "TPU status: $tpu_status"

if [ "$tpu_status" = "READY" ]; then
    if check_extraction_running; then
        log "Extraction already running. Entering monitor mode."
    else
        completed=$(check_extraction_completed 2>/dev/null || echo "UNKNOWN")
        if [ "$completed" = "DONE" ]; then
            log "Extraction already completed!"
            exit 0
        fi
        log "TPU ready but extraction not running. Launching..."
        # Don't re-setup if code already there (saves time on first run)
        gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0 \
            --command="test -f ~/activation-extract/multihost_extract.py && echo EXISTS || echo MISSING" \
            2>/dev/null | tail -1 | grep -q "MISSING" && setup_all_workers
        launch_extraction
    fi
else
    log "TPU not ready. Waiting for recovery..."
    wait_for_tpu_ready
    setup_all_workers
    launch_extraction
fi

# Monitor loop
RECOVERY_COUNT=0
while true; do
    sleep "$POLL_INTERVAL"

    tpu_status=$(check_tpu_status)

    if [ "$tpu_status" != "READY" ]; then
        log "!! TPU PREEMPTED (status: ${tpu_status:-GONE})"

        if wait_for_tpu_ready; then
            RECOVERY_COUNT=$((RECOVERY_COUNT + 1))
            log "Recovery #$RECOVERY_COUNT: Re-setup + relaunch..."
            setup_all_workers
            launch_extraction
            log "Recovery #$RECOVERY_COUNT done. Resuming from GCS checkpoint."
        else
            log "FATAL: TPU did not recover. Exiting."
            exit 1
        fi
        continue
    fi

    if ! check_extraction_running; then
        completed=$(check_extraction_completed 2>/dev/null || echo "UNKNOWN")
        if [ "$completed" = "DONE" ]; then
            log "=========================================="
            log "EXTRACTION COMPLETED SUCCESSFULLY!"
            log "Total recoveries: $RECOVERY_COUNT"
            log "=========================================="
            exit 0
        else
            log "WARNING: Process died but not completed."
            # Check if code still exists (TPU may have been recreated with fresh VMs)
            code_check=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0 \
                --command="test -f ~/activation-extract/multihost_extract.py && echo EXISTS || echo MISSING" \
                2>/dev/null | tail -1)
            if [ "$code_check" = "MISSING" ]; then
                log "Code missing on workers (TPU recreated?). Re-deploying..."
                RECOVERY_COUNT=$((RECOVERY_COUNT + 1))
                setup_all_workers
            fi
            log "Relaunching..."
            launch_extraction
        fi
    else
        log "OK: Extraction running. (recoveries: $RECOVERY_COUNT)"
    fi
done
