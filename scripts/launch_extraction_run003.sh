#!/bin/bash
# =============================================================================
# Extraction Run 003: Layers 15 + 22, Grid-Chunk Pipeline, Batch 2 (50K tasks)
#
# Extracts residual activations from layers 15 and 22 of KathirKs/qwen-2.5-0.5b
# using the grid_chunking pipeline (chunk_size=5120) on v5litepod-64.
#
# Dataset: Next 50K tasks (50001-100000) from barc0/200k_HEAVY
#   -> gs://arc-data-europe-west4/dataset_streams/combined_50k_batch2.jsonl
#
# Memory optimization:
#   - fsdp_size=4 (model_size=1): no hidden_dim splitting, full 896-dim per host
#   - batch_size=16: fits 2 layers x 5120 tokens x 896 dim in HBM
#   - shard_size_gb=1.0: flush to disk/GCS frequently, keep memory bounded
#   - delete_local_after_upload: free disk after GCS upload
#
# Run with nohup:
#   nohup bash scripts/launch_extraction_run003.sh > launch_run003.log 2>&1 &
# =============================================================================

set -uo pipefail

TPU_NAME="node-v5e-64-europe-west4-b"
ZONE="europe-west4-b"
NUM_WORKERS=16
TARBALL="/tmp/activation-extract.tar.gz"
REPO_DIR="/home/kathirks_gc/activation-extract"

# Extraction config
LAYERS="15 22"
MODEL_PATH="KathirKs/qwen-2.5-0.5b"
DATASET_PATH="gs://arc-data-europe-west4/dataset_streams/combined_50k_batch2.jsonl"
GCS_BUCKET="arc-data-europe-west4"
GCS_PREFIX="activations/layer15_22_gridchunk_50k_batch2"
CHECKPOINT_PREFIX="checkpoints/gridchunk_layer15_22_batch2"

# Pipeline config
PIPELINE="grid_chunking"
MAX_SEQ_LENGTH=5120
BATCH_SIZE=16           # Global batch across 16 hosts (1 per host)
FSDP_SIZE=4             # All 4 local devices on FSDP axis -> model_size=1, no hidden_dim split
PREDICTIONS_PER_TASK=8
MAX_TASKS=50000
SHARD_SIZE_GB=1.0

POLL_INTERVAL=300       # Check every 5 minutes
MAX_RECOVERY_WAIT=1800  # Wait up to 30 min for TPU recreation

BRANCH="production/extraction-v2"

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
    log "Creating tarball from $BRANCH branch..."
    cd /home/kathirks_gc

    # Sync sae/ from worktree if it exists
    if [ -d /home/kathirks_gc/sae-worktree/sae/ ]; then
        rsync -a --delete \
            /home/kathirks_gc/sae-worktree/sae/ \
            /home/kathirks_gc/activation-extract/sae/
    fi

    tar czf "$TARBALL" \
        --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='activations' --exclude='checkpoints' --exclude='nohup.out' \
        --exclude='extraction.log' --exclude='launch.log' --exclude='training.log' \
        --exclude='sae_logs' --exclude='sae_checkpoints' --exclude='data' \
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
        pip install -q flax optax 'transformers>=4.38.0' 'jinja2>=3.1.0' torch tqdm gcsfs 'google-cloud-storage>=2.14.0' numpy ml_dtypes 2>/dev/null &&
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
    # Kill any stale processes
    log "Killing stale processes..."
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all \
        --command="pkill -f 'python3.*multihost_extract' 2>/dev/null; sleep 1; fuser -k 5555/tcp 2>/dev/null; true" \
        2>/dev/null || true
    sleep 5

    # Detect worker 0 IP for barrier sync
    local barrier_host
    barrier_host=$(get_worker0_ip)
    log "Worker 0 IP (barrier host): $barrier_host"

    log "Launching extraction on all workers..."
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all --command="
        cd ~/activation-extract &&
        { [ -f extraction.log ] && mv extraction.log extraction.log.prev || true; } &&
        nohup python3 -u multihost_extract.py \
            --topology v5litepod-64 \
            --model_path $MODEL_PATH \
            --dataset_path $DATASET_PATH \
            --max_tasks $MAX_TASKS \
            --pipeline $PIPELINE \
            --predictions_per_task $PREDICTIONS_PER_TASK \
            --layers_to_extract $LAYERS \
            --activation_type residual \
            --batch_size $BATCH_SIZE \
            --max_seq_length $MAX_SEQ_LENGTH \
            --fsdp_size $FSDP_SIZE \
            --upload_to_gcs \
            --gcs_bucket $GCS_BUCKET \
            --gcs_prefix $GCS_PREFIX \
            --shard_size_gb $SHARD_SIZE_GB \
            --delete_local_after_upload \
            --enable_barrier_sync \
            --barrier_controller_host $barrier_host \
            --barrier_port 5555 \
            --checkpoint_gcs_prefix $CHECKPOINT_PREFIX \
            --verbose \
            > extraction.log 2>&1 &
        echo 'Launched on \$(hostname)'
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
log "Extraction Run 003 Launcher"
log "=========================================="
log "TPU: $TPU_NAME ($ZONE)"
log "Model: $MODEL_PATH"
log "Layers: $LAYERS (residual)"
log "Pipeline: $PIPELINE (chunk_size=$MAX_SEQ_LENGTH)"
log "Dataset: $DATASET_PATH"
log "Batch size: $BATCH_SIZE (fsdp_size=$FSDP_SIZE)"
log "GCS output: gs://$GCS_BUCKET/$GCS_PREFIX/"
log "Checkpoints: gs://$GCS_BUCKET/$CHECKPOINT_PREFIX/"
log "=========================================="

# Verify dataset exists on GCS
log "Verifying dataset exists..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0 \
    --command="gcloud storage ls -l $DATASET_PATH 2>/dev/null || echo 'DATASET_MISSING'" \
    2>/dev/null | tail -2 | while read -r line; do
    if echo "$line" | grep -q "DATASET_MISSING"; then
        log "ERROR: Dataset not found at $DATASET_PATH"
        log "Run 'bash scripts/create_dataset_batch2.sh' first"
        exit 1
    fi
    log "  Dataset: $line"
done

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
        log "TPU ready. Setting up and launching..."
        setup_all_workers
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
            log "EXTRACTION RUN 003 COMPLETED!"
            log "Layers: $LAYERS"
            log "Output: gs://$GCS_BUCKET/$GCS_PREFIX/"
            log "Total recoveries: $RECOVERY_COUNT"
            log "=========================================="
            exit 0
        else
            log "WARNING: Process died but not completed."
            code_check=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0 \
                --command="test -f ~/activation-extract/multihost_extract.py && echo EXISTS || echo MISSING" \
                2>/dev/null | tail -1)
            if [ "$code_check" = "MISSING" ]; then
                log "Code missing (TPU recreated?). Re-deploying..."
                RECOVERY_COUNT=$((RECOVERY_COUNT + 1))
                setup_all_workers
            fi
            log "Relaunching..."
            launch_extraction
        fi
    else
        # Show latest progress
        latest=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0 \
            --command="tail -3 ~/activation-extract/extraction.log 2>/dev/null | head -1 || echo ''" 2>/dev/null | tail -1)
        log "OK: Running. Latest: $latest (recoveries: $RECOVERY_COUNT)"
    fi
done
