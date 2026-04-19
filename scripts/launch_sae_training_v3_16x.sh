#!/bin/bash
# =============================================================================
# SAE Training Launcher v3 — 16x expansion
#
# Changes from v2:
#   - dict_size: 7168 (8x) -> 14336 (16x)
#   - k: 32 kept (same active count; sparser %: 0.45% -> 0.22%)
#   - New checkpoint prefix: layer19_topk_896d_v3_16x
#
# Run with nohup:
#   nohup bash scripts/launch_sae_training_v3_16x.sh > sae_training_v3_16x.log 2>&1 &
# =============================================================================

set -uo pipefail

TPU_NAME="node-v5e-64-europe-west4-b"
ZONE="europe-west4-b"
NUM_WORKERS=16
TARBALL="/tmp/activation-extract.tar.gz"
REPO_DIR="/home/kathirks_gc/activation-extract"

# SAE training config
ARCHITECTURE="topk"
HIDDEN_DIM=896
DICT_SIZE=14336       # 16x expansion (896 * 16)
K=32
BATCH_SIZE=4096       # Global batch (256 per host, 64 per device)
NUM_STEPS=150000
LEARNING_RATE=3e-4
LR_WARMUP=1000
DTYPE="bfloat16"

# Dead neuron resampling: stop at step 75K to prevent late-training divergence
DEAD_NEURON_RESAMPLE_STEPS=25000
DEAD_NEURON_RESAMPLE_UNTIL=75000

# Data source: merged 896-dim activations (paired host shards concatenated)
GCS_PATH="gs://arc-data-europe-west4/activations/layer19_merged_50k"
LAYER_INDEX=19

# Logging and checkpointing
LOG_DIR="./sae_logs"
CHECKPOINT_DIR="./sae_checkpoints"
CHECKPOINT_EVERY=5000
LOG_EVERY=100
EVAL_EVERY=1000

POLL_INTERVAL=300
MAX_RECOVERY_WAIT=7200

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

check_training_running() {
    local result
    result=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0 \
        --command="pgrep -af 'python3.*sae.scripts.train' 2>/dev/null | grep -v pgrep | head -1 || echo ''" 2>/dev/null | tail -1)
    [[ -n "$result" ]]
}

check_training_completed() {
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0 \
        --command="grep -q 'Training complete' ~/activation-extract/training.log 2>/dev/null && echo 'DONE' || echo 'NOT_DONE'" \
        2>/dev/null | tail -1
}

create_tarball() {
    log "Creating tarball..."
    cd /home/kathirks_gc

    # Sync sae/ from worktree into main repo for deployment
    rsync -a --delete \
        /home/kathirks_gc/sae-worktree/sae/ \
        /home/kathirks_gc/activation-extract/sae/

    tar czf "$TARBALL" \
        --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='activations' --exclude='checkpoints' --exclude='nohup.out' \
        --exclude='extraction.log' --exclude='launch.log' --exclude='training.log' \
        --exclude='sae_logs' --exclude='sae_checkpoints' \
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

launch_training() {
    local barrier_host
    barrier_host=$(get_worker0_ip)
    log "Worker 0 IP (barrier host): $barrier_host"

    log "Launching SAE training v2 on all workers..."
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all --command="
        cd ~/activation-extract &&
        export PYTHONUNBUFFERED=1 &&
        nohup python3 -u -m sae.scripts.train \
            --architecture $ARCHITECTURE \
            --hidden_dim $HIDDEN_DIM \
            --dict_size $DICT_SIZE \
            --k $K \
            --dtype $DTYPE \
            --source_type pickle \
            --data_dir /tmp/unused \
            --gcs_path '$GCS_PATH' \
            --layer_index $LAYER_INDEX \
            --batch_size $BATCH_SIZE \
            --num_steps $NUM_STEPS \
            --learning_rate $LEARNING_RATE \
            --lr_warmup_steps $LR_WARMUP \
            --lr_decay cosine \
            --shuffle_buffer_size 262144 \
            --dead_neuron_resample_steps $DEAD_NEURON_RESAMPLE_STEPS \
            --dead_neuron_resample_until $DEAD_NEURON_RESAMPLE_UNTIL \
            --log_every $LOG_EVERY \
            --eval_every $EVAL_EVERY \
            --log_dir $LOG_DIR \
            --checkpoint_dir $CHECKPOINT_DIR \
            --checkpoint_every $CHECKPOINT_EVERY \
            --upload_checkpoints_to_gcs \
            --checkpoint_gcs_bucket arc-data-europe-west4 \
            --checkpoint_gcs_prefix sae_checkpoints/layer19_topk_896d_v3_16x \
            --mesh_type data_parallel \
            --enable_barrier_sync \
            --barrier_controller_host $barrier_host \
            --barrier_port 5555 \
            > training.log 2>&1 &
        echo 'Launched on worker'
    " 2>/dev/null | tail -20

    log "Training launched"
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
log "SAE Training Launcher v3 16x (v5e-64)"
log "=========================================="
log "TPU: $TPU_NAME ($ZONE)"
log "Architecture: $ARCHITECTURE (hidden=$HIDDEN_DIM, dict=$DICT_SIZE, k=$K)"
log "Data: $GCS_PATH (layer $LAYER_INDEX)"
log "Training: batch=$BATCH_SIZE, steps=$NUM_STEPS, lr=$LEARNING_RATE"
log "Dead neuron resample: every ${DEAD_NEURON_RESAMPLE_STEPS} steps, stop at step $DEAD_NEURON_RESAMPLE_UNTIL"
log "Checkpoint prefix: sae_checkpoints/layer19_topk_896d_v3_16x"
log "Dtype: $DTYPE"
log "=========================================="

tpu_status=$(check_tpu_status)
log "TPU status: $tpu_status"

if [ "$tpu_status" = "READY" ]; then
    if check_training_running; then
        log "Training already running. Entering monitor mode."
    else
        completed=$(check_training_completed 2>/dev/null || echo "UNKNOWN")
        if [ "$completed" = "DONE" ]; then
            log "Training already completed!"
            exit 0
        fi
        log "TPU ready but training not running. Setting up and launching..."
        setup_all_workers
        launch_training
    fi
else
    log "TPU not ready. Waiting for recovery..."
    wait_for_tpu_ready
    setup_all_workers
    launch_training
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
            launch_training
            log "Recovery #$RECOVERY_COUNT done. Resuming from checkpoint."
        else
            log "FATAL: TPU did not recover. Exiting."
            exit 1
        fi
        continue
    fi

    if ! check_training_running; then
        completed=$(check_training_completed 2>/dev/null || echo "UNKNOWN")
        if [ "$completed" = "DONE" ]; then
            log "=========================================="
            log "SAE TRAINING V3 16x COMPLETED SUCCESSFULLY!"
            log "Total recoveries: $RECOVERY_COUNT"
            log "=========================================="
            exit 0
        else
            log "WARNING: Process died but not completed. Relaunching..."
            launch_training
        fi
    else
        # Show latest training metrics
        latest=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0 \
            --command="tail -1 ~/activation-extract/training.log 2>/dev/null || echo ''" 2>/dev/null | tail -1)
        log "OK: Training running. Latest: $latest (recoveries: $RECOVERY_COUNT)"
    fi
done
