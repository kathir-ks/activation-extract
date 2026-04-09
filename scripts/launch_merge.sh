#!/bin/bash
# =============================================================================
# Launch distributed shard merge across v6e-64 TPU workers
#
# Merges paired host shards (448-dim halves) into full 896-dim activations.
# Uses all 16 v6e workers for parallel processing.
#
# Usage:
#   bash scripts/launch_merge.sh
# =============================================================================

set -uo pipefail

TPU_NAME="node-v5e-64-europe-west4-b"
ZONE="europe-west4-b"
NUM_WORKERS=16

SRC_GCS="gs://arc-data-europe-west4/activations/layer19_gridchunk_50k_v5litepod-64"
DST_GCS="gs://arc-data-europe-west4/activations/layer19_merged_50k"
LAYER_INDEX=19
NUM_PAIRS=8

REPO_DIR="/home/kathirks_gc/activation-extract"
TARBALL="/tmp/activation-extract.tar.gz"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Create tarball
log "Creating tarball..."
cd /home/kathirks_gc

# Sync sae/ from worktree
rsync -a --delete \
    /home/kathirks_gc/sae-worktree/sae/ \
    /home/kathirks_gc/activation-extract/sae/ 2>/dev/null || true

tar czf "$TARBALL" \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='activations' --exclude='checkpoints' --exclude='nohup.out' \
    --exclude='*.log' --exclude='sae_logs' --exclude='sae_checkpoints' \
    activation-extract/
cd "$REPO_DIR"
log "Tarball: $(ls -lh $TARBALL | awk '{print $5}')"

# Deploy to all workers (batches of 4)
log "Deploying to $NUM_WORKERS workers..."
for batch_start in 0 4 8 12; do
    batch_end=$((batch_start + 3))
    if [ $batch_end -ge $NUM_WORKERS ]; then
        batch_end=$((NUM_WORKERS - 1))
    fi
    for w in $(seq $batch_start $batch_end); do
        (
            gcloud compute tpus tpu-vm scp "$TARBALL" \
                "$TPU_NAME":~/activation-extract.tar.gz \
                --zone="$ZONE" --worker="$w" 2>/dev/null
            gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker="$w" --command="
                cd ~ && rm -rf activation-extract && tar xzf activation-extract.tar.gz &&
                pip install -q gcsfs 'google-cloud-storage>=2.14.0' numpy tqdm ml_dtypes 2>/dev/null &&
                echo 'Worker $w ready'
            " 2>/dev/null | tail -1
        ) &
    done
    wait
    log "  Workers $batch_start-$batch_end deployed"
done

# Launch merge on all workers
log "Launching merge across $NUM_WORKERS workers..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all --command="
    cd ~/activation-extract &&
    WORKER_ID=\$(curl -s 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number' -H 'Metadata-Flavor: Google' 2>/dev/null || echo '0') &&
    nohup python3 -u scripts/merge_sharded_activations.py \
        --src_gcs '$SRC_GCS' \
        --dst_gcs '$DST_GCS' \
        --layer_index $LAYER_INDEX \
        --num_pairs $NUM_PAIRS \
        --worker_id \$WORKER_ID \
        --num_workers $NUM_WORKERS \
        > merge.log 2>&1 &
    echo \"Worker \$WORKER_ID: merge launched (PID \$!)\"
" 2>/dev/null | tail -20

log "All workers launched. Monitor with:"
log "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0 --command='tail -20 ~/activation-extract/merge.log'"
log ""
log "Expected time: ~60-70 minutes"
log "Output: $DST_GCS/"
