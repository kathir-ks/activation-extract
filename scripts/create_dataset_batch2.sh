#!/bin/bash
# =============================================================================
# Create dataset batch 2 (tasks 50001-100000) on TPU worker 0
# and upload directly to GCS from europe-west4
#
# Usage (from control machine):
#   bash scripts/create_dataset_batch2.sh
# =============================================================================

set -e

TPU_NAME="node-v5e-64-europe-west4-b"
ZONE="europe-west4-b"
WORK_DIR="activation-extract"
GCS_DEST="gs://arc-data-europe-west4/dataset_streams/combined_50k_batch2.jsonl"

echo "==========================================="
echo "  Create Dataset Batch 2 (tasks 50001-100000)"
echo "==========================================="
echo "  TPU: $TPU_NAME ($ZONE)"
echo "  Output: $GCS_DEST"
echo ""

# Step 1: Ensure latest code is on worker 0
echo "Step 1: Pulling latest code on worker 0..."
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE \
    --worker=0 \
    --command="
        cd ~/$WORK_DIR && git fetch --all -q && git reset --hard origin/fix/dynamic-seq-length -q
        echo 'Code updated on worker 0'
    "

# Step 2: Install deps and create dataset
echo ""
echo "Step 2: Creating dataset on worker 0 (europe-west4)..."
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE \
    --worker=0 \
    --command="
        set -e
        cd ~/$WORK_DIR

        # Install datasets lib if needed
        pip3 install -q datasets tqdm gcsfs google-cloud-storage 2>&1 | tail -1

        # Create the dataset: tasks 50000-100000 (next 50K after batch 1)
        echo 'Converting HF dataset (tasks 50001-100000)...'
        python3 convert_hf_to_arc_format.py \
            --dataset_name 'barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems' \
            --column_name 'examples' \
            --output_file 'data/combined_50k_batch2.jsonl' \
            --max_tasks 50000 \
            --start_index 50000 \
            --end_index 100000 \
            --verbose

        # Show file info
        echo ''
        echo 'Dataset file:'
        ls -lh data/combined_50k_batch2.jsonl
        wc -l data/combined_50k_batch2.jsonl
    "

# Step 3: Upload to GCS
echo ""
echo "Step 3: Uploading to GCS (same region, zero transfer cost)..."
gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE \
    --worker=0 \
    --command="
        cd ~/$WORK_DIR
        gsutil cp data/combined_50k_batch2.jsonl $GCS_DEST
        echo ''
        echo 'Uploaded to GCS:'
        gsutil ls -l $GCS_DEST
    "

echo ""
echo "==========================================="
echo "  Dataset batch 2 created and uploaded"
echo "  Location: $GCS_DEST"
echo "==========================================="
