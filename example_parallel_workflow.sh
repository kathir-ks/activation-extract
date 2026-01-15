#!/bin/bash
# Example: Complete Parallel Worker Workflow
#
# This script demonstrates the complete workflow from dataset creation
# to launching parallel workers for activation extraction.

set -e

echo "=========================================="
echo "PARALLEL WORKER WORKFLOW EXAMPLE"
echo "=========================================="

# ============================================================================
# Step 1: Create Dataset Streams
# ============================================================================

echo ""
echo "Step 1: Creating dataset streams..."
echo "-------------------------------------------"

NUM_WORKERS=4  # Use 4 for testing, scale to 32-64 for production
DATASET_DIR="./data/test_streams"
MAX_SAMPLES=1000  # Small number for testing

python create_dataset_streams.py \
    --num_streams $NUM_WORKERS \
    --output_dir $DATASET_DIR \
    --max_samples $MAX_SAMPLES \
    --verbose

echo ""
echo "✓ Dataset streams created in: $DATASET_DIR"
echo "  Created $NUM_WORKERS streams with ~$((MAX_SAMPLES / NUM_WORKERS)) samples each"

# ============================================================================
# Step 2: Configure Environment
# ============================================================================

echo ""
echo "Step 2: Configuring environment..."
echo "-------------------------------------------"

export GCS_BUCKET="my-activation-bucket"  # Change this to your bucket
export UPLOAD_TO_GCS="false"  # Set to "true" to enable GCS upload
export MODEL_PATH="Qwen/Qwen2.5-0.5B"
export DATASET_DIR="$DATASET_DIR"
export BATCH_SIZE=4
export SHARD_SIZE_GB=0.5  # Smaller for testing

echo "  GCS_BUCKET: $GCS_BUCKET"
echo "  UPLOAD_TO_GCS: $UPLOAD_TO_GCS"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  BATCH_SIZE: $BATCH_SIZE"

# ============================================================================
# Step 3: Launch Workers (Simulated - normally run on separate TPUs)
# ============================================================================

echo ""
echo "Step 3: Launching workers..."
echo "-------------------------------------------"
echo ""
echo "In production, each worker would run on a separate TPU VM."
echo "For this example, we'll show how to launch worker 0 only."
echo ""

# Launch worker 0 as an example
export TPU_WORKER_ID=0

echo "Launching worker $TPU_WORKER_ID..."
echo ""
echo "Command that would be run on each TPU:"
echo "  export TPU_WORKER_ID=$TPU_WORKER_ID"
echo "  ./launch_worker.sh"
echo ""

# Actually launch (comment out if you don't want to run)
# ./launch_worker.sh

echo "=========================================="
echo "WORKFLOW EXAMPLE COMPLETE"
echo "=========================================="
echo ""
echo "To run in production:"
echo ""
echo "1. Create streams (once):"
echo "   python create_dataset_streams.py --num_streams 32 --output_dir ./data/streams"
echo ""
echo "2. On each TPU VM (0 to 31):"
echo "   export TPU_WORKER_ID=\$i  # where i = 0, 1, 2, ... 31"
echo "   export GCS_BUCKET=your-bucket"
echo "   export UPLOAD_TO_GCS=true"
echo "   ./launch_worker.sh"
echo ""
echo "3. Monitor progress:"
echo "   cat checkpoints/checkpoint_worker_\$i.json"
echo "   gsutil ls -lh gs://\$GCS_BUCKET/activations/tpu_\$i/"
echo ""
echo "For detailed instructions, see PARALLEL_WORKERS_GUIDE.md"
echo "=========================================="
