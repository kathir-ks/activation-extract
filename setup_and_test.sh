#!/bin/bash
#
# Setup TPU pod + run tests
# Usage: ./setup_and_test.sh [--test-only] [--extraction-test]
#
# Modes:
#   default:          Setup all workers + run unit tests on worker 0
#   --test-only:      Skip setup, just run unit tests on worker 0
#   --extraction-test: Full setup + run multihost extraction test
#

set -e

# ============================================
# Configuration
# ============================================
TPU_NAME="node-v5e-64-europe-west4-b"
ZONE="europe-west4-b"
REPO_URL="https://github.com/kathir-ks/activation-extract.git"
WORK_DIR="activation-extract"
DATASET_PATH="gs://fineweb-data-europe-west4/datasets/stream_000.jsonl"
GCS_BUCKET="fineweb-data-europe-west4"

# Parse arguments
TEST_ONLY=false
EXTRACTION_TEST=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test-only) TEST_ONLY=true ;;
        --extraction-test) EXTRACTION_TEST=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "==========================================="
echo "  TPU Pod Setup & Test Runner"
echo "==========================================="
echo "  TPU:  $TPU_NAME"
echo "  Zone: $ZONE"
echo "  Mode: $(if $TEST_ONLY; then echo 'test-only'; elif $EXTRACTION_TEST; then echo 'extraction-test'; else echo 'setup+unit-test'; fi)"
echo ""

# ============================================
# Step 1: Setup all workers
# ============================================
if [ "$TEST_ONLY" = false ]; then
    echo "Step 1: Setting up all workers..."
    echo "  Cloning repo + installing deps on all workers..."
    echo ""

    gcloud compute tpus tpu-vm ssh $TPU_NAME \
        --zone=$ZONE \
        --worker=all \
        --command="
            set -e
            echo '--- Worker \$(hostname) ---'

            # Clone or update repo
            if [ -d ~/$WORK_DIR ]; then
                echo 'Updating existing repo...'
                cd ~/$WORK_DIR && git fetch --all && git reset --hard origin/main
            else
                echo 'Cloning repo...'
                git clone $REPO_URL ~/$WORK_DIR
            fi
            cd ~/$WORK_DIR

            # Install deps
            echo 'Installing dependencies...'
            pip3 install --upgrade -r requirements.txt --quiet 2>&1 | tail -2

            # Verify JAX
            python3 -c \"
import jax
print(f'  JAX {jax.__version__} | Devices: {len(jax.devices())} | Backend: {jax.default_backend()}')
\"
            echo '✅ Worker \$(hostname) ready'
        "

    echo ""
    echo "✅ All workers set up"
    echo ""
else
    echo "Step 1: Skipped (--test-only)"
    echo ""
fi

# ============================================
# Step 2: Run unit tests on worker 0
# ============================================
echo "Step 2: Running unit tests on worker 0..."
echo ""

gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE \
    --worker=0 \
    --command="
        cd ~/$WORK_DIR && python3 tests/test_code_fixes.py
    "

UNIT_EXIT=$?
echo ""

if [ $UNIT_EXIT -eq 0 ]; then
    echo "✅ Unit tests passed"
else
    echo "❌ Unit tests failed (exit code: $UNIT_EXIT)"
    exit $UNIT_EXIT
fi

# ============================================
# Step 3: Run extraction test (optional)
# ============================================
if [ "$EXTRACTION_TEST" = true ]; then
    echo ""
    echo "Step 3: Running multihost extraction test..."
    echo ""

    # Clean up stale processes
    echo "  Cleaning stale processes..."
    gcloud compute tpus tpu-vm ssh $TPU_NAME \
        --zone=$ZONE \
        --worker=all \
        --command="pkill -f 'python3.*multihost_extract' 2>/dev/null; sleep 1; fuser -k 5555/tcp 2>/dev/null; echo 'cleaned'" 2>/dev/null || true
    sleep 5

    # Run extraction
    gcloud compute tpus tpu-vm ssh $TPU_NAME \
        --zone=$ZONE \
        --worker=all \
        --command="cd ~/$WORK_DIR && \
            python3 multihost_extract.py \
                --topology v5litepod-64 \
                --dataset_path $DATASET_PATH \
                --max_tasks 100 \
                --gcs_bucket $GCS_BUCKET \
                --batch_size 64 \
                --upload_to_gcs \
                --verbose"

    echo ""
    echo "✅ Extraction test complete"
fi

echo ""
echo "==========================================="
echo "  ✅ ALL DONE"
echo "==========================================="
