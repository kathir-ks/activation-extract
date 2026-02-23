#!/bin/bash
# ============================================================================
# Setup Worker — Prepare a TPU VM for extraction after preemption/recreation
# ============================================================================
#
# Called by auto_recover.sh after TPU becomes READY.
# Runs on each worker via: gcloud tpu-vm ssh ... --worker=all --command="..."
#
# Steps:
#   1. Kill stale Python processes (leftover from previous run)
#   2. Clone or update the repo
#   3. Install Python dependencies + JAX[TPU]
#   4. Verify JAX can see TPU devices
#
# Usage:
#   ./scripts/setup_worker.sh [--branch main] [--repo URL]
# ============================================================================

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/kathir-ks/activation-extract.git}"
BRANCH="${BRANCH:-main}"
WORK_DIR="$HOME/activation-extract"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --branch) BRANCH="$2"; shift 2 ;;
        --repo)   REPO_URL="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

echo "[$(date '+%H:%M:%S')] Setting up worker on $(hostname)..."

# Step 1: Kill stale Python processes from previous extraction runs
echo "[$(date '+%H:%M:%S')] Killing stale Python processes..."
pkill -f "multihost_extract.py" 2>/dev/null || true
pkill -f "extract_activations.py" 2>/dev/null || true
sleep 2
# Force kill if still alive
pkill -9 -f "multihost_extract.py" 2>/dev/null || true
pkill -9 -f "extract_activations.py" 2>/dev/null || true

# Step 2: Clone or update repo
if [ -d "$WORK_DIR/.git" ]; then
    echo "[$(date '+%H:%M:%S')] Updating existing repo..."
    cd "$WORK_DIR"
    git fetch origin
    git reset --hard "origin/$BRANCH"
    git clean -fd
else
    echo "[$(date '+%H:%M:%S')] Cloning repo..."
    rm -rf "$WORK_DIR"
    git clone --branch "$BRANCH" "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi

# Step 3: Install Python dependencies
echo "[$(date '+%H:%M:%S')] Installing Python dependencies..."
pip3 install --upgrade -r requirements.txt --quiet 2>&1 | tail -1

# Install JAX TPU if not already working
if ! python3 -c "import jax; assert len(jax.devices()) > 0" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] Installing JAX[TPU]..."
    pip3 install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --quiet 2>&1 | tail -1
fi

# Step 4: Verify
echo "[$(date '+%H:%M:%S')] Verifying setup..."
python3 -c "
import jax
devices = jax.devices()
print(f'  JAX {jax.__version__} — {len(devices)} devices: {devices[0].platform}')
" || {
    echo "ERROR: JAX verification failed on $(hostname)"
    exit 1
}

# Create directories
mkdir -p checkpoints activations

echo "[$(date '+%H:%M:%S')] Worker $(hostname) ready."
