#!/bin/bash
# ============================================================================
# Setup Worker — Clone repo, install deps, kill stale processes
# ============================================================================
# Called by auto_recover.sh after TPU becomes READY again.
# Designed to be idempotent: safe to run multiple times.
#
# Usage:
#   ./scripts/setup_worker.sh [--branch BRANCH]
# ============================================================================

set -euo pipefail

REPO_URL="https://github.com/kathir-ks/activation-extract.git"
WORK_DIR="$HOME/activation-extract"
BRANCH="main"

while [[ $# -gt 0 ]]; do
    case $1 in
        --branch) BRANCH="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "[setup_worker] $(date '+%Y-%m-%d %H:%M:%S') Starting setup on $(hostname)"

# Kill any stale python processes from previous extraction runs
echo "[setup_worker] Killing stale python3 processes..."
pkill -f "multihost_extract.py" 2>/dev/null || true
pkill -f "extract_activations.py" 2>/dev/null || true
sleep 1

# Clone or update repo
if [ -d "$WORK_DIR/.git" ]; then
    echo "[setup_worker] Updating existing repo..."
    cd "$WORK_DIR"
    git fetch origin
    git reset --hard "origin/$BRANCH"
    git clean -fd
else
    echo "[setup_worker] Cloning repo..."
    rm -rf "$WORK_DIR"
    git clone --branch "$BRANCH" "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi

# Install Python dependencies
echo "[setup_worker] Installing requirements..."
pip3 install --upgrade -r requirements.txt --quiet 2>&1 | tail -5

# Verify JAX can see TPU devices
echo "[setup_worker] Verifying JAX TPU..."
python3 -c "
import jax
devices = jax.devices()
print(f'  JAX {jax.__version__}, {len(devices)} devices: {devices[0].platform}')
" || {
    echo "[setup_worker] JAX verification failed, reinstalling jax[tpu]..."
    pip3 install --upgrade 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --quiet
    python3 -c "import jax; print(f'  JAX {jax.__version__}, {len(jax.devices())} devices')"
}

# Create working directories
mkdir -p "$WORK_DIR/checkpoints" "$WORK_DIR/activations"

echo "[setup_worker] $(date '+%Y-%m-%d %H:%M:%S') Setup complete on $(hostname)"
