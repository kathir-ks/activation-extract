#!/bin/bash
#
# Setup TPU Worker - Clone Repository and Install Dependencies
# This script runs on each TPU VM when it first starts (or after preemption)
#
# Usage: ./setup_tpu_worker.sh [--force]
#

set -e

REPO_URL="https://github.com/kathir-ks/activation-extract.git"
WORK_DIR="$HOME/activation-extract"
FORCE_REINSTALL=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --force) FORCE_REINSTALL=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================="
echo "TPU WORKER SETUP"
echo "=========================================="
echo "Repository: $REPO_URL"
echo "Work directory: $WORK_DIR"
echo "Force reinstall: $FORCE_REINSTALL"
echo ""

# Check if already set up (skip if not forcing)
if [ -d "$WORK_DIR" ] && [ "$FORCE_REINSTALL" = false ]; then
    echo "✓ Repository already exists at $WORK_DIR"
    echo "  Use --force to reinstall"
    echo ""

    # Still update the repo
    echo "Pulling latest changes..."
    cd "$WORK_DIR"
    git pull

    echo ""
    echo "✓ Repository updated"
    echo ""
    echo "Setup complete! Ready to run extraction."
    exit 0
fi

# Step 1: Install system dependencies
echo "Step 1: Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git \
    python3-pip \
    python3-venv \
    wget \
    curl \
    htop

echo "✓ System dependencies installed"
echo ""

# Step 2: Clone or update repository
if [ -d "$WORK_DIR" ]; then
    echo "Step 2: Cleaning and re-cloning repository..."
    rm -rf "$WORK_DIR"
else
    echo "Step 2: Cloning repository..."
fi

git clone "$REPO_URL" "$WORK_DIR"
cd "$WORK_DIR"

echo "✓ Repository cloned"
echo ""

# Step 3: Create Python virtual environment (optional but recommended)
echo "Step 3: Setting up Python environment..."

# Upgrade pip
python3 -m pip install --upgrade pip --quiet

echo "✓ Python environment ready"
echo ""

# Step 4: Install Python dependencies
echo "Step 4: Installing Python dependencies..."
echo "  This may take 5-10 minutes on first install..."

# Install dependencies with specific handling for TPU
pip3 install --upgrade -r requirements.txt --quiet

# Verify JAX TPU installation
echo ""
echo "Verifying JAX TPU installation..."
python3 -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'JAX devices: {jax.devices()}')" || {
    echo "⚠ JAX installation verification failed"
    echo "  Attempting to reinstall JAX for TPU..."
    pip3 install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --quiet
    python3 -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'JAX devices: {jax.devices()}')"
}

echo "✓ Dependencies installed and verified"
echo ""

# Step 5: Create necessary directories
echo "Step 5: Creating directories..."
mkdir -p checkpoints
mkdir -p activations
mkdir -p dataset_streams

echo "✓ Directories created"
echo ""

# Step 6: Test imports
echo "Step 6: Testing imports..."
python3 -c "
import jax
import flax
import transformers
import datasets
from google.cloud import storage
print('✓ All critical imports successful')
print(f'  JAX version: {jax.__version__}')
print(f'  JAX devices: {jax.devices()}')
print(f'  Transformers version: {transformers.__version__}')
" || {
    echo "✗ Import test failed!"
    echo "  Some dependencies may not be installed correctly."
    exit 1
}

echo ""

# Step 7: Display system info
echo "=========================================="
echo "SYSTEM INFORMATION"
echo "=========================================="
echo "Hostname: $(hostname)"
echo "TPU_WORKER_ID: ${TPU_WORKER_ID:-not set}"
echo "Python version: $(python3 --version)"
echo "Disk space:"
df -h "$HOME" | tail -1
echo ""
echo "Memory:"
free -h | grep Mem | awk '{print "  Total: " $2 "  Used: " $3 "  Available: " $7}'
echo ""

echo "=========================================="
echo "✓ SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Repository location: $WORK_DIR"
echo "Ready to run: ./scripts/run_extraction_worker.sh"
echo ""
echo "Quick test:"
echo "  cd $WORK_DIR"
echo "  python3 -c 'import jax; print(jax.devices())'"
echo ""
