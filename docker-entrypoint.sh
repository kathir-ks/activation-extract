#!/bin/bash
# Docker entrypoint script for ARC-AGI distributed inference

set -e

echo "=========================================="
echo "ARC-AGI Distributed Inference Container"
echo "=========================================="

# Check for TPU
echo "Checking for TPU devices..."
python -c "import jax; devices = jax.devices(); print(f'Found {len(devices)} devices: {devices}')" || {
    echo "Warning: No TPU devices found. Running on CPU."
}

# Check if running on TPU VM
if [ -d "/sys/class/tpu" ]; then
    echo "✓ Running on TPU VM"
    export JAX_PLATFORMS=tpu
else
    echo "⚠ Not on TPU VM, using default backend"
fi

# Setup Google Cloud authentication if credentials are mounted
if [ -f "/secrets/gcp-key.json" ]; then
    echo "Setting up GCP authentication..."
    export GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-key.json
    gcloud auth activate-service-account --key-file=/secrets/gcp-key.json
    echo "✓ GCP authentication configured"
fi

# Print environment info
echo ""
echo "Environment Information:"
echo "------------------------"
echo "Python: $(python --version)"
echo "JAX: $(python -c 'import jax; print(jax.__version__)')"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "Working directory: $(pwd)"
echo "Available files:"
ls -lh *.py 2>/dev/null | head -5 || echo "No Python files found"
echo ""

# Execute the command
echo "Executing: $@"
echo "=========================================="
exec "$@"
