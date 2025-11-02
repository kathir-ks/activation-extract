#!/bin/bash
# Helper script to run ARC-AGI inference in Docker on TPU VMs

set -e

# Configuration
IMAGE_NAME="arc-agi-inference:latest"
CONTAINER_NAME="arc-inference-${RANDOM}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "ARC-AGI Docker Runner for TPU VMs"
echo "=========================================="

# Function to print colored messages
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if running on TPU VM
if [ -d "/sys/class/tpu" ]; then
    log_info "Running on TPU VM"
    TPU_FLAGS="--privileged --device=/dev/accel0:/dev/accel0"
else
    log_warn "Not running on TPU VM. TPU support may be limited."
    TPU_FLAGS=""
fi

# Parse arguments
MODE=${1:-help}
shift || true

case $MODE in
    build)
        log_info "Building Docker image..."
        docker build -t $IMAGE_NAME .
        log_info "Build complete!"
        ;;

    transform)
        log_info "Running dataset transformation..."
        docker run --rm \
            -v $(pwd)/arc_data:/workspace/arc_data \
            -v $(pwd)/logs:/workspace/logs \
            $IMAGE_NAME \
            python transform_hf_to_arc.py \
                --output_dir /workspace/arc_data \
                "$@"
        ;;

    simple)
        log_info "Running simple inference..."
        docker run --rm \
            -v $(pwd)/outputs:/workspace/outputs \
            -v $(pwd)/activations:/workspace/activations \
            -v $(pwd)/arc_data:/workspace/arc_data \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            $IMAGE_NAME \
            python simple_extraction_inference.py \
                --dataset_path /workspace/arc_data/arc_format_train.json \
                --output_path /workspace/outputs/predictions.json \
                --activations_dir /workspace/activations \
                "$@"
        ;;

    distributed)
        log_info "Running distributed inference on TPU..."
        docker run --rm \
            $TPU_FLAGS \
            -v $(pwd)/outputs:/workspace/outputs \
            -v $(pwd)/activations:/workspace/activations \
            -v $(pwd)/arc_data:/workspace/arc_data \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            -e JAX_PLATFORMS=tpu \
            $IMAGE_NAME \
            python distributed_inference_with_activations.py \
                --dataset_path /workspace/arc_data/arc_format_train.json \
                --output_filepath /workspace/outputs/submission.json \
                --activations_dir /workspace/activations \
                "$@"
        ;;

    interactive)
        log_info "Starting interactive container..."
        docker run -it --rm \
            $TPU_FLAGS \
            -v $(pwd):/workspace \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            -e JAX_PLATFORMS=tpu \
            --name $CONTAINER_NAME \
            $IMAGE_NAME \
            /bin/bash
        ;;

    test)
        log_info "Running tests..."
        docker run --rm \
            $IMAGE_NAME \
            python test_pipeline.py
        ;;

    compose-up)
        log_info "Starting services with docker-compose..."
        docker-compose up "$@"
        ;;

    compose-down)
        log_info "Stopping docker-compose services..."
        docker-compose down
        ;;

    push)
        if [ -z "$2" ]; then
            log_error "Usage: $0 push <registry/image:tag>"
            exit 1
        fi
        REMOTE_IMAGE=$2
        log_info "Pushing image to $REMOTE_IMAGE..."
        docker tag $IMAGE_NAME $REMOTE_IMAGE
        docker push $REMOTE_IMAGE
        log_info "Push complete!"
        ;;

    pull)
        if [ -z "$2" ]; then
            log_error "Usage: $0 pull <registry/image:tag>"
            exit 1
        fi
        REMOTE_IMAGE=$2
        log_info "Pulling image from $REMOTE_IMAGE..."
        docker pull $REMOTE_IMAGE
        docker tag $REMOTE_IMAGE $IMAGE_NAME
        log_info "Pull complete!"
        ;;

    help|*)
        cat << EOF
Usage: $0 <command> [options]

Commands:
  build                 Build the Docker image
  transform [args]      Run dataset transformation
  simple [args]         Run simple inference
  distributed [args]    Run distributed TPU inference
  interactive           Start interactive container
  test                  Run tests
  compose-up [args]     Start services with docker-compose
  compose-down          Stop docker-compose services
  push <image>          Push image to registry
  pull <image>          Pull image from registry
  help                  Show this help message

Examples:
  # Build image
  $0 build

  # Transform dataset
  $0 transform --max_samples 1000

  # Run simple inference
  $0 simple --model_path YOUR_MODEL --max_tasks 10

  # Run distributed inference on TPU
  $0 distributed --model_path YOUR_MODEL --mesh_shape 8 8

  # Interactive mode
  $0 interactive

  # Run with docker-compose
  $0 compose-up arc-inference

Environment Variables:
  MODEL_PATH           Path to model (default: Qwen/Qwen2.5-0.5B)
  MAX_SAMPLES          Max samples to transform (default: 1000)
  MAX_TASKS            Max tasks for inference (default: 10)
  CLOUD_BUCKET         GCS bucket for outputs
  GCP_KEY_PATH         Path to GCP service account key

Documentation:
  README.md                 - Main documentation
  QUICKSTART.md            - Quick start guide
  DOCKER_MULTIHOST.md      - Multi-host TPU setup (v5e-256, v4-512, etc.)
  QUICK_START_EXTRACTION.md - Activation extraction guide
EOF
        ;;
esac
