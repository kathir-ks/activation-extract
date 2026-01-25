#!/bin/bash
# =============================================================================
# Deploy and Run on v5e TPU Pod
# =============================================================================
#
# This script creates a v5e TPU pod, deploys the code, and runs extraction.
#
# Usage:
#   # Create and run
#   ./configs/deploy_v5e_pod.sh create
#   ./configs/deploy_v5e_pod.sh deploy
#   ./configs/deploy_v5e_pod.sh run
#
#   # Or all in one
#   ./configs/deploy_v5e_pod.sh all
#
#   # Cleanup
#   ./configs/deploy_v5e_pod.sh delete
# =============================================================================

set -e

# =============================================================================
# Configuration - EDIT THESE
# =============================================================================

# TPU Configuration
TPU_NAME="${TPU_NAME:-extraction-v5e-pod}"
ZONE="${ZONE:-us-central2-b}"
TPU_TYPE="${TPU_TYPE:-v5litepod-64}"  # Options: v5litepod-64, v5litepod-128, v5litepod-256
PREEMPTIBLE="${PREEMPTIBLE:-true}"

# GCS Configuration
GCS_BUCKET="${GCS_BUCKET:?Set GCS_BUCKET environment variable}"
DATASET_PATH="${DATASET_PATH:-gs://${GCS_BUCKET}/datasets/arc_tasks.jsonl}"

# Local paths
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-$(pwd)}"
REMOTE_PROJECT_DIR="~/activation-extract"

# Topology mapping
declare -A TOPOLOGY_MAP
TOPOLOGY_MAP["v5litepod-64"]="v5e-64"
TOPOLOGY_MAP["v5litepod-128"]="v5e-128"
TOPOLOGY_MAP["v5litepod-256"]="v5e-256"

TOPOLOGY="${TOPOLOGY_MAP[$TPU_TYPE]:-v5e-64}"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

log_warn() {
    echo -e "\033[1;33m[WARN]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# =============================================================================
# Commands
# =============================================================================

cmd_create() {
    log_info "Creating TPU pod: $TPU_NAME ($TPU_TYPE) in $ZONE"
    
    CREATE_CMD="gcloud compute tpus tpu-vm create $TPU_NAME"
    CREATE_CMD="$CREATE_CMD --zone=$ZONE"
    CREATE_CMD="$CREATE_CMD --accelerator-type=$TPU_TYPE"
    CREATE_CMD="$CREATE_CMD --version=tpu-ubuntu2204-base"
    
    if [ "$PREEMPTIBLE" = "true" ]; then
        CREATE_CMD="$CREATE_CMD --preemptible"
        log_info "Using preemptible instance (70% cheaper)"
    fi
    
    echo "Running: $CREATE_CMD"
    eval $CREATE_CMD
    
    log_info "TPU pod created successfully"
}

cmd_deploy() {
    log_info "Deploying code to all workers..."
    
    # First, create project directory on all workers
    gcloud compute tpus tpu-vm ssh $TPU_NAME \
        --zone=$ZONE \
        --worker=all \
        --command="mkdir -p $REMOTE_PROJECT_DIR"
    
    # Copy project files
    log_info "Copying project files..."
    gcloud compute tpus tpu-vm scp --recurse \
        "$LOCAL_PROJECT_DIR/"* \
        "$TPU_NAME:$REMOTE_PROJECT_DIR/" \
        --zone=$ZONE \
        --worker=all
    
    # Install dependencies on all workers
    log_info "Installing dependencies..."
    gcloud compute tpus tpu-vm ssh $TPU_NAME \
        --zone=$ZONE \
        --worker=all \
        --command="cd $REMOTE_PROJECT_DIR && pip install -q -r requirements.txt"
    
    # Make scripts executable
    gcloud compute tpus tpu-vm ssh $TPU_NAME \
        --zone=$ZONE \
        --worker=all \
        --command="chmod +x $REMOTE_PROJECT_DIR/scripts/*.sh"
    
    log_info "Deployment complete"
}

cmd_run() {
    log_info "Starting extraction on all workers..."
    log_info "Topology: $TOPOLOGY"
    log_info "Dataset: $DATASET_PATH"
    log_info "GCS Bucket: $GCS_BUCKET"
    
    # Build the run command
    RUN_CMD="cd $REMOTE_PROJECT_DIR && "
    RUN_CMD="$RUN_CMD DATASET_PATH=$DATASET_PATH "
    RUN_CMD="$RUN_CMD GCS_BUCKET=$GCS_BUCKET "
    RUN_CMD="$RUN_CMD TOPOLOGY=$TOPOLOGY "
    RUN_CMD="$RUN_CMD ./scripts/launch_multihost.sh"
    
    # Run on all workers
    gcloud compute tpus tpu-vm ssh $TPU_NAME \
        --zone=$ZONE \
        --worker=all \
        --command="$RUN_CMD"
    
    log_info "Extraction complete"
}

cmd_status() {
    log_info "Checking TPU pod status..."
    gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE
}

cmd_ssh() {
    WORKER="${1:-0}"
    log_info "SSH into worker $WORKER..."
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=$WORKER
}

cmd_logs() {
    WORKER="${1:-0}"
    log_info "Fetching logs from worker $WORKER..."
    gcloud compute tpus tpu-vm ssh $TPU_NAME \
        --zone=$ZONE \
        --worker=$WORKER \
        --command="tail -f $REMOTE_PROJECT_DIR/nohup.out 2>/dev/null || echo 'No logs found'"
}

cmd_delete() {
    log_warn "Deleting TPU pod: $TPU_NAME"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --quiet
        log_info "TPU pod deleted"
    else
        log_info "Cancelled"
    fi
}

cmd_all() {
    cmd_create
    sleep 30  # Wait for TPU to be ready
    cmd_deploy
    cmd_run
}

# =============================================================================
# Main
# =============================================================================

case "${1:-help}" in
    create)
        cmd_create
        ;;
    deploy)
        cmd_deploy
        ;;
    run)
        cmd_run
        ;;
    status)
        cmd_status
        ;;
    ssh)
        cmd_ssh "${2:-0}"
        ;;
    logs)
        cmd_logs "${2:-0}"
        ;;
    delete)
        cmd_delete
        ;;
    all)
        cmd_all
        ;;
    *)
        echo "Usage: $0 {create|deploy|run|status|ssh|logs|delete|all}"
        echo ""
        echo "Commands:"
        echo "  create  - Create TPU pod"
        echo "  deploy  - Deploy code to all workers"
        echo "  run     - Start extraction on all workers"
        echo "  status  - Check TPU pod status"
        echo "  ssh [N] - SSH into worker N (default: 0)"
        echo "  logs [N]- View logs from worker N (default: 0)"
        echo "  delete  - Delete TPU pod"
        echo "  all     - Create, deploy, and run"
        echo ""
        echo "Environment Variables:"
        echo "  TPU_NAME      - Name of TPU (default: extraction-v5e-pod)"
        echo "  ZONE          - GCP zone (default: us-central2-b)"
        echo "  TPU_TYPE      - TPU type (default: v5litepod-64)"
        echo "  GCS_BUCKET    - GCS bucket for outputs (required)"
        echo "  DATASET_PATH  - Path to dataset (required for run)"
        echo "  PREEMPTIBLE   - Use preemptible (default: true)"
        exit 1
        ;;
esac
