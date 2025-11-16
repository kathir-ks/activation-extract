#!/bin/bash

################################################################################
# Multi-Host TPU v5e-64 Deployment Script
# Deploys activation extraction to all 8 hosts in parallel
#
# Author: kskathir2003
# Date: 2025-11-16
# Usage: ./deploy_8hosts_parallel.sh
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

################################################################################
# CONFIGURATION
################################################################################

# GCP Configuration
export PROJECT_ID="absolute-axis-470415-g6"
export ZONE="us-central1-a"
export TPU_NAME="v5e-main-1"  # CHANGE THIS to your TPU name

# Docker Image Configuration
export AR_REGION="us-central1"
export AR_REPO="arc-agi-us-central1"  # CHANGE THIS to your repo name
export IMAGE_TAG="activation-extraction"
export IMAGE_PATH="${AR_REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/activation-extraction:${IMAGE_TAG}"

# Dataset Configuration
export DATASET_PATH="gs://fineweb-data-us-central1-a/datasets/arc_barc200k_test_1k.jsonl"
export GCS_BUCKET="fineweb-data-us-central1-a"
export GCS_PREFIX="activations_barc200k_8hosts"

# Model Configuration
export MODEL_PATH="KathirKs/qwen-2.5-0.5b"
export BATCH_SIZE=4
export MAX_SEQ_LENGTH=512
export MAX_TASKS=1000  # Set to empty "" for unlimited

# Multi-Host Configuration
export NUM_HOSTS=8  # v5e-64 has 8 hosts with 8 chips each = 64 total
export MESH_TYPE="2d"  # Options: 1d, 2d, 3d
export COORDINATOR_PORT=8476

# Output Configuration
export SHARD_SIZE_GB=1.0
export COMPRESS_SHARDS="--compress_shards"
export DELETE_LOCAL="--delete_local_after_upload"

# Other
export VERBOSE="--verbose"
export LAYERS_TO_EXTRACT="10 11 12 13 14 15 16 17 18 19 20 21 22 23"

################################################################################
# COLORS FOR OUTPUT
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

################################################################################
# HELPER FUNCTIONS
################################################################################

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${MAGENTA}======================================================================${NC}"
    echo -e "${MAGENTA}$1${NC}"
    echo -e "${MAGENTA}======================================================================${NC}"
}

################################################################################
# PRE-FLIGHT CHECKS
################################################################################

preflight_checks() {
    log_section "Pre-Flight Checks"
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi
    log_success "gcloud CLI found"
    
    # Check if TPU exists
    log_info "Checking if TPU ${TPU_NAME} exists..."
    if ! gcloud compute tpus tpu-vm describe ${TPU_NAME} --zone=${ZONE} &> /dev/null; then
        log_error "TPU ${TPU_NAME} not found in zone ${ZONE}"
        exit 1
    fi
    log_success "TPU ${TPU_NAME} found"
    
    # Check dataset exists
    log_info "Checking if dataset exists..."
    if ! gsutil ls ${DATASET_PATH} &> /dev/null; then
        log_error "Dataset not found: ${DATASET_PATH}"
        log_info "Please create dataset first using:"
        log_info "  python convert_hf_to_arc_format.py --output_file dataset.jsonl"
        exit 1
    fi
    log_success "Dataset found: ${DATASET_PATH}"
    
    # Check GCS bucket exists
    log_info "Checking if GCS bucket exists..."
    if ! gsutil ls gs://${GCS_BUCKET}/ &> /dev/null; then
        log_error "GCS bucket not found: gs://${GCS_BUCKET}/"
        log_info "Creating bucket..."
        gsutil mb -p ${PROJECT_ID} -c STANDARD -l ${ZONE%%-*} gs://${GCS_BUCKET}/
        log_success "Bucket created"
    else
        log_success "GCS bucket found"
    fi
}

################################################################################
# GET COORDINATOR IP
################################################################################

get_coordinator_ip() {
    log_section "Getting Coordinator IP"
    
    log_info "Querying worker-0 for internal IP..."
    COORDINATOR_IP=$(gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
        --zone=${ZONE} \
        --worker=0 \
        --command="hostname -I | awk '{print \$1}'" 2>/dev/null | tr -d '\r\n ')
    
    if [ -z "$COORDINATOR_IP" ]; then
        log_error "Failed to get coordinator IP"
        exit 1
    fi
    
    export COORDINATOR_ADDRESS="${COORDINATOR_IP}:${COORDINATOR_PORT}"
    log_success "Coordinator address: ${COORDINATOR_ADDRESS}"
}

################################################################################
# PULL DOCKER IMAGE ON ALL HOSTS
################################################################################

pull_docker_images() {
    log_section "Pulling Docker Images on All Hosts"
    
    log_info "Authenticating Docker with Artifact Registry..."
    gcloud auth configure-docker ${AR_REGION}-docker.pkg.dev --quiet
    
    local pids=()
    
    for host_id in {0..7}; do
        log_info "Pulling image on host ${host_id}..."
        (
            gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
                --zone=${ZONE} \
                --worker=${host_id} \
                --command="
                    echo 'Host ${host_id}: Configuring Docker auth...'
                    gcloud auth configure-docker ${AR_REGION}-docker.pkg.dev --quiet
                    
                    echo 'Host ${host_id}: Pulling image...'
                    docker pull ${IMAGE_PATH}
                    
                    echo 'Host ${host_id}: Image pulled successfully'
                " 2>&1 | sed "s/^/[Host ${host_id}] /"
        ) &
        pids+=($!)
    done
    
    # Wait for all pulls to complete
    log_info "Waiting for all image pulls to complete..."
    for pid in ${pids[@]}; do
        wait $pid
    done
    
    log_success "All images pulled successfully"
}

################################################################################
# CLEAN UP OLD CONTAINERS
################################################################################

cleanup_old_containers() {
    log_section "Cleaning Up Old Containers"
    
    local pids=()
    
    for host_id in {0..7}; do
        log_info "Cleaning up host ${host_id}..."
        (
            gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
                --zone=${ZONE} \
                --worker=${host_id} \
                --command="
                    # Stop and remove old containers
                    docker stop extraction-host${host_id} 2>/dev/null || true
                    docker rm extraction-host${host_id} 2>/dev/null || true
                    
                    # Clean up old output (optional)
                    # rm -rf ~/data/output 2>/dev/null || true
                    
                    echo 'Host ${host_id}: Cleanup complete'
                " 2>&1 | sed "s/^/[Host ${host_id}] /"
        ) &
        pids+=($!)
    done
    
    # Wait for all cleanups
    for pid in ${pids[@]}; do
        wait $pid
    done
    
    log_success "Cleanup complete on all hosts"
}

################################################################################
# DEPLOY TO ALL HOSTS
################################################################################

deploy_all_hosts() {
    log_section "Deploying to All 8 Hosts"
    
    log_info "Starting deployment..."
    log_info "Coordinator: ${COORDINATOR_ADDRESS}"
    log_info "Mesh type: ${MESH_TYPE}"
    log_info "Dataset: ${DATASET_PATH}"
    log_info "Model: ${MODEL_PATH}"
    
    local pids=()
    
    # Deploy host 0 (coordinator) first
    log_info "${YELLOW}Deploying Host 0 (Coordinator)...${NC}"
    deploy_single_host 0 &
    pids+=($!)
    
    # Wait for coordinator to start
    log_info "Waiting 30 seconds for coordinator to initialize..."
    sleep 30
    
    # Deploy remaining hosts in parallel
    for host_id in {1..7}; do
        log_info "Deploying Host ${host_id}..."
        deploy_single_host ${host_id} &
        pids+=($!)
        sleep 2  # Small delay between deployments
    done
    
    # Wait for all deployments
    log_info "Waiting for all deployments to complete..."
    for pid in ${pids[@]}; do
        wait $pid
    done
    
    log_success "All hosts deployed successfully!"
}

################################################################################
# DEPLOY SINGLE HOST
################################################################################

deploy_single_host() {
    local host_id=$1
    
    # Build max_tasks argument
    local max_tasks_arg=""
    if [ -n "${MAX_TASKS}" ]; then
        max_tasks_arg="--max_tasks ${MAX_TASKS}"
    fi
    
    # Build layers argument
    local layers_arg=""
    if [ -n "${LAYERS_TO_EXTRACT}" ]; then
        layers_arg="--layers_to_extract ${LAYERS_TO_EXTRACT}"
    fi
    
    gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
        --zone=${ZONE} \
        --worker=${host_id} \
        --command="
            echo '========================================================================'
            echo 'Host ${host_id}: Starting extraction container'
            echo '========================================================================'
            
            # Create data directory
            mkdir -p ~/data/output
            
            # Run Docker container
            sudo docker run -d \
                --name extraction-host${host_id} \
                --net=host \
                --privileged \
                -v ~/data:/workspace/data \
                -v ~/.config/gcloud:/root/.config/gcloud:ro \
                -v ~/.cache/huggingface:/cache/huggingface \
                ${IMAGE_PATH} \
                -c \"python /workspace/extract_activations_arc_v5e64.py \
                    --machine_id 0 \
                    --total_machines 1 \
                    --multihost \
                    --coordinator_address ${COORDINATOR_ADDRESS} \
                    --host_id ${host_id} \
                    --num_hosts ${NUM_HOSTS} \
                    --mesh_type ${MESH_TYPE} \
                    --model_path ${MODEL_PATH} \
                    --dataset_path ${DATASET_PATH} \
                    --batch_size ${BATCH_SIZE} \
                    --max_seq_length ${MAX_SEQ_LENGTH} \
                    ${max_tasks_arg} \
                    ${layers_arg} \
                    --output_dir /workspace/data/output \
                    --upload_to_gcs \
                    --gcs_bucket ${GCS_BUCKET} \
                    --gcs_prefix ${GCS_PREFIX}/host_${host_id} \
                    --shard_size_gb ${SHARD_SIZE_GB} \
                    ${COMPRESS_SHARDS} \
                    ${DELETE_LOCAL} \
                    ${VERBOSE}\"
            
            echo 'Host ${host_id}: Container started'
            
            # Wait a moment and check if container is running
            sleep 5
            if docker ps | grep -q extraction-host${host_id}; then
                echo 'Host ${host_id}: ✓ Container running successfully'
            else
                echo 'Host ${host_id}: ✗ Container failed to start'
                docker logs extraction-host${host_id}
                exit 1
            fi
        " 2>&1 | sed "s/^/[Host ${host_id}] /"
    
    return $?
}

################################################################################
# MONITOR PROGRESS
################################################################################

monitor_progress() {
    log_section "Monitoring Progress"
    
    log_info "Checking status on all hosts..."
    echo ""
    
    for host_id in {0..7}; do
        echo -e "${CYAN}Host ${host_id}:${NC}"
        gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
            --zone=${ZONE} \
            --worker=${host_id} \
            --command="
                if docker ps | grep -q extraction-host${host_id}; then
                    echo '  Status: ✓ Running'
                    echo '  Last 3 log lines:'
                    docker logs --tail 3 extraction-host${host_id} 2>&1 | sed 's/^/    /'
                else
                    echo '  Status: ✗ Not running'
                fi
            " 2>/dev/null
        echo ""
    done
    
    log_info "To view detailed logs for a specific host:"
    log_info "  gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=0 --command='docker logs -f extraction-host0'"
    
    log_info "To check GCS output:"
    log_info "  gsutil ls -lh gs://${GCS_BUCKET}/${GCS_PREFIX}/"
}

################################################################################
# VIEW LOGS
################################################################################

view_logs() {
    log_section "Viewing Logs from All Hosts"
    
    local lines=${1:-50}  # Default 50 lines
    
    for host_id in {0..7}; do
        echo ""
        echo -e "${MAGENTA}========== Host ${host_id} (Last ${lines} lines) ==========${NC}"
        gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
            --zone=${ZONE} \
            --worker=${host_id} \
            --command="docker logs --tail ${lines} extraction-host${host_id} 2>&1" 2>/dev/null || true
    done
}

################################################################################
# STOP ALL
################################################################################

stop_all() {
    log_section "Stopping All Containers"
    
    local pids=()
    
    for host_id in {0..7}; do
        log_info "Stopping host ${host_id}..."
        (
            gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
                --zone=${ZONE} \
                --worker=${host_id} \
                --command="
                    docker stop extraction-host${host_id} 2>/dev/null || true
                    docker rm extraction-host${host_id} 2>/dev/null || true
                    echo 'Host ${host_id}: Stopped'
                " 2>&1 | sed "s/^/[Host ${host_id}] /"
        ) &
        pids+=($!)
    done
    
    for pid in ${pids[@]}; do
        wait $pid
    done
    
    log_success "All containers stopped"
}

################################################################################
# CHECK STATUS
################################################################################

check_status() {
    log_section "Container Status on All Hosts"
    
    echo ""
    printf "%-8s %-15s %-20s %-15s\n" "Host" "Status" "Container ID" "Uptime"
    echo "------------------------------------------------------------------------"
    
    for host_id in {0..7}; do
        status=$(gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
            --zone=${ZONE} \
            --worker=${host_id} \
            --command="
                if docker ps --format '{{.ID}}\t{{.Status}}' | grep extraction-host${host_id} &>/dev/null; then
                    docker ps --format '{{.ID}}\t{{.Status}}' | grep extraction-host${host_id}
                else
                    echo 'STOPPED\tN/A'
                fi
            " 2>/dev/null)
        
        container_id=$(echo "$status" | awk '{print $1}')
        uptime=$(echo "$status" | cut -f2-)
        
        if [ "$container_id" == "STOPPED" ]; then
            printf "%-8s %-15s %-20s %-15s\n" "$host_id" "❌ Stopped" "N/A" "N/A"
        else
            printf "%-8s %-15s %-20s %-15s\n" "$host_id" "✓ Running" "$container_id" "$uptime"
        fi
    done
    
    echo ""
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    log_section "TPU v5e-64 8-Host Parallel Deployment"
    echo -e "${CYAN}Date: $(date)${NC}"
    echo -e "${CYAN}User: kskathir2003${NC}"
    echo ""
    
    # Parse command line arguments
    case "${1:-deploy}" in
        deploy)
            preflight_checks
            get_coordinator_ip
            pull_docker_images
            cleanup_old_containers
            deploy_all_hosts
            sleep 10
            monitor_progress
            ;;
        
        monitor)
            monitor_progress
            ;;
        
        logs)
            view_logs "${2:-50}"
            ;;
        
        status)
            check_status
            ;;
        
        stop)
            stop_all
            ;;
        
        restart)
            stop_all
            sleep 5
            preflight_checks
            get_coordinator_ip
            deploy_all_hosts
            sleep 10
            monitor_progress
            ;;
        
        *)
            echo "Usage: $0 {deploy|monitor|logs|status|stop|restart}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Full deployment to all 8 hosts (default)"
            echo "  monitor  - Check current progress on all hosts"
            echo "  logs     - View logs from all hosts (usage: $0 logs [num_lines])"
            echo "  status   - Show container status on all hosts"
            echo "  stop     - Stop all containers"
            echo "  restart  - Stop and redeploy all containers"
            exit 1
            ;;
    esac
    
    log_success "Command completed!"
}

# Run main function
main "$@"