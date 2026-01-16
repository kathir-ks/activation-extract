#!/bin/bash
#
# TPU Management Script - Create, Destroy, Monitor, and Recreate TPUs
# Handles preemptible TPUs across multiple zones with automatic naming
#
# Naming Convention: tpu-{region}-{zone_letter}-{worker_id}
# Example: tpu-us-central1-a-0, tpu-us-central1-b-5, tpu-europe-west4-a-0
#
# Usage:
#   ./manage_tpus.sh create --zones us-central1-a,us-central1-b --workers_per_zone 4
#   ./manage_tpus.sh delete --zones us-central1-a --workers_per_zone 4
#   ./manage_tpus.sh status --zones us-central1-a,us-central1-b
#   ./manage_tpus.sh recreate-preempted --zones us-central1-a --workers_per_zone 4
#

set -e

# Default configuration
TPU_TYPE="v3-8"
ACCELERATOR_TYPE="v3-8"
RUNTIME_VERSION="tpu-ubuntu2204-base"
PREEMPTIBLE=true
PROJECT="${GOOGLE_CLOUD_PROJECT:-}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

function print_usage() {
    cat <<EOF
Usage: $0 <command> [options]

Commands:
  create                 Create new TPUs
  delete                 Delete existing TPUs
  status                 Check status of TPUs
  recreate-preempted     Recreate only preempted TPUs
  list                   List all TPUs

Options:
  --zones ZONES          Comma-separated list of zones (required for create/delete/recreate)
                        Example: us-central1-a,us-central1-b,europe-west4-a
  --workers_per_zone N   Number of workers per zone (required for create/delete/recreate)
  --tpu_type TYPE       TPU type (default: $TPU_TYPE)
  --runtime VERSION     Runtime version (default: $RUNTIME_VERSION)
  --no-preemptible      Create non-preemptible TPUs (not recommended for cost)
  --project PROJECT     GCP project ID (default: from gcloud config)

Naming Convention:
  TPU names follow: tpu-{region}-{zone_letter}-{worker_id}
  Examples:
    - us-central1-a, worker 0  → tpu-us-central1-a-0
    - us-central1-b, worker 3  → tpu-us-central1-b-3
    - europe-west4-a, worker 0 → tpu-europe-west4-a-0

Examples:
  # Create 4 workers in us-central1-a and 4 in us-central1-b (total 8 TPUs)
  $0 create --zones us-central1-a,us-central1-b --workers_per_zone 4

  # Check status across zones
  $0 status --zones us-central1-a,us-central1-b

  # Delete all workers in a zone
  $0 delete --zones us-central1-a --workers_per_zone 4

  # Recreate only preempted TPUs
  $0 recreate-preempted --zones us-central1-a,us-central1-b --workers_per_zone 4

EOF
}

function get_tpu_name() {
    local zone=$1
    local worker_id=$2

    # Extract region and zone letter
    # us-central1-a → us-central1, a
    local region=$(echo "$zone" | rev | cut -d'-' -f2- | rev)
    local zone_letter=$(echo "$zone" | rev | cut -d'-' -f1 | rev)

    echo "tpu-${region}-${zone_letter}-${worker_id}"
}

function create_tpu() {
    local zone=$1
    local worker_id=$2

    local tpu_name=$(get_tpu_name "$zone" "$worker_id")

    echo -e "${YELLOW}Creating TPU: $tpu_name in $zone...${NC}"

    local preempt_flag=""
    if [ "$PREEMPTIBLE" = true ]; then
        preempt_flag="--preemptible"
    fi

    local project_flag=""
    if [ -n "$PROJECT" ]; then
        project_flag="--project=$PROJECT"
    fi

    gcloud compute tpus tpu-vm create "$tpu_name" \
        --zone="$zone" \
        --accelerator-type="$ACCELERATOR_TYPE" \
        --version="$RUNTIME_VERSION" \
        $preempt_flag \
        $project_flag \
        --quiet || {
            echo -e "${RED}✗ Failed to create $tpu_name${NC}"
            return 1
        }

    echo -e "${GREEN}✓ Created $tpu_name${NC}"
    return 0
}

function delete_tpu() {
    local zone=$1
    local worker_id=$2

    local tpu_name=$(get_tpu_name "$zone" "$worker_id")

    # Check if TPU exists
    if ! gcloud compute tpus tpu-vm describe "$tpu_name" --zone="$zone" &>/dev/null; then
        echo -e "${YELLOW}⊘ TPU $tpu_name does not exist (already deleted?)${NC}"
        return 0
    fi

    echo -e "${YELLOW}Deleting TPU: $tpu_name in $zone...${NC}"

    local project_flag=""
    if [ -n "$PROJECT" ]; then
        project_flag="--project=$PROJECT"
    fi

    gcloud compute tpus tpu-vm delete "$tpu_name" \
        --zone="$zone" \
        $project_flag \
        --quiet || {
            echo -e "${RED}✗ Failed to delete $tpu_name${NC}"
            return 1
        }

    echo -e "${GREEN}✓ Deleted $tpu_name${NC}"
    return 0
}

function get_tpu_status() {
    local zone=$1
    local worker_id=$2

    local tpu_name=$(get_tpu_name "$zone" "$worker_id")

    local project_flag=""
    if [ -n "$PROJECT" ]; then
        project_flag="--project=$PROJECT"
    fi

    # Get TPU status
    local status=$(gcloud compute tpus tpu-vm describe "$tpu_name" \
        --zone="$zone" \
        $project_flag \
        --format="value(state)" 2>/dev/null || echo "NOT_FOUND")

    echo "$status"
}

function command_create() {
    local zones=$1
    local workers_per_zone=$2

    if [ -z "$zones" ] || [ -z "$workers_per_zone" ]; then
        echo -e "${RED}Error: --zones and --workers_per_zone required for create${NC}"
        print_usage
        exit 1
    fi

    echo -e "${GREEN}=========================================="
    echo "CREATING TPUs"
    echo -e "==========================================${NC}"
    echo "Zones: $zones"
    echo "Workers per zone: $workers_per_zone"
    echo "TPU type: $TPU_TYPE"
    echo "Runtime: $RUNTIME_VERSION"
    echo "Preemptible: $PREEMPTIBLE"
    echo ""

    IFS=',' read -ra ZONE_ARRAY <<< "$zones"

    local total_tpus=$((${#ZONE_ARRAY[@]} * workers_per_zone))
    echo "Total TPUs to create: $total_tpus"
    echo ""

    local created=0
    local failed=0

    for zone in "${ZONE_ARRAY[@]}"; do
        echo -e "${YELLOW}--- Zone: $zone ---${NC}"

        for worker_id in $(seq 0 $((workers_per_zone - 1))); do
            if create_tpu "$zone" "$worker_id"; then
                ((created++))
            else
                ((failed++))
            fi
            sleep 2  # Avoid rate limiting
        done

        echo ""
    done

    echo -e "${GREEN}=========================================="
    echo "CREATE SUMMARY"
    echo -e "==========================================${NC}"
    echo "✓ Created: $created"
    echo "✗ Failed: $failed"
    echo ""
}

function command_delete() {
    local zones=$1
    local workers_per_zone=$2

    if [ -z "$zones" ] || [ -z "$workers_per_zone" ]; then
        echo -e "${RED}Error: --zones and --workers_per_zone required for delete${NC}"
        print_usage
        exit 1
    fi

    echo -e "${YELLOW}=========================================="
    echo "DELETING TPUs"
    echo -e "==========================================${NC}"
    echo "Zones: $zones"
    echo "Workers per zone: $workers_per_zone"
    echo ""

    IFS=',' read -ra ZONE_ARRAY <<< "$zones"

    local total_tpus=$((${#ZONE_ARRAY[@]} * workers_per_zone))
    echo "Total TPUs to delete: $total_tpus"
    echo ""

    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled."
        exit 0
    fi

    local deleted=0
    local failed=0

    for zone in "${ZONE_ARRAY[@]}"; do
        echo -e "${YELLOW}--- Zone: $zone ---${NC}"

        for worker_id in $(seq 0 $((workers_per_zone - 1))); do
            if delete_tpu "$zone" "$worker_id"; then
                ((deleted++))
            else
                ((failed++))
            fi
        done

        echo ""
    done

    echo -e "${GREEN}=========================================="
    echo "DELETE SUMMARY"
    echo -e "==========================================${NC}"
    echo "✓ Deleted: $deleted"
    echo "✗ Failed: $failed"
    echo ""
}

function command_status() {
    local zones=$1

    if [ -z "$zones" ]; then
        echo -e "${RED}Error: --zones required for status${NC}"
        print_usage
        exit 1
    fi

    echo -e "${GREEN}=========================================="
    echo "TPU STATUS"
    echo -e "==========================================${NC}"
    echo ""

    IFS=',' read -ra ZONE_ARRAY <<< "$zones"

    local project_flag=""
    if [ -n "$PROJECT" ]; then
        project_flag="--project=$PROJECT"
    fi

    for zone in "${ZONE_ARRAY[@]}"; do
        echo -e "${YELLOW}--- Zone: $zone ---${NC}"

        # List all TPUs in this zone matching our naming pattern
        local region=$(echo "$zone" | rev | cut -d'-' -f2- | rev)
        local zone_letter=$(echo "$zone" | rev | cut -d'-' -f1 | rev)
        local pattern="tpu-${region}-${zone_letter}-"

        gcloud compute tpus tpu-vm list \
            --zone="$zone" \
            $project_flag \
            --filter="name:$pattern*" \
            --format="table(name,state,health,accelerator_type)" 2>/dev/null || {
                echo "  No TPUs found or error querying"
            }

        echo ""
    done
}

function command_recreate_preempted() {
    local zones=$1
    local workers_per_zone=$2

    if [ -z "$zones" ] || [ -z "$workers_per_zone" ]; then
        echo -e "${RED}Error: --zones and --workers_per_zone required${NC}"
        print_usage
        exit 1
    fi

    echo -e "${GREEN}=========================================="
    echo "RECREATING PREEMPTED TPUs"
    echo -e "==========================================${NC}"
    echo ""

    IFS=',' read -ra ZONE_ARRAY <<< "$zones"

    local recreated=0
    local skipped=0

    for zone in "${ZONE_ARRAY[@]}"; do
        echo -e "${YELLOW}--- Zone: $zone ---${NC}"

        for worker_id in $(seq 0 $((workers_per_zone - 1))); do
            local status=$(get_tpu_status "$zone" "$worker_id")
            local tpu_name=$(get_tpu_name "$zone" "$worker_id")

            if [ "$status" = "PREEMPTED" ] || [ "$status" = "NOT_FOUND" ]; then
                echo -e "${YELLOW}TPU $tpu_name is $status - recreating...${NC}"

                # Delete if exists
                if [ "$status" = "PREEMPTED" ]; then
                    delete_tpu "$zone" "$worker_id"
                    sleep 5
                fi

                # Recreate
                if create_tpu "$zone" "$worker_id"; then
                    ((recreated++))
                fi
            else
                echo -e "${GREEN}✓ TPU $tpu_name is $status - skipping${NC}"
                ((skipped++))
            fi
        done

        echo ""
    done

    echo -e "${GREEN}=========================================="
    echo "RECREATE SUMMARY"
    echo -e "==========================================${NC}"
    echo "✓ Recreated: $recreated"
    echo "⊘ Skipped (healthy): $skipped"
    echo ""
}

function command_list() {
    echo -e "${GREEN}=========================================="
    echo "ALL TPUs (across all zones)"
    echo -e "==========================================${NC}"
    echo ""

    local project_flag=""
    if [ -n "$PROJECT" ]; then
        project_flag="--project=$PROJECT"
    fi

    gcloud compute tpus tpu-vm list \
        $project_flag \
        --filter="name:tpu-*" \
        --format="table(name,zone,state,health,accelerator_type)" || {
            echo "No TPUs found or error querying"
        }

    echo ""
}

# Parse command
COMMAND=$1
shift || {
    print_usage
    exit 1
}

# Parse options
ZONES=""
WORKERS_PER_ZONE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --zones) ZONES="$2"; shift ;;
        --workers_per_zone) WORKERS_PER_ZONE="$2"; shift ;;
        --tpu_type) TPU_TYPE="$2"; ACCELERATOR_TYPE="$2"; shift ;;
        --runtime) RUNTIME_VERSION="$2"; shift ;;
        --no-preemptible) PREEMPTIBLE=false ;;
        --project) PROJECT="$2"; shift ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; print_usage; exit 1 ;;
    esac
    shift
done

# Execute command
case $COMMAND in
    create)
        command_create "$ZONES" "$WORKERS_PER_ZONE"
        ;;
    delete)
        command_delete "$ZONES" "$WORKERS_PER_ZONE"
        ;;
    status)
        command_status "$ZONES"
        ;;
    recreate-preempted)
        command_recreate_preempted "$ZONES" "$WORKERS_PER_ZONE"
        ;;
    list)
        command_list
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        print_usage
        exit 1
        ;;
esac
