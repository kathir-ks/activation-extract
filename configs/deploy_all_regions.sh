#!/bin/bash
#
# Deploy to All Regions Sequentially
#
# This script deploys to all 4 regions one after another.
# Each deployment runs with monitoring enabled.
#
# RECOMMENDATION: Run each region in a separate terminal for parallel deployment
# and simultaneous monitoring. Use:
#   Terminal 1: bash configs/deploy_us_central1_v5e.sh
#   Terminal 2: bash configs/deploy_europe_west4_v5e.sh
#   Terminal 3: bash configs/deploy_us_east1_v6e.sh
#   Terminal 4: bash configs/deploy_europe_west4_v6e.sh
#

set -e

echo "=========================================="
echo "Multi-Region Deployment"
echo "=========================================="
echo "This will deploy to all 4 regions sequentially."
echo ""
echo "Total capacity:"
echo "  - us-central1-a (v5e-8):   8 workers"
echo "  - europe-west4-b (v5e-8):  8 workers"
echo "  - us-east1-d (v6e-8):      8 workers"
echo "  - europe-west4-a (v6e-8):  8 workers"
echo "  Total: 32 workers"
echo ""
echo "WARNING: Monitoring is enabled for each region."
echo "         This script will wait for you to Ctrl+C each region's monitor."
echo "         Consider running each region in a separate terminal instead."
echo ""

read -p "Continue with sequential deployment? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Region 1: US Central (v5e)
echo "=========================================="
echo "Deploying Region 1: us-central1-a (v5e)"
echo "=========================================="
bash configs/deploy_us_central1_v5e.sh

echo ""
echo "Region 1 deployment complete. Starting Region 2..."
echo ""
sleep 3

# Region 2: Europe West (v5e)
echo "=========================================="
echo "Deploying Region 2: europe-west4-b (v5e)"
echo "=========================================="
bash configs/deploy_europe_west4_v5e.sh

echo ""
echo "Region 2 deployment complete. Starting Region 3..."
echo ""
sleep 3

# Region 3: US East (v6e)
echo "=========================================="
echo "Deploying Region 3: us-east1-d (v6e)"
echo "=========================================="
bash configs/deploy_us_east1_v6e.sh

echo ""
echo "Region 3 deployment complete. Starting Region 4..."
echo ""
sleep 3

# Region 4: Europe West (v6e)
echo "=========================================="
echo "Deploying Region 4: europe-west4-a (v6e)"
echo "=========================================="
bash configs/deploy_europe_west4_v6e.sh

echo ""
echo "=========================================="
echo "All Regions Deployed!"
echo "=========================================="
echo ""
echo "32 TPU workers are now running across 4 regions."
echo "Check progress in each region's monitoring dashboard."
echo ""
