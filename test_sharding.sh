#!/bin/bash
#
# Test sharding system
#
# This script tests the dataset sharding functionality:
# 1. Creates a test dataset
# 2. Shards it into 8 parts with 100MB chunks
# 3. Tests shard claiming and metadata tracking
#

set -e

echo "========================================================================"
echo "Testing Dataset Sharding System"
echo "========================================================================"

# Setup
TEST_DIR="/tmp/sharding_test_$$"
DATASET_FILE="${TEST_DIR}/test_dataset.jsonl"
SHARDED_DIR="${TEST_DIR}/sharded_dataset"

mkdir -p "$TEST_DIR"

echo ""
echo "[1/5] Creating test dataset (100 tasks)..."
python3 - "$DATASET_FILE" <<'EOF'
import json
import sys

# Create 100 test tasks
output_file = sys.argv[1]
num_tasks = 100

with open(output_file, 'w') as f:
    for i in range(num_tasks):
        task = {
            "task_id": f"task_{i:08x}",
            "train": [
                {
                    "input": [[1, 0, 1] for _ in range(5)],
                    "output": [[0, 1, 0] for _ in range(5)]
                }
            ] * 3,  # 3 training examples
            "test": [
                {"input": [[1, 0, 1] for _ in range(5)]}
            ]
        }
        f.write(json.dumps(task) + '\n')

print(f"✓ Created {num_tasks} tasks")
EOF

echo ""
echo "[2/5] Creating sharded dataset (8 shards, ~0.1MB chunks for testing)..."
python create_sharded_dataset.py \
  --input_file "$DATASET_FILE" \
  --output_dir "$SHARDED_DIR" \
  --num_shards 8 \
  --chunk_size_mb 0.1 \
  --verbose

echo ""
echo "[3/5] Checking shard status..."
python shard_manager.py \
  --dataset_dir "$SHARDED_DIR" \
  --action status

echo ""
echo "[4/5] Testing shard claiming..."
python shard_manager.py \
  --dataset_dir "$SHARDED_DIR" \
  --action claim \
  --worker_id "test_worker_1"

echo ""
echo "[5/5] Checking status after claim..."
python shard_manager.py \
  --dataset_dir "$SHARDED_DIR" \
  --action status

echo ""
echo "========================================================================"
echo "Testing Directory Structure"
echo "========================================================================"
echo ""
echo "Created files:"
find "$SHARDED_DIR" -type f | head -20
echo ""
echo "Master metadata:"
cat "$SHARDED_DIR/master_metadata.json"
echo ""
echo "Shard 0 metadata:"
cat "$SHARDED_DIR/shard_000/metadata.json"

echo ""
echo "========================================================================"
echo "✓ Sharding system test complete!"
echo "========================================================================"
echo ""
echo "Test artifacts in: $TEST_DIR"
echo "To clean up: rm -rf $TEST_DIR"
echo ""
