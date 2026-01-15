#!/bin/bash
# Timing analysis script for activation extraction

set -e

echo "========================================="
echo "ACTIVATION EXTRACTION TIMING ANALYSIS"
echo "========================================="
echo ""

# Configuration
DATASET="test_data_small.json"
OUTPUT_DIR="timing_test_output"
TIMING_LOG="timing_results.txt"

# Clean previous runs
rm -rf ${OUTPUT_DIR}
rm -f ${TIMING_LOG}

echo "Configuration:" | tee -a ${TIMING_LOG}
echo "  Dataset: ${DATASET}" | tee -a ${TIMING_LOG}
echo "  Output: ${OUTPUT_DIR}" | tee -a ${TIMING_LOG}
echo "" | tee -a ${TIMING_LOG}

# Test 1: Small run (2 tasks, 5 predictions)
echo "=========================================" | tee -a ${TIMING_LOG}
echo "TEST 1: Small dataset (2 tasks x 5 preds)" | tee -a ${TIMING_LOG}
echo "=========================================" | tee -a ${TIMING_LOG}
echo "Start time: $(date)" | tee -a ${TIMING_LOG}
START_TIME=$(date +%s)

/usr/bin/time -v python3 extract_activations_arc.py \
    --dataset_path ${DATASET} \
    --n_tasks 2 \
    --predictions_per_task 5 \
    --output_dir ${OUTPUT_DIR}_test1 \
    --shard_size_gb 0.001 \
    --compress_shards \
    2>&1 | tee -a ${TIMING_LOG}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "" | tee -a ${TIMING_LOG}
echo "End time: $(date)" | tee -a ${TIMING_LOG}
echo "Duration: ${DURATION} seconds" | tee -a ${TIMING_LOG}
echo "" | tee -a ${TIMING_LOG}

# Count output
NUM_SHARDS=$(ls -1 ${OUTPUT_DIR}_test1/*.pkl.gz 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh ${OUTPUT_DIR}_test1 | cut -f1)
echo "Output:" | tee -a ${TIMING_LOG}
echo "  Shards: ${NUM_SHARDS}" | tee -a ${TIMING_LOG}
echo "  Total size: ${TOTAL_SIZE}" | tee -a ${TIMING_LOG}
echo "" | tee -a ${TIMING_LOG}

# Test 2: Medium run (2 tasks, 10 predictions)
echo "=========================================" | tee -a ${TIMING_LOG}
echo "TEST 2: Medium dataset (2 tasks x 10 preds)" | tee -a ${TIMING_LOG}
echo "=========================================" | tee -a ${TIMING_LOG}
echo "Start time: $(date)" | tee -a ${TIMING_LOG}
START_TIME=$(date +%s)

/usr/bin/time -v python3 extract_activations_arc.py \
    --dataset_path ${DATASET} \
    --n_tasks 2 \
    --predictions_per_task 10 \
    --output_dir ${OUTPUT_DIR}_test2 \
    --shard_size_gb 0.001 \
    --compress_shards \
    2>&1 | tee -a ${TIMING_LOG}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "" | tee -a ${TIMING_LOG}
echo "End time: $(date)" | tee -a ${TIMING_LOG}
echo "Duration: ${DURATION} seconds" | tee -a ${TIMING_LOG}
echo "" | tee -a ${TIMING_LOG}

# Count output
NUM_SHARDS=$(ls -1 ${OUTPUT_DIR}_test2/*.pkl.gz 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh ${OUTPUT_DIR}_test2 | cut -f1)
echo "Output:" | tee -a ${TIMING_LOG}
echo "  Shards: ${NUM_SHARDS}" | tee -a ${TIMING_LOG}
echo "  Total size: ${TOTAL_SIZE}" | tee -a ${TIMING_LOG}
echo "" | tee -a ${TIMING_LOG}

# Extract key metrics
echo "=========================================" | tee -a ${TIMING_LOG}
echo "SUMMARY AND PROJECTIONS" | tee -a ${TIMING_LOG}
echo "=========================================" | tee -a ${TIMING_LOG}

# Parse timing results
echo "" | tee -a ${TIMING_LOG}
echo "Calculating projections..." | tee -a ${TIMING_LOG}

# Simple projection calculation
python3 << 'EOF' | tee -a ${TIMING_LOG}
import json
from pathlib import Path

# Read test results
try:
    with open('timing_test_output_test1/metadata.json') as f:
        test1_meta = json.load(f)
    with open('timing_test_output_test2/metadata.json') as f:
        test2_meta = json.load(f)

    print("\nTest 1 (2 tasks x 5 preds):")
    print(f"  Total samples: {test1_meta['total_samples']}")
    print(f"  Total shards: {test1_meta['total_shards']}")

    print("\nTest 2 (2 tasks x 10 preds):")
    print(f"  Total samples: {test2_meta['total_samples']}")
    print(f"  Total shards: {test2_meta['total_shards']}")

except Exception as e:
    print(f"Error reading metadata: {e}")

# Projection for large datasets
print("\n" + "="*50)
print("PROJECTIONS FOR LARGE-SCALE EXTRACTION")
print("="*50)
print("\nAssuming barc200 dataset characteristics:")
print("  Tasks: 200")
print("  Predictions per task: 100")
print("  Total prompts: 20,000")
print("  Layers: 14")
print("  Total samples: 280,000")
print("\nBased on current timing (assuming linear scaling):")

# Read timing log for actual durations
import re
with open('timing_results.txt') as f:
    log = f.read()

# Extract durations
durations = re.findall(r'Duration: (\d+) seconds', log)
if len(durations) >= 2:
    test1_time = int(durations[0])
    test2_time = int(durations[1])

    # Samples in test runs
    test1_samples = 2 * 5 * 14  # tasks * preds * layers
    test2_samples = 2 * 10 * 14

    # Calculate rate
    rate1 = test1_samples / test1_time if test1_time > 0 else 0
    rate2 = test2_samples / test2_time if test2_time > 0 else 0
    avg_rate = (rate1 + rate2) / 2

    print(f"\n  Test 1: {rate1:.2f} samples/second")
    print(f"  Test 2: {rate2:.2f} samples/second")
    print(f"  Average: {avg_rate:.2f} samples/second")

    # Project for 200 tasks x 100 preds
    large_samples = 200 * 100 * 14
    if avg_rate > 0:
        projected_time = large_samples / avg_rate
        hours = projected_time / 3600
        days = hours / 24

        print(f"\nSingle machine (v4 TPU):")
        print(f"  Total time: {projected_time:.0f} seconds")
        print(f"  Time: {hours:.1f} hours ({days:.2f} days)")

        # 16 machines
        time_16 = projected_time / 16
        hours_16 = time_16 / 3600

        print(f"\n16 machines (8 v5e + 8 v6e):")
        print(f"  Total time: {time_16:.0f} seconds")
        print(f"  Time: {hours_16:.1f} hours ({hours_16/24:.2f} days)")

        # Throughput
        throughput_single = 200 / hours if hours > 0 else 0
        throughput_16 = 200 / hours_16 if hours_16 > 0 else 0

        print(f"\nThroughput:")
        print(f"  Single machine: {throughput_single:.1f} tasks/hour")
        print(f"  16 machines: {throughput_16:.1f} tasks/hour")

EOF

echo "" | tee -a ${TIMING_LOG}
echo "=========================================" | tee -a ${TIMING_LOG}
echo "Timing analysis complete!"
echo "Results saved to: ${TIMING_LOG}"
echo "=========================================" | tee -a ${TIMING_LOG}
