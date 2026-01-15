#!/bin/bash

# Performance verification script for real extraction
# Tests with actual model and measures timing

echo "======================================================================"
echo "PERFORMANCE VERIFICATION - REAL EXTRACTION"
echo "======================================================================"
echo ""
echo "Testing with:"
echo "  - Model: KathirKs/qwen-2.5-0.5b"
echo "  - Dataset: test_gcs_dataset.jsonl"
echo "  - Batch size: 4"
echo "  - Max tasks: 3 (small test)"
echo "  - TPU: 4 devices"
echo ""

OUTPUT_DIR="/tmp/perf_test_$(date +%s)"
mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo "Running extraction with JIT optimizations..."
echo "======================================================================"
echo ""

# Run with timing
time python extract_activations_arc_v5e64.py \
  --dataset_path test_gcs_dataset.jsonl \
  --model_path KathirKs/qwen-2.5-0.5b \
  --batch_size 4 \
  --max_seq_length 512 \
  --max_tasks 3 \
  --output_dir "$OUTPUT_DIR" \
  --gcs_bucket fineweb-data-us-central1-a \
  --verbose 2>&1 | tee "$OUTPUT_DIR/performance.log"

echo ""
echo "======================================================================"
echo "TIMING ANALYSIS"
echo "======================================================================"

# Extract timing information from logs
echo ""
echo "Analyzing compilation logs..."
grep -c "Compiling jit(extract_activations_sharded)" "$OUTPUT_DIR/performance.log" | \
  awk '{if ($1 == 0) print "  ✓ No JIT compilations (using cached compilation)"; else if ($1 == 1) print "  ✓ Single JIT compilation (expected on first run)"; else print "  ⚠ Multiple compilations detected:", $1}'

echo ""
echo "Batch processing info:"
grep "Processing batches" "$OUTPUT_DIR/performance.log" || echo "  (Progress bar hidden in log)"

echo ""
echo "Output:"
ls -lh "$OUTPUT_DIR" | grep -E '\.(pkl|gz)' | awk '{print "  Shard:", $9, "-", $5}'

echo ""
echo "======================================================================"
echo "Performance log saved to: $OUTPUT_DIR/performance.log"
echo "======================================================================"
