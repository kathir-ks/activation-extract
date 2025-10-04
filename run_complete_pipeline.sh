#!/bin/bash
#
# Complete Pipeline Runner for ARC-AGI Mechanistic Interpretability
#
# Usage:
#   ./run_complete_pipeline.sh [model_path] [max_samples]
#
# Example:
#   ./run_complete_pipeline.sh "your-model-path" 100
#

set -e  # Exit on error

# Default parameters
MODEL_PATH="${1:-Qwen/Qwen2.5-0.5B}"
MAX_SAMPLES="${2:-10}"
OUTPUT_DIR="./pipeline_output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================================"
echo "ARC-AGI MECHANISTIC INTERPRETABILITY PIPELINE"
echo "======================================================================"
echo "Model: $MODEL_PATH"
echo "Max samples: $MAX_SAMPLES"
echo "Output: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo "======================================================================"
echo ""

# Create output directories
mkdir -p $OUTPUT_DIR/arc_data
mkdir -p $OUTPUT_DIR/results/activations
mkdir -p $OUTPUT_DIR/logs

# Log file
LOG_FILE="$OUTPUT_DIR/logs/pipeline_$TIMESTAMP.log"

echo "Logs will be saved to: $LOG_FILE"
echo ""

# Function to log and execute
run_step() {
    local step_name=$1
    shift
    echo "----------------------------------------------------------------------"
    echo "STEP: $step_name"
    echo "----------------------------------------------------------------------"
    echo "Command: $@"
    echo ""

    # Run command and tee to log
    if "$@" 2>&1 | tee -a $LOG_FILE; then
        echo "✓ $step_name completed successfully"
        echo ""
    else
        echo "✗ $step_name failed!"
        echo "Check log file: $LOG_FILE"
        exit 1
    fi
}

# Step 1: Test dataset structure
run_step "Testing Dataset Structure" \
    python test_transformation.py

# Step 2: Transform dataset
run_step "Transforming Dataset" \
    python transform_hf_to_arc.py \
        --dataset barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
        --output_dir $OUTPUT_DIR/arc_data \
        --split train \
        --max_samples $MAX_SAMPLES

# Step 3: Run inference with activation extraction
run_step "Running Inference with Activation Extraction" \
    python simple_extraction_inference.py \
        --model_path "$MODEL_PATH" \
        --dataset_path $OUTPUT_DIR/arc_data/arc_format_train.json \
        --output_path $OUTPUT_DIR/results/predictions.json \
        --activations_dir $OUTPUT_DIR/results/activations \
        --max_tasks $MAX_SAMPLES \
        --save_every_n_samples 10

# Step 4: Generate summary report
echo "----------------------------------------------------------------------"
echo "GENERATING SUMMARY REPORT"
echo "----------------------------------------------------------------------"

REPORT_FILE="$OUTPUT_DIR/REPORT_$TIMESTAMP.txt"

cat > $REPORT_FILE << EOF
====================================================================
ARC-AGI MECHANISTIC INTERPRETABILITY PIPELINE REPORT
====================================================================
Timestamp: $TIMESTAMP
Model: $MODEL_PATH
Max Samples: $MAX_SAMPLES

OUTPUTS
--------------------------------------------------------------------
1. Transformed Data: $OUTPUT_DIR/arc_data/
   - ARC format: arc_format_train.json
   - Test outputs: test_outputs_train.json

2. Inference Results: $OUTPUT_DIR/results/
   - Predictions: predictions.json

3. Activations: $OUTPUT_DIR/results/activations/
   - Metadata: metadata.json
   - Batch files: activations_batch_*.pkl

4. Logs: $OUTPUT_DIR/logs/
   - Pipeline log: pipeline_$TIMESTAMP.log

ACTIVATION SUMMARY
--------------------------------------------------------------------
EOF

# Add activation summary from metadata
if [ -f "$OUTPUT_DIR/results/activations/metadata.json" ]; then
    python << PYEOF >> $REPORT_FILE
import json
with open("$OUTPUT_DIR/results/activations/metadata.json") as f:
    meta = json.load(f)
    print(f"Total samples: {meta.get('total_samples', 'N/A')}")
    print(f"Total batches: {meta.get('total_batches', 'N/A')}")
    print(f"Start time: {meta.get('start_time', 'N/A')}")
    print(f"End time: {meta.get('end_time', 'N/A')}")
    print(f"\nFiles:")
    for f in meta.get('files', []):
        print(f"  - {f['filename']}: {f['n_samples']} samples")
PYEOF
fi

cat >> $REPORT_FILE << EOF

NEXT STEPS
--------------------------------------------------------------------
1. Inspect activations:
   python << EOF
   import pickle
   with open('$OUTPUT_DIR/results/activations/activations_batch_000001.pkl', 'rb') as f:
       data = pickle.load(f)
       print(f"Loaded {len(data)} samples")
       print(f"First sample keys: {data[0].keys()}")
       print(f"Activation shape: {data[0]['output_logits'].shape}")
   EOF

2. Load all activations for SAE training:
   See QUICKSTART.md for examples

3. Train Sparse Autoencoder on extracted activations

4. Perform mechanistic interpretability analysis

====================================================================
EOF

cat $REPORT_FILE

echo ""
echo "======================================================================"
echo "PIPELINE COMPLETE!"
echo "======================================================================"
echo "Report saved to: $REPORT_FILE"
echo ""
echo "Quick checks:"
echo "  1. View predictions: cat $OUTPUT_DIR/results/predictions.json | jq"
echo "  2. Check activations: ls -lh $OUTPUT_DIR/results/activations/"
echo "  3. Read full report: cat $REPORT_FILE"
echo ""
echo "For more details, see:"
echo "  - QUICKSTART.md"
echo "  - README_DISTRIBUTED_INFERENCE.md"
echo "======================================================================"
