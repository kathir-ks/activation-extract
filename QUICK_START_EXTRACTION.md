# Quick Start: Extract Activations from Layers 10-23

## Overview

The system is now configured to extract activations from **layers 10-23** (14 layers total) during inference.

## Run on v5e-64 TPU

### Step 1: Transform Dataset

```bash
python transform_hf_to_arc.py \
  --dataset barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
  --output_dir ./arc_data \
  --max_samples 50000
```

**Output**: `./arc_data/arc_format_train.json`

### Step 2: Run Distributed Inference with Activation Extraction

```bash
python distributed_inference_with_activations.py \
  --model_path /path/to/qwen2.5-0.5b \
  --tasks_file ./arc_data/arc_format_train.json \
  --output_dir ./outputs \
  --activations_dir ./activations \
  --batch_size 8 \
  --mesh_shape 8,8 \
  --gcs_bucket gs://your-bucket/activations
```

### Step 3: Check Results

```bash
# View activation files
ls -lh activations/

# Expected output:
# layer_10_batch_000001.pkl
# layer_11_batch_000001.pkl
# ...
# layer_23_batch_000001.pkl
# metadata.json

# Check metadata
cat activations/metadata.json
```

## What Gets Extracted

### Layers
- **Layer 10** to **Layer 23** (inclusive)
- **14 layers total**
- These are the upper-middle to final layers of the model

### Per Sample
- **Shape**: `[896,]` (hidden dimension)
- **Data type**: float32
- **Size**: 3.6 KB per layer
- **Total per sample**: 14 × 3.6 KB = 50.4 KB

### For 50,000 Samples
- **Total storage**: ~2.52 GB
- **Number of files**: ~1,400 pickle files
- **Batch files**: ~100 batches × 14 layers

## Activation File Format

Each pickle file contains:

```python
import pickle

with open('activations/layer_12_batch_000001.pkl', 'rb') as f:
    data = pickle.load(f)

# data = {
#     'layer': 12,
#     'activations': [
#         {
#             'task_id': 'task_001',
#             'sample_idx': 0,
#             'activation': np.array([896,]),
#             'timestamp': '2025-01-15T10:30:00'
#         },
#         # ... 511 more samples
#     ]
# }
```

## Using with Docker

### Build
```bash
cd /home/kathirks_gc/torch_xla/qwen
./run-docker.sh build
```

### Transform
```bash
./run-docker.sh transform
```

### Run Distributed Inference
```bash
./run-docker.sh distributed
```

## Configuration

The default configuration extracts layers 10-23. To change:

### Extract Different Layers

```bash
# Only extract layers 15-23
python distributed_inference_with_activations.py \
  --model_path /path/to/model \
  --tasks_file arc_tasks.json \
  --layers_to_extract 15 16 17 18 19 20 21 22 23
```

### Extract All Layers (0-23)

```bash
python distributed_inference_with_activations.py \
  --model_path /path/to/model \
  --tasks_file arc_tasks.json \
  --layers_to_extract 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
```

## Loading Activations for SAE Training

```python
import pickle
import numpy as np
import glob

# Load all activations for layer 12
layer_12_files = sorted(glob.glob('activations/layer_12_batch_*.pkl'))

all_activations = []
for filepath in layer_12_files:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        for item in data['activations']:
            all_activations.append(item['activation'])

# Stack into training array
X_train = np.stack(all_activations)  # [n_samples, 896]

print(f"Loaded {X_train.shape[0]} activations of dimension {X_train.shape[1]}")
# Output: Loaded 50000 activations of dimension 896

# Now ready for SAE training
# sae = SparseAutoencoder(input_dim=896, latent_dim=8192)
# sae.fit(X_train)
```

## Performance Estimates

### On v5e-64 (64 TPU cores)

- **Batch size**: 8 per device × 64 = 512 samples/iteration
- **Layers extracted**: 14
- **Activations per iteration**: 512 × 14 = 7,168

### For 50,000 samples:

- **Iterations**: ~98
- **Time per iteration**: ~30-60 seconds
- **Total time**: ~50-100 minutes
- **Output size**: ~2.52 GB

### With Cloud Upload:

Add `--gcs_bucket gs://your-bucket/activations` and activations automatically upload to GCS after each batch.

## Verification

Run the example script to verify setup:

```bash
python example_activation_extraction.py
```

This will show:
- Which layers are being extracted (10-23)
- Activation shapes for each layer
- Example generation with extraction

## Summary

✅ **Configured to extract**: Layers 10-23 (14 layers)
✅ **Storage per 50k samples**: ~2.52 GB
✅ **Output format**: Pickle files, one per layer per batch
✅ **Cloud upload**: Optional GCS integration
✅ **Ready for**: SAE training on late-layer representations

## Next Steps

1. Run transformation on full dataset
2. Execute distributed inference with extraction
3. Verify activation files created
4. Load activations for SAE training
5. Train SAE models on each layer

## Documentation

- **Full details**: `README_ACTIVATION_EXTRACTION.md`
- **Layer config**: `LAYERS_CONFIG.md`
- **Implementation**: `ACTIVATION_EXTRACTION_SUMMARY.md`
