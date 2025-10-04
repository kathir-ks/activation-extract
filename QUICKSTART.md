## Quick Start Guide: ARC-AGI Mechanistic Interpretability Pipeline

Complete end-to-end guide for extracting activations from your ARC-AGI model for SAE training.

## 📋 Overview

This pipeline allows you to:
1. Transform HuggingFace datasets to ARC-AGI format
2. Run inference on TPU pods (v4-64, v5e-64)
3. Extract layer activations during inference
4. Store activations locally and/or in cloud storage
5. Use activations for Sparse Autoencoder (SAE) training

##  Step-by-Step Instructions

### Step 1: Test Dataset Structure

First, inspect the HuggingFace dataset structure:

```bash
python test_transformation.py
```

This shows you the structure of the dataset before transformation.

### Step 2: Transform Dataset

Transform the HuggingFace dataset to ARC-AGI format:

```bash
# Transform small sample for testing (10 tasks)
python transform_hf_to_arc.py \
    --dataset barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
    --output_dir ./arc_data \
    --split train \
    --max_samples 10

# Transform full dataset (or larger subset)
python transform_hf_to_arc.py \
    --dataset barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
    --output_dir ./arc_data \
    --split train \
    --max_samples 10000  # Adjust as needed
```

**Output:**
- `arc_data/arc_format_train.json` - Tasks in ARC-AGI format
- `arc_data/test_outputs_train.json` - Ground truth outputs

### Step 3: Run Inference with Activation Extraction

#### Option A: Simple Sequential Inference (Good for testing)

```bash
python simple_extraction_inference.py \
    --model_path YOUR_MODEL_PATH \
    --dataset_path arc_data/arc_format_train.json \
    --output_path results/predictions.json \
    --activations_dir results/activations \
    --max_tasks 100 \
    --save_every_n_samples 50
```

####Option B: Distributed TPU Inference (Production)

```bash
# For v4-64 or v5e-64
python distributed_inference_with_activations.py \
    --model_path YOUR_MODEL_PATH \
    --dataset_path arc_data/arc_format_train.json \
    --output_filepath results/submission.json \
    --activations_dir results/activations \
    --batch_size 8 \
    --extract_activations \
    --layers_to_extract 0 11 23 \
    --mesh_shape 8 8 \
    --save_every_n_batches 10
```

With cloud storage:
```bash
python distributed_inference_with_activations.py \
    --model_path YOUR_MODEL_PATH \
    --dataset_path arc_data/arc_format_train.json \
    --output_filepath results/submission.json \
    --activations_dir results/activations \
    --extract_activations \
    --layers_to_extract 0 11 23 \
    --upload_to_cloud \
    --cloud_bucket gs://your-bucket/arc-activations
```

### Step 4: Load Activations for SAE Training

```python
import pickle
import json
import numpy as np

# Load metadata
with open('results/activations/metadata.json') as f:
    metadata = json.load(f)

print(f"Total samples: {metadata['total_samples']}")
print(f"Total batches: {metadata['total_batches']}")

# Load all activations
all_activations = []

for file_info in metadata['files']:
    filepath = f"results/activations/{file_info['filename']}"
    with open(filepath, 'rb') as f:
        batch_data = pickle.load(f)

    for item in batch_data:
        all_activations.append(item['output_logits'])

# Concatenate
activations_array = np.concatenate(all_activations, axis=0)
print(f"Activations shape: {activations_array.shape}")

# Now use for SAE training
```

## 📁 File Structure

After running the full pipeline:

```
torch_xla/qwen/
├── arc_data/
│   ├── arc_format_train.json          # Transformed dataset
│   └── test_outputs_train.json        # Ground truth outputs
│
├── results/
│   ├── predictions.json                # Model predictions
│   └── activations/
│       ├── metadata.json               # Extraction metadata
│       ├── activations_batch_000001.pkl
│       ├── activations_batch_000002.pkl
│       └── ...
│
└── [all the scripts]
```

## ⚙️ Configuration Options

### Mesh Shapes for Different TPU Configurations

| TPU Type | Cores | Recommended Mesh | Description |
|----------|-------|------------------|-------------|
| v4-8     | 8     | `(1, 8)`        | Single host, model parallel |
| v4-16    | 16    | `(2, 8)`        | 2-way data, 8-way model |
| v4-32    | 32    | `(4, 8)`        | 4-way data, 8-way model |
| v4-64    | 64    | `(8, 8)`        | 8-way data, 8-way model |
| v5e-64   | 64    | `(8, 8)`        | 8-way data, 8-way model |

### Layer Extraction Recommendations

For a 24-layer model (like Qwen2-0.5B):
- **Quick analysis**: `[0, 11, 23]` (first, middle, last)
- **Detailed analysis**: `[0, 5, 11, 17, 23]` (every ~6 layers)
- **Full analysis**: `[0, 1, 2, ..., 23]` (all layers - large storage!)

### Batch Size Guidelines

| TPU Cores | Recommended Batch Size per Core | Total Batch Size |
|-----------|---------------------------------|------------------|
| 8         | 8-16                            | 64-128           |
| 16        | 8-16                            | 128-256          |
| 32        | 8-16                            | 256-512          |
| 64        | 8-16                            | 512-1024         |

## 🐛 Troubleshooting

### Dataset transformation fails
```bash
# Check dataset structure first
python test_transformation.py

# Start with small sample
python transform_hf_to_arc.py --max_samples 1
```

### Out of memory during inference
```bash
# Reduce batch size
--batch_size 4

# Save more frequently
--save_every_n_samples 25
```

### Cloud upload fails
```bash
# Install gcloud SDK
pip install google-cloud-storage

# Authenticate
gcloud auth application-default login

# Verify bucket exists
gsutil ls gs://your-bucket/
```

### TPU not recognized
```bash
# Check TPU availability
python -c "import jax; print(jax.devices())"

# Check TPU type
python -c "import jax; print(len(jax.devices()))"
```

## 💡 Tips & Best Practices

1. **Start Small**: Always test with `--max_tasks 10` first
2. **Save Frequently**: Use small `save_every_n_samples` values (50-100)
3. **Monitor Storage**: Activations can be large! Check disk space
4. **Use Cloud Storage**: For large extractions, upload to GCS automatically
5. **Test Locally First**: Run on CPU/GPU before using expensive TPU time

## 📊 Expected Storage Requirements

For 10,000 tasks with 3 layers:
- **Activations**: ~5-10 GB per layer
- **Total**: ~15-30 GB for 3 layers
- **Metadata**: ~1 MB

For 100,000 tasks:
- **Total**: ~150-300 GB for 3 layers

## 🚀 Performance Expectations

### v4-64 (64 TPU cores)
- **Throughput**: ~5000 tokens/second
- **10K tasks**: ~30-45 minutes
- **100K tasks**: ~5-7 hours

### v5e-64 (64 TPU cores)
- **Throughput**: ~8000 tokens/second
- **10K tasks**: ~20-30 minutes
- **100K tasks**: ~3-5 hours

## 🎯 Next Steps for SAE Training

After extracting activations:

1. **Preprocess activations**:
   ```python
   # Normalize, reshape, etc.
   activations = activations / activations.std()
   ```

2. **Split train/val**:
   ```python
   from sklearn.model_selection import train_test_split
   train_acts, val_acts = train_test_split(activations, test_size=0.1)
   ```

3. **Train SAE**:
   ```python
   # Use your preferred SAE library
   # e.g., https://github.com/openai/sparse_autoencoder
   ```

4. **Analyze features**:
   - Identify interpretable features
   - Measure feature sparsity
   - Correlate with model behavior

## 📖 Additional Resources

- **Full Documentation**: `README_DISTRIBUTED_INFERENCE.md`
- **Architecture Details**: `distributed_inference_with_activations.py`
- **Test Suite**: `test_arc_inference.py`
- **Examples**: `example_usage.py`

## 🆘 Getting Help

If you encounter issues:
1. Check this quickstart guide
2. Read `README_DISTRIBUTED_INFERENCE.md`
3. Inspect sample data with `test_transformation.py`
4. Start with minimal configuration and gradually increase

## ✅ Validation Checklist

Before full run:
- [ ] Dataset transforms correctly (`test_transformation.py`)
- [ ] Small inference works (`--max_tasks 1`)
- [ ] Activations save correctly
- [ ] Cloud upload works (if using)
- [ ] TPU is recognized and available
- [ ] Sufficient storage space available

## 🎉 Success Indicators

You'll know it's working when:
- ✅ Transformation completes without errors
- ✅ Inference progresses with TQDM bar
- ✅ Activation files appear in `activations/` directory
- ✅ Metadata.json contains expected batch count
- ✅ Can load and inspect activations with pickle

Happy mechanistic interpretation! 🔬🧠
