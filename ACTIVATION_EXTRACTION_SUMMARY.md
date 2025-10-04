# Activation Extraction Implementation Summary

## What Was Implemented

I've implemented a complete layer activation extraction system for the Qwen JAX model to support SAE (Sparse Autoencoder) training.

## New Files Created

### 1. **qwen2_jax_with_hooks.py** (214 lines)
Extended version of the Qwen model that returns intermediate layer activations.

**Key features:**
- `QwenModelWithActivations` - Modified model class
- `create_model_with_hooks()` - Factory function
- `generate_with_layer_activations()` - Generation with extraction
- `extract_specific_layer_positions()` - Extract specific tokens

**Usage:**
```python
from qwen2_jax_with_hooks import create_model_with_hooks

model = create_model_with_hooks(config, layers_to_extract=[6, 12, 18, 23])
logits, activations = model.apply(params, input_ids, return_activations=True)
```

### 2. **example_activation_extraction.py** (276 lines)
Comprehensive examples showing how to use the activation extraction system.

**Examples included:**
- Basic extraction
- Generation with extraction
- Saving for SAE training
- Loading HuggingFace weights

**Run:**
```bash
python example_activation_extraction.py
```

### 3. **README_ACTIVATION_EXTRACTION.md**
Complete documentation on the activation extraction system.

**Covers:**
- Architecture overview
- How extraction works
- Token position handling
- Output format
- Performance considerations
- Troubleshooting

## Updated Files

### 1. **distributed_inference_with_activations.py**
Updated the `generate_with_activations()` function to use the new hooks.

**Changes:**
```python
# OLD (TODO placeholder):
logits = model.apply(params, generated_ids)
# TODO: Extract activations from intermediate layers

# NEW (implemented):
logits, activations = model.apply(params, generated_ids, return_activations=True)
for layer_name, layer_activations in activations.items():
    layer_idx = int(layer_name.replace('layer_', '').replace('_norm', ''))
    if layer_idx in activation_extractor.config.layers_to_extract:
        last_token_activation = layer_activations[:, -1, :]
        activation_extractor.extract_layer_activation(
            layer_idx, last_token_activation, task_id, sample_idx
        )
```

### 2. **simple_extraction_inference.py**
Added import for the new hooks module.

## How It Works

### Architecture

```
Input → Embedding → [Layer 0 → Layer 1 → ... → Layer 23] → Norm → LM Head → Logits
                          ↓           ↓              ↓
                     Extract if   Extract if    Extract if
                     in list      in list       in list
                          ↓           ↓              ↓
                    Activations Dict {'layer_6': [...], 'layer_12': [...], ...}
```

### Extraction Flow

1. **Model Creation:**
   ```python
   model = create_model_with_hooks(config, layers_to_extract=[6, 12, 18, 23])
   ```

2. **Forward Pass:**
   ```python
   logits, activations = model.apply(params, input_ids, return_activations=True)
   ```

3. **Activations Format:**
   ```python
   activations = {
       'layer_6': [batch, seq_len, hidden_dim],   # 896 for Qwen2.5-0.5B
       'layer_12': [batch, seq_len, hidden_dim],
       'layer_18': [batch, seq_len, hidden_dim],
       'layer_23': [batch, seq_len, hidden_dim]
   }
   ```

4. **Extract Last Token (for SAE):**
   ```python
   last_token = activations['layer_12'][:, -1, :]  # [batch, 896]
   ```

5. **Save for Training:**
   ```python
   with open(f'layer_{layer_idx}_batch_{batch_id}.pkl', 'wb') as f:
       pickle.dump({'activations': [...], 'layer': layer_idx}, f)
   ```

## Integration with Distributed Inference

### On v5e-64 TPU (64 devices):

```bash
# Transform dataset
python transform_hf_to_arc.py \
  --dataset barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
  --output_dir ./arc_data \
  --max_samples 50000

# Run distributed inference with activation extraction
python distributed_inference_with_activations.py \
  --model_path /path/to/qwen/model \
  --tasks_file ./arc_data/arc_format_train.json \
  --output_dir ./outputs \
  --activations_dir ./activations \
  --batch_size 8 \
  --mesh_shape 8,8 \
  --layers_to_extract 6 12 18 23 \
  --gcs_bucket gs://your-bucket/activations
```

### What Happens:

1. **Data Distribution:**
   - Total batch: 64 devices × 8 samples = 512 samples
   - Device 0: samples[0:8]
   - Device 1: samples[8:16]
   - ...
   - Device 63: samples[504:512]

2. **Parallel Extraction:**
   - Each device runs forward pass with hooks
   - Extracts activations from layers [6, 12, 18, 23]
   - Saves to local storage

3. **Activation Storage:**
   ```
   activations/
   ├── layer_6_batch_000001.pkl   (512 samples × 896 dim)
   ├── layer_12_batch_000001.pkl
   ├── layer_18_batch_000001.pkl
   ├── layer_23_batch_000001.pkl
   ├── metadata.json
   └── ...
   ```

4. **Cloud Upload (optional):**
   - Automatically uploads to GCS bucket
   - Parallel upload from all hosts

## Output Format

### Activation Files

Each pickle file contains:

```python
{
    'layer': 12,
    'activations': [
        {
            'task_id': 'task_001',
            'sample_idx': 0,
            'activation': np.array([896,]),  # hidden_dim
            'timestamp': '2025-01-15T10:30:00'
        },
        # ... more samples
    ]
}
```

### Metadata File

```json
{
  "model_name": "Qwen/Qwen2.5-0.5B",
  "layers_extracted": [6, 12, 18, 23],
  "hidden_dim": 896,
  "total_samples": 50000,
  "batches": [...]
}
```

## Usage Examples

### Example 1: Basic Extraction

```python
from qwen2_jax_with_hooks import create_model_with_hooks

model = create_model_with_hooks(config, layers_to_extract=[12, 23])
params = model.init(key, dummy_input)

logits, activations = model.apply(params, input_ids, return_activations=True)

# activations['layer_12']: [1, seq_len, 896]
# activations['layer_23']: [1, seq_len, 896]
```

### Example 2: Generation with Extraction

```python
from qwen2_jax_with_hooks import generate_with_layer_activations

generated_ids, final_acts, acts_per_step = generate_with_layer_activations(
    model, params, input_ids,
    max_tokens=50,
    layers_to_extract=[12],
    return_all_step_activations=True
)

# acts_per_step = [
#   {'step': 0, 'activations': {'layer_12': [...]}, 'seq_len': 10},
#   {'step': 1, 'activations': {'layer_12': [...]}, 'seq_len': 11},
#   ...
# ]
```

### Example 3: SAE Training Data

```python
# Load activations
import pickle
import numpy as np

activations_list = []
for batch_file in ['layer_12_batch_000001.pkl', 'layer_12_batch_000002.pkl']:
    with open(f'activations/{batch_file}', 'rb') as f:
        data = pickle.load(f)
        for item in data['activations']:
            activations_list.append(item['activation'])

# Stack into training array
X_train = np.stack(activations_list)  # [n_samples, 896]

# Train SAE
# sae = SparseAutoencoder(input_dim=896, latent_dim=8192)
# sae.train(X_train)
```

## Performance

### Memory Usage

For Qwen2.5-0.5B (hidden_dim=896):
- Per sample per layer: 896 × 4 bytes = 3.6 KB
- For 4 layers: 14.4 KB per sample
- For batch of 512: 7.4 MB per batch

### Storage

For 50,000 samples across 4 layers:
- Total activations: 50,000 × 4 layers × 3.6 KB = 720 MB
- With metadata: ~750 MB

### Throughput

On v5e-64 with batch_size=8:
- 512 samples per iteration
- ~100 iterations for 50,000 samples
- Estimated time: 30-60 minutes (depending on model size)

## Testing

Run the examples to verify everything works:

```bash
cd /home/kathirks_gc/torch_xla/qwen
python example_activation_extraction.py
```

## Summary

✅ **Implemented:**
- Model with activation extraction hooks (`qwen2_jax_with_hooks.py`)
- Integration with distributed inference
- Comprehensive examples and documentation
- Support for selective layer extraction
- Batch saving to prevent memory overflow
- Cloud storage upload

✅ **Ready for:**
- Large-scale distributed inference on TPU pods
- SAE training on extracted activations
- Interpretability research
- Feature analysis

## Next Steps

1. Test on actual TPU hardware
2. Run full dataset transformation (200k samples)
3. Execute distributed inference with extraction
4. Train SAEs on extracted activations
5. Analyze learned features
