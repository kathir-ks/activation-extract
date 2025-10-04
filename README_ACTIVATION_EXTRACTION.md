# Layer Activation Extraction for SAE Training

This document explains how the layer activation extraction system works for training Sparse Autoencoders (SAEs).

## Overview

The system extracts intermediate layer activations during inference for SAE training. It's implemented through:

1. **qwen2_jax_with_hooks.py** - Model with activation extraction hooks
2. **distributed_inference_with_activations.py** - Distributed inference with extraction
3. **simple_extraction_inference.py** - Simple sequential extraction

## Architecture

### 1. Model with Hooks (`QwenModelWithActivations`)

The model is modified to return intermediate layer activations:

```python
from qwen2_jax_with_hooks import create_model_with_hooks

# Create model that extracts layers [6, 12, 18, 23]
model = create_model_with_hooks(
    config,
    layers_to_extract=[6, 12, 18, 23]
)

# Forward pass returns both logits and activations
logits, activations = model.apply(params, input_ids, return_activations=True)

# activations = {
#     'layer_6': [batch, seq_len, hidden_dim],
#     'layer_12': [batch, seq_len, hidden_dim],
#     'layer_18': [batch, seq_len, hidden_dim],
#     'layer_23': [batch, seq_len, hidden_dim]
# }
```

### 2. How Activations Are Extracted

During each decoder layer:

```python
# qwen2_jax_with_hooks.py, line 55-65
for i in range(self.config.num_hidden_layers):
    hidden_states = QwenDecoderLayer(self.config, name=f'layers_{i}')(
        hidden_states, attention_mask
    )

    # Extract if this layer is requested
    if return_activations:
        if self.layers_to_extract is None or i in self.layers_to_extract:
            activations[f'layer_{i}'] = hidden_states  # [batch, seq, hidden]
```

### 3. Token Position Extraction

For SAE training, we typically extract the **last token position** (the one being predicted):

```python
# distributed_inference_with_activations.py, line 266-270
for layer_name, layer_activations in activations.items():
    if layer_idx in activation_extractor.config.layers_to_extract:
        # Extract last token position
        last_token_activation = layer_activations[:, -1, :]  # [batch, hidden_dim]
        activation_extractor.extract_layer_activation(
            layer_idx, last_token_activation, task_id, sample_idx
        )
```

## Usage

### Simple Inference with Extraction

```bash
python simple_extraction_inference.py \
  --model_path Qwen/Qwen2.5-0.5B \
  --tasks_file arc_tasks.json \
  --output_dir ./outputs \
  --activations_dir ./activations \
  --layers_to_extract 6 12 18 23
```

### Distributed Inference with Extraction

```bash
python distributed_inference_with_activations.py \
  --model_path /path/to/model \
  --tasks_file arc_tasks.json \
  --output_dir ./outputs \
  --activations_dir ./activations \
  --batch_size 8 \
  --mesh_shape 8,8 \
  --layers_to_extract 6 12 18 23 \
  --gcs_bucket gs://your-bucket/activations
```

## Output Format

### Activation Files

Saved as pickle files in `activations_dir/`:

```
activations/
├── layer_6_batch_000001.pkl
├── layer_12_batch_000001.pkl
├── layer_18_batch_000001.pkl
├── layer_23_batch_000001.pkl
├── layer_6_batch_000002.pkl
└── ...
```

### Each pickle file contains:

```python
{
    'layer': 6,
    'activations': [
        {
            'task_id': 'task_abc123',
            'sample_idx': 0,
            'activation': np.array([896,]),  # hidden_dim
            'timestamp': '2025-01-15T10:30:00'
        },
        ...
    ]
}
```

### Metadata File

`activations/metadata.json`:

```json
{
  "model_name": "Qwen/Qwen2.5-0.5B",
  "layers_extracted": [6, 12, 18, 23],
  "hidden_dim": 896,
  "total_samples": 50000,
  "batches": [
    {
      "batch_id": 1,
      "layer": "layer_6",
      "file": "layer_6_batch_000001.pkl",
      "n_samples": 512
    },
    ...
  ]
}
```

## Loading Activations for SAE Training

```python
import pickle
import numpy as np

# Load a batch
with open('activations/layer_12_batch_000001.pkl', 'rb') as f:
    data = pickle.load(f)

layer = data['layer']
activations = [item['activation'] for item in data['activations']]
activations_array = np.stack(activations)  # [n_samples, hidden_dim]

print(f"Layer {layer}: {activations_array.shape}")
# Layer 12: (512, 896)

# Use for SAE training
# sae.train(activations_array)
```

## Which Layers to Extract?

Common choices for Qwen2.5-0.5B (24 layers):

- **Early layers**: [2, 4, 6] - Lower-level features
- **Middle layers**: [10, 12, 14] - Mid-level representations
- **Late layers**: [18, 20, 22, 23] - High-level abstractions
- **Default**: [6, 12, 18, 23] - Evenly distributed

For larger models, scale proportionally.

## Performance Considerations

### Memory Usage

Extracting activations increases memory usage:

```
Memory per sample per layer = hidden_dim × 4 bytes (float32)
```

For Qwen2.5-0.5B (hidden_dim=896):
- Per sample per layer: 896 × 4 = 3.6 KB
- For 4 layers: 14.4 KB per sample
- For batch of 512: 7.4 MB per batch

### Batch Saving

Activations are saved in batches to prevent memory overflow:

```python
# distributed_inference_with_activations.py
save_every_n_samples: int = 512  # Save after 512 samples
```

### Cloud Upload

Automatically upload to GCS:

```python
--gcs_bucket gs://your-bucket/activations
--upload_to_cloud True
```

## Integration with Distributed Inference

### How Sharding Works with Activations

On a v5e-64 TPU (64 devices):

```python
# Each device processes batch_size samples
# Device 0: samples[0:8]
# Device 1: samples[8:16]
# ...
# Device 63: samples[504:512]

# Each device extracts activations independently
# Activations are saved per-device, then merged
```

### Activation Collection Pattern

```python
# distributed_inference_with_activations.py, line 254-274
for step in range(max_tokens):
    # Forward with extraction
    logits, activations = model.apply(params, generated_ids, return_activations=True)

    # Extract from each layer
    for layer_name, layer_activations in activations.items():
        layer_idx = int(layer_name.replace('layer_', '').replace('_norm', ''))
        if layer_idx in layers_to_extract:
            last_token_activation = layer_activations[:, -1, :]
            activation_extractor.extract_layer_activation(
                layer_idx, last_token_activation, task_id, sample_idx
            )

    # Generate next token
    next_token_id = jnp.argmax(logits[:, -1, :], axis=-1)
    generated_ids = jnp.concatenate([generated_ids, next_token_id], axis=1)
```

## Examples

See `example_activation_extraction.py` for comprehensive examples:

```bash
python example_activation_extraction.py
```

Examples include:
1. Basic layer extraction
2. Generation with extraction
3. Saving for SAE training
4. Using real HuggingFace weights

## Troubleshooting

### "Layer not found in activations"

Make sure the layer is in `layers_to_extract`:

```python
model = create_model_with_hooks(config, layers_to_extract=[6, 12, 18, 23])
```

### High memory usage

Reduce `save_every_n_samples` or extract fewer layers:

```python
save_every_n_samples: int = 256  # Save more frequently
layers_to_extract: List[int] = [12, 23]  # Extract fewer layers
```

### Activations shape mismatch

Ensure you're extracting the right token position:

```python
# For SAE training, use last token
last_token = activations['layer_12'][:, -1, :]  # [batch, hidden_dim]

# For sequence analysis, use all tokens
all_tokens = activations['layer_12']  # [batch, seq_len, hidden_dim]
```

## Next Steps

1. Run extraction on full dataset
2. Load activations for SAE training
3. Train SAE on each layer
4. Analyze learned features
5. Use for interpretability research

## References

- Original Qwen model: `qwen2_jax.py`
- Model with hooks: `qwen2_jax_with_hooks.py`
- Distributed extraction: `distributed_inference_with_activations.py`
- Examples: `example_activation_extraction.py`
