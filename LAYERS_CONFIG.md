# Layer Activation Extraction Configuration

## Current Configuration

**Extracting layers: 10 to 23 (inclusive)**

This extracts activations from the **last 14 layers** of the Qwen2.5-0.5B model (which has 24 layers total, indexed 0-23).

## Layer Breakdown

### Qwen2.5-0.5B Architecture
- **Total layers**: 24 (layers 0-23)
- **Hidden dimension**: 896
- **Extracting**: Layers 10-23 (14 layers)

### Why Layers 10-23?

**Layer 10-13: Mid-level representations**
- Transitioning from low-level to high-level features
- Abstract patterns emerging

**Layer 14-19: High-level features**
- Complex semantic understanding
- Task-specific representations

**Layer 20-23: Final representations**
- Most abstract features
- Close to final predictions
- Layer 23 is the last layer before normalization

## Memory & Storage Impact

### Per Sample
- **Per layer**: 896 × 4 bytes = 3.6 KB
- **14 layers**: 14 × 3.6 KB = 50.4 KB per sample
- **Batch of 512**: 512 × 50.4 KB = 25.8 MB per batch

### For 50,000 Samples
- **Total size**: 50,000 × 50.4 KB = 2.52 GB
- **Files created**: ~100 batch files × 14 layers = ~1,400 files

## Configuration Files

### 1. distributed_inference_with_activations.py
```python
layers_to_extract: List[int] = field(default_factory=lambda: list(range(10, 24)))
```

### 2. simple_extraction_inference.py
```python
def __post_init__(self):
    if self.layers_to_extract is None:
        self.layers_to_extract = list(range(10, 24))
```

### 3. qwen2_jax_with_hooks.py
```python
DEFAULT_SAE_LAYERS = list(range(10, 24))  # Layers 10-23
```

## Usage

### Default (extracts layers 10-23)
```bash
python distributed_inference_with_activations.py \
  --model_path /path/to/model \
  --tasks_file arc_tasks.json \
  --batch_size 8 \
  --mesh_shape 8,8
```

### Custom layers
```bash
python distributed_inference_with_activations.py \
  --model_path /path/to/model \
  --tasks_file arc_tasks.json \
  --layers_to_extract 10 11 12 13 14 15 16 17 18 19 20 21 22 23
```

### Subset of layers (e.g., only 15-23)
```bash
python distributed_inference_with_activations.py \
  --model_path /path/to/model \
  --tasks_file arc_tasks.json \
  --layers_to_extract 15 16 17 18 19 20 21 22 23
```

## Output Structure

### Activation Files
```
activations/
├── layer_10_batch_000001.pkl
├── layer_11_batch_000001.pkl
├── layer_12_batch_000001.pkl
├── layer_13_batch_000001.pkl
├── layer_14_batch_000001.pkl
├── layer_15_batch_000001.pkl
├── layer_16_batch_000001.pkl
├── layer_17_batch_000001.pkl
├── layer_18_batch_000001.pkl
├── layer_19_batch_000001.pkl
├── layer_20_batch_000001.pkl
├── layer_21_batch_000001.pkl
├── layer_22_batch_000001.pkl
├── layer_23_batch_000001.pkl
├── layer_10_batch_000002.pkl
├── ...
└── metadata.json
```

### Metadata
```json
{
  "model_name": "Qwen/Qwen2.5-0.5B",
  "layers_extracted": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
  "hidden_dim": 896,
  "num_layers_extracted": 14,
  "total_samples": 50000,
  "batches": [...]
}
```

## Performance on v5e-64

### Extraction Rate
- **Batch size**: 8 per device × 64 devices = 512 samples/batch
- **Layers extracted**: 14
- **Activations per batch**: 512 × 14 = 7,168 layer activations
- **Storage per batch**: ~25.8 MB

### Estimated Time (50,000 samples)
- **Batches needed**: 50,000 / 512 ≈ 98 batches
- **Time per batch**: ~30-60 seconds (including extraction)
- **Total time**: ~50-100 minutes

### Cloud Upload (optional)
```bash
python distributed_inference_with_activations.py \
  --model_path /path/to/model \
  --tasks_file arc_tasks.json \
  --gcs_bucket gs://your-bucket/activations \
  --upload_to_cloud True
```

## Changing the Configuration

### To extract ALL layers (0-23):
```python
# In distributed_inference_with_activations.py or simple_extraction_inference.py
layers_to_extract: List[int] = field(default_factory=lambda: list(range(0, 24)))
```

### To extract only specific layers:
```python
# E.g., only layers 12, 16, 20, 23
layers_to_extract: List[int] = field(default_factory=lambda: [12, 16, 20, 23])
```

### To extract only the last 5 layers:
```python
layers_to_extract: List[int] = field(default_factory=lambda: list(range(19, 24)))
```

## Verification

To verify which layers will be extracted:

```python
from distributed_inference_with_activations import DistributedARCConfig

config = DistributedARCConfig()
print(f"Extracting layers: {config.layers_to_extract}")
print(f"Number of layers: {len(config.layers_to_extract)}")
# Output:
# Extracting layers: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# Number of layers: 14
```

## Summary

✅ **Current setup**: Extracts layers 10-23 (14 layers)
✅ **Storage**: ~2.52 GB for 50,000 samples
✅ **Files**: ~1,400 pickle files + metadata
✅ **Ready for**: SAE training on late-layer representations
