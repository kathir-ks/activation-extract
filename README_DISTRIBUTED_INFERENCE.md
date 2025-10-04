# Distributed TPU Inference with Activation Extraction

Complete pipeline for running ARC-AGI inference on TPU pods (v4-64, v5e-64) with layer activation extraction for SAE training.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE COMPONENTS                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. DATA TRANSFORMATION                                              │
│     HuggingFace Dataset → ARC-AGI Format                            │
│     ├─ transform_hf_to_arc.py                                       │
│     ├─ Input: HF dataset with 'examples' column                     │
│     └─ Output: arc_format_train.json + test_outputs_train.json     │
│                                                                       │
│  2. DISTRIBUTED INFERENCE                                            │
│     Multi-host TPU Inference with Activation Capture                │
│     ├─ distributed_inference_with_activations.py                    │
│     ├─ Uses JAX pmap/pjit for parallelization                       │
│     ├─ Shards data across TPU cores                                 │
│     └─ Extracts activations during forward pass                     │
│                                                                       │
│  3. ACTIVATION STORAGE                                               │
│     Local + Cloud Storage for Activations                           │
│     ├─ Local: ./activations/*.pkl files                             │
│     ├─ Cloud: GCS bucket (optional)                                 │
│     └─ Metadata: tracks layers, batches, shapes                     │
│                                                                       │
│  4. DATA SHARDING & DISTRIBUTION                                     │
│     ├─ Shard prompts across hosts                                   │
│     ├─ Each host processes subset of data                           │
│     └─ Gather results from all hosts                                │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Transform HuggingFace Dataset

```bash
python transform_hf_to_arc.py \
    --dataset barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems \
    --output_dir ./arc_data \
    --split train \
    --max_samples 10000  # Optional: limit samples
```

**Output:**
- `arc_data/arc_format_train.json` - Tasks in ARC-AGI format
- `arc_data/test_outputs_train.json` - Ground truth outputs for verification

### 2. Run Distributed Inference

#### Single Host (e.g., v4-8)
```bash
python distributed_inference_with_activations.py \
    --model_path your-model-path \
    --dataset_path arc_data/arc_format_train.json \
    --output_filepath results/submission.json \
    --activations_dir results/activations \
    --batch_size 8 \
    --extract_activations \
    --layers_to_extract 0 11 23 \
    --mesh_shape 1 8
```

#### Multi-Host (e.g., v4-64, v5e-64)
```bash
# On each host, run:
python distributed_inference_with_activations.py \
    --model_path your-model-path \
    --dataset_path arc_data/arc_format_train.json \
    --output_filepath results/submission.json \
    --activations_dir results/activations \
    --batch_size 8 \
    --extract_activations \
    --layers_to_extract 0 11 23 \
    --mesh_shape 8 8 \  # 8 data parallel, 8 model parallel
    --upload_to_cloud \
    --cloud_bucket gs://your-bucket/activations
```

## Configuration

### DistributedARCConfig Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_path` | HuggingFace model path | `"Qwen/Qwen2.5-0.5B"` |
| `dataset_path` | ARC-AGI format data path | `'arc_data.json'` |
| `batch_size` | Batch size per device | `8` |
| `mesh_shape` | (data_parallel, model_parallel) | `(1, 1)` |
| `extract_activations` | Enable activation extraction | `True` |
| `layers_to_extract` | Which layers to extract | `[0, 11, 23]` |
| `activations_dir` | Local storage directory | `'./activations'` |
| `save_every_n_batches` | Save frequency | `10` |
| `upload_to_cloud` | Enable cloud upload | `False` |
| `cloud_bucket` | GCS bucket path | `None` |

### Mesh Configuration

#### v4-64 (64 TPU cores)
```python
mesh_shape = (8, 8)  # 8-way data, 8-way model parallelism
# or
mesh_shape = (16, 4)  # 16-way data, 4-way model parallelism
```

#### v5e-64 (64 TPU cores)
```python
mesh_shape = (8, 8)  # Balanced parallelism
```

## Activation Storage Structure

```
activations/
├── metadata.json                    # Metadata about extraction
├── layer_0_batch_000001.pkl        # Layer 0, batch 1
├── layer_0_batch_000002.pkl
├── layer_11_batch_000001.pkl       # Layer 11, batch 1
├── layer_11_batch_000002.pkl
├── layer_23_batch_000001.pkl       # Layer 23, batch 1
└── layer_23_batch_000002.pkl
```

### Activation File Format

Each `.pkl` file contains:
```python
[
    {
        'task_id': 'abc123def',
        'sample_idx': 0,
        'activation': np.ndarray,  # Shape: [batch, seq_len, hidden_size]
        'shape': (8, 512, 896)
    },
    ...
]
```

### Metadata Format

`metadata.json`:
```json
{
    "layers_extracted": [0, 11, 23],
    "model_path": "your-model-path",
    "batch_size": 8,
    "batches": [
        {
            "batch_id": 1,
            "layer": "layer_0",
            "file": "layer_0_batch_000001.pkl",
            "n_samples": 64
        },
        ...
    ]
}
```

## Cloud Storage (GCS)

### Setup

1. Install Google Cloud SDK:
```bash
pip install google-cloud-storage
```

2. Authenticate:
```bash
gcloud auth application-default login
```

3. Create bucket:
```bash
gsutil mb gs://your-activations-bucket
```

4. Run with cloud upload:
```bash
python distributed_inference_with_activations.py \
    --upload_to_cloud \
    --cloud_bucket gs://your-activations-bucket/arc-activations
```

## Data Flow

### Input Pipeline
```
HF Dataset Row
├─ examples: [ex1, ex2, ex3, ex4]
│
└─> Transform
    ├─ ex1, ex2, ex3 → train examples
    └─ ex4 → test example (input only)
        └─ output saved separately for verification
```

### Distributed Processing
```
Data (N tasks)
│
├─> Shard across hosts (H hosts)
│   └─ Each host gets N/H tasks
│
├─> Shard across devices per host (D devices)
│   └─ Each device gets N/(H*D) tasks
│
├─> Process in batches (B batch_size)
│   └─ Forward pass with activation capture
│
└─> Gather results
    ├─ Predictions → submission.json
    └─ Activations → activations/*.pkl
```

## Multi-Host Coordination

### Option 1: Manual Sharding
```bash
# Host 0
python distributed_inference_with_activations.py \
    --shard_id 0 --num_shards 8 ...

# Host 1
python distributed_inference_with_activations.py \
    --shard_id 1 --num_shards 8 ...

# ... Host 7
python distributed_inference_with_activations.py \
    --shard_id 7 --num_shards 8 ...
```

### Option 2: JAX Multi-Host (Automatic)
Uses JAX's built-in multi-host coordination with `jax.distributed.initialize()`.

## Loading Activations for SAE Training

```python
import pickle
import json

# Load metadata
with open('activations/metadata.json') as f:
    metadata = json.load(f)

# Load activations for specific layer
layer_activations = []
for batch_info in metadata['batches']:
    if batch_info['layer'] == 'layer_11':
        with open(f"activations/{batch_info['file']}", 'rb') as f:
            batch_data = pickle.load(f)
            for item in batch_data:
                layer_activations.append(item['activation'])

# Concatenate all activations
import numpy as np
all_activations = np.concatenate(layer_activations, axis=0)
print(f"Total activations shape: {all_activations.shape}")
```

## Performance Estimates

### v4-64 (64 cores)
- **Batch size per core**: 8
- **Total batch size**: 512 (64 * 8)
- **Throughput**: ~5000 tokens/second
- **10K tasks**: ~30-45 minutes (depends on task complexity)

### v5e-64 (64 cores)
- **Batch size per core**: 8
- **Total batch size**: 512
- **Throughput**: ~8000 tokens/second
- **10K tasks**: ~20-30 minutes

## Troubleshooting

### Memory Issues
- Reduce `batch_size`
- Reduce `max_model_len`
- Save activations more frequently (`save_every_n_batches`)

### Slow Performance
- Increase `batch_size` if memory allows
- Use more model parallelism if model is large
- Check data loading bottlenecks

### Cloud Upload Failures
- Check GCS permissions
- Verify bucket exists
- Check network connectivity from TPU

## Next Steps

After extraction:
1. Download activations from GCS (if using cloud storage)
2. Preprocess activations for SAE training
3. Train Sparse Autoencoder on extracted activations
4. Use SAE for mechanistic interpretability analysis

## References

- JAX distributed docs: https://jax.readthedocs.io/en/latest/multi_process.html
- TPU documentation: https://cloud.google.com/tpu/docs
- ARC-AGI: https://github.com/fchollet/ARC-AGI
