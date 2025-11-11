# Activation Shard Format Specification

## Overview

This document defines the standard format for storing model activations extracted during inference. The format is designed to be:
- **Generic**: Works across different models, datasets, and extraction scenarios
- **Efficient**: Supports compression and chunked loading
- **Scalable**: Handles large-scale extraction across distributed systems
- **Interoperable**: Easy to load and process in downstream tasks (SAE training, analysis, etc.)

## Directory Structure

```
output_dir/
├── metadata.json                 # Global metadata about the extraction
├── shard_0001.pkl.gz            # First shard (compressed pickle)
├── shard_0002.pkl.gz            # Second shard
├── ...
└── shard_NNNN.pkl.gz            # Last shard
```

## File Formats

### 1. Metadata File (`metadata.json`)

**Purpose**: Describes the entire extraction job and provides an index to all shards.

**Format**:
```json
{
  "version": "1.0",
  "created_at": "2025-01-09T12:00:00Z",
  "extraction_config": {
    "model_name": "Qwen/Qwen2.5-7B",
    "model_architecture": "qwen2",
    "model_params": {
      "hidden_size": 4096,
      "num_hidden_layers": 28,
      "num_attention_heads": 32,
      "vocab_size": 151936
    },
    "dataset_name": "HuggingFaceFW/fineweb-edu",
    "dataset_config": "sample-10BT",
    "dataset_split": "train",
    "layers_extracted": [15, 16, 17, 18],
    "max_seq_length": 512,
    "batch_size": 16
  },
  "distributed_config": {
    "machine_id": 0,
    "total_machines": 32,
    "multihost": false,
    "num_hosts": 1,
    "mesh_type": "1d"
  },
  "shards": [
    {
      "shard_id": 1,
      "filename": "shard_0001.pkl.gz",
      "size_bytes": 52428800,
      "num_samples": 100,
      "layers": [15, 16, 17, 18],
      "sample_id_range": [0, 99],
      "compressed": true,
      "checksum": "sha256:abcd1234..."
    },
    {
      "shard_id": 2,
      "filename": "shard_0002.pkl.gz",
      "size_bytes": 52428800,
      "num_samples": 100,
      "layers": [15, 16, 17, 18],
      "sample_id_range": [100, 199],
      "compressed": true,
      "checksum": "sha256:efgh5678..."
    }
  ],
  "statistics": {
    "total_shards": 10,
    "total_samples": 1000,
    "total_size_bytes": 524288000,
    "layers_extracted": [15, 16, 17, 18],
    "extraction_duration_seconds": 3600.5
  }
}
```

**Required Fields**:
- `version`: Format version (semantic versioning)
- `created_at`: ISO 8601 timestamp
- `extraction_config`: Model and dataset configuration
- `shards`: List of shard metadata
- `statistics`: Summary statistics

### 2. Shard Files (`shard_*.pkl.gz`)

**Purpose**: Store actual activation data in manageable chunks.

**Format**: Compressed pickle file containing a dictionary:

```python
{
  "layer_15": [
    {
      "sample_idx": 0,
      "activation": np.ndarray,  # Shape: [seq_len, hidden_size], dtype: float32
      "shape": (512, 4096),
      "text_preview": "First 200 chars of input text...",
      "metadata": {
        "token_count": 487,
        "dataset_idx": 12345,
        "machine_id": 0
      }
    },
    {
      "sample_idx": 1,
      "activation": np.ndarray,
      "shape": (512, 4096),
      "text_preview": "Another sample text...",
      "metadata": {...}
    }
  ],
  "layer_16": [...],
  "layer_17": [...],
  "layer_18": [...]
}
```

**Shard Structure**:
- Top-level keys: `layer_{idx}` (string)
- Each layer contains a list of activation dictionaries
- Each activation dict has:
  - `sample_idx`: Global sample ID (unique across all shards)
  - `activation`: NumPy array (float32)
  - `shape`: Tuple of activation dimensions
  - `text_preview`: First 200 characters of input text
  - `metadata`: Optional additional metadata

## Naming Conventions

### File Names
- Shards: `shard_{id:04d}.pkl.gz` (zero-padded to 4 digits)
- Metadata: `metadata.json` (fixed name)

### Layer Keys
- Format: `layer_{idx}` where `idx` is the layer index (0-indexed)
- Examples: `layer_0`, `layer_15`, `layer_27`

### Sample IDs
- Global unique identifier across all machines and shards
- Format: `machine_id * samples_per_machine + local_sample_idx`
- Ensures no collisions in distributed extraction

## Compression

**Default**: gzip compression (`.pkl.gz`)
**Alternative**: Uncompressed pickle (`.pkl`) for faster loading

**Trade-offs**:
- Compressed: ~5-10x smaller, ~2x slower to load
- Uncompressed: Faster loading, larger storage

## Loading API

### Python Example

```python
import json
import pickle
import gzip
from pathlib import Path
import numpy as np

def load_metadata(output_dir):
    """Load extraction metadata"""
    with open(Path(output_dir) / "metadata.json") as f:
        return json.load(f)

def load_shard(shard_path, compressed=True):
    """Load a single shard"""
    if compressed:
        with gzip.open(shard_path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(shard_path, 'rb') as f:
            return pickle.load(f)

def iter_activations(output_dir, layer_idx):
    """Iterate over all activations for a specific layer"""
    metadata = load_metadata(output_dir)
    layer_key = f"layer_{layer_idx}"

    for shard_info in metadata['shards']:
        if layer_idx not in shard_info['layers']:
            continue

        shard_path = Path(output_dir) / shard_info['filename']
        shard_data = load_shard(shard_path, compressed=shard_info['compressed'])

        if layer_key in shard_data:
            for item in shard_data[layer_key]:
                yield item

# Usage
for item in iter_activations("./activations", layer_idx=15):
    activation = item['activation']  # numpy array
    sample_idx = item['sample_idx']
    text = item['text_preview']
    # Process activation...
```

## Best Practices

### Shard Sizing
- **Recommended**: 50-500 MB per shard
- **Min**: 10 MB (avoid too many small files)
- **Max**: 2 GB (stay under GCS object size limits)

### Memory Management
- Load shards one at a time
- Use generators/iterators for large datasets
- Clear activations after processing each shard

### Distributed Extraction
- Each machine writes to separate output directory
- Use unique `machine_id` in metadata
- Ensure `sample_idx` is globally unique
- Merge metadata files after extraction complete

### Validation
- Always include checksums in metadata
- Verify shard integrity after transfer
- Check that all sample IDs are unique

## Compatibility

### PyTorch DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class ActivationDataset(Dataset):
    def __init__(self, output_dir, layer_idx):
        self.output_dir = Path(output_dir)
        self.layer_idx = layer_idx
        self.layer_key = f"layer_{layer_idx}"

        # Load metadata and build index
        with open(self.output_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        # Build index: [(shard_id, item_idx), ...]
        self.index = []
        for shard_info in self.metadata['shards']:
            if layer_idx in shard_info['layers']:
                shard_path = self.output_dir / shard_info['filename']
                shard_data = load_shard(shard_path, compressed=shard_info['compressed'])
                num_items = len(shard_data[self.layer_key])
                for i in range(num_items):
                    self.index.append((shard_info['shard_id'], i))

        # Cache current shard
        self.current_shard_id = None
        self.current_shard = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        shard_id, item_idx = self.index[idx]

        # Load shard if not cached
        if shard_id != self.current_shard_id:
            shard_info = next(s for s in self.metadata['shards'] if s['shard_id'] == shard_id)
            shard_path = self.output_dir / shard_info['filename']
            self.current_shard = load_shard(shard_path, compressed=shard_info['compressed'])
            self.current_shard_id = shard_id

        item = self.current_shard[self.layer_key][item_idx]
        return {
            'activation': item['activation'],
            'sample_idx': item['sample_idx'],
            'text': item['text_preview']
        }

# Usage
dataset = ActivationDataset("./activations", layer_idx=15)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    activations = batch['activation']  # [batch_size, seq_len, hidden_size]
    # Train SAE...
```

## Future Extensions

### Version 2.0 (Planned)
- Support for multiple activation types (pre-layer-norm, post-MLP, attention outputs)
- Token-level metadata (position IDs, attention masks)
- Sparse activation format for memory efficiency
- HDF5 backend option for random access

### Version 3.0 (Proposed)
- Streaming format for real-time processing
- Built-in data augmentation hooks
- Integration with Hugging Face datasets library
