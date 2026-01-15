# Qwen Activation Extraction Framework

A modular, maintainable codebase for extracting activations from Qwen models on TPU infrastructure.

## Features

- **Multi-host TPU Support**: Efficiently distribute extraction across multiple TPU hosts
- **ARC-AGI Integration**: Built-in support for Abstraction and Reasoning Challenge tasks
- **Modular Design**: Clean separation of concerns with well-defined module boundaries
- **Flexible Storage**: Automatic sharding with optional GCS upload

## Installation

```bash
# Clone the repository (if applicable)
cd /path/to/qwen/refactored

# Install dependencies
pip install jax jaxlib flax transformers datasets gcsfs tqdm termcolor jinja2
```

## Quick Start

### 1. Convert Dataset to ARC Format

```bash
# Convert HuggingFace dataset to ARC format
python convert_dataset.py from_hf \
    --dataset_name "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems" \
    --output_file data/arc_tasks.jsonl \
    --max_tasks 1000
```

### 2. Extract Activations

```bash
# Single-host extraction
python extract_activations.py \
    --model_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_path data/arc_tasks.jsonl \
    --output_dir ./activations \
    --layers 4 8 12 16 20

# Multi-host extraction with GCS upload
python extract_activations.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_path data/arc_tasks.jsonl \
    --output_dir ./activations \
    --use_multihost \
    --machine_id 0 \
    --total_machines 4 \
    --upload_to_gcs \
    --gcs_bucket my-bucket
```

### 3. Use as a Library

```python
from refactored import (
    QwenConfig, QwenModel, create_model_with_hooks,
    load_arc_dataset_jsonl, create_grid_encoder,
    ActivationStorage, ExtractionConfig, run_extraction
)

# Quick extraction
config = ExtractionConfig(
    model_path="Qwen/Qwen2.5-0.5B-Instruct",
    dataset_path="data/arc_tasks.jsonl",
    output_dir="./activations"
)
results = run_extraction(config)
```

## Module Structure

```
refactored/
├── __init__.py          # Package exports
├── extract_activations.py  # CLI entrypoint
├── convert_dataset.py      # Dataset conversion CLI
│
├── model/               # Qwen model implementation
│   ├── __init__.py      # Module exports
│   ├── config.py        # QwenConfig dataclass
│   ├── qwen.py          # Core JAX model
│   ├── hooks.py         # Activation extraction hooks
│   ├── kv_cache.py      # KV cache utilities
│   └── sharding.py      # Multi-host TPU sharding
│
├── arc/                 # ARC-AGI integration
│   ├── __init__.py      # Module exports
│   ├── encoders.py      # Grid encoders
│   ├── prompting.py     # Prompt templates
│   └── augmentation.py  # Data augmentation
│
├── data/                # Dataset utilities
│   ├── __init__.py      # Module exports
│   ├── converter.py     # HF to ARC format
│   ├── loader.py        # Dataset loading
│   └── sharding.py      # Shard management
│
└── extraction/          # Extraction pipeline
    ├── __init__.py      # Module exports
    ├── storage.py       # Activation storage
    └── extractor.py     # Extraction logic
```

## Modules

### model/

Core Qwen model implementation in JAX:

- **config.py**: `QwenConfig` dataclass with factory functions
- **qwen.py**: RMSNorm, MLP, Attention, Model implementations
- **hooks.py**: `QwenModelWithActivations` for layer activation extraction
- **kv_cache.py**: Fixed-size KV cache for efficient generation
- **sharding.py**: Multi-host TPU sharding utilities

### arc/

ARC-AGI task processing:

- **encoders.py**: Grid-to-text encoders (minimal, code block, shape, etc.)
- **prompting.py**: Jinja2 templates for various prompt formats
- **augmentation.py**: Geometric transformations, color swapping

### data/

Dataset management:

- **converter.py**: HuggingFace to ARC format conversion
- **loader.py**: JSONL and sharded dataset loading
- **sharding.py**: `ShardManager` for distributed processing

### extraction/

Activation extraction pipeline:

- **storage.py**: `ActivationStorage` with automatic sharding and GCS upload
- **extractor.py**: `ExtractionConfig` and unified extraction runners

## API Reference

### Model

```python
from refactored.model import (
    QwenConfig,           # Configuration dataclass
    QwenModel,            # Core JAX model
    create_model_with_hooks,  # Create model with activation hooks
    config_from_hf,       # Load config from HuggingFace
    convert_hf_to_jax_weights,  # Convert weights
    DEFAULT_SAE_LAYERS,   # Default layers for SAE training
)
```

### ARC

```python
from refactored.arc import (
    create_grid_encoder,      # Factory for grid encoders
    create_prompts_from_task, # Generate prompts from tasks
    apply_data_augmentation,  # Apply augmentations
)
```

### Data

```python
from refactored.data import (
    load_arc_dataset_jsonl,   # Load from JSONL
    convert_hf_dataset_to_arc_format,  # Convert HF datasets
    ShardManager,             # Manage distributed shards
)
```

### Extraction

```python
from refactored.extraction import (
    ExtractionConfig,     # Configuration for extraction
    run_extraction,       # Run extraction pipeline
    ActivationStorage,    # Storage with sharding
)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for developer guidelines.

## License

MIT License - See LICENSE file for details.
