# Contributing to Qwen Activation Extraction Framework

Thank you for your interest in contributing! This document provides guidelines for development.

## Development Setup

1. **Clone and setup environment**:
   ```bash
   cd /path/to/qwen/refactored
   pip install -e ".[dev]"  # If setup.py exists
   # OR install dependencies manually:
   pip install jax jaxlib flax transformers datasets gcsfs tqdm pytest
   ```

2. **Run tests**:
   ```bash
   python -m pytest tests/
   ```

## Code Style

### Python Standards

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use docstrings for all public functions and classes

### Import Order

1. Standard library imports
2. Third-party imports (jax, flax, transformers, etc.)
3. Local imports (relative imports within package)

Example:
```python
import json
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from .config import QwenConfig
from .qwen import QwenModel
```

### Docstrings

Use Google-style docstrings:

```python
def process_batch(
    inputs: jnp.ndarray,
    config: QwenConfig,
    verbose: bool = False
) -> Dict[str, jnp.ndarray]:
    """
    Process a batch of inputs through the model.
    
    Args:
        inputs: Input token IDs with shape [batch, seq_len]
        config: Model configuration
        verbose: Print progress messages
        
    Returns:
        Dictionary containing:
            - 'logits': Output logits [batch, seq_len, vocab]
            - 'hidden': Final hidden states [batch, seq_len, hidden]
            
    Raises:
        ValueError: If input shape is invalid
    """
```

## Module Guidelines

### Adding New Features

1. **Determine the appropriate module**:
   - Model changes → `model/`
   - ARC-related → `arc/`
   - Dataset handling → `data/`
   - Extraction pipeline → `extraction/`

2. **Update module `__init__.py`**:
   - Add new exports to the appropriate `__init__.py`
   - Update `__all__` list

3. **Update main `__init__.py`** (if public API):
   - Add to root package exports for convenience

### Creating New Modules

1. Create module directory with `__init__.py`
2. Define clear public API in `__init__.py`
3. Use `__all__` to control exports
4. Update parent `__init__.py`
5. Add documentation to README.md

## Testing Guidelines

### Test Structure

```
tests/
├── test_model.py       # Model tests
├── test_arc.py         # ARC module tests
├── test_data.py        # Data module tests
└── test_extraction.py  # Extraction tests
```

### Writing Tests

```python
import pytest
from refactored.model import QwenConfig, get_default_config

def test_default_config():
    """Test default configuration values."""
    config = get_default_config()
    assert config.hidden_size == 896
    assert config.num_hidden_layers == 24

def test_config_from_hf():
    """Test loading config from HuggingFace."""
    from refactored.model import config_from_hf
    config = config_from_hf("Qwen/Qwen2.5-0.5B-Instruct")
    assert config.vocab_size > 0
```

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**:
   - Follow code style guidelines
   - Add/update tests
   - Update documentation

3. **Test locally**:
   ```bash
   python -m pytest tests/
   python -c "from refactored import *"  # Check imports
   ```

4. **Submit PR**:
   - Clear description of changes
   - Reference any related issues
   - Ensure CI passes

## Common Tasks

### Adding a New Grid Encoder

1. Create encoder class in `arc/encoders.py`:
   ```python
   class MyNewEncoder(GridEncoder):
       def to_text(self, grid: List[List[int]]) -> str:
           # Implementation
           
       def to_grid(self, text: str) -> List[List[int]]:
           # Implementation
   ```

2. Add to factory in `create_grid_encoder()`

3. Export in `arc/__init__.py`

### Adding a New Extraction Mode

1. Add configuration options to `ExtractionConfig`

2. Implement logic in `extraction/extractor.py`

3. Update CLI in `extract_activations.py`

4. Document in README.md

## Architecture Decisions

### Why JAX/Flax?

- Native TPU support with minimal overhead
- Functional programming model suits activation extraction
- Easy parallelization across devices

### Why Modular Structure?

- Clear separation of concerns
- Easy to test individual components
- Enables code reuse across projects
- Simplifies onboarding for new developers

### Why Sharded Storage?

- Prevents memory issues with large activations
- Enables parallel processing in distributed setups
- Simplifies GCS upload for cloud workflows

## Questions?

Open an issue for any questions about contributing.
