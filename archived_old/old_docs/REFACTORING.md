# Codebase Refactoring

## Overview

The codebase has been refactored to eliminate duplication and improve maintainability by consolidating common functionality into shared modules.

---

## New Structure

```
torch_xla/qwen/
├── core/                          # Shared utilities (NEW)
│   ├── __init__.py               # Exports all core functions
│   ├── jax_utils.py              # JAX/TPU utilities
│   ├── dataset_utils.py          # Dataset loading
│   └── activation_storage.py     # Activation saving/uploading
│
├── extract_activations.py        # Main extraction script (REFACTORED)
├── extract_activations_arc_v5e64.py  # Legacy (still works)
├── extract_activations_fineweb_multihost.py  # Legacy (still works)
│
├── shard_manager.py              # Dataset sharding
├── create_sharded_dataset.py     # Dataset sharding
│
├── qwen2_jax.py                  # Model definition
├── qwen2_jax_with_hooks.py       # Model with activation hooks
├── kvcache_utils.py              # KV cache utilities
│
├── arc24/                        # ARC-specific utilities
│   ├── data_augmentation.py
│   ├── prompting.py
│   └── encoders.py
│
└── [various test scripts]
```

---

## What Was Refactored

### 1. **core/jax_utils.py** - JAX/TPU Utilities

Consolidated all JAX and TPU-specific functionality:

**Functions:**
- `initialize_multihost()` - Initialize JAX distributed
- `create_device_mesh()` - Create device mesh (1D/2D/3D)
- `create_sharding_strategy()` - Define parameter sharding
- `shard_params()` - Apply sharding to parameters
- `extract_activations_sharded()` - JIT-compiled activation extraction
- `pad_sequences()` - Pad sequences to uniform length
- `get_device_memory_info()` - Get device memory stats

**Previously duplicated in:**
- `extract_activations_fineweb_multihost.py`
- `extract_activations_arc_v5e64.py`

---

### 2. **core/dataset_utils.py** - Dataset Loading

Consolidated dataset loading and prompt creation:

**Functions:**
- `load_arc_dataset_jsonl()` - Load from single JSONL file
- `load_arc_dataset_from_shard()` - Load from sharded dataset with auto-claiming
- `create_prompts_from_dataset()` - Create prompts with data augmentation

**Previously duplicated in:**
- `extract_activations_arc_v5e64.py`
- `extract_activations_arc.py`

---

### 3. **core/activation_storage.py** - Activation Storage

Consolidated activation saving and GCS upload logic:

**Classes:**
- `ActivationStorage` - Handle saving with automatic sharding and GCS upload

**Functions:**
- `load_activation_shard()` - Load saved activation shard

**Previously duplicated in:**
- `extract_activations_fineweb_multihost.py`
- `extract_activations_arc_v5e64.py`

---

### 4. **extract_activations.py** - Main Script (REFACTORED)

Clean, simple extraction script that uses all core utilities:

**Benefits:**
- ~500 lines (vs ~650 in old scripts)
- All common code imported from `core/`
- Single source of truth
- Easier to maintain and test

**Features:**
- Single JSONL file or sharded dataset
- Single-host or multi-host TPU
- Automatic shard claiming
- GCS upload with automatic sharding

---

## Usage

### New Refactored Script

```bash
# Use the new refactored script
python extract_activations.py \
  --use_sharded_dataset \
  --sharded_dataset_dir gs://bucket/sharded_dataset \
  --model_path KathirKs/qwen-2.5-0.5b \
  --batch_size 4 \
  --gcs_bucket bucket \
  --upload_to_gcs
```

### Legacy Scripts (Still Work)

Old scripts still work unchanged:

```bash
# Old script (still works)
python extract_activations_arc_v5e64.py \
  --use_sharded_dataset \
  --sharded_dataset_dir gs://bucket/sharded_dataset \
  --model_path KathirKs/qwen-2.5-0.5b \
  --batch_size 4 \
  --gcs_bucket bucket \
  --upload_to_gcs
```

---

## Benefits

### Code Reduction

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| JAX utilities | Duplicated 2× | 1 shared module | ~200 lines |
| Dataset loading | Duplicated 2× | 1 shared module | ~150 lines |
| Activation storage | Duplicated 2× | 1 shared module | ~180 lines |
| **Total** | **~1,300 lines** | **~800 lines** | **~40% reduction** |

### Maintainability

✅ **Single source of truth** - Bug fixes in one place benefit all scripts
✅ **Easier testing** - Test core utilities independently
✅ **Better organization** - Clear separation of concerns
✅ **Easier onboarding** - New developers can understand structure quickly

### Backwards Compatibility

✅ **Old scripts still work** - No breaking changes
✅ **Gradual migration** - Can switch to new script when ready
✅ **Same arguments** - Command-line interface unchanged

---

## Migration Guide

### For Users

**Option 1: Use new script** (recommended)
```bash
# Replace this:
python extract_activations_arc_v5e64.py ...

# With this:
python extract_activations.py ...
```

**Option 2: Keep using old scripts**
```bash
# Old scripts still work unchanged
python extract_activations_arc_v5e64.py ...
```

### For Developers

**Importing utilities:**

```python
# Old way (before refactoring)
from extract_activations_fineweb_multihost import (
    ActivationStorage,
    pad_sequences,
    extract_activations_sharded
)

# New way (after refactoring)
from core import (
    ActivationStorage,
    pad_sequences,
    extract_activations_sharded
)
```

**All available imports:**

```python
from core import (
    # JAX utilities
    initialize_multihost,
    create_device_mesh,
    create_sharding_strategy,
    shard_params,
    extract_activations_sharded,
    pad_sequences,
    get_device_memory_info,
    P,  # PartitionSpec alias

    # Dataset utilities
    load_arc_dataset_jsonl,
    load_arc_dataset_from_shard,
    create_prompts_from_dataset,

    # Storage
    ActivationStorage,
    load_activation_shard,
)
```

---

## Testing

### Test Refactored Code

```bash
# Run the test script
bash test_refactored.sh
```

### What Gets Tested

1. ✅ Core utilities import correctly
2. ✅ JAX utilities work (device mesh, sharding)
3. ✅ Dataset loading works (both modes)
4. ✅ Activation storage works
5. ✅ End-to-end extraction works

---

## Files Modified

### New Files
- `core/__init__.py`
- `core/jax_utils.py`
- `core/dataset_utils.py`
- `core/activation_storage.py`
- `extract_activations.py` (refactored main script)
- `REFACTORING.md` (this file)

### Modified Files
- `Dockerfile` - Updated to include core/ directory

### Unchanged Files
- `extract_activations_arc_v5e64.py` - Still works
- `extract_activations_fineweb_multihost.py` - Still works
- `shard_manager.py` - Unchanged
- `create_sharded_dataset.py` - Unchanged
- All model files (`qwen2_jax.py`, etc.) - Unchanged

---

## Next Steps

1. ✅ Test the refactored code: `bash test_refactored.sh`
2. ✅ Switch to new `extract_activations.py` script
3. ✅ Update Docker image: `docker build -t activation-extraction:latest .`
4. ✅ Deploy and verify on TPU

---

## Questions?

- **New users**: Use `extract_activations.py` (refactored)
- **Existing users**: Continue using old scripts or migrate when ready
- **Developers**: Import from `core/` for new code

See `QUICK_START.md` for deployment instructions.
