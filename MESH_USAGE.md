# Distributed Inference with Proper Mesh Usage

## Overview

The distributed inference pipeline now properly uses JAX's mesh for **2D parallelism** combining:
- **Data parallelism**: Different batches on different devices (across 'data' axis)
- **Model parallelism**: Model weights split across devices (across 'model' axis)

## Key Improvements

### Before (pmap only)
- ❌ Used `pmap` which only does 1D data parallelism
- ❌ Mesh was created but ignored
- ❌ All parameters replicated on all devices (memory inefficient)
- ❌ No model parallelism

### After (Mesh-aware with NamedSharding)
- ✅ Proper 2D mesh parallelism using `jax.jit` with sharding specs
- ✅ Mesh actively used for both data and model dimensions
- ✅ Parameters sharded across 'model' axis (memory efficient)
- ✅ Input data sharded across 'data' axis
- ✅ True distributed computation using both mesh axes

## How It Works

### 1. Mesh Setup
```python
# Creates a 2D mesh: [data=2, model=2] for 4 TPUs
mesh_shape = (2, 2)  # (data_parallelism, model_parallelism)
mesh = Mesh(devices.reshape(2, 2), axis_names=('data', 'model'))
```

### 2. Parameter Sharding
Parameters are sharded according to their type:
- **Embeddings** `[vocab_size, hidden_size]`: Shard hidden_size across 'model' axis
- **Linear layers** `[in_features, out_features]`: Shard out_features across 'model' axis
- **Norms** `[hidden_size]`: Replicated (no sharding)

```python
# Example: Embedding with shape (151936, 896)
spec = PartitionSpec(None, 'model')
# This splits the 896 dimension across 2 devices: 448 per device
```

### 3. Data Sharding
Input batches are sharded across the 'data' axis:
```python
# Input shape: [batch_size, seq_len]
spec = PartitionSpec('data', None)
# Batch dimension split across 'data' axis
```

### 4. Verification

The script outputs sharding info:
```
Verifying parameter sharding:
  embed_tokens: shape=(151936, 896), sharding=PartitionSpec(None, 'model')
  layers_0_mlp_down_proj: shape=(4864, 896), sharding=PartitionSpec(None, 'model')

Sharding verification:
  Input shape: (8, 453)
  Input sharding: PartitionSpec('data', None)
  Params sharding: PartitionSpec(None, 'model')
```

## Usage

Run with the same command as before:

```bash
python distributed_inference_with_activations.py \
  --model_path KathirKs/qwen-2.5-0.5b \
  --dataset_path test_data_small.json \
  --output_filepath test_outputs/predictions.json \
  --activations_dir test_activations/run_$(date +%Y%m%d_%H%M%S) \
  --batch_size 2 \
  --n_tasks 2 \
  --max_output_tokens 500 \
  --predictions_per_task 2 \
  --grid_encoder "GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))" \
  --prompt_version output-from-examples-v0 \
  --extract_activations \
  --layers_to_extract 10 11 12 \
  --mesh_shape 2 2  # This now properly creates 2D parallelism!
```

## Benefits

1. **Memory Efficiency**: Model weights are split across devices instead of replicated
2. **Scalability**: Can handle larger models by sharding across more devices
3. **Better Resource Utilization**: Both data and model axes of the mesh are actively used
4. **Flexibility**: Can adjust mesh_shape based on available devices and model size

## Technical Details

### NamedSharding vs pmap

**Old approach (pmap)**:
- Only supports 1D parallelism
- Requires explicit device replication
- Mesh context ignored

**New approach (NamedSharding + jax.jit)**:
- Supports N-D parallelism via mesh
- Automatic sharding based on PartitionSpec
- Full mesh integration

### Sharding Specifications

**PartitionSpec** defines how each dimension maps to mesh axes:
- `PartitionSpec('data', None)`: Shard dim 0 across 'data', replicate dim 1
- `PartitionSpec(None, 'model')`: Replicate dim 0, shard dim 1 across 'model'
- `PartitionSpec(None)`: Replicate all dimensions

## Validation

Check the output logs for:
1. ✅ "Mesh axes: data=X, model=Y" shows mesh is created
2. ✅ "Verifying parameter sharding" shows PartitionSpec with 'model' axis
3. ✅ "Input sharding: PartitionSpec('data', None)" confirms data sharding
4. ✅ No "replicated" in parameter shardings (except norms)
