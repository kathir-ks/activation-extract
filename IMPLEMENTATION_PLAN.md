# Implementation Plan: MaxText-Style KV Cache + Fully JIT-Compiled Generation

## Goal
Implement fully JIT-compiled generation with:
- Fixed-size KV cache (no concatenation growth)
- Fixed activation extraction (layers 10-23)
- Zero Python overhead during generation
- All buffers stay on TPU until task completion

## Architecture Overview
```
┌─────────────────────────────────────────────────────────┐
│  Pre-allocated Buffers (on TPU)                         │
│  ├─ KV Cache: [24, 2, batch, heads, max_len, dim]      │
│  └─ Activations: [14, max_tokens, batch, hidden]       │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  Fully JIT-Compiled Generation (lax.fori_loop)          │
│  ├─ Prefill: Write to KV cache once                    │
│  └─ Decode: Update cache + activations at each step    │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  Post-Processing (Python, after JIT completes)          │
│  └─ Transfer activations to CPU & save                  │
└─────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Core Infrastructure (Priority 1)

#### 1.1 `kvcache_utils.py` - Cache Management
- `KVCacheConfig` dataclass
- `create_kv_cache_buffers()` - Pre-allocate fixed-size buffers
- `update_kv_cache_ar()` - Update at position using dynamic_update_slice_in_dim
- `get_attention_kv()` - Get valid portion for attention
- `create_activation_buffer()` - Pre-allocate activation storage
- `update_activation_buffer()` - Write at step position

#### 1.2 Modify `qwen2_jax.py` - Fixed KV Cache Support
- Add `use_fixed_cache=True` parameter
- Replace concatenation with dynamic_update_slice_in_dim
- Support both modes for backwards compatibility
- Update QwenAttention to handle fixed cache

### Phase 2: Model Integration (Priority 2)

#### 2.1 Modify `qwen2_jax_with_hooks.py` - Fixed Extraction
- Remove dynamic layers_to_extract
- Hardcode FIXED_EXTRACT_LAYERS = (10, 11, 12, ..., 23)
- Return stacked array instead of dict
- Always extract 14 layers (fixed shape)

#### 2.2 `generate_jitted.py` - Fully JIT Generation
- `prefill_with_cache()` - Initialize cache
- `decode_step_with_cache()` - Single decode step
- `generate_single_task_jitted()` - Main entry (lax.fori_loop)
- All shapes fixed for JIT compilation

### Phase 3: Distributed Integration (Priority 3)

#### 3.1 Update `distributed_kv_cached_inference.py`
- Use generate_single_task_jitted() instead of Python loop
- Create buffers per task
- Async flush activations after task completes

### Phase 4: Testing & Validation (Priority 4)

#### 4.1 Unit Tests (`test_kvcache_fixed.py`)
- test_create_kv_cache_buffers()
- test_update_kv_cache()
- test_activation_buffer()
- test_output_equivalence() - Compare with current implementation
- test_jit_compilation() - Verify shapes stay constant

#### 4.2 Integration Tests (`test_generation_jitted.py`)
- test_single_task_generation()
- test_different_input_lengths()
- test_kv_cache_persistence()

#### 4.3 Performance Tests (`test_performance.py`)
- test_speed_comparison()
- test_throughput() - Target: >30 tokens/sec
- test_memory_usage() - No growth during generation
- test_compilation_time()

#### 4.4 End-to-End Tests (`test_distributed_jitted.py`)
- test_arc_task_with_jitted_generation()
- test_activation_extraction_and_save()

## Key Design Patterns from MaxText

### Pattern 1: Fixed-Shape KV Cache
```python
# Pre-allocate
kv_cache = jnp.zeros((layers, batch, max_len, heads, dim))

# Update at position (not concatenate)
kv_cache = jax.lax.dynamic_update_slice_in_dim(
    kv_cache, new_kv, position, axis=seq_dim
)
```

### Pattern 2: Separate Prefill and AR Caches
```python
cache = {
    'prefill': {'k': ..., 'v': ..., 'length': prompt_len},
    'ar': {'k': ..., 'v': ..., 'index': current_step}
}
```

### Pattern 3: Fixed Activation Buffer
```python
# Pre-allocate
acts = jnp.zeros((14, max_tokens, batch, hidden))

# Update at step
acts = acts.at[:, step, :, :].set(layer_activations)
```

### Pattern 4: lax.fori_loop for Generation
```python
def decode_step(step, carry):
    kv_cache, acts, gen_ids = carry
    # ... forward pass, update cache, update acts
    return (new_cache, new_acts, new_ids)

final_cache, final_acts, final_ids = jax.lax.fori_loop(
    0, max_tokens, decode_step, initial_carry
)
```

## Success Metrics

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Speed | 5-10 tok/s | 30 tok/s | 50 tok/s |
| Compilation | N/A | <10s | <5s |
| Memory/Task | ~200MB | <100MB | <80MB |
| Python Overhead | High | Zero | Zero |
| Correctness | ✓ | 100% | 100% |

## Memory Budget (ARC-AGI)

```
Per Task:
- KV Cache: ~16.5 MB (24 layers, max 1300 tokens)
- Activations: ~55 MB (14 layers, 1100 tokens, 896 hidden)
- Total: ~72 MB per task

TPU v4: 32GB HBM → Can fit 400+ tasks in memory
```

## Implementation Order

1. **Day 1-2:** kvcache_utils.py + unit tests
2. **Day 3:** Modify qwen2_jax.py for fixed cache
3. **Day 4:** Update qwen2_jax_with_hooks.py for fixed extraction
4. **Day 5:** Implement generate_jitted.py
5. **Day 6:** Integration tests + distributed update
6. **Day 7:** Performance validation

## Risk Mitigation

- **Shape Mismatches:** Extensive assertions + multi-length tests
- **Slow Extraction:** Profile, add option to disable
- **Memory Overflow:** Monitor usage, configurable buffer sizes
- **Distributed Issues:** Keep both implementations, feature flag

## Files to Create/Modify

**New Files:**
- `kvcache_utils.py` - Cache management utilities
- `generate_jitted.py` - Fully JIT generation
- `test_kvcache_fixed.py` - Unit tests
- `test_generation_jitted.py` - Integration tests
- `test_performance.py` - Performance benchmarks
- `test_distributed_jitted.py` - End-to-end tests

**Modified Files:**
- `qwen2_jax.py` - Add fixed KV cache support
- `qwen2_jax_with_hooks.py` - Fixed layer extraction
- `distributed_kv_cached_inference.py` - Use new generation

**Preserved Files (backwards compatibility):**
- Keep all existing files working
- Add feature flags to switch implementations
