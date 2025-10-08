# Performance Issue: Slow Token Generation (0.64 tok/s)

## Root Cause Analysis

The distributed inference is generating only **0.64 tokens/second** instead of the expected **50-100 tokens/second** for a 0.5B model on 4 TPUs.

### Primary Cause: **No KV Cache**

The model implementation (`qwen2_jax.py` and `qwen2_jax_with_hooks.py`) does **NOT have a KV (Key-Value) cache**.

**What this means:**
- On each token generation step, the model recomputes attention for ALL previous tokens
- Token 1: Compute attention for 453 tokens (input)
- Token 2: Compute attention for 454 tokens (input + 1 new)
- Token 3: Compute attention for 455 tokens (input + 2 new)
- ...
- Token 500: Compute attention for 953 tokens (input + 499 new)

**Computational complexity:**
- Without KV cache: O(n²) where n = sequence length
- With KV cache: O(n) - linear

For 500 tokens with input length ~450:
- **Without cache**: ~450² + 451² + ... + 950² ≈ **250 million** operations
- **With cache**: ~450 + 451 + ... + 950 ≈ **700 thousand** operations

This is a **350x difference** in computation!

### Secondary Causes:

1. **JIT Recompilation per Sequence Length**
   - Each new sequence length (453, 454, 455, ..., 953) triggers a new JIT compilation
   - With 500 tokens generated, we get 500 different compilations
   - Each compilation takes time (even if cached after first batch)

2. **Python Loop Overhead**
   - The generation loop is in Python, not inside XLA
   - Each iteration requires host-device communication

## Current Performance

- **0.64 tokens/second**
- 500 tokens takes ~13 minutes per sample
- 8 samples takes ~1 hour 43 minutes

## Expected Performance (with KV cache)

- **50-100 tokens/second** (realistic estimate)
- 500 tokens would take ~5-10 seconds per sample
- 8 samples would take ~1-2 minutes total

**Expected speedup: 50-100x**

## Solutions

### Solution 1: Implement KV Cache (RECOMMENDED)

Modify the model to cache attention keys and values:

```python
class QwenDecoderLayerWithCache(nn.Module):
    def __call__(self, hidden_states, cache=None):
        # ... existing code ...

        # Compute Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        if cache is not None:
            # Concatenate with cached K, V
            key = jnp.concatenate([cache['key'], key], axis=1)
            value = jnp.concatenate([cache['value'], value], axis=1)

        # Compute attention with full K, V
        attn_output = attention(query, key, value, mask)

        # Update cache
        new_cache = {'key': key, 'value': value}

        return attn_output, new_cache
```

**Benefits:**
- 50-100x speedup
- Proper autoregressive generation
- Industry standard approach

**Effort:** Medium (requires modifying model architecture)

### Solution 2: Use Existing JAX LLM Libraries

Use libraries that already have KV cache implemented:
- **transformers-jax** (if available)
- **JAXformer**
- **FlaxLM**

**Benefits:**
- Immediate performance gains
- Battle-tested implementations
- Additional optimizations (flash attention, etc.)

**Effort:** Medium (requires model migration)

### Solution 3: Pad to Max Length (Quick Fix)

Pre-pad all sequences to max length and use masking:

```python
def distributed_generate(params, input_ids):
    batch_size = input_ids.shape[0]
    max_len = input_ids.shape[1] + max_tokens

    # Pad to max length
    generated_ids = jnp.pad(input_ids, ((0, 0), (0, max_tokens)))
    position = input_ids.shape[1]

    # Generate with fixed shape
    for i in range(max_tokens):
        logits = model.apply(params, generated_ids[:, :position+i])
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        generated_ids = generated_ids.at[:, position+i].set(next_token)

    return generated_ids
```

**Benefits:**
- Eliminates JIT recompilation
- Can be JIT compiled once

**Drawbacks:**
- Still O(n²) complexity (no KV cache)
- Memory inefficient
- Only ~2-3x speedup

**Effort:** Low

## Recommendation

**Implement KV cache (Solution 1)** - This is the only way to achieve the expected 50-100 tok/s performance.

The mesh-based sharding is working correctly. The bottleneck is purely the lack of KV cache in the model architecture.

## Temporary Workaround

For now, reduce `max_output_tokens` to get faster results:
- Use `--max_output_tokens 50-100` for testing
- Each sample will complete in ~1-2 minutes instead of ~13 minutes

## Verification

After implementing KV cache, you should see:
- ✅ Tokens/second: 50-100 (not 0.64)
- ✅ 500 tokens in ~5-10 seconds (not 13 minutes)
- ✅ No recompilation warnings after first token
- ✅ Linear scaling with sequence length
