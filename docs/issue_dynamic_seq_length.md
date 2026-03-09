# Issue: Fixed Sequence Length Truncation

## Problem

The extraction pipeline uses a fixed `max_seq_length=2048` for all sequences. The `pad_sequences()` function in `core/jax_utils.py:330-335` handles sequences exceeding this limit with right-truncation:

```python
elif len(seq) > max_len:
    # Truncate if too long
    padded.append(seq[:max_len])
```

This keeps only the **first** 2048 tokens and discards the rest.

## Why This Matters

### ARC Prompt Structure

An ARC prompt (using `output-from-examples-v1`) is structured as:

```
[System: "You are a helpful assistant."]           ~8 tokens
[User message start]                               ~50 tokens
  [Training example 1: input grid + output grid]   ~100-500 tokens
  [Training example 2: input grid + output grid]   ~100-500 tokens
  [Training example 3: input grid + output grid]   ~100-500 tokens
  ...
  [Test input grid]                                ~50-200 tokens
[Assistant: "### Output\n\n```grid"]               ~10 tokens
```

With right-truncation, the **test input** (near the end) is the first thing lost. The model sees training examples but not the task it's supposed to solve.

### Impact on Activations

- **Truncated sequences**: Activations represent the model reading incomplete context with no task to solve. The internal representations are fundamentally different from when the model has the complete prompt.
- **Short sequences**: Tokens beyond the actual prompt are padding (`pad_token_id`). Activations at padded positions are meaningless but still stored, wasting space.

### Model Capacity

The Qwen 2.5 0.5B model supports `max_position_embeddings=32768`. The current limit of 2048 uses only **6.25%** of the model's context window.

### Dataset Characteristics (Measured)

Measured on 1000 sampled tasks from the `200k_HEAVY` dataset with v1 prompting:

```
Min:       502    Under 2048:   5.4%
P10:      2945    Under 4096:  17.6%
P25:      5232    Under 8192:  46.0%
Median:   8947    Under 16384: 80.8%
Mean:    11260    Under 32768: 96.8%
P95:     29007
Max:     80847    Exceed 2048:  94.6%
```

**94.6% of prompts exceed 2048 tokens.** The v1 extraction run (530 GB) had nearly all
activations from truncated prompts with missing test inputs.

## Proposed Solutions

### Option 1: Dynamic Sequence Length (Recommended)

Instead of padding all sequences to a fixed length, group sequences by similar lengths and pad within each group:

```
Batch 1: sequences of ~500 tokens  -> pad to 512
Batch 2: sequences of ~1200 tokens -> pad to 1280
Batch 3: sequences of ~3000 tokens -> pad to 3072
```

**Pros**: No wasted compute on padding, no truncation, handles variable-length prompts
**Cons**: More complex batching logic, variable-sized tensors need careful handling with JAX JIT

### Option 2: Increase Fixed Length

Simply increase `max_seq_length` to 4096, 8192, or higher.

**Pros**: Simplest change (one parameter)
**Cons**: Short sequences waste compute/memory on padding. Memory usage scales with max length.

### Option 3: Filter + Increase

Set a reasonable max length (e.g., 4096) and **filter out** sequences that exceed it instead of truncating:

```python
# Instead of truncating
sequences = [s for s in sequences if len(s) <= max_seq_length]
```

**Pros**: All extracted activations are from complete prompts
**Cons**: Loses some data (tasks with very large grids)

### Option 4: Left Truncation

Truncate from the left instead of the right, keeping the test input and answer:

```python
padded.append(seq[-max_len:])  # Keep last max_len tokens
```

**Pros**: Preserves the test input and task context
**Cons**: Loses early training examples, model sees incomplete context from a different angle

## Recommended Approach

**Option 1 (Dynamic Sequence Length)** with **Option 3 (Filter)** as a safety net:

1. Measure the actual token length distribution across the dataset
2. Implement dynamic batching that groups sequences by length buckets
3. Set a hard cap at a reasonable max (e.g., 8192) and filter out anything longer
4. Extract activations only from non-padding positions to save storage

## Files to Modify

| File | Change |
|------|--------|
| `core/jax_utils.py` | Replace fixed `pad_sequences()` with dynamic length support |
| `multihost_extract.py` | Add length-based batching, filtering, per-token activation extraction |
| `extract_activations.py` | Same changes for single-host script |
| `core/activation_storage.py` | Handle variable-length activations in shards |

## Measuring the Problem

Before implementing, measure the distribution:

```python
# On a TPU worker with the dataset:
lengths = [len(tokenizer.encode(p['prompt'])) for p in prompts_data]
print(f"Exceed 2048: {sum(1 for l in lengths if l > 2048)} / {len(lengths)}")
print(f"Exceed 4096: {sum(1 for l in lengths if l > 4096)} / {len(lengths)}")
```

## Branch

Fix is being developed on: `fix/dynamic-seq-length`
