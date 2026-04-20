# Testing Reference

This document covers all test suites, what they validate, and how to run them.

---

## Quick Reference

```bash
# Run all tests locally (CPU backend)
JAX_PLATFORMS=cpu python3 tests/test_code_fixes.py
JAX_PLATFORMS=cpu python3 -m unittest tests.test_resume_and_streams -v
JAX_PLATFORMS=cpu python3 tests/test_e2e_extraction.py
JAX_PLATFORMS=cpu python3 tests/test_jax_vs_hf.py
JAX_PLATFORMS=cpu python3 tests/test_multihost_equivalence.py

# Run on TPU pod (all workers, via setup script)
./setup_and_test.sh --test-only

# Run with pytest
JAX_PLATFORMS=cpu pytest tests/ -v
```

---

## Test Suites

### 1. `tests/test_code_fixes.py` — Core Infrastructure (8 tests)

Validates fundamental building blocks of the pipeline.

| Test | What It Checks |
|------|---------------|
| `pad_sequences` | Padding and truncation logic in `core/jax_utils.py` |
| `ActivationStorage shard creation` | Buffer-to-shard lifecycle |
| `dynamic_batching bucket assignment` | Sequence bucketing efficiency |
| `barrier_sync message format` | Socket barrier protocol |
| `checkpoint save/load roundtrip` | Resume-from-checkpoint fidelity |
| `tokenizer encode/decode` | Qwen tokenizer integration |
| `model config from HF` | HuggingFace config conversion |
| `GCS path construction` | Path formatting for uploads |

### 2. `tests/test_resume_and_streams.py` — Resume & Streaming (15 tests)

Unit tests (via `unittest`) for dataset streaming and checkpoint/resume.

| Test | What It Checks |
|------|---------------|
| Dataset stream iteration | Sequential sample loading from JSONL |
| Stream resume from offset | Checkpoint-based stream restart |
| Multi-stream round-robin | Correct interleaving of multiple streams |
| Shard claiming atomicity | No duplicate claims across workers |
| Checkpoint JSON schema | Required fields present after save |
| Resume state consistency | Loaded state matches saved state |

### 3. `tests/test_e2e_extraction.py` — End-to-End Extraction (5 tests)

Tests the full extraction pipeline from model forward pass to stored shards, using a tiny random-init model (no downloads).

| Test | What It Checks |
|------|---------------|
| `ActivationStorage shard roundtrip` | Store activations → save shard → reload → compare values |
| `Forward pass to shard roundtrip` | Model inference → storage → reload → numerical match (max_diff < 1e-4) |
| `Padding crop correctness` | Stored activations have padding removed, matching actual sequence lengths |
| `Dynamic batching bucket assignment` | Sequences assigned to correct length buckets with expected padding ratios |
| `Metadata JSON consistency` | Shard metadata fields match actual stored data |

### 4. `tests/test_jax_vs_hf.py` — JAX vs HuggingFace Validation (7 tests)

Validates that the JAX/Flax Qwen reimplementation produces numerically equivalent results to the HuggingFace PyTorch reference.

| Test | What It Checks |
|------|---------------|
| `Weight conversion completeness` | All HF parameter keys have JAX counterparts |
| `Logit equivalence` | Top-5 token predictions match between JAX and HF (bfloat16) |
| `Top-5 prediction agreement` | Same top tokens for 3 diverse prompts |
| `Activation shapes and non-triviality` | Extracted activations have correct shapes and non-zero values |
| `Activation type variants` | `residual`, `mlp`, `attn` produce distinct activations |
| `Per-layer hidden state match` | JAX layer outputs match HF `hidden_states[i+1]` (float32, max_diff < 0.01) |
| `Batch consistency` | Single-sample and batched inference produce same activations (relative error < 15%) |

**Key technical details:**
- HuggingFace `hidden_states[0]` = embeddings, `hidden_states[i+1]` = post-layer-i, `hidden_states[-1]` = post-final-norm
- bfloat16 error compounds through 24 layers (~1.4% relative error is normal)
- Per-layer comparison uses float32 for both models to isolate implementation differences from precision differences

### 5. `tests/test_multihost_equivalence.py` — Sharding Correctness (3 tests)

Simulates multi-host sharding on a single machine to verify sharding doesn't corrupt activations.

| Test | What It Checks |
|------|---------------|
| `Sharded vs unsharded equivalence` | Activations through device mesh match plain forward pass (relative error < 5%) |
| `Hidden dim integrity` | Full hidden_dim (896) present on each host after sharding (no fragmentation) |
| `extract_activations_sharded consistency` | `core/jax_utils.py:extract_activations_sharded` matches direct `model.apply` |

**Key technical details:**
- bfloat16 JIT compilation paths differ between sharded/unsharded (non-associative operations), causing small numerical differences
- `extract_activations_sharded` requires a mesh with a `'data'` axis (uses `P('data', None, None)`)
- On CPU with 1 device, sharding is trivially correct — these tests are a smoke test on CPU, meaningful on TPU

---

## Running on TPU Pod

The `setup_and_test.sh` script runs all test suites across all 16 TPU workers:

```bash
# Full setup + tests
./setup_and_test.sh

# Tests only (skip setup, pull latest code)
./setup_and_test.sh --test-only

# Setup + tests + extraction smoke test
./setup_and_test.sh --extraction-test
```

All tests run with `JAX_PLATFORMS=cpu` to avoid TPU lock issues during individual test execution. The extraction test (`--extraction-test`) uses the actual TPU.

---

## pytest Configuration

`pytest.ini` configures:
- Test discovery: `tests/test_*.py`, functions matching `test_*`
- Markers: `slow` (full model loads), `gpu` (TPU/GPU required)
- Output: verbose with short tracebacks

---

## Test Count Summary

| Suite | Tests | Backend |
|-------|-------|---------|
| test_code_fixes | 8 | CPU |
| test_resume_and_streams | 15 | CPU |
| test_e2e_extraction | 5 | CPU |
| test_jax_vs_hf | 7 | CPU |
| test_multihost_equivalence | 3 | CPU (smoke) / TPU (full) |
| **Total** | **38** | |
