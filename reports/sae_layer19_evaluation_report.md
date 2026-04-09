# SAE Layer 19 Evaluation Report

**Date**: April 2026  
**Model**: Qwen 2.5 0.5B — Layer 19 Residual Activations  
**SAE Architecture**: TopK Sparse Autoencoder  

---

## 1. Overview

This report documents the training, evaluation, and feature analysis of a Sparse Autoencoder (SAE) trained on layer 19 residual stream activations from Qwen 2.5 0.5B, extracted from ARC-AGI task prompts. The goal is to decompose the model's internal representations into interpretable features that may reveal how the model processes grid-based reasoning tasks.

---

## 2. Input Data

### 2.1 Activation Extraction
- **Source model**: Qwen 2.5 0.5B (24 transformer layers, hidden_dim=896)
- **Layer extracted**: Layer 19 (residual stream after both attention and MLP)
- **Extraction method**: Single forward pass through JAX/Flax reimplementation on TPU
- **TPU used for extraction**: v5litepod-64 (16 hosts, 64 chips)
- **Dataset**: ARC-AGI tasks encoded as grid chunks with data augmentation (rotations, flips, color maps)

### 2.2 Activation Format
- **Location**: `gs://arc-data-europe-west4/activations/layer19_merged_50k/`
- **Structure**: 8 `pair_XX/` subdirectories containing gzipped pickle shards
- **Total shards**: 2,345 (correctly merged from paired 448-dim halves into 896-dim)
- **Samples per shard**: ~235
- **Tokens per sample**: 5,120
- **Hidden dimension**: 896 (full residual stream width)
- **Dtype**: bfloat16
- **Total token activations**: ~2.82 billion

### 2.3 Merge Process
The original extraction on v5litepod-64 produced 448-dim activation halves due to FSDP model parallelism splitting the hidden dimension across host pairs. These were merged:
- Source shards 0001-0294: 448-dim halves, merged into 896-dim (pair concatenation)
- Source shards 0295-0631: Already 896-dim (from later extraction with corrected FSDP config)
- Only the merged 448-dim shards (2,345 total) were used for training

---

## 3. SAE Training Configuration

### 3.1 Architecture
| Parameter | Value |
|-----------|-------|
| Architecture | TopK |
| Hidden dimension (input) | 896 |
| Dictionary size | 7,168 |
| Expansion ratio | 8x |
| k (top-k sparsity) | 32 |
| Dtype | bfloat16 |
| Normalize decoder | Yes |

### 3.2 Training Hyperparameters
| Parameter | v1 | v2 (improved) |
|-----------|-----|----------------|
| Total steps | 200,000 | 150,000 |
| Batch size (global) | 4,096 | 4,096 |
| Learning rate | 3e-4 | 3e-4 |
| LR schedule | Cosine (10% floor) | Cosine (10% floor) |
| LR warmup | 1,000 steps | 1,000 steps |
| Optimizer | Adam | Adam |
| Gradient clipping | 1.0 | 1.0 |
| Shuffle buffer | 262,144 | 262,144 |
| Dead neuron resample | Every 25K steps | Every 25K steps, **stop at step 75K** |
| Dead neuron window | 10,000 steps | 10,000 steps |

### 3.3 Training Infrastructure
- **TPU**: v5litepod-64 (`node-v5e-64-europe-west4-b`) — 16 hosts, 64 chips
- **Region**: europe-west4 (same region as GCS bucket — zero cross-region transfer)
- **Parallelism**: Pure data-parallel across 64 devices
- **Barrier sync**: Socket-based pre-JAX barrier coordination across 16 hosts
- **Checkpoint storage**: `gs://arc-data-europe-west4/sae_checkpoints/`

### 3.4 Training Data Consumption
- Available tokens: ~2.82 billion
- Tokens consumed: ~614M (v2, 150K steps x 4096 batch)
- Effective epochs: 0.22 (well under 1 epoch — no data repetition)

---

## 4. Training Results

### 4.1 v1 Training — Divergence Discovery

v1 training completed all 200,000 steps but **diverged catastrophically** between steps 150K-200K:

| Step | MSE | Explained Variance | x_hat std | Status |
|------|-----|--------------------|-----------|--------|
| 5,000 | 0.3115 | 89.1% | 1.54 | Improving |
| 10,000 | 0.2922 | 89.7% | 1.54 | Improving |
| 25,000 | 0.2581 | 90.9% | 1.59 | Improving |
| 50,000 | 0.2475 | 91.3% | 1.62 | Improving |
| 75,000 | 0.2429 | 91.5% | 1.62 | Improving |
| 100,000 | 0.2396 | 91.6% | 1.62 | Peak range |
| **125,000** | **0.2384** | **91.6%** | **1.62** | **Best** |
| 150,000 | 0.2487 | 91.3% | 1.64 | Slight degradation |
| **175,000** | **0.4973** | **82.5%** | **1.94** | **Diverging** |
| **200,000** | **11.0354** | **-288%** | **4.66** | **Collapsed** |

### 4.2 Divergence Root Cause Analysis

**Root cause**: Dead neuron resampling destabilizing late training.

Evidence:
- Weight deltas **increased** from 150K-200K despite near-zero LR (3.8% of max at step 175K)
- `b_enc` had an anomalous 16.7% jump at 125K→150K (vs ~5% for other transitions)
- `W_dec` max single-weight deltas exceeded 1.0 starting at step 150K
- Decoder column norms remained stable (~1.003) — not a norm explosion
- No NaN/Inf in parameters — not numerical overflow
- Data not exhausted (only 0.29 epochs used)

**Mechanism**: The `dead_neuron_resample_steps=25000` setting reinitializes dead neurons at steps 125K, 150K, 175K. At these late steps, the learning rate is very low (14.8% and 3.8% of max). Newly resampled neurons inject noise into the decoder but cannot converge fast enough with the low LR, causing cascading instability.

### 4.3 v2 Training — Stable

v2 training with `dead_neuron_resample_until=75000` prevented the divergence:

| Step | MSE | Explained Variance | x_hat std |
|------|-----|--------------------|-----------|
| 5,000 | 0.3115 | 89.1% | 1.54 |
| 25,000 | 0.2591 | 90.9% | 1.59 |
| 50,000 | 0.2501 | 91.2% | 1.62 |
| 75,000 | 0.2451 | 91.4% | 1.62 |
| 100,000 | 0.2445 | 91.4% | 1.63 |
| **125,000** | **0.2428** | **91.5%** | **1.63** |
| **150,000** | **0.2453** | **91.4%** | **1.63** |

The model converges by ~100K steps and remains **stable through the end of training**.

---

## 5. Full Evaluation — v2 Step 125,000

### 5.1 Reconstruction Quality
Evaluated on 24,064,000 tokens from 20 randomly sampled shards:

| Metric | Value |
|--------|-------|
| MSE | 0.1254 |
| **Explained variance** | **94.6%** |
| Input variance | ~2.84 |

### 5.2 Sparsity
| Metric | Value |
|--------|-------|
| L0 (active features per token) | 32.0 |
| L0 fraction | 0.45% |
| Target k | 32 |

The TopK mechanism enforces exactly k=32 active features per token.

### 5.3 Feature Utilization
| Metric | Value |
|--------|-------|
| Active features (on eval data) | 6,744 / 7,168 (94.1%) |
| Dead features | 424 (5.9%) |
| Dead >10K steps (checkpoint tracking) | 31 (0.4%) |
| Max inactive steps | 32,864 |

### 5.4 Feature Density Distribution
| Percentile | Density |
|------------|---------|
| P01 | 0.000000 |
| P05 | 0.000001 |
| P10 | 0.000002 |
| P25 | 0.000010 |
| P50 (median) | 0.000054 |
| P75 | 0.000300 |
| P90 | 0.002518 |
| P95 | 0.014191 |
| P99 | 0.075731 |
| Max | 0.998464 |

The distribution is highly skewed — a few features fire on nearly every token (density ~1.0) while most features fire very rarely (median density 0.005%).

---

## 6. Feature Pattern Analysis

### 6.1 Analysis Setup
- **Samples analyzed**: 200-250 ARC grid chunks from 3-5 diverse shards
- **Tokens analyzed**: ~1,024,000
- **Sequence length**: 5,120 tokens per sample
- **Analysis regions**: 5 segments of 1,024 tokens each (0-1K, 1K-2K, ..., 4K-5K)

### 6.2 Feature Hierarchy

The SAE learns a clear hierarchical feature structure:

#### Tier 1: Universal Features (20 features, active in 80%+ samples)
These fire on virtually every token across all samples. They encode the base structure of ARC grid prompts — grid syntax, number formatting, common token patterns.

**Top 6 dominant features** (99.9% density, co-occur with Jaccard similarity 1.0):

| Feature | Density | Mean Activation |
|---------|---------|-----------------|
| 239 | 99.91% | 23.4 |
| 6613 | 99.91% | 20.0 |
| 1705 | 99.91% | 20.2 |
| 392 | 99.91% | 20.8 |
| 2525 | 99.91% | 19.4 |
| 1844 | 98.61% | 19.3 |

These 6 features always co-occur and have the highest activation magnitudes. They represent the "grid encoding substrate" — the foundational representation that Qwen uses for all ARC prompts regardless of task type.

**Full universal feature set** (20 features): 141, 239, 392, 1142, 1705, 1844, 2029, 2051, 2100, 2269, 2391, 2525, 2998, 3807, 4277, 4661, 5128, 5343, 5421, 6613

All universal features show **UNIFORM** position patterns — they fire equally across all token positions in the sequence.

#### Tier 2: Medium-Selectivity Features (252 features, active in 10-50% of samples)
These discriminate between different types of ARC tasks. Grouped by position pattern:

| Pattern | Count | Description |
|---------|-------|-------------|
| Uniform | 209 | Content/pattern features — fire throughout the sequence |
| Broad | 23 | Slight position gradient, likely grid-size dependent |
| START | 8 | Prompt structure features (first 1K tokens) |
| Region 0-1K | 9 | Input grid features |
| Region 3K-4K | 1 | Late-sequence feature |
| Region 4K-5K | 2 | Output/answer region features |

The **209 uniform medium-selectivity features** are the most interesting for ARC task understanding — they encode patterns that distinguish task types regardless of where in the sequence they appear.

#### Tier 3: Selective Features (69 features, active in 1-5 samples)
Highly specific features that fire only for very particular grid patterns or task configurations. These represent the "long tail" of learned representations.

#### Tier 4: Dead Features (521 features, never active at >1% threshold)
Features that the SAE allocated but never learned to use effectively.

### 6.3 Position-Dependent Features

**Start-of-sequence features** (fire only in tokens 0-1K):
- Features 3894, 7031, 5551, 4117, 4865, 4095, 6414, 2869, 6022, 7057
- These encode prompt headers, task framing, and instruction tokens
- Start density 0.4-0.5%, end density ~0%

**End-of-sequence features** (fire only in tokens 4K-5K):
- Features 5449, 6688, 1708, 4773, 3531, 3338, 2181, 4912, 518, 4324
- These encode output/answer region tokens
- End density 0.03-0.05%, start density ~0%

### 6.4 Feature Co-occurrence

The top 6 universal features (239, 392, 1705, 2525, 6613, 1844) form a **perfectly correlated cluster** — Jaccard similarity 1.0 across all pairs. They fire together on every sample.

Beyond the universal cluster, co-occurrence is more selective and follows the sample clustering structure described below.

### 6.5 Sample Clustering

Using 936 discriminative features (medium + selective), samples were clustered by cosine similarity of their feature fingerprints (threshold 0.7):

**27 clusters** identified from 250 samples (34 unclustered):

| Cluster | Size | Key Distinctive Features | Selectivity |
|---------|------|--------------------------|-------------|
| C4 | 33 | 3546 (9.2x), 4790, 3442 | Largest group |
| C1 | 21 | 4116 (9.9x), 5703 (13.7x), 1461 (8.4x) | Strong signature |
| C13 | 13 | 3333, 1776, 6465 | |
| C10 | 11 | 3442, 3333, 5595 | Related to C13 |
| C2 | 9 | 2201 (10.3x), 4779 (8.2x), 3527 (11.9x) | |
| C0 | 8 | 7037, 3248 (8.8x), 1137 | |
| **C6** | **6** | **1863 (84x), 6180 (127x), 851 (165x)** | **Extremely distinctive** |
| **C5** | **3** | **6599 (51x), 5336 (8.9x)** | **Highly unique** |

**Notable findings:**
- **Cluster 6** has the most extreme feature selectivity observed: Feature 1863 fires at 100% density within the cluster but only 1.2% elsewhere (84x). Features 6180 (127x selectivity) and 851 (165x selectivity) are even more discriminating. This cluster represents a very specific type of ARC pattern.
- **Clusters 1 and 3** are related (cosine similarity 0.81), sharing Feature 4116 as a strong discriminator — suggesting related but distinct grid patterns.
- **Clusters 4, 8, 9** form a super-group (pairwise similarity 0.60-0.70), sharing features 3442 and 4526 — suggesting related transformation types.
- **Clusters 0 and 5** are related (similarity 0.59), sharing features 2094, 1137, 2136.

**Cluster size distribution**: [33, 21, 13, 12, 11, 9, 9, 9, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 4, ...]

### 6.6 Feature Selectivity Distribution

| Samples Active In | Feature Count | Percentage |
|--------------------|---------------|------------|
| 0 samples | 6,934 | 96.7% |
| 1 sample | 28 | 0.4% |
| 2-4 samples | 38 | 0.5% |
| 5-9 samples | 29 | 0.4% |
| 10-19 samples | 37 | 0.5% |
| 20-49 samples | 49 | 0.7% |
| 50-99 samples | 18 | 0.3% |
| 100+ samples | 23 | 0.3% |

The vast majority of features are either dead or very rarely active at the sample level, with a small number of features carrying most of the representational load.

---

## 7. Key Findings

1. **The SAE achieves 94.6% explained variance** with only 32 active features per token (out of 7,168), demonstrating that layer 19 residual stream activations are highly compressible into sparse features.

2. **A hierarchy of features emerges naturally**: 20 universal (grid syntax), 252 medium-selectivity (task-discriminating), 69 highly selective (task-specific), forming a pyramid of abstraction.

3. **6 dominant features account for the base representation**: Features 239, 392, 1705, 2525, 6613, 1844 fire on virtually every token and always co-occur, encoding the fundamental grid prompt structure.

4. **Clear position encoding exists**: Distinct features activate only at the start (prompt/instructions) or end (output/answer) of sequences, showing the SAE captures the prompt structure.

5. **27 sample clusters emerge from feature fingerprints**, with some showing extreme selectivity (up to 165x enrichment for specific features). These clusters likely correspond to different ARC task types or transformation patterns.

6. **Dead neuron resampling causes late-training divergence**: Resampling at low LR injects noise that cannot converge, causing catastrophic reconstruction quality collapse. Stopping resampling in the second half of training prevents this.

---

## 8. Artifacts

### 8.1 Checkpoints
| Run | Location | Best Step |
|-----|----------|-----------|
| v1 | `gs://arc-data-europe-west4/sae_checkpoints/layer19_topk_896d_v5e64/` | step 125,000 |
| v2 | `gs://arc-data-europe-west4/sae_checkpoints/layer19_topk_896d_v2/` | step 125,000 |

### 8.2 Training Data
- Merged activations: `gs://arc-data-europe-west4/activations/layer19_merged_50k/`
- Source activations: `gs://arc-data-europe-west4/activations/layer19_gridchunk_50k_v5litepod-64/`

### 8.3 Analysis Scripts
- `scripts/eval_sae.py` — Full SAE evaluation
- `scripts/diagnose_sae.py` — Checkpoint diagnostics
- `scripts/find_divergence.py` — Divergence point finder
- `scripts/analyze_sae_patterns.py` — Global feature pattern analysis
- `scripts/analyze_deep_dive.py` — Deep-dive: medium features, universal cluster, clustering

---

## 9. Technical Fixes Applied

During this work, several multi-host TPU issues were discovered and fixed:

1. **Device mesh ordering** (`sae/training/distributed.py`): v5litepod-64 requires devices ordered by (process_index, local_device_id) for contiguous host subcubes.

2. **Parameter replication** (`sae/training/distributed.py`): `jax.device_put` with replicated sharding hangs on multi-host pods. Fixed to use `jax.make_array_from_single_device_arrays`.

3. **bfloat16 deserialization** (`sae/training/checkpointing.py`, `sae/data/pickle_source.py`): `ml_dtypes` must be imported before any numpy unpickling of bfloat16 arrays.

4. **Dead neuron resample cutoff** (`sae/configs/training.py`, `sae/training/trainer.py`): Added `dead_neuron_resample_until` parameter to stop resampling after a specified step, preventing late-training divergence.
