# SAE Training Runs & Data Inventory

This document records all extraction runs, SAE training experiments, GCS data locations, and evaluation results.

---

## 1. Activation Extraction Runs

### Run 001: Layer 12, v1 Prompting, 50K Tasks

| Parameter | Value |
|-----------|-------|
| Date | 2026-03-08 |
| Model | `KathirKs/qwen-2.5-0.5b` (Qwen 2.5 0.5B fine-tuned on ARC) |
| TPU | v5litepod-64 (16 workers x 4 chips) |
| Layer | 12 (residual) |
| Dataset | `barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems` (50K tasks) |
| Prompt version | `output-from-examples-v1` |
| Predictions/task | 8 |
| Total sequences | 400,000 |
| Max seq length | 2048 |
| GCS output | `gs://arc-data-europe-west4/activations/layer12_v1_50k_v5litepod-64/` |
| Total shards | 688 (43 x 16 hosts) |
| Size | ~530 GB (compressed) |
| Status | **Completed** |

Full details: [docs/extraction_run_001.md](extraction_run_001.md)

### Run 002: Layer 19, Grid-Chunk Pipeline, 50K Tasks

| Parameter | Value |
|-----------|-------|
| Date | 2026-03 |
| Model | `KathirKs/qwen-2.5-0.5b` |
| TPU | v5litepod-64 |
| Layer | 19 (residual) |
| Pipeline | `grid_chunking` (continuous token stream, chunk_size=5120) |
| Dataset | `combined_50k.jsonl` from ARC-AGI |
| Max seq length | 5120 |
| Batch size | 16 |
| GCS output (raw) | `gs://arc-data-europe-west4/activations/layer19_gridchunk_50k_v5litepod-64/` |
| GCS output (merged) | `gs://arc-data-europe-west4/activations/layer19_merged_50k/` |
| Status | **Completed + Merged** |

**Merge details:**
- Raw extraction produced 448-dim halves (FSDP model parallelism split hidden_dim across host pairs)
- Shards 0001-0294: 448-dim, merged into 896-dim via pair concatenation
- Shards 0295-0631: Already 896-dim (later extraction with corrected FSDP config)
- Merged data: 8 `pair_XX/` directories, 2,345 total shards
- Tokens: ~2.82 billion, dtype: bfloat16

---

## 2. SAE Training Runs

All runs target **Layer 19 residual stream** (hidden_dim=896, Qwen 2.5 0.5B).

### Common Configuration

| Parameter | Value |
|-----------|-------|
| Training data | `gs://arc-data-europe-west4/activations/layer19_merged_50k/` |
| TPU | `node-v5e-64-europe-west4-b` (v5litepod-64, 16 hosts, 64 chips) |
| Region | europe-west4-b |
| Parallelism | Data-parallel across 64 devices |
| Batch size | 4,096 (global), 256/host, 64/device |
| Learning rate | 3e-4 (cosine decay, 10% floor) |
| LR warmup | 1,000 steps |
| Optimizer | Adam, grad clip 1.0 |
| Shuffle buffer | 262,144 vectors |
| Checkpoint frequency | Every 5,000 steps |
| Dtype | bfloat16 |

### v1 — TopK 8x (original)

| Parameter | Value |
|-----------|-------|
| Launcher | `scripts/launch_sae_training_v6e.sh` |
| Architecture | TopK |
| Dict size | 7,168 (8x expansion) |
| k | 32 |
| Steps | 200,000 |
| Dead neuron resample | Every 25K steps (no cutoff) |
| Checkpoint prefix | `sae_checkpoints/layer19_topk_896d_v5e64` |
| **Status** | **Diverged at 175K-200K** |

**Divergence trajectory:**

| Step | MSE | Explained Variance | Status |
|------|-----|--------------------|--------|
| 125,000 | 0.2384 | 91.6% | Best |
| 150,000 | 0.2487 | 91.3% | Slight degradation |
| 175,000 | 0.4973 | 82.5% | Diverging |
| 200,000 | 11.0354 | -288% | Collapsed |

**Root cause:** Dead neuron resampling at steps 125K, 150K, 175K injects noise at low LR that cannot converge, causing cascading instability.

### v2 — TopK 8x (stable, improved)

| Parameter | Value |
|-----------|-------|
| Launcher | `scripts/launch_sae_training_v2.sh` |
| Architecture | TopK |
| Dict size | 7,168 (8x) |
| k | 32 |
| Steps | 150,000 |
| Dead neuron resample | Every 25K steps, **stop at step 75K** |
| Checkpoint prefix | `sae_checkpoints/layer19_topk_896d_v2` |
| **Status** | **Completed, stable** |

**Training trajectory:**

| Step | MSE | Explained Variance |
|------|-----|--------------------|
| 5,000 | 0.3115 | 89.1% |
| 25,000 | 0.2591 | 90.9% |
| 50,000 | 0.2501 | 91.2% |
| 75,000 | 0.2451 | 91.4% |
| 100,000 | 0.2445 | 91.4% |
| **125,000** | **0.2428** | **91.5%** |

**Full evaluation (step 125K on 24M tokens):**

| Metric | Value |
|--------|-------|
| MSE | 0.1254 |
| **Explained variance** | **94.6%** |
| L0 (active features) | 32.0 |
| L0 fraction | 0.45% |
| Active features | 6,744 / 7,168 (94.1%) |
| Dead features | 424 (5.9%) |
| Data consumed | ~614M tokens (0.22 epochs) |

### v3 — TopK 16x (larger dictionary)

| Parameter | Value |
|-----------|-------|
| Launcher | `scripts/launch_sae_training_v3_16x.sh` |
| Architecture | TopK |
| Dict size | **14,336 (16x)** |
| k | 32 |
| Steps | 150,000 |
| Dead neuron resample | Stop at 75K |
| Checkpoint prefix | `sae_checkpoints/layer19_topk_896d_v3_16x` |
| **Status** | **Pending evaluation** |

**Purpose:** A/B against v2 — does 2x more features improve reconstruction or feature quality? Same k=32, so sparsity fraction drops from 0.45% to 0.22%.

### v4 — JumpReLU 8x (architecture comparison)

| Parameter | Value |
|-----------|-------|
| Launcher | `scripts/launch_sae_training_v4_jumprelu.sh` |
| Architecture | **JumpReLU** |
| Dict size | 7,168 (8x, matches v2) |
| L1 coefficient | 3e-3 (L0 penalty with STE) |
| Steps | 150,000 |
| Dead neuron resample | Stop at 75K |
| Checkpoint prefix | `sae_checkpoints/layer19_jumprelu_896d_v4` |
| **Status** | **Pending evaluation** |

**Purpose:** Direct architecture comparison against v2. Same dict size, same data. JumpReLU uses learnable per-feature thresholds instead of hard top-k constraint.

---

## 3. Feature Analysis Results (v2 Step 125K)

From `scripts/analyze_sae_patterns.py` and `scripts/analyze_deep_dive.py`.

### Feature Hierarchy
- **20 universal features**: Fire on 80%+ of all samples (grid syntax encoding)
- **252 medium-selectivity features**: Discriminate between ARC task types (10-50% of samples)
- **69 selective features**: Highly specific to 1-5 samples
- **521+ dead features**: Never active above 1% threshold

### Top 6 Dominant Features
Features 239, 392, 1705, 2525, 6613, 1844 — fire on 99.9% of tokens with Jaccard similarity 1.0. Encode the fundamental grid prompt structure.

### Position-Dependent Features
- **Start features** (tokens 0-1K): 3894, 7031, 5551, 4117, 4865 — prompt headers
- **End features** (tokens 4K-5K): 5449, 6688, 1708, 4773 — output/answer region

### Sample Clustering
27 clusters from 250 samples, using 936 discriminative features:
- **Cluster 6**: Most extreme selectivity (feature 851 at 165x enrichment)
- **Clusters 1+3**: Related (similarity 0.81), share feature 4116
- **Clusters 4+8+9**: Super-group (similarity 0.60-0.70)

Full report: [reports/sae_layer19_evaluation_report.md](../reports/sae_layer19_evaluation_report.md)

---

## 4. GCS Data Map

```
gs://arc-data-europe-west4/
├── dataset_streams/
│   └── combined_50k.jsonl                    # ARC-AGI 50K tasks (1.26 GB)
├── activations/
│   ├── layer12_v1_50k_v5litepod-64/          # Layer 12, 688 shards, 400K acts
│   ├── layer19_gridchunk_50k_v5litepod-64/   # Layer 19 raw (448-dim halves)
│   │   ├── host_00/ ... host_15/
│   └── layer19_merged_50k/                   # Layer 19 merged (896-dim)
│       ├── pair_00/ ... pair_07/             # 2,345 shards, ~2.82B tokens
├── sae_checkpoints/
│   ├── layer19_topk_896d_v5e64/              # v1 (diverged)
│   ├── layer19_topk_896d_v2/                 # v2 (stable, best)
│   ├── layer19_topk_896d_v3_16x/             # v3 (16x, pending eval)
│   └── layer19_jumprelu_896d_v4/             # v4 (JumpReLU, pending eval)
├── sae_analysis/
│   ├── v2_collect/collected.pkl.gz           # Collected features
│   └── v2_color/                             # Color analysis
└── checkpoints/
    ├── checkpoint_v5litepod-64_host_XX.json  # Extraction progress
    └── gridchunk_layer19/                    # Grid chunk cache
```

---

## 5. Checkpoint Format

Each SAE checkpoint directory (`step_XXXXXXXX/`) contains:
```
step_00125000/
├── metadata.json           # Config, step number, total_tokens, param_keys
├── params/
│   ├── W_enc.npy           # [hidden_dim, dict_size]
│   ├── b_enc.npy           # [dict_size]
│   ├── W_dec.npy           # [dict_size, hidden_dim]
│   └── b_dec.npy           # [hidden_dim]
├── opt_state/
│   ├── opt_0000.npy        # Adam moment estimates
│   └── ...
├── dead_neuron_steps.npy   # Per-feature dead step counter
└── latest_step.json        # Marker file
```

---

## 6. Shard Format (Activations)

Each `.pkl.gz` shard is a gzipped pickle containing:
```python
{
    layer_idx: [              # e.g., 19
        {
            'sample_idx': int,
            'activation': np.ndarray,   # [seq_len, hidden_dim=896], bfloat16
            'shape': tuple,
            'text_preview': str
        },
        ...
    ]
}
```
