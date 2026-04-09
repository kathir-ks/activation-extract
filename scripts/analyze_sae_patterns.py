#!/usr/bin/env python3
"""
Analyze SAE feature patterns across 100+ ARC-AGI samples.

Loads activation shards, runs through SAE, and analyzes:
1. Global feature statistics (most/least common features)
2. Per-sample feature fingerprints
3. Token-position patterns (beginning/middle/end of sequence)
4. Feature co-occurrence and clustering
5. Most selective/interpretable features

Runs on CPU on a europe-west4 TPU worker (same region as GCS data).
"""
import os, json, gzip, pickle, subprocess, time, sys
import ml_dtypes
import numpy as np
from collections import Counter, defaultdict

TMP = "/tmp/sae_analysis"
CKPT_PREFIX = "sae_checkpoints/layer19_topk_896d_v2"
CKPT_STEP = 125000
DATA_GCS = "gs://arc-data-europe-west4/activations/layer19_merged_50k"
LAYER = 19
K = 32
DICT_SIZE = 7168
NUM_SHARDS = 3       # ~700 samples (235 each), use more shards from different pairs
MAX_SAMPLES = 200    # Cap total samples analyzed


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print("[%s] %s" % (ts, msg), flush=True)


def gcs_cp(src, dst):
    subprocess.run(["gcloud", "storage", "cp", src, dst], capture_output=True, check=True)


def load_sae_params():
    """Download and load SAE checkpoint."""
    d = "%s/step_%08d" % (TMP, CKPT_STEP)
    if not os.path.exists(d):
        subprocess.run(["gcloud", "storage", "cp", "-r",
            "gs://arc-data-europe-west4/%s/step_%08d" % (CKPT_PREFIX, CKPT_STEP),
            TMP], capture_output=True, check=True)
    params = {}
    with open(d + "/metadata.json") as f:
        meta = json.load(f)
    for k in meta["param_keys"]:
        a = np.load(d + "/params/%s.npy" % k)
        if a.dtype.kind == "V" and a.dtype.itemsize == 2:
            a = a.view(ml_dtypes.bfloat16)
        params[k] = a.astype(np.float32)
    return params


def sae_encode_topk(params, x):
    """Encode activations through SAE, return sparse codes. x: [tokens, 896]"""
    x = x.astype(np.float32)
    z_pre = (x - params["b_dec"]) @ params["W_enc"] + params["b_enc"]
    # TopK
    topk_idx = np.argpartition(-z_pre, K, axis=-1)[:, :K]
    z = np.zeros_like(z_pre)
    for i in range(len(z)):
        vals = np.maximum(z_pre[i, topk_idx[i]], 0)
        z[i, topk_idx[i]] = vals
    return z  # [tokens, 7168]


def load_samples(shard_paths):
    """Load activation samples from multiple shards."""
    all_samples = []
    for sp in shard_paths:
        local = "%s/shard_%d.pkl.gz" % (TMP, len(all_samples))
        gcs_cp(sp, local)
        with gzip.open(local, "rb") as f:
            data = pickle.load(f)
        layer_data = data.get(LAYER) or data.get(str(LAYER))
        for s in layer_data:
            all_samples.append({
                "activation": s["activation"],
                "text_preview": s.get("text_preview", ""),
                "sample_idx": s.get("sample_idx", -1),
            })
        os.remove(local)
        if len(all_samples) >= MAX_SAMPLES:
            break
    return all_samples[:MAX_SAMPLES]


def main():
    os.makedirs(TMP, exist_ok=True)

    log("=" * 70)
    log("SAE FEATURE PATTERN ANALYSIS")
    log("=" * 70)
    log("Checkpoint: v2 step %d" % CKPT_STEP)
    log("Data: %s" % DATA_GCS)

    # Load SAE
    log("\nLoading SAE...")
    params = load_sae_params()
    log("  W_enc: %s, W_dec: %s" % (params["W_enc"].shape, params["W_dec"].shape))

    # Select shards from different pairs for diversity
    shard_paths = [
        "%s/pair_00/shard_0010.pkl.gz" % DATA_GCS,
        "%s/pair_03/shard_0050.pkl.gz" % DATA_GCS,
        "%s/pair_06/shard_0100.pkl.gz" % DATA_GCS,
    ]

    log("\nLoading samples from %d shards..." % len(shard_paths))
    samples = load_samples(shard_paths)
    log("  Loaded %d samples" % len(samples))

    # ================================================================
    # ANALYSIS 1: Global feature activation statistics
    # ================================================================
    log("\n" + "=" * 70)
    log("1. GLOBAL FEATURE STATISTICS")
    log("=" * 70)

    # Track per-feature: total fires, total activation magnitude
    feature_fire_count = np.zeros(DICT_SIZE, dtype=np.float64)
    feature_activation_sum = np.zeros(DICT_SIZE, dtype=np.float64)
    total_tokens = 0
    per_sample_top_features = []  # top features per sample

    for i, sample in enumerate(samples):
        acts = sample["activation"]  # [seq_len, 896]
        # Process in chunks to avoid OOM
        chunk_size = 1024
        sample_feature_sum = np.zeros(DICT_SIZE, dtype=np.float64)
        sample_fire_count = np.zeros(DICT_SIZE, dtype=np.float64)

        for start in range(0, len(acts), chunk_size):
            end = min(start + chunk_size, len(acts))
            z = sae_encode_topk(params, acts[start:end])
            active = (z != 0).astype(np.float32)
            feature_fire_count += active.sum(axis=0)
            feature_activation_sum += np.abs(z).sum(axis=0)
            sample_feature_sum += np.abs(z).sum(axis=0)
            sample_fire_count += active.sum(axis=0)
            total_tokens += len(z)

        # Top features for this sample (by total activation)
        top_idx = np.argsort(-sample_feature_sum)[:50]
        per_sample_top_features.append({
            "idx": i,
            "preview": sample["text_preview"][:60],
            "top_features": top_idx.tolist(),
            "top_activations": sample_feature_sum[top_idx].tolist(),
            "top_fire_counts": sample_fire_count[top_idx].tolist(),
        })

        if (i + 1) % 50 == 0:
            log("  Processed %d/%d samples (%d tokens)" % (i + 1, len(samples), total_tokens))

    log("  Total: %d samples, %d tokens" % (len(samples), total_tokens))

    # Global feature density
    feature_density = feature_fire_count / total_tokens
    mean_activation = feature_activation_sum / np.maximum(feature_fire_count, 1)
    ever_active = feature_fire_count > 0

    log("\n  Active features: %d / %d (%.1f%%)" % (
        ever_active.sum(), DICT_SIZE, ever_active.mean() * 100))
    log("  Dead features: %d" % (~ever_active).sum())

    # Top 30 most common features
    top30 = np.argsort(-feature_density)[:30]
    log("\n  Top 30 most common features:")
    log("  %6s  %12s  %12s  %12s" % ("Feat#", "Density", "MeanAct", "TotalFires"))
    for f in top30:
        log("  %6d  %12.6f  %12.4f  %12d" % (
            f, feature_density[f], mean_activation[f], int(feature_fire_count[f])))

    # Bottom 30 (rarest active features)
    active_features = np.where(ever_active)[0]
    if len(active_features) > 30:
        rarest = active_features[np.argsort(feature_density[active_features])[:30]]
        log("\n  30 rarest active features:")
        log("  %6s  %12s  %12s  %12s" % ("Feat#", "Density", "MeanAct", "TotalFires"))
        for f in rarest:
            log("  %6d  %12.6f  %12.4f  %12d" % (
                f, feature_density[f], mean_activation[f], int(feature_fire_count[f])))

    # ================================================================
    # ANALYSIS 2: Token position patterns
    # ================================================================
    log("\n" + "=" * 70)
    log("2. TOKEN POSITION PATTERNS")
    log("=" * 70)

    # Divide sequence into 5 regions: [0-1024), [1024-2048), etc.
    regions = [(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096), (4096, 5120)]
    region_names = ["tokens 0-1K", "tokens 1K-2K", "tokens 2K-3K", "tokens 3K-4K", "tokens 4K-5K"]
    region_feature_density = np.zeros((len(regions), DICT_SIZE), dtype=np.float64)
    region_token_count = np.zeros(len(regions), dtype=np.float64)

    for sample in samples[:50]:  # Use 50 samples for position analysis
        acts = sample["activation"]
        for r_idx, (start, end) in enumerate(regions):
            if start >= len(acts):
                break
            actual_end = min(end, len(acts))
            z = sae_encode_topk(params, acts[start:actual_end])
            active = (z != 0).astype(np.float32)
            region_feature_density[r_idx] += active.sum(axis=0)
            region_token_count[r_idx] += len(z)

    # Normalize
    for r_idx in range(len(regions)):
        if region_token_count[r_idx] > 0:
            region_feature_density[r_idx] /= region_token_count[r_idx]

    # Find features that vary most across positions
    density_std = np.std(region_feature_density, axis=0)
    density_mean = np.mean(region_feature_density, axis=0)
    # Coefficient of variation (high = position-dependent)
    cv = density_std / np.maximum(density_mean, 1e-10)
    position_sensitive = np.argsort(-cv)[:20]

    log("\n  Top 20 position-sensitive features (highest variation across token positions):")
    log("  %6s  %10s | %s" % ("Feat#", "CV", " | ".join("%-8s" % n for n in region_names)))
    for f in position_sensitive:
        densities = " | ".join("%-8.5f" % region_feature_density[r, f] for r in range(len(regions)))
        log("  %6d  %10.2f | %s" % (f, cv[f], densities))

    # Features that fire mainly at the start vs end
    start_heavy = region_feature_density[0] / np.maximum(region_feature_density[-1], 1e-10)
    end_heavy = region_feature_density[-1] / np.maximum(region_feature_density[0], 1e-10)

    start_feats = np.argsort(-start_heavy)[:10]
    end_feats = np.argsort(-end_heavy)[:10]

    log("\n  Top 10 start-of-sequence features (fire more at tokens 0-1K):")
    for f in start_feats:
        if feature_density[f] > 1e-6:
            log("    Feature %d: start_density=%.5f, end_density=%.5f, ratio=%.1f" % (
                f, region_feature_density[0, f], region_feature_density[-1, f], start_heavy[f]))

    log("\n  Top 10 end-of-sequence features (fire more at tokens 4K-5K):")
    for f in end_feats:
        if feature_density[f] > 1e-6:
            log("    Feature %d: start_density=%.5f, end_density=%.5f, ratio=%.1f" % (
                f, region_feature_density[0, f], region_feature_density[-1, f], end_heavy[f]))

    # ================================================================
    # ANALYSIS 3: Feature co-occurrence
    # ================================================================
    log("\n" + "=" * 70)
    log("3. FEATURE CO-OCCURRENCE (sample level)")
    log("=" * 70)

    # Build binary sample x feature matrix (does feature fire in >1% of tokens for this sample?)
    threshold = 0.01  # feature fires in >1% of tokens
    sample_feature_matrix = np.zeros((len(samples), DICT_SIZE), dtype=np.float32)
    for i, sf in enumerate(per_sample_top_features):
        # Use fire counts / seq_len as density
        for j, feat_idx in enumerate(sf["top_features"][:50]):
            if sf["top_fire_counts"][j] / 5120.0 > threshold:
                sample_feature_matrix[i, feat_idx] = 1

    # Find most co-occurring pairs among top 100 features
    top100_feats = np.argsort(-feature_density)[:100]
    sub_matrix = sample_feature_matrix[:, top100_feats]

    log("\n  Co-occurrence among top 100 features (Jaccard similarity):")
    cooccur_pairs = []
    for i in range(len(top100_feats)):
        for j in range(i + 1, len(top100_feats)):
            both = np.sum((sub_matrix[:, i] > 0) & (sub_matrix[:, j] > 0))
            either = np.sum((sub_matrix[:, i] > 0) | (sub_matrix[:, j] > 0))
            if either > 0:
                jaccard = both / either
                if both >= 5:  # at least 5 samples
                    cooccur_pairs.append((top100_feats[i], top100_feats[j], jaccard, int(both)))

    cooccur_pairs.sort(key=lambda x: -x[2])
    log("  Top 20 co-occurring feature pairs:")
    log("  %6s  %6s  %10s  %8s" % ("Feat A", "Feat B", "Jaccard", "Samples"))
    for a, b, j, n in cooccur_pairs[:20]:
        log("  %6d  %6d  %10.4f  %8d" % (a, b, j, n))

    # ================================================================
    # ANALYSIS 4: Per-sample fingerprints and clustering
    # ================================================================
    log("\n" + "=" * 70)
    log("4. SAMPLE FINGERPRINTS")
    log("=" * 70)

    # Show top features for a few representative samples
    log("\n  Feature fingerprints for 10 samples:")
    for sf in per_sample_top_features[:10]:
        top5 = sf["top_features"][:5]
        top5_act = sf["top_activations"][:5]
        feats_str = ", ".join("%d(%.0f)" % (f, a) for f, a in zip(top5, top5_act))
        log("  Sample %3d [%s]: %s" % (sf["idx"], sf["preview"][:40], feats_str))

    # Check if samples cluster by their top features
    # Use top-20 features as fingerprint
    fingerprints = np.zeros((len(samples), 20), dtype=np.int32)
    for i, sf in enumerate(per_sample_top_features):
        fingerprints[i] = sf["top_features"][:20]

    # Count how often the same feature appears as #1 across samples
    top1_counter = Counter(sf["top_features"][0] for sf in per_sample_top_features)
    log("\n  Most common #1 feature across samples:")
    for feat, count in top1_counter.most_common(15):
        log("    Feature %d: #1 in %d/%d samples (%.1f%%)" % (
            feat, count, len(samples), count / len(samples) * 100))

    # ================================================================
    # ANALYSIS 5: Feature selectivity
    # ================================================================
    log("\n" + "=" * 70)
    log("5. FEATURE SELECTIVITY")
    log("=" * 70)

    # For each active feature, in how many samples does it fire?
    samples_per_feature = (sample_feature_matrix > 0).sum(axis=0)

    # Highly selective: fires in few samples
    selective = np.where((samples_per_feature > 0) & (samples_per_feature <= 5))[0]
    log("\n  Selective features (active in <=5 samples): %d / %d active" % (
        len(selective), ever_active.sum()))

    # Universal: fires in most samples
    universal = np.where(samples_per_feature >= len(samples) * 0.8)[0]
    log("  Universal features (active in >=80%% samples): %d" % len(universal))
    if len(universal) > 0:
        log("  Universal feature IDs: %s" % str(universal.tolist()[:30]))

    # Medium selectivity: fires in 10-50% of samples
    medium = np.where((samples_per_feature >= len(samples) * 0.1) &
                      (samples_per_feature <= len(samples) * 0.5))[0]
    log("  Medium-selectivity features (10-50%% of samples): %d" % len(medium))

    # Distribution
    log("\n  Feature selectivity distribution:")
    bins = [0, 1, 2, 5, 10, 20, 50, 100, len(samples)]
    for i in range(len(bins) - 1):
        count = np.sum((samples_per_feature >= bins[i]) & (samples_per_feature < bins[i + 1]))
        log("    %3d-%3d samples: %4d features" % (bins[i], bins[i + 1] - 1, count))

    # ================================================================
    # SUMMARY
    # ================================================================
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log("  Samples analyzed: %d" % len(samples))
    log("  Tokens analyzed: %d" % total_tokens)
    log("  Active features: %d / %d (%.1f%%)" % (ever_active.sum(), DICT_SIZE, ever_active.mean() * 100))
    log("  Universal features: %d" % len(universal))
    log("  Selective features (<=5 samples): %d" % len(selective))
    log("  Position-sensitive features: see top 20 above")
    log("  Feature co-occurrence: see top 20 pairs above")
    log("")
    log("Analysis complete.")


if __name__ == "__main__":
    main()
