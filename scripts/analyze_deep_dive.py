#!/usr/bin/env python3
"""
Deep-dive SAE feature analysis:
1. Medium-selectivity features (what patterns they correspond to)
2. Universal cluster analysis (what tokens trigger dominant features)
3. Sample clustering by feature fingerprints
"""
import os, json, gzip, pickle, subprocess, time
import ml_dtypes
import numpy as np
from collections import Counter, defaultdict

TMP = "/tmp/sae_deep"
CKPT_STEP = 125000
DATA_GCS = "gs://arc-data-europe-west4/activations/layer19_merged_50k"
LAYER = 19
K = 32
DICT_SIZE = 7168
SEQ_LEN = 5120


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print("[%s] %s" % (ts, msg), flush=True)


def gcs_cp(src, dst):
    subprocess.run(["gcloud", "storage", "cp", src, dst], capture_output=True, check=True)


def load_sae_params():
    d = "%s/step_%08d" % (TMP, CKPT_STEP)
    if not os.path.exists(d):
        subprocess.run(["gcloud", "storage", "cp", "-r",
            "gs://arc-data-europe-west4/sae_checkpoints/layer19_topk_896d_v2/step_%08d" % CKPT_STEP,
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


def sae_encode(params, x):
    x = x.astype(np.float32)
    z_pre = (x - params["b_dec"]) @ params["W_enc"] + params["b_enc"]
    topk_idx = np.argpartition(-z_pre, K, axis=-1)[:, :K]
    z = np.zeros_like(z_pre)
    for i in range(len(z)):
        z[i, topk_idx[i]] = np.maximum(z_pre[i, topk_idx[i]], 0)
    return z


def load_samples(shard_paths, max_samples=200):
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
        if len(all_samples) >= max_samples:
            break
    return all_samples[:max_samples]


def main():
    os.makedirs(TMP, exist_ok=True)

    log("Loading SAE checkpoint...")
    params = load_sae_params()

    # Load samples from diverse pairs
    shard_paths = [
        "%s/pair_00/shard_0010.pkl.gz" % DATA_GCS,
        "%s/pair_01/shard_0030.pkl.gz" % DATA_GCS,
        "%s/pair_02/shard_0050.pkl.gz" % DATA_GCS,
        "%s/pair_04/shard_0070.pkl.gz" % DATA_GCS,
        "%s/pair_06/shard_0090.pkl.gz" % DATA_GCS,
    ]

    log("Loading samples from %d shards..." % len(shard_paths))
    samples = load_samples(shard_paths, max_samples=250)
    log("Loaded %d samples" % len(samples))

    # Pre-compute per-sample feature profiles
    log("\nEncoding all samples through SAE...")
    sample_profiles = []  # per-sample: feature_density [DICT_SIZE]
    sample_position_profiles = []  # per-sample: [5 regions x DICT_SIZE]

    regions = [(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096), (4096, 5120)]
    region_names = ["0-1K", "1K-2K", "2K-3K", "3K-4K", "4K-5K"]

    for i, sample in enumerate(samples):
        acts = sample["activation"]
        seq_len = len(acts)
        feature_density = np.zeros(DICT_SIZE, dtype=np.float64)
        region_density = np.zeros((len(regions), DICT_SIZE), dtype=np.float64)

        for r_idx, (start, end) in enumerate(regions):
            if start >= seq_len:
                break
            actual_end = min(end, seq_len)
            z = sae_encode(params, acts[start:actual_end])
            active = (z != 0).astype(np.float32)
            feature_density += active.sum(axis=0)
            region_density[r_idx] = active.mean(axis=0)

        feature_density /= seq_len
        sample_profiles.append(feature_density)
        sample_position_profiles.append(region_density)

        if (i + 1) % 50 == 0:
            log("  Encoded %d/%d samples" % (i + 1, len(samples)))

    sample_profiles = np.array(sample_profiles)  # [n_samples, DICT_SIZE]
    log("  Done. Shape: %s" % str(sample_profiles.shape))

    # Identify feature categories
    samples_per_feature = (sample_profiles > 0.01).sum(axis=0)  # >1% density threshold
    n = len(samples)

    universal_mask = samples_per_feature >= n * 0.8
    medium_mask = (samples_per_feature >= n * 0.1) & (samples_per_feature <= n * 0.5)
    selective_mask = (samples_per_feature >= 1) & (samples_per_feature <= 5)

    universal_feats = np.where(universal_mask)[0]
    medium_feats = np.where(medium_mask)[0]
    selective_feats = np.where(selective_mask)[0]

    # ================================================================
    # ANALYSIS 2: Universal cluster (what triggers dominant features)
    # ================================================================
    log("\n" + "=" * 70)
    log("ANALYSIS 2: UNIVERSAL FEATURE CLUSTER")
    log("=" * 70)

    log("\n  Universal features (active in >=80%% of samples): %d" % len(universal_feats))
    log("  IDs: %s" % str(universal_feats.tolist()))

    # For each universal feature, analyze position pattern
    log("\n  Position patterns for universal features:")
    log("  %6s  %10s | %s" % ("Feat#", "MeanDens", " | ".join("%-7s" % r for r in region_names)))

    for f in universal_feats:
        mean_dens = sample_profiles[:, f].mean()
        pos_pattern = [np.mean([sp[r_idx, f] for sp in sample_position_profiles]) for r_idx in range(len(regions))]
        pos_str = " | ".join("%-7.4f" % p for p in pos_pattern)
        log("  %6d  %10.4f | %s" % (f, mean_dens, pos_str))

    # Classify universal features
    log("\n  Classification:")
    for f in universal_feats:
        pos_pattern = [np.mean([sp[r_idx, f] for sp in sample_position_profiles]) for r_idx in range(len(regions))]
        total = sum(pos_pattern)
        if total < 1e-6:
            continue
        normalized = [p / total for p in pos_pattern]
        max_region = np.argmax(normalized)
        uniformity = np.std(normalized)

        if uniformity < 0.05:
            label = "UNIFORM (fires equally across all positions)"
        elif normalized[0] > 0.4:
            label = "START-HEAVY (concentrated in first tokens)"
        elif normalized[-1] > 0.4:
            label = "END-HEAVY (concentrated in last tokens)"
        elif normalized[max_region] > 0.3:
            label = "REGION-%s focused" % region_names[max_region]
        else:
            label = "BROAD (slight variation)"

        log("    Feature %d: %s (density=%.4f)" % (f, label, sample_profiles[:, f].mean()))

    # Activation magnitude analysis for top 5 universal features
    top5_universal = [239, 6613, 1705, 392, 2525]
    log("\n  Activation magnitude distribution for top universal features:")
    for f in top5_universal:
        all_acts = []
        for sample in samples[:30]:
            z = sae_encode(params, sample["activation"][:512])  # first 512 tokens
            acts_f = z[:, f]
            all_acts.extend(acts_f[acts_f > 0].tolist())
        if all_acts:
            arr = np.array(all_acts)
            log("    Feature %d: mean=%.2f, std=%.2f, min=%.2f, max=%.2f, P50=%.2f, P95=%.2f" % (
                f, arr.mean(), arr.std(), arr.min(), arr.max(),
                np.percentile(arr, 50), np.percentile(arr, 95)))

    # ================================================================
    # ANALYSIS 1: Medium-selectivity features
    # ================================================================
    log("\n" + "=" * 70)
    log("ANALYSIS 1: MEDIUM-SELECTIVITY FEATURES")
    log("=" * 70)

    log("\n  Medium-selectivity features (10-50%% of samples): %d" % len(medium_feats))

    # For each medium feature, characterize when it fires
    log("\n  Detailed profile for each medium-selectivity feature:")
    log("  %6s  %6s  %10s  %s" % ("Feat#", "nSamp", "MeanDens", "Position Pattern"))

    medium_details = []
    for f in medium_feats:
        n_samples_active = int(samples_per_feature[f])
        mean_dens = sample_profiles[sample_profiles[:, f] > 0.01, f].mean() if n_samples_active > 0 else 0

        # Position pattern (averaged over samples where it's active)
        active_mask = sample_profiles[:, f] > 0.01
        if active_mask.sum() > 0:
            pos = [np.mean([sample_position_profiles[j][r, f] for j in range(len(samples)) if active_mask[j]])
                   for r in range(len(regions))]
        else:
            pos = [0] * len(regions)

        total = sum(pos) + 1e-10
        pos_norm = [p / total for p in pos]
        dominant_region = np.argmax(pos_norm)

        if np.std(pos_norm) < 0.05:
            pos_label = "uniform"
        elif pos_norm[0] > 0.35:
            pos_label = "START"
        elif pos_norm[-1] > 0.35:
            pos_label = "END"
        elif pos_norm[dominant_region] > 0.3:
            pos_label = "region-%s" % region_names[dominant_region]
        else:
            pos_label = "broad"

        pos_str = " ".join("%.3f" % p for p in pos)
        log("  %6d  %6d  %10.4f  %s [%s]" % (f, n_samples_active, mean_dens, pos_str, pos_label))

        medium_details.append({
            "feat": f,
            "n_samples": n_samples_active,
            "mean_density": mean_dens,
            "pos_label": pos_label,
            "active_samples": np.where(active_mask)[0].tolist(),
        })

    # Group medium features by position pattern
    log("\n  Medium features grouped by position pattern:")
    pattern_groups = defaultdict(list)
    for md in medium_details:
        pattern_groups[md["pos_label"]].append(md["feat"])

    for pattern, feats in sorted(pattern_groups.items(), key=lambda x: -len(x[1])):
        log("    %s: %d features — %s" % (pattern, len(feats), str(feats[:15]) + ("..." if len(feats) > 15 else "")))

    # Show which samples share medium features
    log("\n  Sample groupings by medium features (features active in 20-40%% of samples):")
    interesting_medium = [md for md in medium_details if 0.2 * n <= md["n_samples"] <= 0.4 * n]
    for md in interesting_medium[:10]:
        active_previews = [samples[j]["text_preview"][:50] for j in md["active_samples"][:5]]
        inactive_previews = [samples[j]["text_preview"][:50] for j in range(len(samples))
                             if j not in md["active_samples"]][:3]
        log("    Feature %d (active in %d samples):" % (md["feat"], md["n_samples"]))
        log("      Active in:   %s" % str(active_previews))
        log("      Inactive in: %s" % str(inactive_previews))

    # ================================================================
    # ANALYSIS 3: Sample clustering
    # ================================================================
    log("\n" + "=" * 70)
    log("ANALYSIS 3: SAMPLE CLUSTERING BY FEATURE FINGERPRINTS")
    log("=" * 70)

    # Use medium + selective features for clustering (most discriminative)
    cluster_feats = np.concatenate([medium_feats, selective_feats])
    log("\n  Using %d discriminative features for clustering" % len(cluster_feats))

    # Normalize sample profiles to unit vectors
    cluster_matrix = sample_profiles[:, cluster_feats]
    norms = np.linalg.norm(cluster_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    cluster_normed = cluster_matrix / norms

    # Compute pairwise cosine similarity
    n_samples = len(samples)
    log("  Computing pairwise similarity (%d x %d)..." % (n_samples, n_samples))
    sim_matrix = cluster_normed @ cluster_normed.T

    # Simple clustering: find groups of similar samples
    # Greedy: pick seed, add all samples with sim > threshold
    threshold = 0.7
    assigned = set()
    clusters = []

    for seed in range(n_samples):
        if seed in assigned:
            continue
        similar = np.where(sim_matrix[seed] > threshold)[0]
        similar = [s for s in similar if s not in assigned]
        if len(similar) >= 3:  # minimum cluster size
            clusters.append(similar)
            assigned.update(similar)

    # Add remaining as singletons
    remaining = [i for i in range(n_samples) if i not in assigned]

    log("  Found %d clusters (threshold=%.1f), %d unclustered" % (
        len(clusters), threshold, len(remaining)))

    # Show clusters
    for ci, cluster in enumerate(clusters[:15]):
        previews = [samples[j]["text_preview"][:50] for j in cluster[:5]]
        # Find distinctive features for this cluster
        cluster_mean = sample_profiles[cluster].mean(axis=0)
        non_cluster = [i for i in range(n_samples) if i not in cluster]
        if non_cluster:
            non_cluster_mean = sample_profiles[non_cluster].mean(axis=0)
        else:
            non_cluster_mean = np.zeros(DICT_SIZE)
        diff = cluster_mean - non_cluster_mean
        distinctive = np.argsort(-diff)[:5]
        distinctive_feats = [(int(f), float(cluster_mean[f]), float(non_cluster_mean[f])) for f in distinctive]

        log("\n  Cluster %d (%d samples):" % (ci, len(cluster)))
        log("    Samples: %s" % str(previews))
        log("    Distinctive features (feat, cluster_density, other_density):")
        for f, cd, od in distinctive_feats:
            log("      Feature %d: %.4f in cluster vs %.4f elsewhere (%.1fx)" % (
                f, cd, od, cd / max(od, 1e-6)))

    # Inter-cluster similarity
    if len(clusters) >= 2:
        log("\n  Inter-cluster similarity (cosine of mean fingerprints):")
        cluster_means = []
        for cluster in clusters[:10]:
            mean_vec = cluster_normed[cluster].mean(axis=0)
            mean_vec /= max(np.linalg.norm(mean_vec), 1e-8)
            cluster_means.append(mean_vec)

        for i in range(min(len(cluster_means), 10)):
            for j in range(i + 1, min(len(cluster_means), 10)):
                sim = np.dot(cluster_means[i], cluster_means[j])
                log("    Cluster %d vs %d: %.4f" % (i, j, sim))

    # Overall structure summary
    log("\n  Cluster size distribution:")
    sizes = sorted([len(c) for c in clusters], reverse=True)
    log("    %s" % str(sizes[:20]))

    log("\n" + "=" * 70)
    log("ANALYSIS COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
