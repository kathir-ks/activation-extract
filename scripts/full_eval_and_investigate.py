#!/usr/bin/env python3
"""
Full eval on step 125K + investigate divergence cause.
Part 1: Detailed eval metrics on step 125K (20 shards)
Part 2: Investigate divergence (LR schedule, data exhaustion, weight dynamics)
"""
import os, json, gzip, pickle, subprocess, time, math
import ml_dtypes
import numpy as np

TMP = "/tmp/sae_eval"
GCS_BUCKET = "arc-data-europe-west4"
GCS_PREFIX = "sae_checkpoints/layer19_topk_896d_v5e64"
DATA_GCS = "gs://arc-data-europe-west4/activations/layer19_merged_50k"
LAYER = 19
HIDDEN_DIM = 896
DICT_SIZE = 7168
K = 32


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print("[%s] %s" % (ts, msg), flush=True)


def gcs_cp(src, dst):
    subprocess.run(["gcloud", "storage", "cp", src, dst], capture_output=True, check=True)


def gcs_ls(path, pattern="shard_*.pkl.gz"):
    r = subprocess.run(["gcloud", "storage", "ls", os.path.join(path, pattern)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return []
    return sorted([l.strip() for l in r.stdout.strip().split("\n") if l.strip()])


def download_ckpt(step):
    d = "%s/step_%08d" % (TMP, step)
    if not os.path.exists(d):
        gcs = "gs://%s/%s/step_%08d" % (GCS_BUCKET, GCS_PREFIX, step)
        subprocess.run(["gcloud", "storage", "cp", "-r", gcs, TMP],
                       capture_output=True, check=True)
    return d


def load_params(d):
    with open(os.path.join(d, "metadata.json")) as f:
        meta = json.load(f)
    params = {}
    for k in meta["param_keys"]:
        a = np.load(os.path.join(d, "params", "%s.npy" % k))
        if a.dtype.kind == "V" and a.dtype.itemsize == 2:
            a = a.view(ml_dtypes.bfloat16)
        params[k] = a
    return params, meta


def sae_forward(params, x):
    """Full SAE forward pass in numpy. Returns x_hat, z, z_pre."""
    W_enc = params["W_enc"].astype(np.float32)
    b_enc = params["b_enc"].astype(np.float32)
    W_dec = params["W_dec"].astype(np.float32)
    b_dec = params["b_dec"].astype(np.float32)
    x = x.astype(np.float32)

    z_pre = (x - b_dec) @ W_enc + b_enc
    topk = np.argpartition(-z_pre, K, axis=-1)[:, :K]
    z = np.zeros_like(z_pre)
    for i in range(len(z)):
        z[i, topk[i]] = np.maximum(z_pre[i, topk[i]], 0)
    x_hat = z @ W_dec + b_dec
    return x_hat, z, z_pre


def load_shard_acts(local_path):
    with gzip.open(local_path, "rb") as f:
        data = pickle.load(f)
    layer_data = data.get(LAYER) or data.get(str(LAYER))
    if layer_data is None:
        return None
    return np.concatenate([s["activation"] for s in layer_data], axis=0)


# ================================================================
# PART 1: Full eval on step 125K
# ================================================================
def part1_full_eval():
    log("=" * 70)
    log("PART 1: FULL EVALUATION — Step 125,000")
    log("=" * 70)

    ckpt_dir = download_ckpt(125000)
    params, meta = load_params(ckpt_dir)
    log("Checkpoint: step=%d, tokens=%s" % (meta["step"], meta.get("total_tokens", "N/A")))

    # List all shards and select 20 random ones
    log("Listing shards...")
    all_entries = subprocess.run(
        ["gcloud", "storage", "ls", DATA_GCS + "/"],
        capture_output=True, text=True
    ).stdout.strip().split("\n")

    sub_dirs = []
    for e in all_entries:
        e = e.strip().rstrip("/")
        if e and any(p in e.split("/")[-1] for p in ("host_", "pair_")):
            sub_dirs.append(e)

    all_shards = []
    for sdir in sub_dirs:
        all_shards.extend(gcs_ls(sdir))
    log("Total shards: %d across %d subdirs" % (len(all_shards), len(sub_dirs)))

    np.random.seed(42)
    eval_indices = np.random.choice(len(all_shards), min(20, len(all_shards)), replace=False)
    eval_shards = [all_shards[i] for i in sorted(eval_indices)]

    # Load data
    log("Loading %d eval shards..." % len(eval_shards))
    all_acts = []
    for i, sp in enumerate(eval_shards):
        local = os.path.join(TMP, "eval_%d.pkl.gz" % i)
        gcs_cp(sp, local)
        acts = load_shard_acts(local)
        if acts is not None:
            all_acts.append(acts)
        os.remove(local)
        if (i + 1) % 5 == 0:
            log("  Loaded %d/%d shards" % (i + 1, len(eval_shards)))

    eval_data = np.concatenate(all_acts, axis=0)
    log("Eval data: %s (%s)" % (eval_data.shape, eval_data.dtype))

    # Run evaluation in batches
    log("Running forward passes...")
    BS = 1024
    n = eval_data.shape[0]
    all_mse = []
    all_ev = []
    feature_fire_count = np.zeros(DICT_SIZE, dtype=np.float64)
    total_samples = 0

    for start in range(0, n, BS):
        end = min(start + BS, n)
        batch = eval_data[start:end]
        x_hat, z, z_pre = sae_forward(params, batch)
        x_f32 = batch.astype(np.float32)

        mse = np.mean((x_f32 - x_hat) ** 2)
        var = np.var(x_f32)
        ev = 1.0 - np.var(x_f32 - x_hat) / max(var, 1e-8)
        all_mse.append(mse)
        all_ev.append(ev)

        active = (z != 0).astype(np.float32)
        feature_fire_count += active.sum(axis=0)
        total_samples += len(batch)

        if (start // BS + 1) % 500 == 0:
            log("  Batch %d/%d" % (start // BS + 1, n // BS + 1))

    # Aggregate metrics
    avg_mse = np.mean(all_mse)
    avg_ev = np.mean(all_ev)
    feature_density = feature_fire_count / total_samples
    ever_active = feature_fire_count > 0
    dead_frac = 1.0 - ever_active.mean()

    log("")
    log("=" * 70)
    log("EVALUATION RESULTS — Step 125,000")
    log("=" * 70)
    log("  Tokens evaluated:      %d" % total_samples)
    log("  Eval shards:           %d" % len(eval_shards))
    log("")
    log("  --- Reconstruction Quality ---")
    log("  MSE:                   %.6f" % avg_mse)
    log("  Explained variance:    %.4f (%.1f%%)" % (avg_ev, avg_ev * 100))
    log("  Input variance:        %.4f" % np.var(eval_data.astype(np.float32)))
    log("")
    log("  --- Sparsity ---")
    log("  L0 (active features):  32.0 / %d (k=%d)" % (DICT_SIZE, K))
    log("  L0 fraction:           %.4f" % (32.0 / DICT_SIZE))
    log("")
    log("  --- Feature Utilization ---")
    log("  Dead features (0 fires): %d / %d (%.1f%%)" % (
        int((~ever_active).sum()), DICT_SIZE, dead_frac * 100))
    log("  Active features:       %d / %d" % (int(ever_active.sum()), DICT_SIZE))
    if ever_active.any():
        active_density = feature_density[ever_active]
        log("  Mean feature density:  %.6f" % active_density.mean())
        log("  Median feature density:%.6f" % np.median(active_density))
        log("  Max feature density:   %.6f" % feature_density.max())
        log("")
        log("  --- Feature Density Percentiles (active features only) ---")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            log("  P%02d: %.6f" % (p, np.percentile(active_density, p)))

    # Dead neuron tracking from checkpoint
    dead_path = os.path.join(ckpt_dir, "dead_neuron_steps.npy")
    if os.path.exists(dead_path):
        dns = np.load(dead_path)
        if dns.dtype.kind == "V":
            dns = dns.view(np.int32)
        log("")
        log("  --- Dead Neuron Tracking (from checkpoint) ---")
        for threshold in [1000, 5000, 10000, 25000]:
            cnt = (dns >= threshold).sum()
            log("  Dead (>%dk steps):     %d / %d (%.1f%%)" % (
                threshold // 1000, cnt, DICT_SIZE, cnt / DICT_SIZE * 100))
        log("  Max inactive steps:    %d" % dns.max())

    log("=" * 70)
    return avg_ev


# ================================================================
# PART 2: Investigate divergence cause
# ================================================================
def part2_investigate():
    log("")
    log("=" * 70)
    log("PART 2: DIVERGENCE INVESTIGATION")
    log("=" * 70)

    # --- A. Training config analysis ---
    log("")
    log("--- A. Training Configuration ---")
    log("  Architecture: TopK (k=%d)" % K)
    log("  Hidden dim: %d, Dict size: %d (%.0fx expansion)" % (
        HIDDEN_DIM, DICT_SIZE, DICT_SIZE / HIDDEN_DIM))
    log("  Batch size: 4096, LR: 3e-4, Warmup: 1000 steps")
    log("  LR decay: cosine, Total steps: 200,000")
    log("  Dtype: bfloat16")

    # Cosine LR at divergence points
    lr_base = 3e-4
    warmup = 1000
    total = 200000
    log("")
    log("  LR schedule at key points:")
    for step in [1000, 25000, 50000, 100000, 125000, 150000, 175000, 200000]:
        if step < warmup:
            lr = lr_base * step / warmup
        else:
            progress = (step - warmup) / (total - warmup)
            lr = lr_base * 0.5 * (1 + math.cos(math.pi * progress))
        log("    Step %6d: LR = %.2e (%.1f%% of max)" % (step, lr, lr / lr_base * 100))

    # --- B. Data exhaustion check ---
    log("")
    log("--- B. Data Exhaustion Check ---")
    # 2345 shards, 235 samples each, 5120 tokens per sample
    # With 16 hosts, each gets 2345/16 = ~146 shards
    # Per host: 146 * 235 * 5120 = ~175M tokens
    # Batch: 4096 global = 256 per host
    # Steps to exhaust one pass: 175M / 256 = ~685K tokens/step... wait
    # Actually batch_size is token count per step globally
    # 4096 tokens per step globally, 200K steps = 819M tokens total
    total_tokens_available = 2345 * 235 * 5120
    tokens_trained = 200000 * 4096
    epochs = tokens_trained / total_tokens_available
    log("  Available tokens: %d (~%.1fB)" % (total_tokens_available, total_tokens_available / 1e9))
    log("  Tokens consumed:  %d (~%.1fM)" % (tokens_trained, tokens_trained / 1e6))
    log("  Effective epochs:  %.2f" % epochs)
    log("  Data was%s exhausted" % ("" if epochs > 1 else " NOT"))

    # Per-host calculation with shuffle buffer
    shards_per_host = 2345 // 16  # ~146
    tokens_per_host = shards_per_host * 235 * 5120
    tokens_per_step_per_host = 4096 // 16  # 256
    steps_to_exhaust = tokens_per_host / tokens_per_step_per_host
    log("")
    log("  Per host (16 hosts):")
    log("    Shards: ~%d" % shards_per_host)
    log("    Tokens: ~%.0fM" % (tokens_per_host / 1e6))
    log("    Tokens/step: %d" % tokens_per_step_per_host)
    log("    Steps to exhaust: ~%.0fK" % (steps_to_exhaust / 1e3))

    # --- C. Weight dynamics around divergence ---
    log("")
    log("--- C. Weight Dynamics Around Divergence ---")

    steps_to_check = [100000, 125000, 150000, 175000, 200000]
    prev_params = None
    prev_step = None

    for step in steps_to_check:
        d = download_ckpt(step)
        params, meta = load_params(d)

        if prev_params is not None:
            log("")
            log("  Step %d -> %d:" % (prev_step, step))
            for key in ["W_enc", "W_dec", "b_enc", "b_dec"]:
                a = params[key].astype(np.float32)
                b = prev_params[key].astype(np.float32)
                diff = a - b
                log("    %s: delta_norm=%.4f (%.2f%% of norm), max_delta=%.4f" % (
                    key,
                    np.linalg.norm(diff),
                    np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-8) * 100,
                    np.max(np.abs(diff)),
                ))

        prev_params = params
        prev_step = step

    # --- D. Decoder column norms (should be ~1 if normalize_decoder=True) ---
    log("")
    log("--- D. Decoder Column Norms ---")
    for step in [125000, 175000, 200000]:
        d = download_ckpt(step)
        params, _ = load_params(d)
        W_dec = params["W_dec"].astype(np.float32)
        col_norms = np.linalg.norm(W_dec, axis=1)  # [dict_size]
        log("  Step %d: mean=%.4f std=%.4f min=%.4f max=%.4f" % (
            step, col_norms.mean(), col_norms.std(), col_norms.min(), col_norms.max()))

    # --- E. Gradient explosion indicator: b_enc change ---
    log("")
    log("--- E. Bias Drift (indicator of instability) ---")
    for step in [5000, 50000, 100000, 125000, 150000, 175000, 200000]:
        d = download_ckpt(step)
        params, _ = load_params(d)
        b_enc = params["b_enc"].astype(np.float32)
        b_dec = params["b_dec"].astype(np.float32)
        log("  Step %6d: b_enc(mean=%.4f, std=%.4f, norm=%.4f) b_dec(mean=%.4f, std=%.4f, norm=%.4f)" % (
            step, b_enc.mean(), b_enc.std(), np.linalg.norm(b_enc),
            b_dec.mean(), b_dec.std(), np.linalg.norm(b_dec)))

    log("")
    log("=" * 70)
    log("INVESTIGATION COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True)
    ev = part1_full_eval()
    part2_investigate()
