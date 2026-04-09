#!/usr/bin/env python3
"""Find the exact checkpoint step where SAE training diverged."""
import os, json, gzip, pickle, subprocess, time
import ml_dtypes
import numpy as np

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print("[%s] %s" % (ts, msg), flush=True)

def gcs_cp(src, dst):
    subprocess.run(["gcloud", "storage", "cp", src, dst], capture_output=True, check=True)

def download_ckpt(step, tmp="/tmp/sae_diag"):
    d = "%s/step_%08d" % (tmp, step)
    if not os.path.exists(d):
        gcs = "gs://arc-data-europe-west4/sae_checkpoints/layer19_topk_896d_v5e64/step_%08d" % step
        subprocess.run(["gcloud", "storage", "cp", "-r", gcs, tmp], capture_output=True, check=True)
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
    return params

def quick_eval(params, x):
    W_enc = params["W_enc"].astype(np.float32)
    b_enc = params["b_enc"].astype(np.float32)
    W_dec = params["W_dec"].astype(np.float32)
    b_dec = params["b_dec"].astype(np.float32)
    x = x.astype(np.float32)
    z_pre = (x - b_dec) @ W_enc + b_enc
    topk = np.argpartition(-z_pre, 32, axis=-1)[:, :32]
    z = np.zeros_like(z_pre)
    for i in range(len(z)):
        z[i, topk[i]] = np.maximum(z_pre[i, topk[i]], 0)
    x_hat = z @ W_dec + b_dec
    mse = np.mean((x - x_hat)**2)
    var = np.var(x)
    ev = 1.0 - np.var(x - x_hat) / max(var, 1e-8)
    return mse, ev, np.std(x_hat)

# Load test data
os.makedirs("/tmp/sae_diag", exist_ok=True)
log("Loading test data...")
gcs_cp("gs://arc-data-europe-west4/activations/layer19_merged_50k/pair_00/shard_0001.pkl.gz",
       "/tmp/sae_diag/test.pkl.gz")
with gzip.open("/tmp/sae_diag/test.pkl.gz", "rb") as f:
    data = pickle.load(f)
acts = np.concatenate([s["activation"] for s in data[19]], axis=0)[:2048]
log("Test data: %s (var=%.4f)" % (acts.shape, np.var(acts.astype(np.float32))))

# Check multiple steps
steps = [5000, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000]
log("")
log("%-10s %10s %10s %10s" % ("Step", "MSE", "ExplVar", "xhat_std"))
log("-" * 45)
for step in steps:
    try:
        d = download_ckpt(step)
        p = load_params(d)
        mse, ev, xhat_std = quick_eval(p, acts)
        log("%-10d %10.4f %10.4f %10.4f" % (step, mse, ev, xhat_std))
    except Exception as e:
        log("%-10d ERROR: %s" % (step, e))

log("")
log("Done.")
