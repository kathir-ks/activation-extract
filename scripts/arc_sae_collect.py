#!/usr/bin/env python3
"""
Phase 1: Run Qwen 2.5 + trained SAE on N ARC tasks, collect per-token SAE
feature activations with rich per-token metadata. Saves results to GCS as a
pickle bundle that phase 2/3 analysis scripts consume.

Runs entirely on CPU (PyTorch for Qwen, numpy for SAE) so it does NOT interfere
with an active JAX/TPU training job on the same worker.

Usage (on a TPU worker, nice'd to avoid disturbing training):
    nice -n 19 python3 scripts/arc_sae_collect.py \
        --sae_gcs_prefix sae_checkpoints/layer19_topk_896d_v2 \
        --sae_arch topk --sae_dict_size 7168 --sae_k 32 \
        --qwen_model Qwen/Qwen2.5-0.5B \
        --arc_jsonl /home/kathirks_gc/activation-extract/arc_formatted_challenges.jsonl \
        --num_tasks 25 \
        --layer_index 19 \
        --output_gcs gs://arc-data-europe-west4/sae_analysis/v2_collect
"""

import argparse
import gzip
import json
import os
import pickle
import subprocess
import sys
import time
from typing import Any, Dict, List

import ml_dtypes  # noqa: F401 — register bfloat16 with numpy
import numpy as np


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# SAE loading / encoding
# ---------------------------------------------------------------------------

def gcs_cp(src: str, dst: str) -> None:
    subprocess.run(["gcloud", "storage", "cp", src, dst], capture_output=True, check=True)


def gcs_cp_r(src: str, dst: str) -> None:
    subprocess.run(["gcloud", "storage", "cp", "-r", src, dst], capture_output=True, check=True)


def download_sae_checkpoint(gcs_bucket: str, gcs_prefix: str, step: int, tmp_dir: str) -> str:
    os.makedirs(tmp_dir, exist_ok=True)
    if step == 0:
        marker = f"gs://{gcs_bucket}/{gcs_prefix}/latest_step.json"
        local = os.path.join(tmp_dir, "latest_step.json")
        gcs_cp(marker, local)
        with open(local) as f:
            step = json.load(f)["step"]
    step_dir_name = f"step_{step:08d}"
    local_step = os.path.join(tmp_dir, step_dir_name)
    if not os.path.exists(local_step):
        gcs_cp_r(f"gs://{gcs_bucket}/{gcs_prefix}/{step_dir_name}", tmp_dir)
    return local_step


def load_sae_params(step_dir: str) -> Dict[str, np.ndarray]:
    with open(os.path.join(step_dir, "metadata.json")) as f:
        meta = json.load(f)
    params: Dict[str, np.ndarray] = {}
    for k in meta["param_keys"]:
        a = np.load(os.path.join(step_dir, "params", f"{k}.npy"))
        if a.dtype.kind == "V" and a.dtype.itemsize == 2:
            a = a.view(ml_dtypes.bfloat16)
        params[k] = a.astype(np.float32)
    return params


def sae_encode_topk(x: np.ndarray, params: Dict[str, np.ndarray], k: int) -> np.ndarray:
    """TopK SAE encode. x: [N, hidden_dim]. Returns z: [N, dict_size]."""
    z_pre = (x.astype(np.float32) - params["b_dec"]) @ params["W_enc"] + params["b_enc"]
    topk_idx = np.argpartition(-z_pre, k, axis=-1)[:, :k]
    z = np.zeros_like(z_pre)
    for i in range(z.shape[0]):
        z[i, topk_idx[i]] = np.maximum(z_pre[i, topk_idx[i]], 0)
    return z


def sae_encode_jumprelu(x: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    """JumpReLU SAE encode. x: [N, hidden_dim]. Returns z: [N, dict_size]."""
    z_pre = (x.astype(np.float32) - params["b_dec"]) @ params["W_enc"] + params["b_enc"]
    threshold = np.exp(params["log_threshold"])  # [dict_size]
    mask = z_pre > threshold
    z = np.where(mask, z_pre, 0.0)
    return z


# ---------------------------------------------------------------------------
# ARC task loading / prompt rendering (minimal, no chat template)
# ---------------------------------------------------------------------------

def load_arc_tasks(jsonl_path: str, num_tasks: int) -> List[Dict[str, Any]]:
    tasks = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
            if len(tasks) >= num_tasks:
                break
    return tasks


def grid_to_text(grid: List[List[int]]) -> str:
    return "\n".join("".join(str(x) for x in row) for row in grid)


def render_task_minimal(task: Dict[str, Any]) -> Dict[str, Any]:
    """Render a single ARC task as a minimal training prompt: all train examples
    then a test input, all as plain digit grids. Returns the prompt text plus a
    structured manifest of grid blocks with their character offsets."""
    parts: List[str] = []
    blocks: List[Dict[str, Any]] = []

    header = f"Task {task.get('task_id', '?')}\n\n"
    parts.append(header)

    def push_grid(grid: List[List[int]], tag: str) -> None:
        start = sum(len(p) for p in parts)
        text = grid_to_text(grid)
        end = start + len(text)
        parts.append(text + "\n\n")
        blocks.append({
            "tag": tag,
            "char_start": start,
            "char_end": end,
            "grid": [list(row) for row in grid],
            "rows": len(grid),
            "cols": len(grid[0]) if grid else 0,
        })

    for i, ex in enumerate(task.get("train", [])):
        parts.append(f"Example {i + 1} input\n")
        push_grid(ex["input"], f"train_{i}_input")
        parts.append(f"Example {i + 1} output\n")
        push_grid(ex["output"], f"train_{i}_output")

    for i, ex in enumerate(task.get("test", [])):
        parts.append(f"Test {i + 1} input\n")
        push_grid(ex["input"], f"test_{i}_input")
        if "output" in ex and ex["output"]:
            parts.append(f"Test {i + 1} output\n")
            push_grid(ex["output"], f"test_{i}_output")

    prompt = "".join(parts)
    return {"prompt": prompt, "blocks": blocks}


# ---------------------------------------------------------------------------
# Token-level grid cell mapping
# ---------------------------------------------------------------------------

def classify_tokens(
    prompt: str,
    blocks: List[Dict[str, Any]],
    offsets: List[tuple],
    token_strs: List[str],
) -> List[Dict[str, Any]]:
    """For each token, emit metadata:
        - token_idx, token_str, char_start, char_end
        - is_single_digit, digit_value (0..9 or None)
        - block_tag (which grid block) or None
        - row, col (within the grid) when inside a grid
        - is_newline, is_separator (header/newlines between grids)
    Assumes MinimalGridEncoder: each cell is a single digit char; rows separated
    by '\n'. Tokens that fall inside a grid block's char range and are a single
    digit get (row, col) from the prompt char layout.
    """
    # Pre-compute per-block char→(row, col) map for O(1) lookup
    block_char_to_rc: Dict[int, Dict[int, tuple]] = {}
    for bi, b in enumerate(blocks):
        cs, ce = b["char_start"], b["char_end"]
        mapping = {}
        row = 0
        col = 0
        for c_off, ch in enumerate(prompt[cs:ce]):
            abs_pos = cs + c_off
            if ch == "\n":
                row += 1
                col = 0
                continue
            mapping[abs_pos] = (row, col)
            col += 1
        block_char_to_rc[bi] = mapping

    tokens = []
    for tok_idx, ((ch_s, ch_e), tstr) in enumerate(zip(offsets, token_strs)):
        if ch_s is None or ch_e is None or ch_s == ch_e:
            # special token (BOS/EOS/etc.) with no char span
            tokens.append({
                "token_idx": tok_idx,
                "token_str": tstr,
                "char_start": ch_s if ch_s is not None else -1,
                "char_end": ch_e if ch_e is not None else -1,
                "is_single_digit": False,
                "digit_value": None,
                "block_tag": None,
                "block_idx": None,
                "row": None,
                "col": None,
                "is_newline": False,
                "is_in_grid": False,
            })
            continue

        raw = prompt[ch_s:ch_e]
        is_nl = raw == "\n"
        is_digit = (len(raw) == 1 and raw.isdigit())
        digit_val = int(raw) if is_digit else None

        # Find which block (if any) this token falls in
        block_idx = None
        block_tag = None
        row = None
        col = None
        for bi, b in enumerate(blocks):
            if b["char_start"] <= ch_s < b["char_end"]:
                block_idx = bi
                block_tag = b["tag"]
                if is_digit and ch_s in block_char_to_rc[bi]:
                    row, col = block_char_to_rc[bi][ch_s]
                break

        tokens.append({
            "token_idx": tok_idx,
            "token_str": tstr,
            "char_start": int(ch_s),
            "char_end": int(ch_e),
            "is_single_digit": bool(is_digit),
            "digit_value": digit_val,
            "block_tag": block_tag,
            "block_idx": block_idx,
            "row": row,
            "col": col,
            "is_newline": bool(is_nl),
            "is_in_grid": block_idx is not None,
        })
    return tokens


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-token SAE activations on ARC tasks")
    parser.add_argument("--sae_gcs_bucket", default="arc-data-europe-west4")
    parser.add_argument("--sae_gcs_prefix", required=True,
                        help="e.g., sae_checkpoints/layer19_topk_896d_v2")
    parser.add_argument("--sae_step", type=int, default=0, help="0 = use latest_step.json")
    parser.add_argument("--sae_arch", default="topk", choices=["topk", "jumprelu"])
    parser.add_argument("--sae_hidden_dim", type=int, default=896)
    parser.add_argument("--sae_dict_size", type=int, required=True)
    parser.add_argument("--sae_k", type=int, default=32)
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--layer_index", type=int, default=19)
    parser.add_argument("--arc_jsonl", required=True)
    parser.add_argument("--num_tasks", type=int, default=25)
    parser.add_argument("--max_seq_len", type=int, default=3072)
    parser.add_argument("--tmp_dir", default="/tmp/arc_sae_collect")
    parser.add_argument("--output_gcs", required=True,
                        help="e.g., gs://arc-data-europe-west4/sae_analysis/v2_collect")
    parser.add_argument("--output_name", default="collected.pkl.gz")
    args = parser.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)

    log("=" * 70)
    log("ARC SAE Collection — Phase 1")
    log("=" * 70)
    log(f"SAE: {args.sae_gcs_prefix} arch={args.sae_arch} dict={args.sae_dict_size}")
    log(f"Model: {args.qwen_model} layer {args.layer_index}")
    log(f"Tasks: {args.num_tasks} from {args.arc_jsonl}")
    log(f"Output: {args.output_gcs}/{args.output_name}")

    # 1. Download SAE
    log("Downloading SAE checkpoint...")
    step_dir = download_sae_checkpoint(
        args.sae_gcs_bucket, args.sae_gcs_prefix, args.sae_step, args.tmp_dir,
    )
    params = load_sae_params(step_dir)
    with open(os.path.join(step_dir, "metadata.json")) as f:
        ckpt_meta = json.load(f)
    log(f"  Loaded SAE params (step {ckpt_meta.get('step')}): keys={list(params.keys())}")
    log(f"  W_enc shape: {params['W_enc'].shape}, b_enc shape: {params['b_enc'].shape}")

    # 2. Load Qwen via HuggingFace (CPU)
    log("Loading Qwen model via HuggingFace (CPU)...")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Force CPU to avoid any accidental accelerator use
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 4))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    hidden_dim = model.config.hidden_size
    log(f"  Loaded {args.qwen_model}: hidden_dim={hidden_dim}")
    assert hidden_dim == args.sae_hidden_dim, (
        f"Model hidden_dim {hidden_dim} != SAE hidden_dim {args.sae_hidden_dim}"
    )

    # 3. Load ARC tasks
    log("Loading ARC tasks...")
    tasks = load_arc_tasks(args.arc_jsonl, args.num_tasks)
    log(f"  Loaded {len(tasks)} tasks")

    # 4. For each task, render + tokenize + forward + SAE encode
    collected: List[Dict[str, Any]] = []
    t0 = time.time()

    for ti, task in enumerate(tasks):
        render = render_task_minimal(task)
        prompt = render["prompt"]
        blocks = render["blocks"]

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=args.max_seq_len,
        )
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        offsets_pt = enc["offset_mapping"][0].tolist()
        offsets = [(o[0], o[1]) for o in offsets_pt]
        seq_len = int(input_ids.shape[1])

        token_ids = input_ids[0].tolist()
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]

        token_meta = classify_tokens(prompt, blocks, offsets, token_strs)

        # Forward pass, grab layer `layer_index` residual stream
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        # hidden_states tuple: len = num_layers + 1 (embeddings first)
        # Qwen convention: hidden_states[i] = output of layer i-1 for i >= 1.
        # We follow extract_activations.py semantics: layer_index k means the
        # residual stream at the k-th transformer block's output, which is
        # hidden_states[k+1] when hidden_states[0] is embeddings.
        hs = out.hidden_states[args.layer_index + 1]  # [1, seq_len, H]
        acts_np = hs[0].cpu().numpy().astype(np.float32)  # [seq_len, H]

        # SAE encode
        if args.sae_arch == "topk":
            z = sae_encode_topk(acts_np, params, args.sae_k)
        else:
            z = sae_encode_jumprelu(acts_np, params)

        # Keep only non-zero feature entries per token as (feat_idx, value) pairs
        # to keep the output bundle small.
        sparse_per_token: List[List[List[float]]] = []
        for i in range(seq_len):
            nz = np.nonzero(z[i])[0]
            pairs = [[int(f), float(z[i, f])] for f in nz]
            sparse_per_token.append(pairs)

        collected.append({
            "task_id": task.get("task_id", f"task_{ti}"),
            "prompt": prompt,
            "blocks": blocks,
            "tokens": token_meta,
            "token_ids": token_ids,
            "sparse_features": sparse_per_token,  # list[list[[feat_idx, value]]]
        })

        elapsed = time.time() - t0
        eta = elapsed / (ti + 1) * (len(tasks) - ti - 1)
        log(f"  [{ti+1}/{len(tasks)}] {task.get('task_id','?')}: "
            f"seq_len={seq_len} digit_toks={sum(t['is_single_digit'] for t in token_meta)} "
            f"({elapsed:.1f}s elapsed, ETA {eta:.0f}s)")

    log(f"Collection complete in {time.time() - t0:.1f}s")

    # 5. Save bundle to local, upload to GCS
    out_local = os.path.join(args.tmp_dir, args.output_name)
    bundle = {
        "meta": {
            "sae_gcs_prefix": args.sae_gcs_prefix,
            "sae_arch": args.sae_arch,
            "sae_hidden_dim": args.sae_hidden_dim,
            "sae_dict_size": args.sae_dict_size,
            "sae_k": args.sae_k,
            "sae_step": ckpt_meta.get("step"),
            "qwen_model": args.qwen_model,
            "layer_index": args.layer_index,
            "num_tasks": len(collected),
        },
        "tasks": collected,
    }
    log(f"Writing bundle to {out_local}...")
    with gzip.open(out_local, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    sz_mb = os.path.getsize(out_local) / 1e6
    log(f"  Bundle size: {sz_mb:.1f} MB")

    log(f"Uploading to {args.output_gcs}/{args.output_name}...")
    subprocess.run(
        ["gcloud", "storage", "cp", out_local, f"{args.output_gcs}/{args.output_name}"],
        check=True,
    )
    log("Upload complete.")
    log("=" * 70)
    log("Phase 1 DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()
