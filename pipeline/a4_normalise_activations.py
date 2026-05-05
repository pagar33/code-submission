
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import json
import os
import time

import h5py
import numpy as np
from dotenv import load_dotenv

import config

load_dotenv()

try:
    from artifact_store import upload_activation
    _ARTIFACT_STORE_AVAILABLE = True
except ImportError:
    _ARTIFACT_STORE_AVAILABLE = False


def _maybe_upload(model: str, source: str, output_path: str):
    """Upload to W&B artifact registry if configured (best-effort, never fatal)."""
    if not _ARTIFACT_STORE_AVAILABLE:
        return
    if not os.getenv("WANDB_API_KEY"):
        return
    try:
        upload_activation(model, source, output_path)
    except Exception as exc:
        print(f"[artifact_store] Upload failed (non-fatal): {exc}")


def log_run(script: str, start_time: float, status: str, error: str = ""):
    end_time = time.time()
    entry = {
        "script": script,
        "start_time": start_time,
        "end_time": end_time,
        "status": status,
        "error": error,
    }
    with open("run_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _read_h5_activations(path: str) -> "np.ndarray":
    """Read activations from an HDF5 file, handling bfloat16 stored by older extraction runs."""
    with h5py.File(path, "r") as f:
        dset = f["activations"]
        try:
            return dset[:]
        except ValueError:
            # bfloat16 fast-reader bug — bypass using NATIVE_FLOAT memory type
            shape = dset.shape
            buf = np.empty(shape, dtype=np.float32)
            fspace = dset.id.get_space()
            mspace = h5py.h5s.create_simple(shape)
            dset.id.read(mspace, fspace, buf, h5py.h5t.NATIVE_FLOAT)
            return buf


def _normalise_one(model: str, input_path: str, output_path: str, stats_path: str):
    """Normalise a single raw activations file and save norm + stats."""
    for _attempt in range(6):
        try:
            acts = _read_h5_activations(input_path)
            break
        except BlockingIOError:
            if _attempt < 5:
                _w = 2 ** _attempt
                print(f"HDF5 file locked (errno 11), retrying in {_w}s (attempt {_attempt+1}/6)…")
                time.sleep(_w)
            else:
                raise

    hidden_dim = config.MODELS[model]["hidden_dim"]
    if acts.shape[1] != hidden_dim:
        raise ValueError(f"Activation hidden_dim mismatch for {model}: {acts.shape[1]} vs {hidden_dim}")

    mean = acts.mean(axis=0)
    std = acts.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    acts_norm = (acts - mean) / std

    original_mean_l2   = float(np.linalg.norm(acts, axis=1).mean())
    normalised_mean_l2 = float(np.linalg.norm(acts_norm, axis=1).mean())
    print(f"  Before — mean L2 norm: {original_mean_l2:.2f}")
    print(f"  After  — mean L2 norm: {normalised_mean_l2:.2f}")

    with h5py.File(output_path, "w") as f:
        f.create_dataset("activations", data=acts_norm.astype(np.float32))

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "original_mean_l2": original_mean_l2,
        "normalised_mean_l2": normalised_mean_l2,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f)
    print(f"  Saved: {output_path}")


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(config.MODELS.keys()))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--source",
        default=None,
        help="Normalise a per-source file: {model}_{source}_activations.h5 → {model}_{source}_activations_norm.h5",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated list of source names to normalise in one pass.",
    )
    parser.add_argument("--pooling",    default="last_token", help="Pooling strategy (informational; passed through from A2).")
    parser.add_argument("--layer-hook", default="resid_post", help="Hook point (informational; passed through from A2).")
    args = parser.parse_args()

    model = args.model

    # ── Multi-source mode ──
    if args.sources:
        source_list = [s.strip() for s in args.sources.split(",") if s.strip()]
        any_done = False
        for src in source_list:
            safe_src = src.replace("/", "_").replace(" ", "_")
            input_path  = os.path.join(config.ACTIVATIONS_DIR, f"{model}_{safe_src}_activations.h5")
            output_path = os.path.join(config.ACTIVATIONS_DIR, f"{model}_{safe_src}_activations_norm.h5")
            stats_path  = os.path.join(config.ACTIVATIONS_DIR, f"{model}_{safe_src}_norm_stats.json")
            if not os.path.exists(input_path):
                print(f"[skip] {src}: raw activations not found at {input_path}")
                continue
            if os.path.exists(output_path) and os.path.exists(stats_path) and not args.force:
                print(f"[skip] {src}: normalised activations already exist.")
                continue
            print(f"\n[{src}] Normalising {input_path}")
            _normalise_one(model, input_path, output_path, stats_path)
            _maybe_upload(model, src, output_path)
            any_done = True
        log_run("normalise_activations.py", start_time, "success" if any_done else "skipped")
        return 0

    # ── Single-source / full-corpus mode ──
    if args.source:
        safe_source = args.source.replace("/", "_").replace(" ", "_")
        input_path  = os.path.join(config.ACTIVATIONS_DIR, f"{model}_{safe_source}_activations.h5")
        output_path = os.path.join(config.ACTIVATIONS_DIR, f"{model}_{safe_source}_activations_norm.h5")
        stats_path  = os.path.join(config.ACTIVATIONS_DIR, f"{model}_{safe_source}_norm_stats.json")
    else:
        input_path  = os.path.join(config.ACTIVATIONS_DIR, f"{model}_activations.h5")
        output_path = os.path.join(config.ACTIVATIONS_DIR, f"{model}_activations_norm.h5")
        stats_path  = os.path.join(config.ACTIVATIONS_DIR, f"{model}_norm_stats.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing raw activations at {input_path}")

    if os.path.exists(output_path) and os.path.exists(stats_path) and not args.force:
        print("Normalized activations already exist. Use --force to regenerate.")
        log_run("normalise_activations.py", start_time, "skipped")
        return 0

    _normalise_one(model, input_path, output_path, stats_path)
    _maybe_upload(model, args.source or "default", output_path)
    print(f"Saved norm stats to {stats_path}")
    log_run("normalise_activations.py", start_time, "success")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log_run("normalise_activations.py", time.time(), "error", str(e))
        raise
