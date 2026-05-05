#!/usr/bin/env python3
"""
A4b – Auto-Discover SAE features via co-activation clustering.

Algorithm
---------
1. Load SAE + activations for (model, ef).
2. Run SAE forward pass (batched) on up to --max-passages passages to get
   feature activation matrix  F  [n_sam × n_features].
3. Compute per-feature activation variance; discard features whose variance
   is below --min-var (they fire on everything or nothing – not interesting).
4. If the number of surviving features exceeds --max-features, keep the
   --max-features with the highest mean absolute activation.
5. Normalise each surviving feature's activation vector to unit length.
6. Compute the pairwise cosine-similarity matrix → distance = 1 − similarity.
7. Cluster with HDBSCAN(metric="precomputed", min_cluster_size=--min-cluster).
   Falls back to DBSCAN if hdbscan is not installed.
8. Post-process: merge clusters whose centroid cosine similarity exceeds
   --merge-thresh.
9. For each cluster:
     a. Rank passages by sum of feature activations across cluster members.
     b. top-20 = most-activating passages, bottom-20 = least-activating.
     c. Load passage texts from data/corpus.jsonl.
     d. Build CAA vector = mean(raw_hidden[top_idx]) - mean(raw_hidden[bot_idx]).
     e. Optionally name via LLM (prompted with top/bottom passages).
     f. Compute confidence = mean intra-cluster cosine similarity.
10. Write  features/{model}{ef_tag}_autodiscovered.json
    and extend  features/{model}{ef_tag}_feature_labels.json  with
    auto-discovered entries.

Outputs
-------
features/{model}_ef{ef}_autodiscovered.json
    [
      {
        "cluster_id":    int,
        "label":         str,          # LLM-assigned name or "cluster_N"
        "source":        "auto_discovered",
        "domain":        str,          # same as label
        "confidence":    float,        # mean intra-cluster cosine similarity
        "feature_indices": [...],      # SAE feature IDs in this cluster
        "n_features":    int,
        "top_passages":  ["...", ...], # 20 most-activating passage texts
        "bottom_passages": [...],      # 20 least-activating passage texts
        "caa_vector":    [...],        # hidden-dim CAA vector (float32)
      },
      ...
    ]
"""

from __future__ import annotations


import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

import config
from a3_train_sae import TopKSAE


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log_run(script: str, start_time: float, status: str, error: str = "") -> None:
    entry = {
        "script": script,
        "start_time": start_time,
        "end_time": time.time(),
        "status": status,
        "error": error,
    }
    with open("run_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _find_sae_checkpoint(model_name: str, ef: int) -> str:
    """Find the best SAE checkpoint for (model_name, ef)."""
    import re
    sae_dir = config.SAE_DIR
    ef_tag = "" if ef == config.SAE_EXPANSION_FACTOR else f"_ef{ef}"
    candidates = []
    for fname in os.listdir(sae_dir):
        if not fname.endswith(".pt"):
            continue
        if model_name not in fname:
            continue
        expected_prefix = f"{model_name}{ef_tag}_sae_step"
        if not fname.startswith(expected_prefix):
            continue
        m = re.search(r"step(\d+)", fname)
        if m:
            candidates.append((int(m.group(1)), fname))
    if candidates:
        _, fname = max(candidates)
        return os.path.join(sae_dir, fname)
    for fallback in [
        os.path.join(sae_dir, f"{model_name}{ef_tag}_sae.pt"),
        *([] if ef_tag else [os.path.join(sae_dir, f"{model_name}_sae.pt")]),
    ]:
        if os.path.exists(fallback):
            return fallback
    raise FileNotFoundError(
        f"No SAE checkpoint found for {model_name} ef={ef} in {sae_dir}"
    )


def _load_activations(model_name: str):
    """Load (or assemble from per-source files) normalised hidden activations.

    Returns:
        acts: np.ndarray of shape (n_total, hidden_dim)
        source_info: None for single-file loads, or list of (source_tag, n_rows)
                     in the same order as the concatenated rows.
    """
    import glob as _glob
    acts_dir = config.ACTIVATIONS_DIR
    bare = os.path.join(acts_dir, f"{model_name}_activations_norm.h5")
    if os.path.exists(bare):
        with h5py.File(bare, "r") as h5:
            acts = h5["activations"][:]
        return acts, None
    pattern = os.path.join(acts_dir, f"{model_name}_*_activations_norm.h5")
    per_source = sorted(_glob.glob(pattern))
    if not per_source:
        raise FileNotFoundError(
            f"No activations found at {bare} "
            f"or per-source files matching {pattern}"
        )
    chunks = []
    source_info = []
    prefix = model_name + "_"
    suffix = "_activations_norm.h5"
    for p in per_source:
        with h5py.File(p, "r") as h5:
            arr = h5["activations"][:]
        chunks.append(arr)
        fname = os.path.basename(p)
        # Extract source tag: strip "{model_name}_" prefix and "_activations_norm.h5" suffix
        if fname.startswith(prefix) and fname.endswith(suffix):
            source_tag = fname[len(prefix):-len(suffix)]
        else:
            source_tag = fname
        source_info.append((source_tag, arr.shape[0]))
        print(f"[autodiscover] Loaded {fname} ({arr.shape[0]} rows)")
    acts = np.concatenate(chunks, axis=0)
    print(f"[autodiscover] Combined {len(per_source)} files → {acts.shape[0]} rows")
    return acts, source_info


def _load_corpus_texts(needed_indices: set, source_info=None) -> dict:
    """Load passage texts for a set of global corpus indices.

    When source_info is provided (list of (source_tag, n_rows) in concatenation
    order), tries to load texts from per-source JSONL files named
    {source_tag}.jsonl in DATA_DIR or DATA_DIR/datasets/.
    Falls back to scanning a single corpus.jsonl if per-source files aren't found.

    Returns a dict mapping global corpus index → text string.
    """
    data_dir = config.DATA_DIR
    texts: dict = {}

    # ── Try per-source JSONL files ─────────────────────────────────────────
    if source_info:
        any_source_found = False
        global_offset = 0
        for source_tag, n_rows in source_info:
            local_needed = {
                i - global_offset
                for i in needed_indices
                if global_offset <= i < global_offset + n_rows
            }
            if local_needed:
                jsonl_path = None
                for candidate in [
                    os.path.join(data_dir, f"{source_tag}.jsonl"),
                    os.path.join(data_dir, "datasets", f"{source_tag}.jsonl"),
                ]:
                    if os.path.exists(candidate):
                        jsonl_path = candidate
                        break
                if jsonl_path:
                    any_source_found = True
                    with open(jsonl_path, encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if i in local_needed:
                                try:
                                    texts[i + global_offset] = json.loads(line)["text"]
                                except Exception:
                                    texts[i + global_offset] = ""
                else:
                    print(f"[autodiscover] Warning: no JSONL found for source '{source_tag}' "
                          f"(tried {source_tag}.jsonl in {data_dir})")
            global_offset += n_rows
        if any_source_found:
            return texts

    # ── Fallback: single combined corpus.jsonl ─────────────────────────────
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    if not os.path.exists(corpus_path):
        print(f"[autodiscover] Warning: corpus.jsonl not found at {corpus_path}")
        print(f"[autodiscover] Hint: place per-source JSONL files as "
              f"{{source_tag}}.jsonl in {data_dir}/ to enable passage texts.")
        return {}
    with open(corpus_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in needed_indices:
                try:
                    texts[i] = json.loads(line)["text"]
                except Exception:
                    texts[i] = ""
            if len(texts) >= len(needed_indices):
                break
    return texts


# ---------------------------------------------------------------------------
# SAE forward pass to build feature activation matrix
# ---------------------------------------------------------------------------

def _compute_feature_acts(
    acts: np.ndarray,
    model_name: str,
    ef: int,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Run the SAE forward pass on `acts` and return the sparse feature activation
    matrix  F  shaped [n, n_features].  Float32 on CPU.
    """
    ckpt_path = _find_sae_checkpoint(model_name, ef)
    hidden_dim = config.MODELS[model_name]["hidden_dim"]
    n_features = hidden_dim * ef
    sae_topk = config.MODELS[model_name].get("sae_topk", 32)
    # Scale topk proportionally when ef differs from the default
    if ef != config.SAE_EXPANSION_FACTOR:
        base_sparsity = sae_topk / (hidden_dim * config.SAE_EXPANSION_FACTOR)
        sae_topk = max(int(n_features * base_sparsity), 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = TopKSAE(hidden_dim, n_features, sae_topk).to(device)
    sae.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    sae.eval()
    print(f"[autodiscover] SAE loaded ({n_features:,} features, topk={sae_topk}) on {device}")

    n = acts.shape[0]
    F = np.zeros((n, n_features), dtype=np.float32)

    for i in range(0, n, batch_size):
        sl = slice(i, min(i + batch_size, n))
        batch = torch.from_numpy(acts[sl]).float().to(device)
        with torch.no_grad():
            _, sparse = sae(batch)
        F[i : i + sparse.shape[0]] = sparse.float().cpu().numpy()
        bno = i // batch_size + 1
        total = (n + batch_size - 1) // batch_size
        if bno % 10 == 0 or bno == total:
            print(f"[autodiscover] Forward pass batch {bno}/{total}")

    return F


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def _cosine_sim_matrix(V: np.ndarray) -> np.ndarray:
    """Compute n×n cosine similarity matrix for rows of V (float32)."""
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0.0] = 1e-9
    V_norm = V / norms
    return (V_norm @ V_norm.T).clip(-1.0, 1.0)


def _hdbscan_cluster(dist_matrix: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """
    Cluster features using HDBSCAN on a precomputed distance matrix.
    Falls back to DBSCAN if hdbscan is not available.
    """
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="precomputed",
            cluster_selection_epsilon=0.0,
        )
        labels = clusterer.fit_predict(dist_matrix.astype(np.float64))
        print(f"[autodiscover] HDBSCAN: {len(set(labels)) - (1 if -1 in labels else 0)} clusters "
              f"({np.sum(labels == -1)} noise points)")
        return labels
    except ImportError:
        print("[autodiscover] hdbscan not installed — falling back to DBSCAN. "
              "Install with: pip install hdbscan")
        from sklearn.cluster import DBSCAN
        eps_candidate = float(np.percentile(dist_matrix, 10))
        eps_candidate = max(0.05, min(0.5, eps_candidate))
        clusterer = DBSCAN(
            eps=eps_candidate,
            min_samples=min_cluster_size,
            metric="precomputed",
        )
        labels = clusterer.fit_predict(dist_matrix)
        print(f"[autodiscover] DBSCAN: {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
        return labels


def _merge_similar_clusters(
    labels: np.ndarray,
    centroid_sims: np.ndarray,
    merge_threshold: float,
) -> np.ndarray:
    """
    merge_threshold > 0: merge cluster pairs whose centroid cosine similarity
    exceeds the threshold (greedy single-linkage).
    """
    unique = [c for c in sorted(set(labels)) if c != -1]
    if len(unique) < 2:
        return labels
    # Union-find
    parent = {c: c for c in unique}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, ci in enumerate(unique):
        for j, cj in enumerate(unique):
            if j <= i:
                continue
            if centroid_sims[i, j] >= merge_threshold:
                union(ci, cj)

    remap = {}
    for c in unique:
        root = find(c)
        if root not in remap:
            remap[root] = len(remap)
        remap[c] = remap[root]

    new_labels = np.where(labels == -1, -1, np.vectorize(remap.get)(labels, -1))
    n_before = len(unique)
    n_after = len(set(remap.values()))
    if n_after < n_before:
        print(f"[autodiscover] Merged {n_before} → {n_after} clusters "
              f"(threshold={merge_threshold:.2f})")
    return new_labels


# ---------------------------------------------------------------------------
# LLM naming: handled browser-side via Puter.js (puter.ai.chat)
# ---------------------------------------------------------------------------
# Cluster naming is intentionally NOT done in Python.  The browser reads the
# autodiscovered.json output, calls puter.ai.chat() for each pending cluster,
# and POSTs the resulting names back via /api/pipeline/auto_discover/apply_names.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="A4b: auto-discover SAE features via co-activation clustering"
    )
    parser.add_argument("--model", required=True,
                        help="Model name (must match config.MODELS key)")
    parser.add_argument("--ef", type=int, default=None,
                        help="SAE expansion factor (default: model's sae_ef or config.SAE_EXPANSION_FACTOR)")
    parser.add_argument("--min-var", type=float, default=0.01,
                        dest="min_var",
                        help="Minimum per-feature activation variance to keep (default 0.01)")
    parser.add_argument("--min-cluster", type=int, default=5,
                        dest="min_cluster",
                        help="HDBSCAN/DBSCAN min_cluster_size (default 5)")
    parser.add_argument("--merge-thresh", type=float, default=0.85,
                        dest="merge_thresh",
                        help="Centroid cosine similarity threshold for merging clusters (default 0.85)")
    parser.add_argument("--llm", type=str, default="skip",
                        help="LLM model ID for cluster naming, or 'skip' to disable (default skip)")
    parser.add_argument("--max-passages", type=int, default=10_000,
                        dest="max_passages",
                        help="Max number of passages used for feature extraction (default 10000)")
    parser.add_argument("--max-features", type=int, default=8_000,
                        dest="max_features",
                        help="Max active features to cluster (keeps top-N by mean activation, default 8000)")
    parser.add_argument("--top-k-passages", type=int, default=20,
                        dest="top_k",
                        help="Top/bottom passage count per cluster (default 20)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    model_name = args.model
    if model_name not in config.MODELS:
        print(f"[autodiscover] ERROR: Unknown model '{model_name}'. "
              f"Available: {sorted(config.MODELS.keys())}")
        sys.exit(1)

    model_cfg = config.MODELS[model_name]
    ef: int = args.ef or model_cfg.get("sae_ef") or config.SAE_EXPANSION_FACTOR
    ef_tag = "" if ef == config.SAE_EXPANSION_FACTOR else f"_ef{ef}"

    feat_dir = config.FEATURES_DIR
    os.makedirs(feat_dir, exist_ok=True)

    out_path = os.path.join(feat_dir, f"{model_name}{ef_tag}_autodiscovered.json")
    labels_path = os.path.join(feat_dir, f"{model_name}{ef_tag}_feature_labels.json")

    if os.path.exists(out_path) and not args.force:
        print(f"[autodiscover] Output already exists at {out_path} "
              f"(use --force to overwrite). Exiting.")
        return

    _set_seed(42)

    # ── 1. Load activations ────────────────────────────────────────────────
    print(f"[autodiscover] Loading activations for {model_name} …")
    acts, source_info = _load_activations(model_name)
    n_total = acts.shape[0]
    print(f"[autodiscover] Loaded {n_total:,} passages, hidden_dim={acts.shape[1]}")

    # Subsample if needed
    if n_total > args.max_passages:
        rng = np.random.default_rng(42)
        sample_idx = np.sort(
            rng.choice(n_total, size=args.max_passages, replace=False)
        )
        acts_sample = acts[sample_idx]
        print(f"[autodiscover] Subsampled to {args.max_passages:,} passages")
    else:
        sample_idx = np.arange(n_total)
        acts_sample = acts

    # ── 2. SAE forward pass ────────────────────────────────────────────────
    print(f"[autodiscover] Running SAE forward pass (ef={ef}) …")
    F = _compute_feature_acts(acts_sample, model_name, ef)
    n_sam, n_features = F.shape
    print(f"[autodiscover] Feature matrix: {n_sam:,} × {n_features:,}")

    # ── 3. Activity variance filter ────────────────────────────────────────
    feat_var = F.var(axis=0)
    active_mask = feat_var >= args.min_var
    active_indices = np.where(active_mask)[0]
    print(f"[autodiscover] Features passing variance filter ≥{args.min_var}: "
          f"{active_mask.sum():,} / {n_features:,}")

    if active_mask.sum() == 0:
        print("[autodiscover] No features passed the variance filter. "
              "Try lowering --min-var.")
        _log_run("auto_discover.py", start_time, "error",
                 "no features passed variance filter")
        sys.exit(1)

    # ── 4. Reduce to top-N by mean activation if too many ─────────────────
    F_active = F[:, active_indices]
    if F_active.shape[1] > args.max_features:
        mean_act = F_active.mean(axis=0)
        top_local = np.argpartition(mean_act, -args.max_features)[-args.max_features:]
        top_local = np.sort(top_local)
        active_indices = active_indices[top_local]
        F_active = F[:, top_local]
        print(f"[autodiscover] Reduced to top-{args.max_features:,} active features "
              f"by mean activation")

    n_active = F_active.shape[1]
    print(f"[autodiscover] Clustering {n_active:,} features …")

    # ── 5. Cosine similarity + HDBSCAN ────────────────────────────────────
    # Cluster features (rows of F_active.T) → each feature is a n_sam-dim vector
    sim_matrix = _cosine_sim_matrix(F_active.T.astype(np.float32))   # [n_active, n_active]
    dist_matrix = (1.0 - sim_matrix).clip(0.0, 2.0).astype(np.float64)

    cluster_labels = _hdbscan_cluster(dist_matrix, args.min_cluster)

    # ── 6. Merge similar clusters ─────────────────────────────────────────
    unique_clusters = sorted(set(cluster_labels) - {-1})
    if args.merge_thresh < 1.0 and len(unique_clusters) > 1:
        # Build centroids (mean normalised feature vec per cluster)
        centroids = np.zeros((len(unique_clusters), n_sam), dtype=np.float32)
        for i, cid in enumerate(unique_clusters):
            members = np.where(cluster_labels == cid)[0]
            centroids[i] = F_active.T[members].mean(axis=0)
        centroid_sim = _cosine_sim_matrix(centroids)
        cluster_labels = _merge_similar_clusters(
            cluster_labels, centroid_sim, args.merge_thresh
        )
        unique_clusters = sorted(set(cluster_labels) - {-1})

    print(f"[autodiscover] Final cluster count: {len(unique_clusters)}")

    if len(unique_clusters) == 0:
        print("[autodiscover] No clusters found. The variance filter or cluster "
              "size may be too strict. Try --min-var 0.001 or --min-cluster 3.")
        _log_run("auto_discover.py", start_time, "error", "no clusters found")
        sys.exit(1)

    # ── 7. Load corpus texts ───────────────────────────────────────────────
    print("[autodiscover] Loading corpus passage texts …")
    needed_indices = set(sample_idx.tolist())
    corpus_texts = _load_corpus_texts(needed_indices, source_info=source_info)
    print(f"[autodiscover] Loaded {len(corpus_texts):,} passage texts")

    # Map sample positions to texts via original corpus line index
    def _get_text(sample_pos: int) -> str:
        orig_idx = int(sample_idx[sample_pos])
        return corpus_texts.get(orig_idx, "")

    # ── 8. Build sample→source lookup ──────────────────────────────────────
    sample_source: list = []
    if source_info:
        breakpoints = []
        offset = 0
        for tag, nrows in source_info:
            breakpoints.append((offset, offset + nrows, tag))
            offset += nrows
        for pos in range(n_sam):
            orig = int(sample_idx[pos])
            src = "unknown"
            for start, end, stag in breakpoints:
                if start <= orig < end:
                    src = stag
                    break
            sample_source.append(src)

    # ── 9. Build cluster records ───────────────────────────────────────────
    results: List[dict] = []
    used_labels: set = set()

    for cid in unique_clusters:
        members_local = np.where(cluster_labels == cid)[0]         # indices in F_active
        members_global = active_indices[members_local].tolist()     # original feature IDs

        # Per-cluster activation score per passage = mean activation across members
        cluster_acts = F_active[:, members_local].mean(axis=1)     # [n_sam]

        top_k = min(args.top_k, n_sam // 2)
        if top_k < 1:
            top_k = 1

        top_idx = np.argpartition(cluster_acts, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(cluster_acts[top_idx])[::-1]].tolist()

        bottom_idx = np.argpartition(cluster_acts, top_k)[:top_k]
        bottom_idx = bottom_idx[np.argsort(cluster_acts[bottom_idx])].tolist()

        top_texts = [_get_text(i) for i in top_idx]
        bot_texts = [_get_text(i) for i in bottom_idx]

        # Source distribution over top passages
        if sample_source:
            from collections import Counter
            top_srcs = Counter(sample_source[i] for i in top_idx)
            total_top = sum(top_srcs.values()) or 1
            source_dist = {k: round(v / total_top, 3)
                           for k, v in top_srcs.most_common()}
        else:
            source_dist = {}

        # Intra-cluster confidence = mean pairwise cosine similarity
        if len(members_local) > 1:
            sub_sim = sim_matrix[np.ix_(members_local, members_local)]
            n_m = len(members_local)
            confidence = float(
                (sub_sim.sum() - n_m) / max(n_m * (n_m - 1), 1)
            )
        else:
            confidence = 1.0

        # CAA vector: mean(raw_hidden[top_idx]) - mean(raw_hidden[bottom_idx])
        top_acts_raw = np.stack([acts_sample[i] for i in top_idx]).mean(axis=0)
        bot_acts_raw = np.stack([acts_sample[i] for i in bottom_idx]).mean(axis=0)
        caa_vector = (top_acts_raw - bot_acts_raw).tolist()

        # Labels start as cluster_N; browser renames via Puter.js after job completes
        label = f"cluster_{cid}"
        used_labels.add(label)

        results.append({
            "cluster_id":        int(cid),
            "label":             label,
            "source":            "auto_discovered",
            "domain":            label,
            "confidence":        round(confidence, 4),
            "feature_indices":   members_global,
            "n_features":        len(members_global),
            "top_passages":      [t[:500] for t in top_texts if t],
            "bottom_passages":   [t[:500] for t in bot_texts if t],
            "source_distribution": source_dist,
            "llm_prompt":        "",
            "llm_response":      "",
            "llm_naming_pending": args.llm != "skip",
            "caa_vector":        [round(v, 6) for v in caa_vector],
        })

        print(f"[autodiscover] Cluster {cid}: label='{label}' "
              f"({len(members_global)} features, confidence={confidence:.3f})")

    # ── 10. Write autodiscovered.json ──────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[autodiscover] Saved {len(results)} clusters → {out_path}")
    # Also write a root-level copy so local server can pull it even if
    # features/ is a symlink on the worker (path-traversal check blocks symlinks).
    root_copy = os.path.join(os.path.dirname(feat_dir),
                             f"{model_name}{ef_tag}_autodiscovered.json")
    try:
        import shutil as _shutil
        _shutil.copy2(out_path, root_copy)
    except Exception as _e:
        print(f"[autodiscover] Warning: could not write root copy: {_e}")

    # ── 11. Extend feature_labels.json ────────────────────────────────────
    existing: dict = {"features": [], "domain_counts": {}}
    if os.path.exists(labels_path):
        try:
            existing = json.load(open(labels_path))
        except Exception:
            pass

    # Remove any previous auto-discovered entries
    existing_feats = [
        f for f in existing.get("features", [])
        if f.get("source") != "auto_discovered"
    ]

    # Build new auto-discovered feature entries (one per feature in each cluster)
    new_entries = []
    for cluster in results:
        for fid in cluster["feature_indices"]:
            new_entries.append({
                "feature_id":     fid,
                "domain":         cluster["label"],
                "label":          cluster["label"],
                "source":         "auto_discovered",
                "confidence":     cluster["confidence"],
                "cluster_id":     cluster["cluster_id"],
            })

    existing["features"] = existing_feats + new_entries

    # Recompute domain_counts
    domain_counts: dict = {}
    for feat in existing["features"]:
        d = feat.get("domain", "other")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    existing["domain_counts"] = domain_counts

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"[autodiscover] Extended feature labels → {labels_path} "
          f"(+{len(new_entries)} entries)")

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    summary = {
        "model":          model_name,
        "ef":             ef,
        "n_passages":     n_sam,
        "n_features_total": n_features,
        "n_features_active": n_active,
        "n_clusters":     len(results),
        "min_var":        args.min_var,
        "min_cluster":    args.min_cluster,
        "merge_thresh":   args.merge_thresh,
        "llm":            args.llm,
        "elapsed_s":      round(elapsed, 1),
        "output":         out_path,
    }
    print(f"\n[autodiscover] Done in {elapsed:.1f}s — {len(results)} clusters discovered\n"
          f"  {json.dumps(summary, indent=2)}")

    summary_path = os.path.join(feat_dir, f"{model_name}{ef_tag}_autodiscovered_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _log_run("auto_discover.py", start_time, "done")


if __name__ == "__main__":
    main()
