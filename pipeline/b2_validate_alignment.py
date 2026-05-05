
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import glob
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from scipy import stats as scipy_stats
from scipy.linalg import orthogonal_procrustes as _orthogonal_procrustes

import config

_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# Reject auto-generated concept labels (concept_1, cluster42, unknown_3…)
# so that unsupervised A4b discoveries never pollute B2 domain scoring.
_UNNAMED_CONCEPT_RE = re.compile(
    r'^(concept|cluster|topic|unknown)[_\s]*\d',
    re.IGNORECASE,
)


def _is_named_concept(label: str) -> bool:
    if not label:
        return False
    return not _UNNAMED_CONCEPT_RE.match(label)


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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (r, p-value)."""
    if x.std() == 0 or y.std() == 0:
        return 0.0, 1.0
    r, p = scipy_stats.pearsonr(x, y)
    return float(r), float(p)


def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (rho, p-value)."""
    if x.std() == 0 or y.std() == 0:
        return 0.0, 1.0
    r, p = scipy_stats.spearmanr(x, y)
    return float(r), float(p)


def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient rho_c."""
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = x.mean(), y.mean()
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    sxy = np.cov(x, y)[0, 1]
    denom = sx2 + sy2 + (mx - my) ** 2
    if denom == 0:
        return 1.0
    return float(2.0 * sxy / denom)


def _shuffle_null(x: np.ndarray, y: np.ndarray, n_perm: int = 500, seed: int = 0) -> float:
    """Two-sided permutation p-value for Pearson r."""
    rng = np.random.default_rng(seed)
    obs_r = np.corrcoef(x, y)[0, 1] if (x.std() > 0 and y.std() > 0) else 0.0
    count = 0
    y_perm = y.copy()
    for _ in range(n_perm):
        rng.shuffle(y_perm)
        r = np.corrcoef(x, y_perm)[0, 1] if (x.std() > 0 and y_perm.std() > 0) else 0.0
        if abs(r) >= abs(obs_r):
            count += 1
    return float(count) / n_perm


def _bh_correction(pvalues: List[float], alpha: float = 0.05) -> List[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvalues)
    if n == 0:
        return []
    order = np.argsort(pvalues)
    pvals = np.array(pvalues)
    adjusted = np.zeros(n)
    for i, idx in enumerate(order):
        adjusted[idx] = pvals[idx] * n / (i + 1)
    # Enforce monotonicity from the end
    for i in range(n - 2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i + 1]])
    return np.clip(adjusted, 0.0, 1.0).tolist()


def _coactivation_rate(
    a_all: np.ndarray,
    b_all: np.ndarray,
    percentile: int = 90,
) -> float:
    """Co-activation rate rho_c at the given percentile threshold.

    Computes the fraction of corpus passages where BOTH features activate
    above their respective per-feature P{percentile} thresholds simultaneously.
    rho_c is a property of a FEATURE PAIR, not of a concept in isolation.

    Args:
        a_all: activations for feature a across ALL passages (1-D array)
        b_all: activations for feature b across ALL passages (1-D array)
        percentile: detection threshold percentile of non-zero activations (e.g. 90)

    Returns:
        float in [0, 1] — fraction of passages where both features are active.
    """
    nonzero_a = a_all[a_all > 0]
    nonzero_b = b_all[b_all > 0]
    thresh_a = float(np.percentile(nonzero_a, percentile)) if len(nonzero_a) > 0 else 0.0
    thresh_b = float(np.percentile(nonzero_b, percentile)) if len(nonzero_b) > 0 else 0.0
    n = min(len(a_all), len(b_all))
    if n == 0:
        return 0.0
    both_active = (a_all[:n] > thresh_a) & (b_all[:n] > thresh_b)
    return float(both_active.mean())


def _coactivation_rate_clustered(
    decoder_cols_a: np.ndarray,
    decoder_cols_b: np.ndarray,
    all_acts_a: np.ndarray,
    all_acts_b: np.ndarray,
    percentile: int = 90,
) -> float:
    """Cluster-pooled co-activation rate.

    A passage is 'active' for a cluster if ANY member feature exceeds the P90
    threshold. This accounts for feature splitting across models.

    Args:
        decoder_cols_a: (n_passages, n_cluster_features_a) activation matrix for
                        all features in model A's cluster
        decoder_cols_b: (n_passages, n_cluster_features_b) activation matrix for
                        all features in model B's cluster
        all_acts_a: same as decoder_cols_a (alias for clarity)
        all_acts_b: same as decoder_cols_b (alias for clarity)
        percentile: detection threshold percentile of non-zero activations

    Returns:
        float in [0, 1]
    """
    # Pool across cluster: max activation per passage
    pool_a = decoder_cols_a.max(axis=1) if decoder_cols_a.ndim == 2 else decoder_cols_a
    pool_b = decoder_cols_b.max(axis=1) if decoder_cols_b.ndim == 2 else decoder_cols_b
    return _coactivation_rate(pool_a, pool_b, percentile=percentile)


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d between two samples."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def _mutual_info_bins(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """Mutual information via equal-width histogram binning."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if x.std() < 1e-10 or y.std() < 1e-10:
        return 0.0
    hist, _, _ = np.histogram2d(x, y, bins=n_bins)
    pxy = hist / hist.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    mask = pxy > 0
    mi = float(np.sum(pxy[mask] * np.log(pxy[mask] / (px * py)[mask])))
    return max(mi, 0.0)


def _cca_weights(x: np.ndarray, y: np.ndarray, n_components: int = 64,
                 device: Optional[torch.device] = None):
    """CCA projection weights via whitened cross-covariance SVD (GPU-accelerated).
    Returns x_w (d_x × k) and y_w (d_y × k): feature-to-CCA-direction weights."""
    dev = device if device is not None else _DEVICE
    dtype = torch.float64  # keep float64 for numerical stability
    X = torch.from_numpy(x.astype(np.float64)).to(dtype).to(dev)
    Y = torch.from_numpy(y.astype(np.float64)).to(dtype).to(dev)
    n = X.shape[0]
    reg = 1e-6
    Cxx = (X.T @ X) / (n - 1) + torch.eye(X.shape[1], dtype=dtype, device=dev) * reg
    Cyy = (Y.T @ Y) / (n - 1) + torch.eye(Y.shape[1], dtype=dtype, device=dev) * reg
    Cxy = (X.T @ Y) / (n - 1)
    ex, Vx = torch.linalg.eigh(Cxx)
    ey, Vy = torch.linalg.eigh(Cyy)
    ex = ex.clamp(min=1e-8)
    ey = ey.clamp(min=1e-8)
    Cxx_isqrt = Vx @ torch.diag(ex ** -0.5) @ Vx.T
    Cyy_isqrt = Vy @ torch.diag(ey ** -0.5) @ Vy.T
    K = Cxx_isqrt @ Cxy @ Cyy_isqrt
    U, S, Vh = torch.linalg.svd(K, full_matrices=False)
    k = min(n_components, len(S))
    x_w = (Cxx_isqrt @ U[:, :k]).cpu().numpy()   # (d_x, k)
    y_w = (Cyy_isqrt @ Vh[:k].T).cpu().numpy()   # (d_y, k)
    return x_w, y_w


def _batch_shuffle_null_gpu(
    all_a: np.ndarray,   # (N_pairs, n_samples)
    all_b: np.ndarray,   # (N_pairs, n_samples)
    n_perm: int = 1000,
    seed: int = 0,
    chunk: int = 50,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Vectorised permutation p-values for all pairs simultaneously on GPU.
    For each pair i, shuffles all_b[i] n_perm times and counts |r| >= |obs_r|.
    Returns p-values array of shape (N_pairs,)."""
    dev = device if device is not None else _DEVICE
    A = torch.from_numpy(all_a).float().to(dev)   # (N, S)
    B = torch.from_numpy(all_b).float().to(dev)   # (N, S)

    def _pearson_3d(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """X, Y: (C, N, S) -> (C, N) abs Pearson r."""
        Xm = X - X.mean(dim=2, keepdim=True)
        Ym = Y - Y.mean(dim=2, keepdim=True)
        num = (Xm * Ym).sum(dim=2)
        denom = Xm.norm(dim=2) * Ym.norm(dim=2) + 1e-10
        return (num / denom).abs()  # (C, N)

    # Observed |r| for each pair: (N,)
    obs_r = _pearson_3d(A.unsqueeze(0), B.unsqueeze(0)).squeeze(0)  # (N,)

    N, S = A.shape
    counts = torch.zeros(N, device=dev)
    rng = np.random.default_rng(seed)

    dispatched = 0
    while dispatched < n_perm:
        c = min(chunk, n_perm - dispatched)
        # Build (c, N, S) random permutation indices on CPU, transfer to GPU
        rand_np = rng.uniform(size=(c, N, S)).astype(np.float32)
        perm_idx = torch.from_numpy(
            np.argsort(rand_np, axis=2).astype(np.int64)
        ).to(dev)                                         # (c, N, S)
        B_exp = B.unsqueeze(0).expand(c, -1, -1)          # (c, N, S)
        B_perm = B_exp.gather(2, perm_idx)                # (c, N, S)
        A_exp = A.unsqueeze(0).expand(c, -1, -1)          # (c, N, S)
        r_perm = _pearson_3d(A_exp, B_perm)               # (c, N)
        counts += (r_perm >= obs_r.unsqueeze(0)).float().sum(dim=0)
        dispatched += c

    return (counts / n_perm).cpu().numpy()


def _multi_gpu_perm_test(
    all_a: np.ndarray, all_b: np.ndarray, n_perm: int, seed: int = 0, chunk: int = 100
) -> np.ndarray:
    """Split pairs across all available GPUs and run permutation tests in parallel threads."""
    if _NUM_GPUS <= 1:
        return _batch_shuffle_null_gpu(all_a, all_b, n_perm=n_perm, seed=seed,
                                       chunk=chunk, device=_DEVICE)
    N = all_a.shape[0]
    splits = np.array_split(np.arange(N), _NUM_GPUS)
    results: List[Optional[np.ndarray]] = [None] * _NUM_GPUS

    def _run(gpu_idx: int, indices: np.ndarray) -> None:
        if len(indices) == 0:
            results[gpu_idx] = np.array([], dtype=np.float32)
            return
        dev = torch.device(f"cuda:{gpu_idx}")
        # Derive seed from the first pair index in this shard so results are
        # identical regardless of how many GPUs are used.
        shard_seed = seed + int(indices[0])
        results[gpu_idx] = _batch_shuffle_null_gpu(
            all_a[indices], all_b[indices],
            n_perm=n_perm, seed=shard_seed, chunk=chunk, device=dev,
        )

    threads = [threading.Thread(target=_run, args=(i, splits[i]))
               for i in range(_NUM_GPUS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Reassemble in original order
    out = np.empty(N, dtype=np.float32)
    for i, indices in enumerate(splits):
        if len(indices) > 0:
            out[indices] = results[i]
    return out


def _load_raw_activations(model_name: str) -> np.ndarray:
    """Load raw (pre-SAE) normalised hidden states for model_name.
    Uses pre-allocated output to avoid doubling peak RAM during concatenation."""
    plain_path = os.path.join(config.ACTIVATIONS_DIR, f"{model_name}_activations_norm.h5")
    if os.path.exists(plain_path):
        with h5py.File(plain_path, "r") as h5:
            return h5["activations"][:].astype(np.float32)
    pattern = os.path.join(config.ACTIVATIONS_DIR, f"{model_name}_*_activations_norm.h5")
    source_files = sorted(glob.glob(pattern))
    if not source_files:
        raise FileNotFoundError(
            f"No raw activations found for {model_name} in {config.ACTIVATIONS_DIR}"
        )
    print(f"[step6/sae-free] Aggregating {len(source_files)} source files for {model_name}")
    # Two-pass: first gather shapes, then pre-allocate and copy chunk-by-chunk to
    # avoid doubling peak RAM (np.concatenate holds all chunks + output simultaneously).
    sizes, dim = [], None
    for sf in source_files:
        with h5py.File(sf, "r") as h5:
            sh = h5["activations"].shape
            sizes.append(sh[0])
            dim = sh[1]
    out = np.empty((sum(sizes), dim), dtype=np.float32)
    offset = 0
    for sf, n in zip(source_files, sizes):
        with h5py.File(sf, "r") as h5:
            chunk = h5["activations"][:].astype(np.float32)
        out[offset:offset + n] = chunk
        del chunk
        offset += n
    return out


def _load_raw_activations_subsample(model_name: str, n_rows: int,
                                    rng: np.random.Generator) -> np.ndarray:
    """Load only n_rows random rows of raw activations (for CCA when probes are cached).
    Reads only the needed rows from H5 — avoids loading the full matrix."""
    plain_path = os.path.join(config.ACTIVATIONS_DIR, f"{model_name}_activations_norm.h5")
    if os.path.exists(plain_path):
        with h5py.File(plain_path, "r") as h5:
            total = h5["activations"].shape[0]
            idx = np.sort(rng.choice(total, size=min(n_rows, total), replace=False))
            return h5["activations"][idx].astype(np.float32)
    pattern = os.path.join(config.ACTIVATIONS_DIR, f"{model_name}_*_activations_norm.h5")
    source_files = sorted(glob.glob(pattern))
    if not source_files:
        raise FileNotFoundError(
            f"No raw activations found for {model_name} in {config.ACTIVATIONS_DIR}"
        )
    sizes, dim = [], None
    for sf in source_files:
        with h5py.File(sf, "r") as h5:
            sh = h5["activations"].shape
            sizes.append(sh[0])
            dim = sh[1]
    total = sum(sizes)
    idx = np.sort(rng.choice(total, size=min(n_rows, total), replace=False))
    parts = []
    offset = 0
    for sf, n in zip(source_files, sizes):
        file_idx = idx[(idx >= offset) & (idx < offset + n)] - offset
        if len(file_idx) > 0:
            with h5py.File(sf, "r") as h5:
                parts.append(h5["activations"][file_idx].astype(np.float32))
        offset += n
    return np.concatenate(parts, axis=0) if parts else np.empty((0, dim), dtype=np.float32)


def _train_linear_probe(
    acts: np.ndarray, pos_idx: np.ndarray, neg_idx: np.ndarray,
    device: torch.device = _DEVICE,
) -> Optional[np.ndarray]:
    """Fit a ridge logistic regression probe on pos vs neg passages (GPU).
    Returns the weight vector (hidden_dim,), or None if too few samples."""
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos < 5 or n_neg < 5:
        return None
    # Subsample to at most 5000 each to keep training fast
    rng = np.random.default_rng(42)
    p_idx = pos_idx[rng.choice(n_pos, size=min(n_pos, 5000), replace=False)]
    n_idx = neg_idx[rng.choice(n_neg, size=min(n_neg, 5000), replace=False)]
    # Clip indices to corpus size
    max_idx = acts.shape[0] - 1
    p_idx = p_idx[p_idx <= max_idx]
    n_idx = n_idx[n_idx <= max_idx]
    if len(p_idx) < 2 or len(n_idx) < 2:
        return None
    # Build data on GPU
    X = torch.from_numpy(
        np.concatenate([acts[p_idx], acts[n_idx]], axis=0).astype(np.float32)
    ).to(device)  # (n_samples, d)
    y = torch.cat([
        torch.ones(len(p_idx), dtype=torch.float32, device=device),
        torch.zeros(len(n_idx), dtype=torch.float32, device=device),
    ])
    # Standardise features on GPU
    mu = X.mean(dim=0)
    sig = X.std(dim=0) + 1e-8
    X = (X - mu) / sig
    # Closed-form ridge: w = (X^T X + λI)^{-1} X^T y  (λ=0.01 for stability)
    lam = 0.01
    n, d = X.shape
    if n < d:
        # Kernel trick: (X X^T + λI)^{-1}
        K = X @ X.T + lam * torch.eye(n, dtype=X.dtype, device=device)
        alpha = torch.linalg.solve(K, y)
        w = X.T @ alpha
    else:
        A = X.T @ X + lam * torch.eye(d, dtype=X.dtype, device=device)
        w = torch.linalg.solve(A, X.T @ y)
    norm = w.norm()
    if norm < 1e-10:
        return None
    return (w / norm).cpu().numpy().astype(np.float64)


def _rsa_score(mat_a: np.ndarray, mat_b: np.ndarray, sample: np.ndarray) -> float:
    """Representational Similarity Analysis: Spearman r between RDMs."""
    a = mat_a[sample]
    b = mat_b[sample]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    rdm_a = 1.0 - a_norm @ a_norm.T
    rdm_b = 1.0 - b_norm @ b_norm.T
    idx_upper = np.triu_indices(len(sample), k=1)
    va = rdm_a[idx_upper]
    vb = rdm_b[idx_upper]
    if va.std() == 0 or vb.std() == 0:
        return 0.0
    r, _ = scipy_stats.spearmanr(va, vb)
    return float(r)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_labels() -> Dict[int, Dict[str, int]]:
    labels_path = os.path.join(config.DATA_DIR, "corpus_labels.jsonl")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels file at {labels_path}")
    labels = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            labels[i] = json.loads(line)
    return labels


def _load_feature_matrix(model_name: str, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    # Try plain name first (legacy), then any EF-tagged variant
    idx_path = os.path.join(config.FEATURES_DIR, f"{model_name}_top{topk}_feature_idx.npy")
    mat_path = os.path.join(config.FEATURES_DIR, f"{model_name}_top{topk}_feature_acts.npy")
    if not os.path.exists(idx_path) or not os.path.exists(mat_path):
        # Search for EF-tagged files: {model_name}_ef*_top*_feature_acts.npy
        candidates = sorted(glob.glob(
            os.path.join(config.FEATURES_DIR, f"{model_name}_*_top*_feature_acts.npy")
        ))
        if not candidates:
            raise FileNotFoundError(f"Missing feature matrix for {model_name} (run Step 5 first)")
        mat_path = candidates[-1]  # pick most recent
        idx_path = mat_path.replace("_feature_acts.npy", "_feature_idx.npy")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Missing feature index for {model_name}: {idx_path}")
    idx = np.load(idx_path)
    mat = np.load(mat_path)
    return idx, mat


def _sample_indices(pos_idx: np.ndarray, neg_idx: np.ndarray, n: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    pos = rng.choice(pos_idx, size=min(n, len(pos_idx)), replace=False)
    neg = rng.choice(neg_idx, size=min(n, len(neg_idx)), replace=False)
    return np.concatenate([pos, neg]), pos, neg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aligned-path", default="",
                        help="Path to aligned_pairs.jsonl (default: alignment/aligned_pairs.jsonl)")
    parser.add_argument("--output-path", default="",
                        help="Path to write validation_results.jsonl (default: alignment/validation_results.jsonl)")
    parser.add_argument("--pair-filter", default="",
                        help="Restrict to specific model pairs e.g. 'gpt2-llama,gemma-llama'")
    parser.add_argument("--pearson-thresh", type=float, default=0.5,
                        help="Minimum |Pearson r| for validation_pass (default: 0.5)")
    parser.add_argument("--n-perms", type=int, default=200,
                        help="Permutations for shuffle null test (default: 200, 0=skip)")
    parser.add_argument("--rsa", action="store_true", help="Compute RSA score per pair (slower)")
    parser.add_argument("--deconfound-k", type=int, default=0, dest="deconfound_k",
                        help="Remove top-k PCA components from activations before validation (0=skip)")
    parser.add_argument("--sae-free", action="store_true", dest="sae_free",
                        help="Train linear probes on raw hidden states; compute saefree_procrustes_cosine per pair")
    parser.add_argument("--info-theory", action="store_true", dest="info_theory",
                        help="Compute mutual information between feature pair activation profiles")
    parser.add_argument("--random-init-baseline", action="store_true", dest="random_init_baseline",
                        help="Compute shuffled-activation baseline for threshold_status calibration")
    args = parser.parse_args()

    set_seed(42)
    print(f"[step6] Device: {_DEVICE}" + (
        f" ({torch.cuda.get_device_name(0)})" if _DEVICE.type == "cuda" else ""
    ))
    aligned_path = args.aligned_path or os.path.join(config.ALIGNMENT_DIR, "aligned_pairs.jsonl")
    results_path = args.output_path or os.path.join(config.ALIGNMENT_DIR, "validation_results.jsonl")

    if os.path.exists(results_path) and not args.force:
        print(f"{os.path.basename(results_path)} exists. Use --force to recompute.")
        log_run("step6_validate_alignment.py", start_time, "skipped")
        return 0

    pairs = [json.loads(l) for l in open(aligned_path, "r", encoding="utf-8") if l.strip()]
    if not pairs:
        raise ValueError(f"{aligned_path} is empty")

    # Auto-detect models from aligned_pairs (no hardcoded list)
    model_names_in_pairs: set = set()
    for p in pairs:
        model_names_in_pairs.add(p["a_model"])
        model_names_in_pairs.add(p["b_model"])

    # Apply pair filter
    pair_filter_set: set = set()
    if args.pair_filter:
        for item in args.pair_filter.split(","):
            item = item.strip()
            # Match against known model names — do NOT use split("-",1) because
            # model names contain dashes (e.g. gpt2-large, deepseek-llm-7b).
            for _cma in config.MODELS:
                if item.startswith(_cma + "-"):
                    _cmb = item[len(_cma) + 1:]
                    if _cmb in config.MODELS:
                        pair_filter_set.add((_cma, _cmb))
                        break
    if pair_filter_set:
        pairs = [p for p in pairs if (p["a_model"], p["b_model"]) in pair_filter_set
                 or (p["b_model"], p["a_model"]) in pair_filter_set]

    labels = _load_labels()

    # ── Build pos/neg index arrays for ALL corpus domains dynamically ─────────
    # corpus_labels.jsonl contains 15+ domain keys (math_gsm8k, code_python, etc.)
    # Pos = passages where that domain value > 0.5 (from that domain)
    # Neg = passages where that domain value is 0 or absent
    _SKIP_KEYS = {"id", "source"}
    _all_corpus_domain_keys: set = set()
    for _row in labels.values():
        _all_corpus_domain_keys.update(k for k in _row.keys() if k not in _SKIP_KEYS)

    _CORPUS_DOMAIN_POS_NEG: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for _dom in sorted(_all_corpus_domain_keys):
        _pos = np.array([i for i, row in labels.items() if (row.get(_dom) or 0) > 0.5], dtype=np.int64)
        _neg = np.array([i for i, row in labels.items() if (row.get(_dom) or 0) <= 0.5], dtype=np.int64)
        if len(_pos) >= 10 and len(_neg) >= 10:
            _CORPUS_DOMAIN_POS_NEG[_dom] = (_pos, _neg)
    print(f"[step6] Corpus domains loaded: {sorted(_CORPUS_DOMAIN_POS_NEG.keys())}")

    # Legacy aliases (kept for backward compat; values now come from _CORPUS_DOMAIN_POS_NEG)
    pos_sent, neg_sent = _CORPUS_DOMAIN_POS_NEG.get("sentiment",
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))

    # ── Load per-model feature → domain lookup from feature_labels.json ──────
    _feature_domain_lookup: Dict[str, Dict[int, str]] = {}
    for _mname in sorted(model_names_in_pairs):
        _ef_path = os.path.join(config.FEATURES_DIR, f"{_mname}_ef*_feature_labels.json")
        _candidates = sorted(glob.glob(_ef_path))
        _plain_path = os.path.join(config.FEATURES_DIR, f"{_mname}_feature_labels.json")
        _lpath = _candidates[-1] if _candidates else (_plain_path if os.path.exists(_plain_path) else None)
        if _lpath:
            try:
                _ldata = json.load(open(_lpath, "r", encoding="utf-8"))
                _feature_domain_lookup[_mname] = {
                    int(e["feature_id"]): e["domain"]
                    for e in _ldata.get("features", [])
                    if "feature_id" in e and "domain" in e
                }
                print(f"[step6] Feature domain lookup loaded for {_mname}: "
                      f"{len(_feature_domain_lookup[_mname])} features")
            except Exception as _exc:
                print(f"[step6] Could not load feature labels for {_mname}: {_exc}")
                _feature_domain_lookup[_mname] = {}
        else:
            print(f"[step6] No feature_labels.json found for {_mname}")
            _feature_domain_lookup[_mname] = {}

    feature_idx: Dict[str, np.ndarray] = {}
    feature_mat: Dict[str, np.ndarray] = {}
    topk = config.TOP_FEATURES_FOR_ALIGNMENT
    for model_name in sorted(model_names_in_pairs):
        try:
            idx, mat = _load_feature_matrix(model_name, topk)
            feature_idx[model_name] = idx
            feature_mat[model_name] = mat
        except FileNotFoundError as e:
            print(f"[warn] Could not load feature matrix for {model_name}: {e}")

    idx_to_col = {
        m: {int(fid): i for i, fid in enumerate(feature_idx[m])}
        for m in feature_idx
    }

    # Optional: deconfound top-k PCA components from feature activation matrices (GPU SVD)
    if args.deconfound_k > 0:
        for m in list(feature_mat.keys()):
            mat_t = torch.from_numpy(feature_mat[m].astype(np.float32)).to(_DEVICE)
            k = min(args.deconfound_k, mat_t.shape[0] - 1, mat_t.shape[1])
            U, S, Vt = torch.linalg.svd(mat_t, full_matrices=False)
            proj = (U[:, :k] * S[:k]) @ Vt[:k, :]
            feature_mat[m] = (mat_t - proj).cpu().numpy()
            del mat_t, U, S, Vt, proj
            if _DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[deconfound_k={k}] Removed top-{k} PCA components from {m}")

    # ── Enrich missing domain fields on pairs using feature_labels lookup ─────
    # Only accept a domain label if it is a real semantic concept—not an
    # auto-generated placeholder like "concept_1" or "cluster_42" produced
    # by unsupervised A4b discovery (gpt2 generates many of these).
    for p in pairs:
        if p.get("domain") and _is_named_concept(p["domain"]):
            continue
        # Clear any unnamed domain that may have leaked in from aligned_pairs.jsonl
        if p.get("domain") and not _is_named_concept(p["domain"]):
            p["domain"] = ""
        a_dom = _feature_domain_lookup.get(p["a_model"], {}).get(int(p["a_feature"]))
        b_dom = _feature_domain_lookup.get(p["b_model"], {}).get(int(p["b_feature"]))
        # Skip unnamed labels from either model
        a_dom = a_dom if _is_named_concept(a_dom or "") else None
        b_dom = b_dom if _is_named_concept(b_dom or "") else None
        # Prefer matching domains; then whichever is in corpus pos/neg; then any label
        if a_dom and a_dom == b_dom:
            p["domain"] = a_dom
        elif a_dom and a_dom in _CORPUS_DOMAIN_POS_NEG:
            p["domain"] = a_dom
        elif b_dom and b_dom in _CORPUS_DOMAIN_POS_NEG:
            p["domain"] = b_dom
        elif a_dom:
            p["domain"] = a_dom
        elif b_dom:
            p["domain"] = b_dom
        # else: leave as missing → will be caught as unknown_domain below

    # Pre-build domain-model feature clusters for cluster-pooled co-activation
    domain_model_clusters: Dict[tuple, List[int]] = defaultdict(list)
    for p in pairs:
        dom = p.get("domain", "unknown")
        domain_model_clusters[(p["a_model"], dom)].append(int(p["a_feature"]))
        domain_model_clusters[(p["b_model"], dom)].append(int(p["b_feature"]))

    # ── CCA + Procrustes precomputation ───────────────────────────────────────
    # For each unique model pair: fit CCA on full feature matrices, then fit one
    # Procrustes rotation across all concept direction vectors for that pair.
    # This produces a scientifically valid procrustes_cosine_cca per aligned pair.
    _CCA_COMPONENTS = 64
    _pair_procrustes_cos: dict = {}  # (a_model, b_model, a_feat, b_feat) -> float

    # Step 1: fit CCA weights per model pair
    _model_pair_cca_weights: dict = {}  # (a_model, b_model) -> (x_w, y_w)
    for p in pairs:
        key = (p["a_model"], p["b_model"])
        if key in _model_pair_cca_weights:
            continue
        ma, mb = key
        if ma not in feature_mat or mb not in feature_mat:
            continue
        try:
            # Standardise feature matrices on GPU (zero mean, unit std per column)
            _Xt = torch.from_numpy(feature_mat[ma]).to(torch.float64).to(_DEVICE)
            _Yt = torch.from_numpy(feature_mat[mb]).to(torch.float64).to(_DEVICE)
            _Xt = (_Xt - _Xt.mean(0)) / (_Xt.std(0) + 1e-8)
            _Yt = (_Yt - _Yt.mean(0)) / (_Yt.std(0) + 1e-8)
            X = _Xt.cpu().numpy(); del _Xt
            Y = _Yt.cpu().numpy(); del _Yt
            nc = min(_CCA_COMPONENTS, X.shape[1], Y.shape[1])
            x_w, y_w = _cca_weights(X, Y, n_components=nc)
            _model_pair_cca_weights[key] = (x_w, y_w)
            print(f"[step6] CCA fitted for {ma}-{mb} ({nc} components)")
        except Exception as exc:
            print(f"[step6] CCA failed for {ma}-{mb}: {exc}")

    # Step 2: collect direction vectors per model pair
    _pair_dirs: dict = defaultdict(list)  # key -> list of (a_feat, b_feat, a_dir, b_dir)
    for p in pairs:
        key = (p["a_model"], p["b_model"])
        if key not in _model_pair_cca_weights:
            continue
        ma, mb = key
        a_feat_i = int(p["a_feature"])
        b_feat_i = int(p["b_feature"])
        a_col_i = idx_to_col.get(ma, {}).get(a_feat_i)
        b_col_i = idx_to_col.get(mb, {}).get(b_feat_i)
        if a_col_i is None or b_col_i is None:
            continue
        x_w, y_w = _model_pair_cca_weights[key]
        _pair_dirs[key].append((a_feat_i, b_feat_i, x_w[a_col_i, :], y_w[b_col_i, :]))

    # Step 3: fit Procrustes per model pair and store per-pair cosines
    for key, pair_list in _pair_dirs.items():
        if len(pair_list) < 2:
            for a_feat_i, b_feat_i, a_dir, b_dir in pair_list:
                na = np.linalg.norm(a_dir); nb = np.linalg.norm(b_dir)
                cos = float(np.dot(a_dir / (na + 1e-8), b_dir / (nb + 1e-8)))
                _pair_procrustes_cos[(key[0], key[1], a_feat_i, b_feat_i)] = cos
            continue
        A_mat = np.stack([d[2] for d in pair_list])  # (n_pairs, nc)
        B_mat = np.stack([d[3] for d in pair_list])  # (n_pairs, nc)
        try:
            R, _ = _orthogonal_procrustes(A_mat, B_mat)
            A_rot = A_mat @ R
            for i, (a_feat_i, b_feat_i, _, _) in enumerate(pair_list):
                a_r = A_rot[i]; b_r = B_mat[i]
                na = np.linalg.norm(a_r); nb = np.linalg.norm(b_r)
                cos = float(np.dot(a_r / (na + 1e-8), b_r / (nb + 1e-8)))
                _pair_procrustes_cos[(key[0], key[1], a_feat_i, b_feat_i)] = cos
        except Exception as exc:
            print(f"[step6] Procrustes failed for {key}: {exc}")
            for a_feat_i, b_feat_i, a_dir, b_dir in pair_list:
                na = np.linalg.norm(a_dir); nb = np.linalg.norm(b_dir)
                cos = float(np.dot(a_dir / (na + 1e-8), b_dir / (nb + 1e-8)))
                _pair_procrustes_cos[(key[0], key[1], a_feat_i, b_feat_i)] = cos
    # ─────────────────────────────────────────────────────────────────────────

    # ── SAE-free probe precomputation ─────────────────────────────────────────
    # When --sae-free: load raw hidden states, train per-domain linear probes,
    # fit CCA + Procrustes on probe weight vectors per model pair.
    _saefree_procrustes_cos: dict = {}  # (a_model, b_model, domain) -> float

    if args.sae_free:
        # Two-phase approach to minimise peak CPU RAM:
        #   Phase 1 — for each model that still needs probes, load its full raw acts,
        #             train all domain probes, then FREE immediately.
        #   Phase 2 — for each model pair, load only a tiny 10k-row subsample for CCA.
        # Peak RAM per iteration: feature_mat (~7.5 GB) + one full model (~12 GB max)
        # instead of two full models at once (~24 GB+).

        probe_weights: Dict[tuple, np.ndarray] = {}  # (model, domain) -> weight vec
        model_pairs_seen: List[tuple] = list(dict.fromkeys(
            (p["a_model"], p["b_model"]) for p in pairs
        ))
        _all_probe_doms = set(_CORPUS_DOMAIN_POS_NEG.keys())

        for _mp_idx, (ma, mb) in enumerate(model_pairs_seen):
            gpu_id = _mp_idx % max(1, _NUM_GPUS)
            sae_dev = torch.device(f"cuda:{gpu_id}") if _NUM_GPUS > 0 else torch.device("cpu")
            print(f"[step6/sae-free] Processing pair {ma}-{mb} on cuda:{gpu_id}")

            # ── Phase 1: probe training (load full acts only for uncached models) ──
            _pair_load_ok = True
            for _m in (ma, mb):
                if all((_m, d) in probe_weights for d in _all_probe_doms):
                    continue  # all probes already cached for this model
                try:
                    _full_acts = _load_raw_activations(_m)
                    print(f"[step6/sae-free] Loaded {_m}: {_full_acts.shape}")
                except FileNotFoundError as exc:
                    print(f"[step6/sae-free] Warning: {exc}")
                    _pair_load_ok = False
                    break
                for dom, (p_idx, n_idx) in _CORPUS_DOMAIN_POS_NEG.items():
                    if (_m, dom) not in probe_weights:
                        w = _train_linear_probe(_full_acts, p_idx, n_idx, device=sae_dev)
                        if w is not None:
                            probe_weights[(_m, dom)] = w
                            print(f"[step6/sae-free] Probe: ({_m}, {dom}) dim={w.shape[0]}")
                del _full_acts  # free immediately — don't hold two models at once
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not _pair_load_ok:
                continue

            shared_doms = [
                dom for dom in _CORPUS_DOMAIN_POS_NEG
                if (ma, dom) in probe_weights and (mb, dom) in probe_weights
            ]
            if not shared_doms:
                continue

            # ── Phase 2: CCA on tiny 10k-row subsample (negligible RAM) ──────────
            _N_SUB = 10000
            _rng_sub = np.random.default_rng(7)
            try:
                raw_a_sub = _load_raw_activations_subsample(ma, _N_SUB, _rng_sub)
                raw_b_sub = _load_raw_activations_subsample(mb, _N_SUB, _rng_sub)
                n_sub = min(raw_a_sub.shape[0], raw_b_sub.shape[0])
                # Standardise raw activation subsamples on the target GPU
                _Xs = torch.from_numpy(raw_a_sub[:n_sub]).to(torch.float64).to(sae_dev)
                _Ys = torch.from_numpy(raw_b_sub[:n_sub]).to(torch.float64).to(sae_dev)
                del raw_a_sub, raw_b_sub
                _Xs = (_Xs - _Xs.mean(0)) / (_Xs.std(0) + 1e-8)
                _Ys = (_Ys - _Ys.mean(0)) / (_Ys.std(0) + 1e-8)
                X_sub = _Xs.cpu().numpy(); del _Xs
                Y_sub = _Ys.cpu().numpy(); del _Ys
                nc = min(64, X_sub.shape[1], Y_sub.shape[1])
                x_w_raw, y_w_raw = _cca_weights(X_sub, Y_sub, n_components=nc, device=sae_dev)
                del X_sub, Y_sub
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[step6/sae-free] CCA {ma}-{mb} ({nc} components) on cuda:{gpu_id}")
            except Exception as exc:
                print(f"[step6/sae-free] CCA failed {ma}-{mb}: {exc}")
                continue

            # Project domain probe vectors into CCA subspace and fit Procrustes
            A_probe = np.stack([probe_weights[(ma, d)] for d in shared_doms])  # (D, h_a)
            B_probe = np.stack([probe_weights[(mb, d)] for d in shared_doms])  # (D, h_b)
            # Project probe directions into CCA subspace on GPU
            _At = torch.from_numpy(A_probe.astype(np.float32)).to(sae_dev)
            _Bt = torch.from_numpy(B_probe.astype(np.float32)).to(sae_dev)
            _xwt = torch.from_numpy(x_w_raw.astype(np.float32)).to(sae_dev)
            _ywt = torch.from_numpy(y_w_raw.astype(np.float32)).to(sae_dev)
            A_cca = (_At @ _xwt).cpu().numpy().astype(np.float64)  # (D, nc)
            B_cca = (_Bt @ _ywt).cpu().numpy().astype(np.float64)  # (D, nc)
            del _At, _Bt, _xwt, _ywt

            if len(shared_doms) >= 2:
                try:
                    R, _ = _orthogonal_procrustes(A_cca, B_cca)
                    A_rot = A_cca @ R
                except Exception:
                    A_rot = A_cca
            else:
                A_rot = A_cca

            for i, dom in enumerate(shared_doms):
                a_r = A_rot[i]; b_r = B_cca[i]
                na = np.linalg.norm(a_r); nb = np.linalg.norm(b_r)
                cos = float(np.dot(a_r / (na + 1e-8), b_r / (nb + 1e-8)))
                _saefree_procrustes_cos[(ma, mb, dom)] = cos

    # ─────────────────────────────────────────────────────────────────────────

    # ── Batch GPU permutation-test pre-computation ────────────────────────────
    # _sample_indices uses seed 42 each call, so all pairs with the same domain
    # get identical row-index sets.  Pre-compute those sets once, then stack all
    # valid (a_vals, b_vals) rows into a single GPU batch and run n_perms shuffle
    # tests simultaneously.
    _DOMAIN_SAMPLES: Dict[str, np.ndarray] = {}
    for _dom, (_pi, _ni) in _CORPUS_DOMAIN_POS_NEG.items():
        _s, _, _ = _sample_indices(_pi, _ni, n=50)
        _DOMAIN_SAMPLES[_dom] = _s

    _pair_shuffle_p: Dict[int, Optional[float]] = {}  # index in pairs -> p-value

    if args.n_perms > 0:
        _batch_a_list: List[np.ndarray] = []
        _batch_b_list: List[np.ndarray] = []
        _batch_pair_idx: List[int] = []

        for _pi, _p in enumerate(pairs):
            _dom = _p.get("domain", "unknown")
            _am = _p["a_model"]
            _bm = _p["b_model"]
            _af = int(_p["a_feature"])
            _bf = int(_p["b_feature"])
            if _dom not in _DOMAIN_SAMPLES:
                continue
            if _am not in feature_mat or _bm not in feature_mat:
                continue
            if _af not in idx_to_col.get(_am, {}) or _bf not in idx_to_col.get(_bm, {}):
                continue
            _sall = _DOMAIN_SAMPLES[_dom]
            _ac = idx_to_col[_am][_af]
            _bc = idx_to_col[_bm][_bf]
            _batch_a_list.append(feature_mat[_am][_sall, _ac].astype(np.float32))
            _batch_b_list.append(feature_mat[_bm][_sall, _bc].astype(np.float32))
            _batch_pair_idx.append(_pi)

        if _batch_pair_idx:
            print(f"[step6] Running {args.n_perms} permutations for "
                  f"{len(_batch_pair_idx)} pairs across {max(1, _NUM_GPUS)} GPU(s) ...")
            _all_a_np = np.stack(_batch_a_list)   # (N_valid, S)
            _all_b_np = np.stack(_batch_b_list)   # (N_valid, S)
            _pvals = _multi_gpu_perm_test(_all_a_np, _all_b_np,
                                          n_perm=args.n_perms, seed=0)
            for _i, _orig_idx in enumerate(_batch_pair_idx):
                _pair_shuffle_p[_orig_idx] = float(_pvals[_i])
            print(f"[step6] Permutation tests done.")
    # ─────────────────────────────────────────────────────────────────────────

    # ── GPU-batched precomputation: rho_c (P75/P90/P95), cluster-pooled rho_c,
    # Cohen's d, and direction means — computed once for all pairs on GPU,
    # then looked up O(1) per pair in the loop below.
    # Collect only the specific columns actually needed across all pairs (much
    # smaller than the full topk width — avoids a huge nanpercentile over all
    # features for each model).
    print("[step6] Precomputing per-feature activation thresholds ...")
    _needed_cols: Dict[str, set] = {m: set() for m in feature_mat}
    for _p in pairs:
        _paf = int(_p["a_feature"]); _pbf = int(_p["b_feature"])
        _ma_n = _p["a_model"]; _mb_n = _p["b_model"]
        if _paf in idx_to_col.get(_ma_n, {}):
            _needed_cols.setdefault(_ma_n, set()).add(idx_to_col[_ma_n][_paf])
        if _pbf in idx_to_col.get(_mb_n, {}):
            _needed_cols.setdefault(_mb_n, set()).add(idx_to_col[_mb_n][_pbf])

    _feat_thresh: Dict[str, Dict[int, np.ndarray]] = {}
    for _m, _mat in feature_mat.items():
        _feat_thresh[_m] = {}
        _ncols = np.array(sorted(_needed_cols.get(_m, set())), dtype=np.int64)
        if len(_ncols) == 0:
            for _pct in (75, 90, 95):
                _feat_thresh[_m][_pct] = np.zeros(0, dtype=np.float32)
            continue
        _mat_sub = _mat[:, _ncols].astype(np.float32)  # (P, n_needed)
        # Compute threshold percentiles on GPU via torch.nanquantile
        _mat_sub_t = torch.from_numpy(_mat_sub).to(_DEVICE)
        del _mat_sub
        _mat_sub_nan_t = torch.where(_mat_sub_t > 0, _mat_sub_t, torch.full_like(_mat_sub_t, float('nan')))
        del _mat_sub_t
        _col_thresh_rows = []
        for _q in (0.75, 0.90, 0.95):
            _t = torch.nanquantile(_mat_sub_nan_t, _q, dim=0)
            torch.nan_to_num_(_t, nan=0.0)
            _col_thresh_rows.append(_t.cpu().numpy().astype(np.float32))
        del _mat_sub_nan_t
        if _DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        _col_thresh = np.stack(_col_thresh_rows)  # (3, n_needed)
        # Store as full-width arrays indexed by original column id for O(1) lookup
        _full_width = _mat.shape[1]
        for _i_pct, _pct in enumerate((75, 90, 95)):
            _t_full = np.zeros(_full_width, dtype=np.float32)
            _t_full[_ncols] = _col_thresh[_i_pct]
            _feat_thresh[_m][_pct] = _t_full

    # Pre-compute cluster-pooled active masks: (model, domain) -> (n_passages,) bool
    _cluster_pool_active: Dict[tuple, np.ndarray] = {}
    for (_cmname, _cdom), _cfids in domain_model_clusters.items():
        if _cmname not in feature_mat:
            continue
        _cvalid = list({idx_to_col[_cmname][_f] for _f in _cfids
                        if _f in idx_to_col.get(_cmname, {})})
        if not _cvalid:
            continue
        # Pool cluster columns and compute P90 threshold on GPU
        _cmat_t = torch.from_numpy(
            feature_mat[_cmname][:, _cvalid].astype(np.float32)
        ).to(_DEVICE)
        _cpool_t = _cmat_t.max(dim=1).values
        del _cmat_t
        _cnz_t = _cpool_t[_cpool_t > 0]
        _cthresh = float(_cnz_t.quantile(0.90).item()) if _cnz_t.numel() > 0 else 0.0
        _cluster_pool_active[(_cmname, _cdom)] = (_cpool_t > _cthresh).cpu().numpy()
        del _cpool_t, _cnz_t

    # Batch all rho_c, Cohen's d, and direction means on GPU, grouped by model pair.
    print("[step6] Batching rho_c and Cohen's d on GPU ...")
    _precomp: Dict[int, Dict] = {}  # pair_idx -> metric dict
    _by_model_pair: Dict[tuple, List[int]] = {}
    for _pi, _p in enumerate(pairs):
        _pdom = _p.get("domain", "unknown")
        if _pdom not in _CORPUS_DOMAIN_POS_NEG:
            continue
        if _p["a_model"] not in feature_mat or _p["b_model"] not in feature_mat:
            continue
        _paf = int(_p["a_feature"])
        _pbf = int(_p["b_feature"])
        if _paf not in idx_to_col.get(_p["a_model"], {}) or _pbf not in idx_to_col.get(_p["b_model"], {}):
            continue
        _mpkey = (_p["a_model"], _p["b_model"])
        if _mpkey not in _by_model_pair:
            _by_model_pair[_mpkey] = []
        _by_model_pair[_mpkey].append(_pi)

    for (_ma, _mb), _pidxs in _by_model_pair.items():
        _a_cols_arr = np.array([idx_to_col[_ma][int(pairs[_i]["a_feature"])] for _i in _pidxs])
        _b_cols_arr = np.array([idx_to_col[_mb][int(pairs[_i]["b_feature"])] for _i in _pidxs])
        _domains_arr = [pairs[_i].get("domain", "unknown") for _i in _pidxs]
        _np = len(_pidxs)

        # Load feature columns for this model pair onto GPU
        _A = torch.from_numpy(feature_mat[_ma][:, _a_cols_arr].astype(np.float32)).to(_DEVICE)  # (P, n)
        _B = torch.from_numpy(feature_mat[_mb][:, _b_cols_arr].astype(np.float32)).to(_DEVICE)

        # Thresholds per pair-column
        _ta75 = torch.from_numpy(_feat_thresh[_ma][75][_a_cols_arr]).to(_DEVICE)  # (n,)
        _ta90 = torch.from_numpy(_feat_thresh[_ma][90][_a_cols_arr]).to(_DEVICE)
        _ta95 = torch.from_numpy(_feat_thresh[_ma][95][_a_cols_arr]).to(_DEVICE)
        _tb75 = torch.from_numpy(_feat_thresh[_mb][75][_b_cols_arr]).to(_DEVICE)
        _tb90 = torch.from_numpy(_feat_thresh[_mb][90][_b_cols_arr]).to(_DEVICE)
        _tb95 = torch.from_numpy(_feat_thresh[_mb][95][_b_cols_arr]).to(_DEVICE)

        _act_a75 = _A > _ta75.unsqueeze(0)  # (P, n) bool
        _act_a90 = _A > _ta90.unsqueeze(0)
        _act_a95 = _A > _ta95.unsqueeze(0)
        _act_b75 = _B > _tb75.unsqueeze(0)
        _act_b90 = _B > _tb90.unsqueeze(0)
        _act_b95 = _B > _tb95.unsqueeze(0)

        _rho75 = (_act_a75 & _act_b75).float().mean(dim=0).cpu().numpy()  # (n,)
        _rho90 = (_act_a90 & _act_b90).float().mean(dim=0).cpu().numpy()
        _rho95 = (_act_a95 & _act_b95).float().mean(dim=0).cpu().numpy()
        del _act_a75, _act_a90, _act_a95, _act_b75, _act_b90, _act_b95
        del _ta75, _ta90, _ta95, _tb75, _tb90, _tb95
        del _A, _B
        torch.cuda.empty_cache()

        # Cohen's d and direction means: group by domain for pos/neg indexing
        _cohen_d_out = np.zeros(_np, dtype=np.float64)
        _a_pos_out = np.zeros(_np, dtype=np.float64)
        _a_neg_out = np.zeros(_np, dtype=np.float64)
        _b_pos_out = np.zeros(_np, dtype=np.float64)
        _b_neg_out = np.zeros(_np, dtype=np.float64)

        _by_dom2: Dict[str, List[int]] = {}
        for _li2, _d2 in enumerate(_domains_arr):
            if _d2 not in _by_dom2:
                _by_dom2[_d2] = []
            _by_dom2[_d2].append(_li2)

        for _d2, _lis2 in _by_dom2.items():
            if _d2 not in _CORPUS_DOMAIN_POS_NEG:
                continue
            _pos2, _neg2 = _CORPUS_DOMAIN_POS_NEG[_d2]
            _li2_arr = np.array(_lis2)
            _ac2 = _a_cols_arr[_li2_arr]
            _bc2 = _b_cols_arr[_li2_arr]
            # Slice pos/neg rows for needed columns only
            _Ap2 = torch.from_numpy(feature_mat[_ma][np.ix_(_pos2, _ac2)].astype(np.float32)).to(_DEVICE)  # (n_pos, k)
            _An2 = torch.from_numpy(feature_mat[_ma][np.ix_(_neg2, _ac2)].astype(np.float32)).to(_DEVICE)
            _Bp2 = torch.from_numpy(feature_mat[_mb][np.ix_(_pos2, _bc2)].astype(np.float32)).to(_DEVICE)
            _Bn2 = torch.from_numpy(feature_mat[_mb][np.ix_(_neg2, _bc2)].astype(np.float32)).to(_DEVICE)
            _npos2 = float(_Ap2.shape[0])
            _nneg2 = float(_An2.shape[0])
            _apm2 = _Ap2.mean(dim=0); _anm2 = _An2.mean(dim=0)
            _bpm2 = _Bp2.mean(dim=0); _bnm2 = _Bn2.mean(dim=0)
            _ps_a = torch.sqrt(
                ((_npos2 - 1) * _Ap2.var(dim=0, unbiased=True) + (_nneg2 - 1) * _An2.var(dim=0, unbiased=True))
                / (_npos2 + _nneg2 - 2 + 1e-10)
            )
            _ps_b = torch.sqrt(
                ((_npos2 - 1) * _Bp2.var(dim=0, unbiased=True) + (_nneg2 - 1) * _Bn2.var(dim=0, unbiased=True))
                / (_npos2 + _nneg2 - 2 + 1e-10)
            )
            _cd2 = (((_apm2 - _anm2) / (_ps_a + 1e-10)) + ((_bpm2 - _bnm2) / (_ps_b + 1e-10))).cpu().numpy() / 2.0
            _cohen_d_out[_li2_arr] = _cd2
            _a_pos_out[_li2_arr] = _apm2.cpu().numpy()
            _a_neg_out[_li2_arr] = _anm2.cpu().numpy()
            _b_pos_out[_li2_arr] = _bpm2.cpu().numpy()
            _b_neg_out[_li2_arr] = _bnm2.cpu().numpy()
            del _Ap2, _An2, _Bp2, _Bn2
        torch.cuda.empty_cache()

        for _li3, _pi3 in enumerate(_pidxs):
            _d3 = _domains_arr[_li3]
            _precomp[_pi3] = {
                "rho_c_p75": float(_rho75[_li3]),
                "rho_c_p90": float(_rho90[_li3]),
                "rho_c_p95": float(_rho95[_li3]),
                "rho_c_p90_clustered": float(
                    (_cluster_pool_active[(_ma, _d3)] & _cluster_pool_active[(_mb, _d3)]).mean()
                    if (_ma, _d3) in _cluster_pool_active and (_mb, _d3) in _cluster_pool_active
                    else float(_rho90[_li3])
                ),
                "cohen_d": float(_cohen_d_out[_li3]),
                "a_pos_mean": float(_a_pos_out[_li3]),
                "a_neg_mean": float(_a_neg_out[_li3]),
                "b_pos_mean": float(_b_pos_out[_li3]),
                "b_neg_mean": float(_b_neg_out[_li3]),
            }
    print("[step6] GPU precomputation complete.")
    # ─────────────────────────────────────────────────────────────────────────

    results: List[Dict] = []
    updated_pairs: List[Dict] = []
    all_pearson_pvals: List[float] = []

    for _pair_loop_idx, p in enumerate(pairs):
        if _pair_loop_idx % 100 == 0:
            print(f"[step6] Validating pair {_pair_loop_idx}/{len(pairs)} ...")
        domain = p.get("domain", "unknown")
        a_model = p["a_model"]
        b_model = p["b_model"]
        a_feat = int(p["a_feature"])
        b_feat = int(p["b_feature"])

        if domain in _CORPUS_DOMAIN_POS_NEG:
            pos_idx_d, neg_idx_d = _CORPUS_DOMAIN_POS_NEG[domain]
        else:
            result = {
                "pair": f"{a_model}-{b_model}",
                "a_feature": a_feat, "b_feature": b_feat,
                "domain": domain,
                "pearson_r": 0.0, "pearson_p": 1.0,
                "spearman_rho": 0.0, "spearman_p": 1.0,
                "ccc": 0.0, "shuffle_null_p": 1.0,
                "direction_agreement": False,
                "validation_pass": False,
                "reason": "unknown_domain",
            }
            results.append(result)
            p.update(result)
            updated_pairs.append(p)
            continue

        if a_model not in feature_mat or b_model not in feature_mat:
            result = {
                "pair": f"{a_model}-{b_model}",
                "a_feature": a_feat, "b_feature": b_feat,
                "domain": domain,
                "pearson_r": 0.0, "pearson_p": 1.0,
                "spearman_rho": 0.0, "spearman_p": 1.0,
                "ccc": 0.0, "shuffle_null_p": 1.0,
                "direction_agreement": False,
                "validation_pass": False,
                "reason": "feature_matrix_missing",
            }
            results.append(result)
            p.update(result)
            updated_pairs.append(p)
            continue

        if a_feat not in idx_to_col.get(a_model, {}) or b_feat not in idx_to_col.get(b_model, {}):
            result = {
                "pair": f"{a_model}-{b_model}",
                "a_feature": a_feat, "b_feature": b_feat,
                "domain": domain,
                "pearson_r": 0.0, "pearson_p": 1.0,
                "spearman_rho": 0.0, "spearman_p": 1.0,
                "ccc": 0.0, "shuffle_null_p": 1.0,
                "direction_agreement": False,
                "validation_pass": False,
                "reason": "feature_not_in_topk",
            }
            results.append(result)
            p.update(result)
            updated_pairs.append(p)
            continue

        sample_all, _, _ = _sample_indices(pos_idx_d, neg_idx_d, n=50)

        a_col = idx_to_col[a_model][a_feat]
        b_col = idx_to_col[b_model][b_feat]

        a_vals = feature_mat[a_model][sample_all, a_col]
        b_vals = feature_mat[b_model][sample_all, b_col]

        # All heavy metrics looked up from GPU-precomputed dict (O(1) per pair).
        _pc = _precomp.get(_pair_loop_idx, {})
        rho_c_p75 = _pc.get("rho_c_p75", 0.0)
        rho_c_p90 = _pc.get("rho_c_p90", 0.0)
        rho_c_p95 = _pc.get("rho_c_p95", 0.0)
        rho_c_p90_clustered = _pc.get("rho_c_p90_clustered", 0.0)
        cohen_d_val = _pc.get("cohen_d", 0.0)
        a_pos_mean = _pc.get("a_pos_mean", 0.0)
        a_neg_mean = _pc.get("a_neg_mean", 0.0)
        b_pos_mean = _pc.get("b_pos_mean", 0.0)
        b_neg_mean = _pc.get("b_neg_mean", 0.0)

        # Procrustes cosine: CCA-projected + Procrustes-rotated direction cosine.
        # Full corpus column still needed for plain_cos fallback + mutual info + baseline.
        a_all_corpus = feature_mat[a_model][:, a_col]
        b_all_corpus = feature_mat[b_model][:, b_col]
        a_profile = a_all_corpus.astype(np.float64)
        b_profile = b_all_corpus.astype(np.float64)
        a_pnorm = float(np.linalg.norm(a_profile))
        b_pnorm = float(np.linalg.norm(b_profile))
        plain_cos = float(np.dot(a_profile / (a_pnorm + 1e-8), b_profile / (b_pnorm + 1e-8)))
        procrustes_cosine_cca = _pair_procrustes_cos.get(
            (a_model, b_model, a_feat, b_feat), plain_cos
        )

        # Underpowered: n_passages for this domain < 500
        n_passages_domain = len(pos_idx_d) + len(neg_idx_d)
        underpowered = n_passages_domain < 500

        # Threshold status and optional shuffled-activation baseline
        threshold_status = "working_threshold"
        baseline_procrustes_cos: Optional[float] = None
        baseline_rho_c_p90: Optional[float] = None
        if args.random_init_baseline and a_pnorm > 1e-8 and b_pnorm > 1e-8:
            rng_bl = np.random.default_rng(seed=int((a_feat + b_feat) % (2 ** 31)))
            a_shuffled = a_profile.copy()
            rng_bl.shuffle(a_shuffled)
            a_sh_norm = float(np.linalg.norm(a_shuffled))
            baseline_procrustes_cos = (
                float(np.dot(a_shuffled / a_sh_norm, b_profile / b_pnorm))
                if a_sh_norm > 1e-8 else 0.0
            )
            baseline_rho_c_p90 = _coactivation_rate(
                a_shuffled.astype(np.float32), b_all_corpus, percentile=90
            )
            if procrustes_cosine_cca >= 0.70 and baseline_procrustes_cos < 0.3:
                threshold_status = "random_baseline_calibrated"

        # Mutual information (optional)
        mutual_info: Optional[float] = None
        if args.info_theory:
            mutual_info = _mutual_info_bins(a_profile, b_profile)

        pearson_r, pearson_p = _pearson(a_vals, b_vals)
        spearman_rho, spearman_p = _spearman(a_vals, b_vals)
        ccc_val = _ccc(a_vals, b_vals)
        shuffle_p = _pair_shuffle_p.get(_pair_loop_idx) if args.n_perms > 0 else None
        direction_agreement = (
            ((a_pos_mean > a_neg_mean) and (b_pos_mean > b_neg_mean)) or
            ((a_pos_mean < a_neg_mean) and (b_pos_mean < b_neg_mean))
        )
        sign_flip = pearson_r < 0
        validation_pass = abs(pearson_r) >= args.pearson_thresh
        all_pearson_pvals.append(pearson_p)

        # rho_c fold enrichment: observed / shuffled-activation null.
        # baseline_rho_c_p90 is the rho_c computed on a shuffled copy of feature a
        # (i.e. independence null). Fold enrichment is the primary publication metric.
        _bl_rho = baseline_rho_c_p90 if (baseline_rho_c_p90 is not None and baseline_rho_c_p90 > 1e-10) else None
        rho_c_fold_enrichment_p90 = float(rho_c_p90 / _bl_rho) if _bl_rho else None

        result = {
            "pair": f"{a_model}-{b_model}",
            "a_feature": a_feat,
            "b_feature": b_feat,
            "domain": domain,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "ccc": ccc_val,
            "shuffle_null_p": shuffle_p,
            "direction_agreement": direction_agreement,
            "a_pos_mean": a_pos_mean,
            "a_neg_mean": a_neg_mean,
            "b_pos_mean": b_pos_mean,
            "b_neg_mean": b_neg_mean,
            "rho_c_p75": rho_c_p75,
            "rho_c_p90": rho_c_p90,
            "rho_c_p95": rho_c_p95,
            "rho_c_p90_clustered": rho_c_p90_clustered,
            "rho_c_fold_enrichment_p90": rho_c_fold_enrichment_p90,
            "procrustes_cosine_cca": procrustes_cosine_cca,
            "cohen_d": cohen_d_val,
            "underpowered": underpowered,
            "threshold_status": threshold_status,
            "validation_pass": validation_pass,
            "sign_flip": sign_flip,
            **({
                "baseline_procrustes_cos": baseline_procrustes_cos,
                "baseline_rho_c_p90": baseline_rho_c_p90,
            } if args.random_init_baseline else {}),
            **({
                "mutual_info": mutual_info,
            } if args.info_theory else {}),
            **({
                "saefree_procrustes_cosine": _saefree_procrustes_cos.get((a_model, b_model, domain)),
                "sae_artefact_warning": (
                    _saefree_procrustes_cos.get((a_model, b_model, domain), 1.0) < 0.50
                ),
            } if args.sae_free else {}),
        }
        results.append(result)
        p.update(result)
        updated_pairs.append(p)

    # Apply BH correction to all Pearson p-values and update validation_pass
    if all_pearson_pvals:
        bh_adjusted = _bh_correction(all_pearson_pvals)
        bh_idx = 0
        for r in results:
            if "reason" not in r:
                r["pearson_p_bh"] = bh_adjusted[bh_idx]
                r["validation_pass"] = r["validation_pass"] and (bh_adjusted[bh_idx] < 0.05)
                bh_idx += 1

    # RSA: one score per distinct model pair if requested
    if args.rsa:
        pair_models = set(r["pair"] for r in results)
        rsa_scores: Dict[str, float] = {}
        for pair_key in pair_models:
            # Model names may contain hyphens (e.g. "gpt2-large"), so we cannot
            # simply split on the first "-". Match against known feature_mat keys.
            ma, mb = None, None
            for _cma in feature_mat:
                if pair_key.startswith(_cma + "-"):
                    _cmb = pair_key[len(_cma) + 1:]
                    if _cmb in feature_mat:
                        ma, mb = _cma, _cmb
                        break
            if ma is None or mb is None:
                continue
            # RSA sample: balanced from ALL corpus domains (30 passages per domain
            # side, seeded per domain for reproducibility).  30 per side is the
            # minimum for stable RDM estimation (Kriegeskorte 2008 recommends ≥24
            # stimuli); with 15 domains this yields ~900 passages and ~404 K
            # pairwise distances in the Spearman r — sufficient for a NeurIPS claim.
            _RSA_N_PER_SIDE = 30
            _rsa_parts: List[np.ndarray] = []
            for _rdom, (_rpi, _rni) in _CORPUS_DOMAIN_POS_NEG.items():
                _rng = np.random.default_rng(hash(_rdom) % (2 ** 31))
                _rsa_parts.append(_rng.choice(_rpi, size=min(_RSA_N_PER_SIDE, len(_rpi)), replace=False))
                _rsa_parts.append(_rng.choice(_rni, size=min(_RSA_N_PER_SIDE, len(_rni)), replace=False))
            rsa_sample = np.unique(np.concatenate(_rsa_parts)) if _rsa_parts else None
            if rsa_sample is not None:
                rsa_scores[pair_key] = _rsa_score(feature_mat[ma], feature_mat[mb], rsa_sample)
        for r in results:
            r["rsa_score"] = rsa_scores.get(r["pair"])

    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Rewrite aligned_pairs with updated validation fields
    with open(aligned_path, "w", encoding="utf-8") as f:
        for p in updated_pairs:
            f.write(json.dumps(p) + "\n")

    passed = sum(1 for r in results if r.get("validation_pass"))
    print(f"Validation passed: {passed}/{len(results)} (BH-corrected Pearson |r| >= {args.pearson_thresh})")
    print(f"[step6] Output: {results_path}")

    # Auto-generate undirected pair summary (b2_results.json)
    try:
        import summarise_b2
        summarise_b2.run()
    except Exception as _e:
        print(f"[step6] Warning: summarise_b2 failed: {_e}")

    log_run("step6_validate_alignment.py", start_time, "success")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log_run("step6_validate_alignment.py", time.time(), "error", str(e))
        raise
