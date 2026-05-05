
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import json
import os
import random
import re
import tempfile
import time
import math
import multiprocessing
import glob
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
import torch

# sanitize wandb import to avoid local ./wandb directory shadowing
import sys as _sys, os as _os
_cwd = _os.getcwd()
if '' in _sys.path:
    _sys.path.remove('')
if _cwd in _sys.path:
    _sys.path.remove(_cwd)
# append cwd at end so site-packages wins over local wandb dir
_sys.path.append(_cwd)
import wandb
from sklearn.linear_model import Ridge  # unused directly; sklearn required for linear_sum_assignment compat
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment

try:
    from kan import KAN
    _KAN_AVAILABLE = True
except Exception:
    KAN = None
    _KAN_AVAILABLE = False

import config
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

WANDB_RUN = None

# ---------------------------------------------------------------------------
# Memory reporting utility — call _mem(tag) anywhere for a one-line snapshot
# of CPU RAM and GPU VRAM.  psutil is optional; degrades gracefully.
# ---------------------------------------------------------------------------
try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:
    _psutil = None
    _PSUTIL = False

def _mem(tag: str = "", device: "torch.device | None" = None) -> None:
    """Print a compact RAM/VRAM snapshot.  Safe to call from any thread/process."""
    parts = [f"[MEM:{tag}]" if tag else "[MEM]"]
    if _PSUTIL:
        vm = _psutil.virtual_memory()
        parts.append(f"RAM {vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB "
                     f"({vm.percent:.0f}% used, {vm.available/1e9:.1f} GB free)")
    if torch.cuda.is_available():
        di = device.index if (device is not None and device.type == "cuda") else torch.cuda.current_device()
        alloc  = torch.cuda.memory_allocated(di) / 1e9
        reserv = torch.cuda.memory_reserved(di)  / 1e9
        total  = torch.cuda.get_device_properties(di).total_memory / 1e9
        parts.append(f"GPU[{di}] alloc={alloc:.2f} GB reserved={reserv:.2f} GB total={total:.1f} GB")
    print(" | ".join(parts), flush=True)


from a3_train_sae import TopKSAE


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
    torch.manual_seed(seed)


def _find_sae_checkpoint(model_name: str, sae_dir: str, ef: int) -> str:
    """Find the latest SAE checkpoint for (model, expansion_factor)."""
    ef_tag = "" if ef == config.SAE_EXPANSION_FACTOR else f"_ef{ef}"
    candidates = []
    for fname in os.listdir(sae_dir):
        if not fname.endswith(".pt"):
            continue
        if model_name not in fname:
            continue
        if "step" not in fname:
            continue
        expected_prefix = f"{model_name}{ef_tag}_sae_step"
        if not fname.startswith(expected_prefix):
            continue
        m = re.search(r"step(\d+)", fname)
        if m:
            candidates.append((int(m.group(1)), fname))
    if candidates:
        _, fname = max(candidates, key=lambda x: x[0])
        return os.path.join(sae_dir, fname)
    for fallback in [
        os.path.join(sae_dir, f"{model_name}{ef_tag}_sae.pt"),
        *([] if ef_tag else [os.path.join(sae_dir, f"{model_name}_sae.pt")]),
    ]:
        if os.path.exists(fallback):
            return fallback
    raise FileNotFoundError(f"No SAE checkpoint for {model_name} ef={ef} in {sae_dir}")


def _load_activations(model_name: str) -> np.ndarray:
    """Load normalized activations for model_name.
    Tries plain {model}_activations_norm.h5 first; if missing, concatenates all
    {model}_*_activations_norm.h5 per-source files found in ACTIVATIONS_DIR."""
    _mem(f"before_load_acts:{model_name}")
    plain_path = os.path.join(config.ACTIVATIONS_DIR, f"{model_name}_activations_norm.h5")
    if os.path.exists(plain_path):
        with h5py.File(plain_path, "r") as h5:
            acts = h5["activations"][:]
        _mem(f"after_load_acts:{model_name} shape={acts.shape}")
        return acts

    # Aggregate all per-source files for this model
    pattern = os.path.join(config.ACTIVATIONS_DIR, f"{model_name}_*_activations_norm.h5")
    source_files = sorted(glob.glob(pattern))
    if not source_files:
        raise FileNotFoundError(
            f"No activations found for {model_name} — missing {plain_path} and "
            f"no per-source files matching {pattern}"
        )
    print(f"[step5] Aggregating {len(source_files)} source activation files for {model_name}")
    # Two-pass: first gather shapes, then pre-allocate and copy chunk-by-chunk.
    # Avoids np.concatenate which doubles peak RAM (keeps all chunks + output simultaneously).
    sizes, dim = [], None
    for sf in source_files:
        with h5py.File(sf, "r") as h5:
            sh = h5["activations"].shape
            sizes.append(sh[0]); dim = sh[1]
    acts = np.empty((sum(sizes), dim), dtype=np.float32)
    offset = 0
    for sf, n in zip(source_files, sizes):
        with h5py.File(sf, "r") as h5:
            chunk = h5["activations"][:].astype(np.float32)
        acts[offset:offset + n] = chunk; del chunk; offset += n
    size_gb = acts.nbytes / 1e9
    print(f"[step5] {model_name}: activations shape={acts.shape} ({size_gb:.2f} GB in RAM)")
    _mem(f"after_load_acts:{model_name}")
    return acts


def _select_topk_features(acts: np.ndarray, sae: TopKSAE, n_features: int, topk: int, batch_size: int) -> np.ndarray:
    dev = next(sae.parameters()).device
    _mem(f"select_topk_start dev={dev} n_feats={n_features} topk={topk}", device=dev)
    sum_abs = np.zeros(n_features, dtype=np.float64)
    n = acts.shape[0]
    sae.eval()
    for i in range(0, n, batch_size):
        batch = torch.from_numpy(acts[i:i + batch_size]).float().to(dev)
        with torch.no_grad():
            _, sparse = sae(batch)
            sum_abs += sparse.abs().sum(dim=0).cpu().numpy()
    top_idx = np.argsort(sum_abs)[-topk:][::-1]
    return np.ascontiguousarray(top_idx)


# Registry of memmap temp files to delete after each pair finishes.
_MEMMAP_TMPFILES: list = []


def _compute_feature_matrix(acts: np.ndarray, sae: TopKSAE, feature_idx: np.ndarray, batch_size: int) -> np.ndarray:
    """Compute SAE feature activations stored in a float16 memmap file on disk.
    This is the permanent OOM fix: the matrix is never fully resident in CPU RAM.
    The returned memmap is read-only; .float() in _train_mlp casts batches to fp32.
    Callers must NOT call np.asarray() on the full matrix — always slice in batches."""
    dev = next(sae.parameters()).device
    feature_idx = np.ascontiguousarray(feature_idx)
    feat_t = torch.from_numpy(feature_idx).long().to(dev)
    n = acts.shape[0]
    k = feature_idx.shape[0]
    projected_gb = n * k * 2 / 1e9  # float16 = 2 bytes
    print(f"[step5/matrix] allocating memmap {n}×{k} float16 = {projected_gb:.2f} GB on disk")
    _mem(f"before_memmap n={n} k={k} dev={dev}", device=dev)
    # Write to a temp file on NVMe — avoids allocating n×k floats in RAM
    tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False, dir="/tmp")
    tmp.close()
    _MEMMAP_TMPFILES.append(tmp.name)
    out = np.memmap(tmp.name, dtype=np.float16, mode="w+", shape=(n, k))
    sae.eval()
    _log_every = max(1, n // (batch_size * 5))  # log progress ~5 times
    for _bi, i in enumerate(range(0, n, batch_size)):
        batch = torch.from_numpy(acts[i:i + batch_size]).float().to(dev)
        with torch.no_grad():
            _, sparse = sae(batch)
            out[i:i + batch_size] = sparse[:, feat_t].cpu().to(torch.float16).numpy()
        out.flush()  # keep disk in sync; allows OS to evict pages from RAM
        if _bi % _log_every == 0:
            _mem(f"matrix_fill {i}/{n} rows done", device=dev)
    _mem(f"after_memmap_fill n={n} k={k}", device=dev)
    return out


def _load_selected_features(model_name: str) -> np.ndarray:
    labels_path = os.path.join(config.FEATURES_DIR, f"{model_name}_feature_labels.json")
    if not os.path.exists(labels_path):
        return np.array([], dtype=np.int64)
    data = json.load(open(labels_path, "r", encoding="utf-8"))
    selected = data.get("selected_feature_ids", [])
    if not selected:
        return np.array([], dtype=np.int64)
    return np.array(selected, dtype=np.int64)


def _corr_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # x, y are standardized (zero mean, unit std)
    n = x.shape[0]
    return np.abs((x.T @ y) / max(n - 1, 1))


def _cca_gpu(x_s: np.ndarray, y_s: np.ndarray, n_components: int, device: torch.device) -> tuple:
    """GPU-accelerated CCA via whitened cross-covariance SVD.
    Replaces sklearn CCA — orders of magnitude faster on large matrices."""
    xt = torch.from_numpy(x_s).float().to(device)
    yt = torch.from_numpy(y_s).float().to(device)
    n = xt.shape[0]
    reg = 1e-6
    cxx = (xt.T @ xt) / (n - 1) + torch.eye(xt.shape[1], device=device) * reg
    cyy = (yt.T @ yt) / (n - 1) + torch.eye(yt.shape[1], device=device) * reg
    cxy = (xt.T @ yt) / (n - 1)
    ex, vx = torch.linalg.eigh(cxx)
    ey, vy = torch.linalg.eigh(cyy)
    ex = ex.clamp(min=1e-8)
    ey = ey.clamp(min=1e-8)
    cxx_isqrt = vx @ torch.diag(ex ** -0.5) @ vx.T
    cyy_isqrt = vy @ torch.diag(ey ** -0.5) @ vy.T
    k_mat = cxx_isqrt @ cxy @ cyy_isqrt
    u, s, vh = torch.linalg.svd(k_mat, full_matrices=False)
    k = min(n_components, len(s))
    x_w = (cxx_isqrt @ u[:, :k]).cpu().numpy()
    y_w = (cyy_isqrt @ vh[:k].T).cpu().numpy()
    w = x_w @ y_w.T
    s_cca = torch.abs(cxy).cpu().numpy()
    return w, s_cca, x_w, y_w


def _build_mlp_model(in_dim: int, out_dim: int, hidden_dim: int, dropout: float = 0.1):
    """3-layer MLP bridge with dropout for regularisation.
    dropout=0.1 prevents overfitting when params >> training rows."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=dropout),
        torch.nn.Linear(hidden_dim, out_dim),
    )


def _train_mlp(x: np.ndarray, y: np.ndarray, hidden_dim: int, epochs: int, batch_size: int, lr: float, pearson_weight: float, device: torch.device = None,
               x_mean=None, x_std=None, y_mean=None, y_std=None,
               x_val: np.ndarray = None, y_val: np.ndarray = None,
               label: str = "", dropout: float = 0.1, weight_decay: float = 1e-5):
    """Train MLP bridge.  Pass x_val/y_val to get per-epoch val loss in logs.
    label is a short string shown in log lines (e.g. 'fwd A→B')."""
    if batch_size <= 0:
        batch_size = 256
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NEVER pre-load x/y to GPU — they may be memmaps of 100+ GB.
    # Instead, load each mini-batch on-the-fly from the numpy/memmap array.
    # Optional mean/std tensors allow zero-copy normalisation per batch.
    _xm = torch.from_numpy(x_mean).to(device) if x_mean is not None else None
    _xs = torch.from_numpy(x_std).to(device) if x_std is not None else None
    _ym = torch.from_numpy(y_mean).to(device) if y_mean is not None else None
    _ys = torch.from_numpy(y_std).to(device) if y_std is not None else None
    # Pre-load val set to GPU (it's small: 2000 rows)
    _xv = torch.from_numpy(np.asarray(x_val, dtype=np.float32)).to(device) if x_val is not None else None
    _yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).to(device) if y_val is not None else None
    if _xv is not None and _xm is not None:
        _xv = (_xv - _xm) / _xs
    if _yv is not None and _ym is not None:
        _yv = (_yv - _ym) / _ys

    model = _build_mlp_model(x.shape[1], y.shape[1], hidden_dim, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    tag = f"[MLP{' ' + label if label else ''}]"
    print(f"{tag} model {x.shape[1]}→{hidden_dim}→{y.shape[1]} params={n_params:,} "
          f"train_rows={x.shape[0]:,} batch={batch_size} epochs={epochs} lr={lr} "
          f"dropout={dropout} wd={weight_decay}")
    _mem(f"mlp_init in={x.shape[1]} hidden={hidden_dim} out={y.shape[1]}", device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine annealing: smoothly decays lr → lr/100 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)
    loss_fn = torch.nn.MSELoss()
    _log_every = max(1, epochs // 10)  # log ~10 times

    n = x.shape[0]
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            b_sorted = np.sort(b)  # memmap reads are faster with sorted indices
            xb = torch.from_numpy(np.asarray(x[b_sorted], dtype=np.float32)).to(device)
            yb = torch.from_numpy(np.asarray(y[b_sorted], dtype=np.float32)).to(device)
            if _xm is not None:
                xb = (xb - _xm) / _xs
            if _ym is not None:
                yb = (yb - _ym) / _ys
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if pearson_weight > 0:
                loss = loss + pearson_weight * _pearson_penalty(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if epoch % _log_every == 0 or epoch == epochs - 1:
            model.eval()
            train_loss = epoch_loss / max(n_batches, 1)
            lr_now = scheduler.get_last_lr()[0]
            if _xv is not None:
                with torch.no_grad():
                    val_pred = model(_xv)
                    val_mse = loss_fn(val_pred, _yv).item()
                    # Pearson on flattened outputs
                    vp_f = val_pred.flatten()
                    vy_f = _yv.flatten()
                    vp_c = vp_f - vp_f.mean()
                    vy_c = vy_f - vy_f.mean()
                    val_r = (vp_c @ vy_c / (vp_c.norm() * vy_c.norm()).clamp_min(1e-8)).item()
                print(f"{tag} epoch {epoch:3d}/{epochs} | "
                      f"train_loss={train_loss:.4f} | val_mse={val_mse:.4f} val_r={val_r:.4f} | lr={lr_now:.2e}")
            else:
                print(f"{tag} epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | lr={lr_now:.2e}")
            if epoch % (max(_log_every, 20)) == 0:
                _mem(f"mlp_epoch_{epoch}", device=device)
    model.eval()
    return model


def _pearson_penalty(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = pred.flatten()
    y = target.flatten()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (vx.norm() * vy.norm()).clamp_min(1e-8)
    corr = (vx @ vy) / denom
    return 1.0 - corr


def _batched_mean_std(arr: np.ndarray, batch_size: int = 256):
    """Compute per-feature mean and std from a large/memmap array in batches.
    Returns (mean, std) as float32 arrays of shape (n_features,).
    Never loads the full array into RAM.  Uses float32 accumulators to avoid
    the 2× memory penalty of float64 (precision is sufficient for normalisation)."""
    n, d = arr.shape
    _mem(f"batched_mean_std start n={n} d={d}")
    # Single-pass: accumulate sum and sum-of-squares in float32.
    # batch_size=256 keeps each chunk ≤ 256×d×4 B (≤4 MB at d=4096).
    s1 = np.zeros(d, dtype=np.float32)
    s2 = np.zeros(d, dtype=np.float32)
    for i in range(0, n, batch_size):
        chunk = np.asarray(arr[i:i + batch_size], dtype=np.float32)
        s1 += chunk.sum(axis=0)
        s2 += (chunk * chunk).sum(axis=0)  # avoids a temporary copy vs chunk**2
    mean_f = s1 / n
    std_f = np.sqrt(np.maximum(s2 / n - mean_f ** 2, 1e-8))
    _mem(f"batched_mean_std done n={n} d={d}")
    return mean_f, std_f


KAN_PROGRESS_EVERY = 50


def _train_kan(x: np.ndarray, y: np.ndarray, steps: int, lr: float, pearson_weight: float, ckpt_prefix: str = None, device: torch.device = None):
    if not _KAN_AVAILABLE:
        raise RuntimeError("pykan not installed. Run: pip install pykan")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_t = torch.from_numpy(x).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    model = KAN([x.shape[1], 100, y.shape[1]], grid=3, k=3).to(device)
    try:
        model.speed()
    except Exception:
        pass
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    n = x.shape[0]
    for step in range(steps):
        if step % KAN_PROGRESS_EVERY == 0:
            print(f"[KAN] step {step}/{steps}")
            try:
                if WANDB_RUN is not None:
                    wandb.log({"kan_loss": float(loss.item())}, step=step)
            except Exception:
                pass
        idx = torch.randint(0, n, (min(512, n),))
        xb = x_t[idx]
        yb = y_t[idx]
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        if pearson_weight > 0:
            loss = loss + pearson_weight * _pearson_penalty(pred, yb)
        loss.backward()
        opt.step()
    return model


def _count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------
# Regex to detect auto-generated / unnamed concept labels (concept_1, cluster42…)
# Features with these names are excluded from the candidate pool so that
# unsupervised gpt2 A4b labels like "concept1", "concept42" never pollute B1.
# ---------------------------------------------------------------------------
_UNNAMED_CONCEPT_RE = re.compile(
    r'^(concept|cluster|topic|unknown)[_\s]*\d',
    re.IGNORECASE,
)


def _is_named_concept(label: str) -> bool:
    """Return True iff label is a real semantic concept (not auto-generated)."""
    if not label:
        return False
    return not _UNNAMED_CONCEPT_RE.match(label)


def _svcca(x_s: np.ndarray, y_s: np.ndarray, n_sv: int, n_cca_components: int,
           device: torch.device):
    """SVCCA — project each feature space to its top-n_sv right singular vectors,
    then run CCA on the projected (lower-noise) subspaces.

    Returns s_svcca (K_x × K_y) score matrix in the original feature space.
    Reference: Raghu et al. (2017) arXiv:1706.05806
    """
    K_x, K_y = x_s.shape[1], y_s.shape[1]
    n_sv_x = min(n_sv, x_s.shape[0] - 1, K_x)
    n_sv_y = min(n_sv, y_s.shape[0] - 1, K_y)

    xt = torch.from_numpy(x_s.astype(np.float32)).to(device)
    yt = torch.from_numpy(y_s.astype(np.float32)).to(device)

    # Right singular vectors span the feature subspace with highest variance
    _, _, Vht_x = torch.linalg.svd(xt, full_matrices=False)   # (min,K_x)
    _, _, Vht_y = torch.linalg.svd(yt, full_matrices=False)
    Vx = Vht_x[:n_sv_x].T.cpu().numpy()   # (K_x, n_sv_x)
    Vy = Vht_y[:n_sv_y].T.cpu().numpy()   # (K_y, n_sv_y)
    del xt, yt, Vht_x, Vht_y
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Project passages into SV subspace, then run CCA
    x_sv = x_s @ Vx   # (n_passages, n_sv_x)
    y_sv = y_s @ Vy   # (n_passages, n_sv_y)
    n_comp = min(n_cca_components, n_sv_x, n_sv_y)
    _, _, x_w_sv, y_w_sv = _cca_gpu(x_sv, y_sv, n_comp, device)
    # x_w_sv: (n_sv_x, n_comp); map back to original feature space
    x_w_full = Vx @ x_w_sv   # (K_x, n_comp)
    y_w_full = Vy @ y_w_sv   # (K_y, n_comp)

    # Score matrix: |cosine similarity of CCA loadings| in original feature space
    xwt = torch.from_numpy(x_w_full.astype(np.float32)).to(device)
    ywt = torch.from_numpy(y_w_full.astype(np.float32)).to(device)
    xwn = xwt / (xwt.norm(dim=1, keepdim=True).clamp_min(1e-8))
    ywn = ywt / (ywt.norm(dim=1, keepdim=True).clamp_min(1e-8))
    s_svcca = (xwn @ ywn.T).abs().cpu().numpy()   # (K_x, K_y)
    del xwt, ywt, xwn, ywn
    return s_svcca


def _mutual_nn_scores(x_w: np.ndarray, y_w: np.ndarray, device: torch.device) -> np.ndarray:
    """Mutual Nearest Neighbours in CCA-aligned feature loading space.

    For each feature i in model A: find best-matching feature j in model B
    (forward NN).  For each j in model B: find best-matching i in model A
    (backward NN).  Keep only pairs where both directions agree.  Asymmetric
    correspondences are discarded — they are likely noise or polysemous features.

    x_w, y_w: (K_x, n_comp) and (K_y, n_comp) CCA weight matrices returned by
               _cca_gpu (each row = one feature's loading on CCA directions).
    Returns score matrix (K_x × K_y): MNN pairs get their cosine score, others 0.
    Reference: Conneau et al. (2018) arXiv:1811.01124
    """
    xwt = torch.from_numpy(x_w.astype(np.float32)).to(device)   # (K_x, k)
    ywt = torch.from_numpy(y_w.astype(np.float32)).to(device)   # (K_y, k)
    xwn = xwt / (xwt.norm(dim=1, keepdim=True).clamp_min(1e-8))
    ywn = ywt / (ywt.norm(dim=1, keepdim=True).clamp_min(1e-8))

    S = xwn @ ywn.T                          # (K_x, K_y) cosine similarities
    fwd = S.argmax(dim=1)                    # (K_x,) best j for each i
    bwd = S.argmax(dim=0)                    # (K_y,) best i for each j

    K_x = xwt.shape[0]
    i_idx = torch.arange(K_x, device=device)
    mnn_mask = (bwd[fwd] == i_idx)           # bidirectional agreement

    score_mat = torch.zeros(K_x, ywt.shape[0], device=device)
    j_sel = fwd[mnn_mask]
    score_mat[i_idx[mnn_mask], j_sel] = S[i_idx[mnn_mask], j_sel].clamp(min=0.0)
    del xwt, ywt, xwn, ywn, S, fwd, bwd
    return score_mat.cpu().numpy()


def _align_pair(x: np.ndarray, y: np.ndarray, cca_components: int, mlp_hidden: int,
                mlp_epochs: int, mlp_batch: int, mlp_lr: float, pearson_weight: float,
                kan_steps: int, kan_lr: float, svcca_sv: int = 64,
                mlp_dropout: float = 0.1, mlp_weight_decay: float = 1e-5,
                methods: frozenset = None, device: torch.device = None,
                x_full: np.ndarray = None, y_full: np.ndarray = None):
    """Align a model pair.

    x, y        — top-K feature activation matrices (n_passages × K).  Used for
                  CCA, Procrustes, SVCCA, Mutual-NN, and composite scoring.
    x_full, y_full — SAE activation matrices capped at max_mlp_features
                  ever-active features (n_passages × M, M ≤ max_mlp_features).
                  When provided the MLP bridge is trained on these (technically
                  correct per B1 spec).  Both forward (a→b) and reverse (b→a)
                  bridges are trained here and returned.  MLP is excluded from
                  the composite pair-matching score (output space ≠ top-K space).
    """
    if methods is None:
        methods = frozenset({"cca", "procrustes", "mlp"})
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Analytical methods use top-K feature matrices ──────────────────────
    # Use float32 standardisation — StandardScaler outputs float64 which doubles
    # memory (e.g. 739K×871 float64 = 5.1 GB vs float32 = 2.6 GB).
    x_mean = x.mean(axis=0, keepdims=True); x_std = x.std(axis=0, keepdims=True) + 1e-8
    y_mean = y.mean(axis=0, keepdims=True); y_std = y.std(axis=0, keepdims=True) + 1e-8
    x_s = ((x - x_mean) / x_std).astype(np.float32)
    y_s = ((y - y_mean) / y_std).astype(np.float32)
    del x_mean, x_std, y_mean, y_std

    n = x_s.shape[0]
    n_val = min(2000, max(1, n // 5))
    idx = np.arange(n)
    train_idx = idx[:-n_val]
    val_idx = idx[-n_val:]
    x_train, y_train = x_s[train_idx], y_s[train_idx]
    x_val, y_val = x_s[val_idx], y_s[val_idx]

    # CCA mapping (GPU-accelerated whitened SVD)
    _x_w_cca = _y_w_cca = None   # saved for Mutual-NN downstream
    if "cca" in methods or "mutual_nn" in methods or "mutual-nn" in methods:
        n_comp = min(cca_components, x_s.shape[1], y_s.shape[1])
        w, s_cca, _x_w_cca, _y_w_cca = _cca_gpu(x_s, y_s, n_comp, device)
    else:
        w = np.zeros((x_s.shape[1], y_s.shape[1]), dtype=np.float32)
        s_cca = np.zeros((x_s.shape[1], y_s.shape[1]), dtype=np.float32)

    # SVCCA: SVD projection then CCA — more noise-robust for high-dim SAE spaces
    if "svcca" in methods:
        s_svcca = _svcca(x_s, y_s, n_sv=svcca_sv, n_cca_components=cca_components, device=device)
    else:
        s_svcca = np.zeros((x_s.shape[1], y_s.shape[1]), dtype=np.float32)

    # Mutual-NN: bidirectional nearest neighbours in CCA loading space
    _mnn_available = ("mutual_nn" in methods or "mutual-nn" in methods) and _x_w_cca is not None
    if _mnn_available:
        s_mnn = _mutual_nn_scores(_x_w_cca, _y_w_cca, device)
    else:
        s_mnn = np.zeros((x_s.shape[1], y_s.shape[1]), dtype=np.float32)

    # Procrustes mapping (requires same feature dims)
    if "procrustes" in methods and x_s.shape[1] == y_s.shape[1]:
        r, _ = orthogonal_procrustes(x_s, y_s)
        x_pro = x_s @ r
        s_pro = _corr_matrix(x_pro, y_s)
    else:
        r = None
        s_pro = np.zeros((x_s.shape[1], y_s.shape[1]), dtype=np.float32)

    # ── MLP bridge — use full SAE activations when available ──────────────
    # Using the full SAE activation vector (EF × hidden_dim scalars) is the
    # technically correct input per the B1 spec.  CCA/Procrustes still use the
    # top-K matrices above — those methods benefit from dimensionality reduction.
    using_full = x_full is not None and y_full is not None
    if using_full:
        # Compute mean/std in batches — never materialise the full matrix in RAM.
        print(f"[MLP] computing normalisation stats for full SAE activations ...")
        xf_mean, xf_std = _batched_mean_std(x_full)
        yf_mean, yf_std = _batched_mean_std(y_full)
        n_mlp = x_full.shape[0]
        n_val_mlp = min(2000, max(1, n_mlp // 5))
        # Pass raw memmaps to _train_mlp; normalisation happens per mini-batch.
        x_mlp_train = x_full[:-n_val_mlp]
        y_mlp_train = y_full[:-n_val_mlp]
        x_mlp_val   = x_full[-n_val_mlp:]
        y_mlp_val   = y_full[-n_val_mlp:]
        print(f"[MLP] training on full SAE activations: {x_full.shape[1]}-d → {y_full.shape[1]}-d")
    else:
        xf_mean = xf_std = yf_mean = yf_std = None
        x_mlp_train, y_mlp_train = x_train, y_train
        x_mlp_val,   y_mlp_val   = x_val,   y_val

    mlp = _train_mlp(x_mlp_train, y_mlp_train, mlp_hidden, mlp_epochs, mlp_batch, mlp_lr, pearson_weight, device=device,
                     x_mean=xf_mean, x_std=xf_std, y_mean=yf_mean, y_std=yf_std,
                     x_val=x_mlp_val, y_val=y_mlp_val, label="fwd",
                     dropout=mlp_dropout, weight_decay=mlp_weight_decay)

    # s_mlp for composite scoring only makes sense when MLP operates in top-K space
    if using_full:
        # MLP output is full-dim — cannot contribute to the (K×K) matching matrix.
        # Feature pair matching uses CCA + Procrustes only (more principled anyway).
        s_mlp = np.zeros((x_s.shape[1], y_s.shape[1]), dtype=np.float32)
    else:
        with torch.no_grad():
            y_pred = mlp(torch.from_numpy(x_s).float().to(device)).cpu().numpy()
        s_mlp = _corr_matrix(x_s, y_pred)

    # MLP validation metrics (final)
    with torch.no_grad():
        val_chunks = []
        for _vi in range(0, x_mlp_val.shape[0], 256):
            _xvb = torch.from_numpy(np.asarray(x_mlp_val[_vi:_vi+256], dtype=np.float32)).to(device)
            if xf_mean is not None:
                _xvb = (_xvb - torch.from_numpy(xf_mean).to(device)) / torch.from_numpy(xf_std).to(device)
            val_chunks.append(mlp(_xvb).cpu().numpy())
        y_val_pred = np.concatenate(val_chunks, axis=0)
    y_mlp_val_np = np.asarray(y_mlp_val, dtype=np.float32)
    if yf_mean is not None:
        y_mlp_val_np = (y_mlp_val_np - yf_mean) / yf_std
    mlp_val_mse = float(np.mean((y_val_pred - y_mlp_val_np) ** 2))
    mlp_val_corr = float(np.corrcoef(y_val_pred.flatten(), y_mlp_val_np.flatten())[0, 1]) if y_mlp_val_np.size > 1 else 0.0

    # Reverse MLP (b→a) — trained alongside forward when full activations available
    rev_mlp_state = None
    if using_full:
        print(f"[MLP] training reverse bridge: {y_full.shape[1]}-d \u2192 {x_full.shape[1]}-d")
        rev_mlp = _train_mlp(y_mlp_train, x_mlp_train, mlp_hidden, mlp_epochs, mlp_batch, mlp_lr, pearson_weight, device=device,
                             x_mean=yf_mean, x_std=yf_std, y_mean=xf_mean, y_std=xf_std,
                             x_val=y_mlp_val, y_val=x_mlp_val, label="rev",
                             dropout=mlp_dropout, weight_decay=mlp_weight_decay)
        rev_mlp_state = rev_mlp.state_dict()
        del rev_mlp

    # KAN mapping (optional, uses top-K space)
    kan_state = None
    kan_val_mse = None
    kan_val_corr = None
    kan_params = None
    kan_train_sec = None
    if _KAN_AVAILABLE and "kan" in methods:
        t0 = time.time()
        kan = _train_kan(x_train, y_train, kan_steps, kan_lr, pearson_weight, device=device)
        kan_train_sec = time.time() - t0
        with torch.no_grad():
            y_val_pred_k = kan(torch.from_numpy(x_val).float().to(device)).cpu().numpy()
        kan_val_mse = float(np.mean((y_val_pred_k - y_val) ** 2))
        kan_val_corr = float(np.corrcoef(y_val_pred_k.flatten(), y_val.flatten())[0, 1]) if y_val.size > 1 else 0.0
        kan_state = kan.state_dict()
        kan_params = _count_params(kan)

    # Composite score for feature-pair matching (top-K space only)
    # MLP excluded when using full-dim inputs (can't produce a K×K score matrix).
    active_score_mats = []
    if "cca" in methods:
        active_score_mats.append(s_cca)
    if "svcca" in methods:
        active_score_mats.append(s_svcca)
    if "procrustes" in methods and r is not None:
        active_score_mats.append(s_pro)
    if _mnn_available:
        active_score_mats.append(s_mnn)
    if "mlp" in methods and not using_full:
        active_score_mats.append(s_mlp)
    s_comp = sum(active_score_mats) / len(active_score_mats) if active_score_mats else _corr_matrix(x_s, y_s)

    # ── Pair extraction: MNN-first, Hungarian fallback ─────────────────────
    # MNN pairs are one-to-one by construction (argmax in both directions) and
    # only keep features that genuinely prefer each other — the standard method
    # for cross-model feature matching (Conneau et al. 2018).
    # Hungarian maximises the global total score which forces 1:1 assignment
    # even for unmatched features, diluting the output with weak pairs.
    # We use MNN pairs when available; fall back to Hungarian otherwise.
    pairs = []
    if _mnn_available:
        # Extract every (i,j) where MNN score > 0
        mnn_i, mnn_j = np.nonzero(s_mnn)
        for i, j in zip(mnn_i.tolist(), mnn_j.tolist()):
            pairs.append({
                "a_idx": int(i),
                "b_idx": int(j),
                "score": float(s_comp[i, j]),
                "cca": float(s_cca[i, j]),
                "svcca": float(s_svcca[i, j]),
                "procrustes": float(s_pro[i, j]),
                "mnn": float(s_mnn[i, j]),
                "mlp": float(s_mlp[i, j]),
            })
        # Sort by composite score descending for consistent output
        pairs.sort(key=lambda p: p["score"], reverse=True)
        print(f"[step5] MNN extracted {len(pairs)} bidirectionally-confirmed pairs")
    else:
        # Fallback: Hungarian assignment on composite score matrix
        row_ind, col_ind = linear_sum_assignment(-s_comp)
        for i, j in zip(row_ind, col_ind):
            pairs.append({
                "a_idx": int(i),
                "b_idx": int(j),
                "score": float(s_comp[i, j]),
                "cca": float(s_cca[i, j]),
                "svcca": float(s_svcca[i, j]),
                "procrustes": float(s_pro[i, j]),
                "mnn": float(s_mnn[i, j]),
                "mlp": float(s_mlp[i, j]),
            })
        print(f"[step5] Hungarian assigned {len(pairs)} pairs (MNN not available)")

    return {
        "pairs": pairs,
        "s_cca": s_cca,
        "s_svcca": s_svcca,
        "s_pro": s_pro,
        "s_mnn": s_mnn,
        "s_mlp": s_mlp,
        "s_comp": s_comp,
        "cca_weights": w,
        "procrustes_r": r,
        "mlp_state": mlp.state_dict(),
        "rev_mlp_state": rev_mlp_state,
        "mlp_in_dim": x_full.shape[1] if using_full else x_train.shape[1],
        "mlp_out_dim": y_full.shape[1] if using_full else y_train.shape[1],
        "mlp_src_mean": xf_mean,
        "mlp_src_std": xf_std,
        "mlp_tgt_mean": yf_mean,
        "mlp_tgt_std": yf_std,
        "mlp_val_mse": mlp_val_mse,
        "mlp_val_pearson": mlp_val_corr,
        "kan_state": kan_state,
        "kan_val_mse": kan_val_mse,
        "kan_val_pearson": kan_val_corr,
        "kan_params": kan_params,
        "kan_train_sec": kan_train_sec,
    }


def _pair_worker(worker_args):
    """Align a single model pair in a worker process (spawn-safe).

    Computes SAE activation matrices on-the-fly (no disk caching) for MLP
    training.  Selects the top-`max_mlp_features` ever-active features (by
    cumulative absolute activation) to keep memory bounded — for EF128 models
    this avoids holding a 524K-wide dense matrix in RAM.  With the default
    50 000 cap, each matrix is ~2 GB (10K × 50K × float32), keeping peak worker
    RAM under 10 GB per pair even for the largest models.

    Top-K matrices (loaded from cached .npy files) are still used for CCA,
    Procrustes, SVCCA, and Mutual-NN pair-matching scores.
    """
    a, b, gpu_idx, mat_a, mat_b, sae_info_a, sae_info_b, pair_params = worker_args
    print(f"[step5/worker] START pair {a}→{b} gpu_idx={gpu_idx}")
    _mem(f"worker_start:{a}→{b}")
    print(f"[step5/worker] top-K matrices: A={mat_a.shape} B={mat_b.shape} (shared from main process)")
    _mem(f"worker_start:{a}→{b}")
    if torch.cuda.is_available():
        dev_count = torch.cuda.device_count()
        actual_idx = gpu_idx if gpu_idx < dev_count else 0
        torch.cuda.set_device(actual_idx)  # pin this thread to its assigned GPU
        dev = torch.device(f"cuda:{actual_idx}")
    else:
        dev = torch.device("cpu")

    max_mlp_features = pair_params.get("max_mlp_features", 4_096)
    max_passages = pair_params.get("max_passages", 100_000)

    def _compute_capped(model_name, info):
        """Compute SAE activations, keeping only top-max_mlp_features ever-active
        feature columns (by summed absolute activation across corpus).  This
        caps memory and MLP width while preserving the most informative features.
        Returns (matrix, feature_indices) where matrix is (n_passages × M)."""
        _mem(f"capped_start:{model_name}", device=dev)
        acts = _load_activations(model_name)
        # Optionally subsample rows — disabled by default (max_passages=0) for NeurIPS quality.
        # float16 storage halves RAM so the full corpus fits on an A100 without this.
        original_n = acts.shape[0]
        if 0 < max_passages < original_n:
            rng = np.random.default_rng(42)
            idx = rng.choice(original_n, size=int(max_passages), replace=False)
            idx.sort()
            acts = acts[idx]
            print(f"[step5/mlp] {model_name}: subsampled {len(acts):,}/{original_n:,} passages")
            _mem(f"after_subsample:{model_name} rows={len(acts)}", device=dev)

        # Hard memory guard: auto-clamp max_mlp_features so the float16 memmap
        # stays under 8 GB (avoids OOM if user passes a large --max-mlp-features).
        # Use a local _cap so Python doesn't treat max_mlp_features as a local
        # variable (which would cause UnboundLocalError on the first read).
        _mlp_gb_limit = 8.0
        _max_feats_for_limit = int(_mlp_gb_limit * 1e9 / (acts.shape[0] * 2))  # float16 = 2 bytes
        _cap_requested = max_mlp_features if max_mlp_features > 0 else _max_feats_for_limit
        if _cap_requested > _max_feats_for_limit:
            print(f"[step5/mlp] {model_name}: clamping max_mlp_features "
                  f"{_cap_requested:,} → {_max_feats_for_limit:,} "
                  f"(8 GB / ({acts.shape[0]:,} rows × 2 B) = {_max_feats_for_limit:,} cols)")
            _effective_cap = _max_feats_for_limit
        else:
            _effective_cap = _cap_requested

        print(f"[step5/mlp] {model_name}: loading SAE checkpoint {info['ckpt_path']}")
        _mem(f"before_sae_load:{model_name}", device=dev)
        sae = TopKSAE(info["hidden_dim"], info["n_features"], info["sae_topk"]).to(dev)
        try:
            sae.load_state_dict(torch.load(info["ckpt_path"], map_location=dev, weights_only=True))
        except TypeError:
            sae.load_state_dict(torch.load(info["ckpt_path"], map_location=dev))
        _mem(f"after_sae_load:{model_name} hidden={info['hidden_dim']} n_feats={info['n_features']}", device=dev)

        n_feats = info["n_features"]
        cap = min(int(_effective_cap), n_feats)

        if cap >= n_feats:
            # Small enough to use all features
            idx_all = np.arange(n_feats, dtype=np.int64)
            mat_full = _compute_feature_matrix(acts, sae, idx_all, batch_size=512)
            active_idx = idx_all
        else:
            # Two-pass: first pass computes per-feature cumulative activation sum
            # to identify the top-cap ever-active features; second pass extracts
            # only those columns (avoids allocating the full n_feats-wide matrix).
            print(f"[step5/mlp] {model_name}: selecting top-{cap} of {n_feats} features by activation")
            active_idx = _select_topk_features(acts, sae, n_feats, cap, batch_size=512)
            _mem(f"after_topk_select:{model_name} cap={cap}", device=dev)
            mat_full = _compute_feature_matrix(acts, sae, active_idx, batch_size=512)

        del acts  # free raw activations immediately — no longer needed after matrix fill
        sae.cpu()
        del sae
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        gb = mat_full.nbytes / 1e9
        print(f"[step5/mlp] {model_name}: matrix {mat_full.shape} memmap float16 ({gb:.1f} GB on disk)")
        _mem(f"capped_done:{model_name}", device=dev)
        return mat_full, active_idx

    print(f"[step5] {a}→{b}: computing capped SAE activations for MLP training (cap={max_mlp_features:,})")
    try:
        mat_a_full, active_idx_a = _compute_capped(a, sae_info_a)
        _mem(f"worker_after_capped_A:{a}", device=dev)
        mat_b_full, active_idx_b = _compute_capped(b, sae_info_b)
        _mem(f"worker_after_capped_B:{b}", device=dev)

        # Strip worker-only keys from pair_params before passing to _align_pair
        pp = {k: v for k, v in pair_params.items() if k not in ("max_mlp_features", "max_passages")}
        _mem(f"worker_before_align:{a}→{b}", device=dev)
        result = _align_pair(mat_a, mat_b, x_full=mat_a_full, y_full=mat_b_full, device=dev, **pp)
        result["src_active_idx"] = active_idx_a
        result["tgt_active_idx"] = active_idx_b
        _mem(f"worker_after_align:{a}→{b}", device=dev)
    except RuntimeError as _oom_err:
        # Catch CUDA/CPU OOM and print full diagnostic before re-raising
        if "out of memory" in str(_oom_err).lower() or "oom" in str(_oom_err).lower():
            print(f"\n{'='*70}")
            print(f"[OOM] DETECTED in pair {a}→{b}")
            _mem(f"OOM_DUMP pair={a}→{b}", device=dev)
            if torch.cuda.is_available():
                try:
                    print(torch.cuda.memory_summary(device=dev, abbreviated=False))
                except Exception:
                    pass
            print(traceback.format_exc())
            print(f"{'='*70}\n")
        raise

    # Clean up memmap temp files for this pair
    del mat_a_full, mat_b_full
    for _f in list(_MEMMAP_TMPFILES):
        try:
            os.unlink(_f)
            _MEMMAP_TMPFILES.remove(_f)
        except OSError:
            pass

    print(f"[step5/worker] DONE pair {a}→{b}")
    _mem(f"worker_done:{a}→{b}", device=dev)
    return a, b, result


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt2", "gemma", "llama"])
    parser.add_argument("--expansion-factor", "--ef", type=int, default=config.SAE_EXPANSION_FACTOR,
                        dest="expansion_factor", help="SAE expansion factor to use for all models (default fallback)")
    parser.add_argument("--model-efs", default="", dest="model_efs",
                        help="Per-model EF overrides, format: model1:ef1,model2:ef2 (e.g. gpt2:64,llama:128)")
    parser.add_argument("--topk", type=int, default=config.TOP_FEATURES_FOR_ALIGNMENT)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cca-components", type=int, default=128)
    parser.add_argument("--mlp-hidden", type=int, default=8192,
                        help="MLP bridge hidden layer width. Default 8192 (recommended for EF≥64).")
    parser.add_argument("--max-mlp-features", type=int, default=4_096, dest="max_mlp_features",
                        help="Cap on ever-active SAE features used for MLP bridge input/output. "
                             "Prevents OOM: default 4096 gives ~800 MB memmap per model at 100 K passages. "
                             "Set to 0 to use all features (will OOM for EF128 models). "
                             "Only affects the MLP bridge; CCA/Procrustes always use the top-K label set.")
    parser.add_argument("--max-passages", type=int, default=100_000, dest="max_passages",
                        help="Maximum passage rows used for the MLP bridge matrix. "
                             "Randomly subsampled if corpus is larger. "
                             "Default 100 000 — sufficient for bridge quality, avoids 74 GB memmaps. "
                             "Set to 0 to use the full corpus (may OOM). "
                             "CCA/Procrustes/SVCCA always use the full cached top-K matrices.")
    parser.add_argument("--svcca-sv", type=int, default=64, dest="svcca_sv",
                        help="Number of top singular vectors used in SVCCA projection (default 64).")
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-batch", type=int, default=256)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-dropout", type=float, default=0.1, dest="mlp_dropout",
                        help="Dropout probability in MLP bridge hidden layer. Default 0.1. Set to 0 to disable.")
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-5, dest="mlp_weight_decay",
                        help="Adam weight_decay for MLP bridge. Default 1e-5. Set to 0 to disable.")
    parser.add_argument("--pearson-weight", type=float, default=0.1)
    parser.add_argument("--kan-steps", type=int, default=500)
    parser.add_argument("--kan-lr", type=float, default=1e-3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-align", action="store_true", help="recompute alignment but reuse cached feature matrices")
    parser.add_argument("--run-id", default="", help="Optional run identifier; produces aligned_pairs_{run_id}.jsonl to avoid overwrites")
    parser.add_argument("--confidence-thresh", type=float, default=config.ALIGNMENT_CONFIDENCE_THRESHOLD,
                        dest="confidence_thresh", help="Min composite score for a pair to be written to aligned_pairs.jsonl")
    parser.add_argument("--train-mode", default="paired", choices=["paired", "independent"],
                        dest="train_mode", help="paired = both models share concept sets; independent = allow unmatched sets")
    parser.add_argument("--specificity", type=float, default=0.0,
                        help="Min per-feature specificity score (from feature_labels.json) to include. 0 = off.")
    parser.add_argument("--headroom", type=float, default=1.5,
                        help="Candidate pool multiplier for top-K feature selection (e.g. 1.5 = select top 1.5×K then trim).")
    parser.add_argument("--methods", default="cca,procrustes,mlp,mutual_nn",
                        help="Comma-separated alignment methods to use in score composite: cca,procrustes,mlp,kan,mutual_nn")
    parser.add_argument("--num-gpus", type=int, default=0, dest="num_gpus",
                        help="Number of GPUs for parallel pair alignment. 0 = use all available.")
    args = parser.parse_args()

    global WANDB_RUN
    # Only attempt live W&B logging if explicitly opted in via WANDB_MODE=online.
    # Default to disabled to avoid hanging on team-only accounts (403 timeout).
    _wandb_mode = os.getenv("WANDB_MODE", "disabled")
    try:
        WANDB_RUN = wandb.init(
            project=os.getenv("WANDB_PROJECT", "sae_align"),
            entity=os.getenv("WANDB_ENTITY") or None,
            name=os.getenv("WANDB_RUN_NAME", "step5_align_features"),
            mode=_wandb_mode,
            reinit="finish_previous",
        )
        if _wandb_mode != "disabled":
            wandb.config.update({"topk": args.topk, "cca_components": args.cca_components, "mlp_hidden": args.mlp_hidden, "mlp_epochs": args.mlp_epochs, "mlp_lr": args.mlp_lr, "kan_steps": args.kan_steps, "kan_lr": args.kan_lr, "pearson_weight": args.pearson_weight, "confidence_thresh": args.confidence_thresh, "methods": args.methods, "num_gpus": args.num_gpus})
    except Exception as e:
        print(f"[wandb] Disabling W&B logging ({e})")
        WANDB_RUN = wandb.init(mode="disabled")

    set_seed(42)
    os.makedirs(config.ALIGNMENT_DIR, exist_ok=True)
    os.makedirs(config.FEATURES_DIR, exist_ok=True)

    # Build per-model feature matrices
    feature_mats = {}
    feature_indices = {}
    feature_mat_paths = {}
    # SAE info passed to workers so they can compute full activation matrices
    sae_infos = {}
    # Parse per-model EF overrides: "gpt2:64,llama:128"
    model_efs: dict = {}
    if args.model_efs:
        for token in args.model_efs.split(","):
            token = token.strip()
            if ":" in token:
                mname, efval = token.rsplit(":", 1)
                try:
                    model_efs[mname.strip()] = int(efval.strip())
                except ValueError:
                    pass
    for model_name in args.models:
        ef = model_efs.get(model_name, args.expansion_factor)
        ef_tag = "" if ef == config.SAE_EXPANSION_FACTOR else f"_ef{ef}"
        model_cfg = config.MODELS[model_name]
        hidden_dim = model_cfg["hidden_dim"]
        n_features = hidden_dim * ef
        sae_topk = model_cfg["sae_topk"]
        if ef != config.SAE_EXPANSION_FACTOR:
            base_sparsity = sae_topk / (hidden_dim * config.SAE_EXPANSION_FACTOR)
            sae_topk = max(int(n_features * base_sparsity), 32)

        ckpt_path = _find_sae_checkpoint(model_name, config.SAE_DIR, ef)

        # Use EF-specific feature labels if they exist, else fall back to default
        ef_labels_path = os.path.join(config.FEATURES_DIR, f"{model_name}{ef_tag}_feature_labels.json")
        default_labels_path = os.path.join(config.FEATURES_DIR, f"{model_name}_feature_labels.json")
        labels_path = ef_labels_path if os.path.exists(ef_labels_path) else default_labels_path
        selected = np.array([], dtype=np.int64)
        feature_labels_data = {}
        if os.path.exists(labels_path):
            feature_labels_data = json.load(open(labels_path, "r", encoding="utf-8"))
            sel_raw = feature_labels_data.get("selected_feature_ids", [])
            if sel_raw:
                selected = np.array(sel_raw, dtype=np.int64)
                print(f"[step5] {model_name}: using {len(selected)} selected features for alignment")
        use_selected = selected.size > 0
        k = int(selected.size) if use_selected else args.topk

        idx_path = os.path.join(config.FEATURES_DIR, f"{model_name}{ef_tag}_top{k}_feature_idx.npy")
        mat_path = os.path.join(config.FEATURES_DIR, f"{model_name}{ef_tag}_top{k}_feature_acts.npy")

        # ── Fast path: load from cache without touching activations or SAE ──
        idx, mat = None, None
        if os.path.exists(idx_path) and os.path.exists(mat_path) and (not args.force or args.force_align):
            idx = np.load(idx_path)
            mat = np.load(mat_path)
            if idx.size == 0 or mat.size == 0:
                print(f"[step5] Empty cache for {model_name}; recomputing.")
                idx, mat = None, None
            else:
                print(f"[step5] {model_name}: loaded cached top-K matrices idx={idx.shape} mat={mat.shape} (skipping activation load)")

        # ── Slow path: load activations + SAE and compute matrices ──
        if idx is None:
            acts = _load_activations(model_name)
            _extract_dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            _mem(f"main_before_sae_load:{model_name} hidden={hidden_dim} n_feats={n_features}", device=_extract_dev)
            sae = TopKSAE(hidden_dim, n_features, sae_topk).to(_extract_dev)
            try:
                sae.load_state_dict(torch.load(ckpt_path, map_location=_extract_dev, weights_only=True))
            except TypeError:
                sae.load_state_dict(torch.load(ckpt_path, map_location=_extract_dev))
            _mem(f"main_after_sae_load:{model_name}", device=_extract_dev)

            if use_selected:
                idx = np.ascontiguousarray(selected)
            else:
                candidate_k = max(int(args.topk * max(args.headroom, 1.0)), args.topk)
                idx = _select_topk_features(acts, sae, n_features, candidate_k, args.batch_size)
                if args.specificity > 0.0:
                    spec_scores = feature_labels_data.get("specificity_scores",
                                  feature_labels_data.get("confidence_scores", {}))
                    if isinstance(spec_scores, dict) and spec_scores:
                        filtered = np.array([i for i in idx
                                             if float(spec_scores.get(str(i), 0.0)) >= args.specificity],
                                            dtype=np.int64)
                        if filtered.size > 0:
                            idx = filtered
                        else:
                            print(f"[WARNING] Specificity filter removed all features for {model_name}; ignoring.")
                idx = np.ascontiguousarray(idx[:args.topk])
            mat = _compute_feature_matrix(acts, sae, idx, args.batch_size)
            np.save(idx_path, idx)
            np.save(mat_path, mat)

            # Free GPU VRAM after extraction — alignment workers will claim their own devices
            sae.cpu()
            del sae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        k = idx.size  # re-sync after any headroom/specificity trimming
        if mat.shape[1] != k:
            raise ValueError(f"Feature matrix shape mismatch for {model_name}: {mat.shape} (expected (*, {k}))")

        feature_mat_paths[model_name] = mat_path
        feature_indices[model_name] = idx
        feature_mats[model_name] = mat
        sae_infos[model_name] = {
            "ckpt_path": ckpt_path,
            "hidden_dim": hidden_dim,
            "n_features": n_features,
            "sae_topk": sae_topk,
        }
        _mem(f"main_after_model:{model_name} mat={mat.shape}")

    aligned_path = os.path.join(config.ALIGNMENT_DIR, f"aligned_pairs{'_' + args.run_id if args.run_id else ''}.jsonl")
    aligned_path_tmp = aligned_path + ".tmp"  # written atomically; renamed on success
    summary_path = os.path.join(config.ALIGNMENT_DIR, f"alignment_summary{'_' + args.run_id if args.run_id else ''}.json")

    # --force / --force-align: write to a .tmp file first, rename to final only
    # when the entire run succeeds — protects existing data if the job crashes.
    # Plain resume (no flags): append directly to the live file.
    if args.force or args.force_align:
        with open(aligned_path_tmp, "w", encoding="utf-8") as _:
            pass
        _active_aligned_path = aligned_path_tmp
    else:
        if not os.path.exists(aligned_path):
            with open(aligned_path, "w", encoding="utf-8") as _:
                pass
        _active_aligned_path = aligned_path

    summary = {
        "run_id": args.run_id or "",
        "aligned_path": os.path.basename(aligned_path),
        "topk": args.topk,
        "threshold": args.confidence_thresh,
        "cca_components": args.cca_components,
        "mlp_hidden": args.mlp_hidden,
        "mlp_epochs": args.mlp_epochs,
        "mlp_batch": args.mlp_batch,
        "mlp_lr": args.mlp_lr,
        "pearson_weight": args.pearson_weight,
        "kan_steps": args.kan_steps,
        "kan_lr": args.kan_lr,
        "model_efs": {m: model_efs.get(m, args.expansion_factor) for m in args.models},
        "pairs": [],
    }

    comparison = []

    # ── Multi-GPU parallel alignment ──────────────────────────────────────
    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpus = min(args.num_gpus if args.num_gpus > 0 else n_cuda, n_cuda) or 1
    # Normalise method names: accept both "mutual-nn" and "mutual_nn", "svcca" etc.
    _raw_methods = [m.strip().lower().replace("-", "_") for m in args.methods.split(",") if m.strip()]
    methods_set = frozenset(_raw_methods)
    max_mlp = args.max_mlp_features if args.max_mlp_features > 0 else 10_000_000
    pair_params = dict(
        cca_components=args.cca_components,
        mlp_hidden=args.mlp_hidden,
        mlp_epochs=args.mlp_epochs,
        mlp_batch=args.mlp_batch,
        mlp_lr=args.mlp_lr,
        mlp_dropout=args.mlp_dropout,
        mlp_weight_decay=args.mlp_weight_decay,
        pearson_weight=args.pearson_weight,
        kan_steps=args.kan_steps,
        kan_lr=args.kan_lr,
        svcca_sv=args.svcca_sv,
        methods=methods_set,
        max_mlp_features=max_mlp,
        max_passages=args.max_passages if args.max_passages > 0 else 10_000_000,
    )

    models = args.models
    all_pairs = [(models[i], models[j])
                 for i in range(len(models))
                 for j in range(i + 1, len(models))]

    print(f"[step5] Aligning {len(all_pairs)} pairs using {n_gpus} GPU(s).")

    # ── Per-pair checkpoint dir — skip already-completed pairs on restart ──
    ckpt_dir = os.path.join(config.ALIGNMENT_DIR, "pair_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    def _ckpt_path(a, b):
        return os.path.join(ckpt_dir, f"{a}__{b}.done")

    pending = [(a, b) for a, b in all_pairs if not os.path.exists(_ckpt_path(a, b))]
    skipped = len(all_pairs) - len(pending)
    if skipped:
        print(f"[step5] Skipping {skipped} already-completed pair(s) (checkpoint found).")

    # --force-align: rescore all pairs using existing feature matrices + saved
    # MLPs, without retraining.  Pairs with .done checkpoints are re-scored
    # cheaply (CCA+MNN only, no MLP training) and written to aligned_pairs.jsonl.
    rescore_pairs = []
    if args.force_align:
        rescore_pairs = [(a, b) for a, b in all_pairs if os.path.exists(_ckpt_path(a, b))]
        if rescore_pairs:
            print(f"[step5] --force-align: will rescore {len(rescore_pairs)} completed pair(s) "
                  f"(CCA+MNN only, no MLP retraining).")

    # Assign GPUs round-robin to pending pairs
    # Pass actual numpy arrays instead of file paths to avoid redundant np.load
    # in each worker (ThreadPoolExecutor shares the same process, so this is
    # zero-copy — workers read the same memory the main process already loaded).
    worker_args_pending = [
        (a, b, pi % n_gpus, feature_mats[a], feature_mats[b],
         sae_infos[a], sae_infos[b], pair_params)
        for pi, (a, b) in enumerate(pending)
    ]

    _write_lock = threading.Lock()

    def _save_pair_result(a, b, result):
        """Write MLPs, JSONL entries, and summary for one completed pair."""
        mlp_path = os.path.join(config.ALIGNMENT_DIR, f"mlp_{a}_to_{b}.pt")
        torch.save(result["mlp_state"], mlp_path)
        print(f"[step5] Saved forward bridge ({result['mlp_in_dim']}-d \u2192 {result['mlp_out_dim']}-d): {os.path.basename(mlp_path)}")
        # Save feature indices and normalisation stats so step7 translation mode
        # can reconstruct exactly the same feature subset the MLP was trained on.
        if result.get("src_active_idx") is not None:
            np.save(mlp_path.replace(".pt", "_src_idx.npy"), result["src_active_idx"])
            np.save(mlp_path.replace(".pt", "_tgt_idx.npy"), result["tgt_active_idx"])
        if result.get("mlp_src_mean") is not None:
            np.savez(mlp_path.replace(".pt", "_src_stats.npz"),
                     mean=result["mlp_src_mean"], std=result["mlp_src_std"])
            np.savez(mlp_path.replace(".pt", "_tgt_stats.npz"),
                     mean=result["mlp_tgt_mean"], std=result["mlp_tgt_std"])

        rev_mlp_path = os.path.join(config.ALIGNMENT_DIR, f"mlp_{b}_to_{a}.pt")
        if result.get("rev_mlp_state") is not None:
            torch.save(result["rev_mlp_state"], rev_mlp_path)
            print(f"[step5] Saved reverse bridge ({result['mlp_out_dim']}-d \u2192 {result['mlp_in_dim']}-d): {os.path.basename(rev_mlp_path)}")
            # Save reverse bridge indices/stats (src and tgt swapped)
            if result.get("tgt_active_idx") is not None:
                np.save(rev_mlp_path.replace(".pt", "_src_idx.npy"), result["tgt_active_idx"])
                np.save(rev_mlp_path.replace(".pt", "_tgt_idx.npy"), result["src_active_idx"])
            if result.get("mlp_tgt_mean") is not None:
                np.savez(rev_mlp_path.replace(".pt", "_src_stats.npz"),
                         mean=result["mlp_tgt_mean"], std=result["mlp_tgt_std"])
                np.savez(rev_mlp_path.replace(".pt", "_tgt_stats.npz"),
                         mean=result["mlp_src_mean"], std=result["mlp_src_std"])
        else:
            print(f"[WARNING] No reverse MLP state returned for {b}\u2192{a}; skipping.")

        kan_path = None
        kan_plot_path = None
        if result.get("kan_state") is not None:
            kan_path = os.path.join(config.ALIGNMENT_DIR, f"kan_{a}_to_{b}.pt")
            torch.save(result["kan_state"], kan_path)
            try:
                import matplotlib.pyplot as plt
                kan = KAN([feature_mats[a].shape[1], 100, feature_mats[b].shape[1]])
                kan.load_state_dict(result["kan_state"])
                kan.plot()
                kan_plot_path = os.path.join(config.ALIGNMENT_DIR, f"kan_{a}_to_{b}_plot.png")
                plt.savefig(kan_plot_path, dpi=150, bbox_inches="tight")
                plt.close()
            except Exception:
                kan_plot_path = None

        kept = 0
        with open(_active_aligned_path, "a", encoding="utf-8") as f:
            for p in result["pairs"]:
                if p["score"] < args.confidence_thresh:
                    continue
                kept += 1
                a_feat = int(feature_indices[a][p["a_idx"]])
                b_feat = int(feature_indices[b][p["b_idx"]])
                entry = {
                    "a_model": a, "b_model": b,
                    "a_feature": a_feat, "b_feature": b_feat,
                    "score": p["score"], "cca": p["cca"],
                    "svcca": p.get("svcca", 0.0), "procrustes": p["procrustes"],
                    "mnn": p.get("mnn", 0.0), "mlp": p["mlp"],
                }
                f.write(json.dumps(entry) + "\n")
                entry_rev = {
                    "a_model": b, "b_model": a,
                    "a_feature": b_feat, "b_feature": a_feat,
                    "score": p["score"], "cca": p["cca"],
                    "svcca": p.get("svcca", 0.0), "procrustes": p["procrustes"],
                    "mnn": p.get("mnn", 0.0), "mlp": p["mlp"],
                }
                f.write(json.dumps(entry_rev) + "\n")

        summary["pairs"].append({"a_model": a, "b_model": b, "n_pairs": kept, "mlp_path": os.path.basename(mlp_path)})
        summary["pairs"].append({"a_model": b, "b_model": a, "n_pairs": kept, "mlp_path": os.path.basename(rev_mlp_path)})

    def _run_pair_and_save(worker_args):
        """Run _pair_worker and immediately write results + checkpoint."""
        a, b, *_ = worker_args
        result_tuple = _pair_worker(worker_args)
        a, b, result = result_tuple
        with _write_lock:
            _save_pair_result(a, b, result)
            # Write checkpoint marker so this pair is skipped on any future restart
            with open(_ckpt_path(a, b), "w") as _cf:
                _cf.write(json.dumps({"a": a, "b": b, "ts": time.time()}))
            print(f"[step5] Checkpoint saved for {a}→{b}")
        return a, b, result

    # Use ThreadPoolExecutor so all threads share the same process and CUDA context.
    # Each thread calls torch.cuda.set_device(gpu_idx) inside _pair_worker, which
    # is reliable within a single process. multiprocessing.spawn breaks in remote
    # job runners that restrict subprocess creation.
    if len(worker_args_pending) == 0:
        print("[step5] All pairs already checkpointed — nothing to compute.")
        pair_results = []
    elif n_gpus > 1 and len(worker_args_pending) > 1:
        with ThreadPoolExecutor(max_workers=min(n_gpus, len(worker_args_pending))) as ex:
            futures = {ex.submit(_run_pair_and_save, wa): wa for wa in worker_args_pending}
            pair_results = []
            for fut in as_completed(futures):
                try:
                    pair_results.append(fut.result())
                except Exception as exc:
                    wa = futures[fut]
                    print(f"[step5][ERROR] pair {wa[0]}→{wa[1]} failed: {exc}")
    else:
        pair_results = [_run_pair_and_save(wa) for wa in worker_args_pending]

    # ── Rescore completed pairs when --force-align ─────────────────────────
    # Re-run CCA + MNN scoring on existing feature matrices without retraining
    # MLPs.  Writes updated entries to aligned_pairs.jsonl.
    if rescore_pairs:
        _rescore_dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        _rescore_methods = methods_set
        for a, b in rescore_pairs:
            print(f"[step5] Rescoring {a}→{b} (CCA+MNN, no retraining)...")
            x_s = feature_mats[a].astype(np.float32)
            y_s = feature_mats[b].astype(np.float32)
            n_comp = min(pair_params["cca_components"], x_s.shape[1], y_s.shape[1])
            _, s_cca, _x_w_cca, _y_w_cca = _cca_gpu(x_s, y_s, n_comp, _rescore_dev)
            s_svcca = np.zeros((x_s.shape[1], y_s.shape[1]), dtype=np.float32)
            s_pro = np.zeros((x_s.shape[1], y_s.shape[1]), dtype=np.float32)
            s_mlp = np.zeros((x_s.shape[1], y_s.shape[1]), dtype=np.float32)
            _mnn_ok = ("mutual_nn" in _rescore_methods) and _x_w_cca is not None
            s_mnn = _mutual_nn_scores(_x_w_cca, _y_w_cca, _rescore_dev) if _mnn_ok else np.zeros_like(s_cca)

            active = []
            if "cca" in _rescore_methods: active.append(s_cca)
            if _mnn_ok: active.append(s_mnn)
            s_comp_r = sum(active) / len(active) if active else s_cca

            mlp_path = os.path.join(config.ALIGNMENT_DIR, f"mlp_{a}_to_{b}.pt")
            rev_mlp_path = os.path.join(config.ALIGNMENT_DIR, f"mlp_{b}_to_{a}.pt")

            kept = 0
            with open(_active_aligned_path, "a", encoding="utf-8") as f:
                if _mnn_ok:
                    mnn_i, mnn_j = np.nonzero(s_mnn)
                    _pairs_r = sorted(
                        [{"a_idx": int(i), "b_idx": int(j),
                          "score": float(s_comp_r[i, j]),
                          "cca": float(s_cca[i, j]),
                          "svcca": 0.0, "procrustes": 0.0,
                          "mnn": float(s_mnn[i, j]), "mlp": 0.0}
                         for i, j in zip(mnn_i.tolist(), mnn_j.tolist())],
                        key=lambda p: p["score"], reverse=True)
                    print(f"[step5] {a}→{b}: MNN extracted {len(_pairs_r)} pairs")
                else:
                    ri, ci = linear_sum_assignment(-s_comp_r)
                    _pairs_r = [{"a_idx": int(i), "b_idx": int(j),
                                 "score": float(s_comp_r[i, j]),
                                 "cca": float(s_cca[i, j]),
                                 "svcca": 0.0, "procrustes": 0.0,
                                 "mnn": 0.0, "mlp": 0.0}
                                for i, j in zip(ri, ci)]
                for p in _pairs_r:
                    if p["score"] < args.confidence_thresh:
                        continue
                    kept += 1
                    a_feat = int(feature_indices[a][p["a_idx"]])
                    b_feat = int(feature_indices[b][p["b_idx"]])
                    entry = {"a_model": a, "b_model": b,
                             "a_feature": a_feat, "b_feature": b_feat,
                             "score": p["score"], "cca": p["cca"],
                             "svcca": 0.0, "procrustes": p["procrustes"],
                             "mnn": p["mnn"], "mlp": p["mlp"]}
                    f.write(json.dumps(entry) + "\n")
                    entry_rev = {"a_model": b, "b_model": a,
                                 "a_feature": b_feat, "b_feature": a_feat,
                                 "score": p["score"], "cca": p["cca"],
                                 "svcca": 0.0, "procrustes": p["procrustes"],
                                 "mnn": p["mnn"], "mlp": p["mlp"]}
                    f.write(json.dumps(entry_rev) + "\n")
            print(f"[step5] {a}→{b}: {kept} pairs written (rescore)")
            summary["pairs"].append({"a_model": a, "b_model": b, "n_pairs": kept, "mlp_path": os.path.basename(mlp_path)})
            summary["pairs"].append({"a_model": b, "b_model": a, "n_pairs": kept, "mlp_path": os.path.basename(rev_mlp_path)})

    # ── Comparison / summary entries for all completed pairs ───────────────
    for a, b, result in pair_results:
        # Comparison entry \u2014 use actual MLP in/out dims from result
        mlp_params = _count_params(_build_mlp_model(result.get("mlp_in_dim", feature_mats[a].shape[1]), result.get("mlp_out_dim", feature_mats[b].shape[1]), args.mlp_hidden))
        comp = {
            "pair": f"{a}-{b}",
            "mlp": {
                "val_mse": result.get("mlp_val_mse"),
                "val_pearson": result.get("mlp_val_pearson"),
                "n_parameters": mlp_params,
                "train_time_sec": None,
            },
            "kan": {
                "val_mse": result.get("kan_val_mse"),
                "val_pearson": result.get("kan_val_pearson"),
                "n_parameters": result.get("kan_params"),
                "train_time_sec": result.get("kan_train_sec"),
            },
        }
        # Winner and advantage
        if comp["kan"]["val_mse"] is not None and comp["mlp"]["val_mse"] is not None:
            if comp["kan"]["val_mse"] < comp["mlp"]["val_mse"]:
                comp["winner"] = "kan"
            elif comp["kan"]["val_mse"] > comp["mlp"]["val_mse"]:
                comp["winner"] = "mlp"
            else:
                comp["winner"] = "tie"
            comp["kan_advantage_pct"] = 100.0 * (comp["mlp"]["val_mse"] - comp["kan"]["val_mse"]) / max(comp["mlp"]["val_mse"], 1e-8)
        else:
            comp["winner"] = "mlp"
            comp["kan_advantage_pct"] = 0.0
        comparison.append(comp)

        print(f"Aligned {a}->{b}: kept {kept} pairs")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    comparison_path = os.path.join(config.ALIGNMENT_DIR, "kan_vs_mlp_comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Atomic swap: rename .tmp → final only now that everything succeeded.
    # If the job crashed earlier, aligned_pairs.jsonl is untouched.
    if _active_aligned_path != aligned_path:
        os.replace(aligned_path_tmp, aligned_path)
        print(f"[step5] Atomically replaced {os.path.basename(aligned_path)} with rescored output.")

    print(f"[step5] Output: {aligned_path}")
    log_run("step5_align_features.py", start_time, "success")
    return 0


def save_mlp_sidecars():
    """Regenerate sidecar index/stat files for all existing MLP bridges without retraining.
    Run with: python step5_align_features.py --save-mlp-sidecars --models ...
    Reads each mlp_A_to_B.pt, infers feature count from weight dims,
    recomputes top-k ever-active feature indices and normalisation stats,
    saves _src_idx.npy, _tgt_idx.npy, _src_stats.npz, _tgt_stats.npz."""
    import glob as _glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(config.MODELS.keys()))
    parser.add_argument("--model-efs", default="", dest="model_efs")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-passages", type=int, default=100_000, dest="max_passages")
    args = parser.parse_args()

    model_efs: dict = {}
    if args.model_efs:
        for token in args.model_efs.split(","):
            if ":" in token:
                mn, ev = token.rsplit(":", 1)
                try: model_efs[mn.strip()] = int(ev.strip())
                except ValueError: pass

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Cache: model_name → (active_idx, mean, std)  to avoid recomputing per pair
    _cache: dict = {}

    def _get_model_idx_stats(model_name: str, cap: int):
        cache_key = (model_name, cap)
        if cache_key in _cache:
            return _cache[cache_key]
        ef = model_efs.get(model_name, config.MODELS[model_name].get("sae_ef", config.SAE_EXPANSION_FACTOR))
        model_cfg = config.MODELS[model_name]
        hidden_dim = model_cfg["hidden_dim"]
        n_features = hidden_dim * ef
        ckpt_path = _find_sae_checkpoint(model_name, config.SAE_DIR, ef)
        print(f"[sidecars] {model_name}: loading SAE {ckpt_path}")
        sae = TopKSAE(hidden_dim, n_features, model_cfg["sae_topk"]).to(dev)
        try:
            sae.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=True))
        except TypeError:
            sae.load_state_dict(torch.load(ckpt_path, map_location=dev))
        sae.eval()
        acts = _load_activations(model_name)
        original_n = acts.shape[0]
        if 0 < args.max_passages < original_n:
            rng = np.random.default_rng(42)
            idx_rows = rng.choice(original_n, size=int(args.max_passages), replace=False)
            idx_rows.sort()
            acts = acts[idx_rows]
            print(f"[sidecars] {model_name}: subsampled {len(acts):,}/{original_n:,} passages")
        effective_cap = min(cap, n_features)
        print(f"[sidecars] {model_name}: selecting top-{effective_cap} features")
        active_idx = _select_topk_features(acts, sae, n_features, effective_cap, args.batch_size)
        mat = _compute_feature_matrix(acts, sae, active_idx, args.batch_size)
        mean_v, std_v = _batched_mean_std(mat)
        sae.cpu(); del sae, acts, mat
        if dev.type == "cuda": torch.cuda.empty_cache()
        _cache[cache_key] = (active_idx, mean_v, std_v)
        return active_idx, mean_v, std_v

    mlp_files = _glob.glob(os.path.join(config.ALIGNMENT_DIR, "mlp_*_to_*.pt"))
    if not mlp_files:
        print("[sidecars] No MLP files found in", config.ALIGNMENT_DIR)
        return
    print(f"[sidecars] Found {len(mlp_files)} MLP files to process")
    for mlp_path in sorted(mlp_files):
        src_idx_path   = mlp_path.replace(".pt", "_src_idx.npy")
        tgt_idx_path   = mlp_path.replace(".pt", "_tgt_idx.npy")
        src_stats_path = mlp_path.replace(".pt", "_src_stats.npz")
        tgt_stats_path = mlp_path.replace(".pt", "_tgt_stats.npz")
        if (os.path.exists(src_idx_path) and os.path.exists(tgt_idx_path)
                and os.path.exists(src_stats_path) and os.path.exists(tgt_stats_path)):
            print(f"[sidecars] {os.path.basename(mlp_path)}: sidecars already exist, skipping")
            continue
        # Parse model names from filename: mlp_{src}_to_{tgt}.pt
        base = os.path.basename(mlp_path)[len("mlp_"):-len(".pt")]  # e.g. gpt2-large_to_llama
        # Find the split point: "_to_" between two known model names
        src_name = tgt_name = None
        for candidate in args.models:
            if base.startswith(candidate + "_to_"):
                src_name = candidate
                tgt_name = base[len(candidate) + len("_to_"):]
                break
        if src_name is None or tgt_name is None or src_name not in config.MODELS or tgt_name not in config.MODELS:
            print(f"[sidecars] Could not parse model names from {os.path.basename(mlp_path)}, skipping")
            continue
        # Read MLP input/output dims from saved weights
        state = torch.load(mlp_path, map_location="cpu")
        in_dim  = state["0.weight"].shape[1]
        w_last_key = "3.weight" if "3.weight" in state else "2.weight"
        out_dim = state[w_last_key].shape[0]
        print(f"[sidecars] {src_name}→{tgt_name}: in={in_dim} out={out_dim}")
        try:
            src_idx, src_mean, src_std = _get_model_idx_stats(src_name, in_dim)
            tgt_idx, tgt_mean, tgt_std = _get_model_idx_stats(tgt_name, out_dim)
        except Exception as exc:
            print(f"[sidecars] ERROR for {src_name}→{tgt_name}: {exc}")
            continue
        np.save(src_idx_path, src_idx)
        np.save(tgt_idx_path, tgt_idx)
        np.savez(src_stats_path, mean=src_mean, std=src_std)
        np.savez(tgt_stats_path, mean=tgt_mean, std=tgt_std)
        print(f"[sidecars] Saved sidecars for {src_name}→{tgt_name}")
    print("[sidecars] Done.")


if __name__ == "__main__":
    import sys as _sys
    if "--save-mlp-sidecars" in _sys.argv:
        _sys.argv.remove("--save-mlp-sidecars")
        save_mlp_sidecars()
    else:
        try:
            raise SystemExit(main())
        except Exception as e:
            log_run("step5_align_features.py", time.time(), "error", str(e))
            raise
