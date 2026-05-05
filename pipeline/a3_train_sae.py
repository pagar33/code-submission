
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import bisect
import collections
import json
import os
import queue
import threading
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import random
import re
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv

try:
    import wandb
except Exception:
    wandb = None

import config

try:
    from artifact_store import get_activation_path as _get_activation_path
    _ARTIFACT_STORE_AVAILABLE = True
except ImportError:
    _ARTIFACT_STORE_AVAILABLE = False
    def _get_activation_path(model, source, dest_dir=None):
        raise ImportError("artifact_store not available")


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


def _h5_read_rows(path: str, rows: np.ndarray) -> np.ndarray:
    """Read specific rows from an h5 activations file, handling bfloat16 and duplicate indices."""
    # h5py requires strictly increasing indices — deduplicate, then map back
    unique_rows, inverse = np.unique(rows.astype(np.int64), return_inverse=True)
    with h5py.File(path, "r") as f:
        dset = f["activations"]
        try:
            data = dset[unique_rows]
        except ValueError:
            # bfloat16 stored — use NATIVE_FLOAT bypass
            all_rows = np.empty((dset.shape[0], dset.shape[1]), dtype=np.float32)
            fspace = dset.id.get_space()
            mspace = h5py.h5s.create_simple(dset.shape)
            dset.id.read(mspace, fspace, all_rows, h5py.h5t.NATIVE_FLOAT)
            data = all_rows[unique_rows]
    return data[inverse]  # restore original row order including duplicates


class H5ActivationStore:
    """Lazy activation store — reads batches from h5 files without loading all into RAM.
    Keeps file handles open for the lifetime of the store to eliminate per-step open/close overhead."""

    def __init__(self, file_paths: list[str]):
        self.file_paths = file_paths
        self._fh: list = []       # persistent h5py.File handles
        self._dsets: list = []    # persistent dataset references
        self._is_bf16: list = []  # bfloat16 flag per file, detected once at init
        self.sizes: list[int] = []
        self._hidden_dim: int = 0
        for p in file_paths:
            fh = h5py.File(p, "r")
            dset = fh["activations"]
            self._fh.append(fh)
            self._dsets.append(dset)
            s = dset.shape
            self.sizes.append(s[0])
            self._hidden_dim = s[1]
            # detect bfloat16 once — avoid per-step try/except
            try:
                dset[np.array([0], dtype=np.int64)]
                self._is_bf16.append(False)
            except (ValueError, TypeError):
                self._is_bf16.append(True)
        self.cumulative = np.concatenate([[0], np.cumsum(self.sizes)])
        self.n_total = int(self.cumulative[-1])

    def close(self):
        for fh in self._fh:
            try:
                fh.close()
            except Exception:
                pass
        self._fh = []
        self._dsets = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def shape(self):
        return (self.n_total, self._hidden_dim)

    def _read_from_dset(self, fi: int, rows: np.ndarray) -> np.ndarray:
        """Read specific rows from already-open dataset fi, handling duplicates and bfloat16."""
        unique_rows, inverse = np.unique(rows.astype(np.int64), return_inverse=True)
        dset = self._dsets[fi]
        if self._is_bf16[fi]:
            all_rows = np.empty((dset.shape[0], dset.shape[1]), dtype=np.float32)
            fspace = dset.id.get_space()
            mspace = h5py.h5s.create_simple(dset.shape)
            dset.id.read(mspace, fspace, all_rows, h5py.h5t.NATIVE_FLOAT)
            data = all_rows[unique_rows]
        else:
            data = dset[unique_rows]
        return data[inverse]

    def __getitem__(self, idx):
        """idx: 1-D LongTensor or ndarray of global row indices."""
        if isinstance(idx, torch.Tensor):
            idx_np = idx.cpu().numpy().astype(np.int64)
        else:
            idx_np = np.asarray(idx, dtype=np.int64)
        # vectorized file-index lookup — much faster than Python bisect loop
        file_idxs = np.searchsorted(self.cumulative[1:], idx_np, side='right').astype(np.int64)
        local_idxs = idx_np - self.cumulative[file_idxs].astype(np.int64)
        result = np.empty((len(idx_np), self._hidden_dim), dtype=np.float32)
        for fi in np.unique(file_idxs):
            mask = file_idxs == fi
            rows = local_idxs[mask]
            result[mask] = self._read_from_dset(int(fi), rows)
        return torch.from_numpy(result).float()


class ShuffleBufferSampler:
    """Sequential-read shuffle buffer — avoids random NFS seeks entirely.

    NFS random access is slow (one seek per row). Sequential reads are 10-50x
    faster. This class:
      1. Reads large sequential chunks from h5 into a RAM buffer (background thread)
      2. Shuffles the buffer in-place
      3. Training samples batches from the buffer (pure RAM → instant)
      4. Refills when buffer drops below half

    Memory: buffer_rows × dim × 4 bytes
      buffer_rows=200_000, dim=4096 → ~3.2 GB RAM (safe on A100 pod)

    Quality: at buffer_rows=200k the sampling diversity is effectively
    equivalent to true random — covers ~27% of llama's 739k rows per fill,
    spanning all domains.
    """

    def __init__(self, store_files: list[str], n_total: int, hidden_dim: int,
                 buffer_rows: int = 200_000):
        self._store_files = store_files
        self._n_total = n_total
        self._hidden_dim = hidden_dim
        self._buffer_rows = buffer_rows

        # Shared buffer and cursor — only background thread writes, main reads
        self._buf: np.ndarray | None = None
        self._buf_size: int = 0          # valid rows currently in buffer
        self._buf_pos: int = 0           # next row to hand out
        self._lock = threading.Lock()
        self._fill_event = threading.Event()  # signals background to fill
        self._ready_event = threading.Event() # signals main buffer is ready
        self._stop = threading.Event()

        # Sequential read position across the whole dataset (wraps around)
        self._read_pos: int = 0
        # Open own file handle for sequential reads
        self._store = H5ActivationStore(store_files)

        # Initial blocking fill so training can start immediately
        self._fill_buffer_sync()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _read_sequential_chunk(self, n_rows: int) -> np.ndarray:
        """Read n_rows sequentially from the dataset (wraps around at end)."""
        end = self._read_pos + n_rows
        if end <= self._n_total:
            chunk = self._store[np.arange(self._read_pos, end, dtype=np.int64)].numpy()
            self._read_pos = end % self._n_total
        else:
            # Wrap around
            part1 = self._store[np.arange(self._read_pos, self._n_total, dtype=np.int64)].numpy()
            remainder = n_rows - len(part1)
            part2 = self._store[np.arange(0, remainder, dtype=np.int64)].numpy()
            chunk = np.concatenate([part1, part2], axis=0)
            self._read_pos = remainder
        return chunk

    def _fill_buffer_sync(self):
        """Fill buffer synchronously (used for initial fill)."""
        chunk = self._read_sequential_chunk(self._buffer_rows)
        np.random.shuffle(chunk)  # shuffle in-place so sampling is non-sequential
        with self._lock:
            self._buf = chunk
            self._buf_size = len(chunk)
            self._buf_pos = 0

    def _worker(self):
        """Background thread: refills buffer when main thread signals low."""
        while not self._stop.is_set():
            self._fill_event.wait(timeout=1.0)
            if self._stop.is_set():
                break
            self._fill_event.clear()
            # Read new chunk sequentially (fast NFS sequential I/O)
            chunk = self._read_sequential_chunk(self._buffer_rows)
            np.random.shuffle(chunk)
            with self._lock:
                self._buf = chunk
                self._buf_size = len(chunk)
                self._buf_pos = 0
            self._ready_event.set()

    def next_batch(self, batch_size: int) -> torch.Tensor:
        """Return next batch of batch_size rows from the buffer."""
        with self._lock:
            pos = self._buf_pos
            end = pos + batch_size
            if end <= self._buf_size:
                batch = self._buf[pos:end].copy()
                self._buf_pos = end
                remaining = self._buf_size - end
            else:
                # Wrap within buffer (rare, only if batch_size > buffer_rows)
                batch = np.concatenate([
                    self._buf[pos:],
                    self._buf[:batch_size - (self._buf_size - pos)]
                ], axis=0)
                self._buf_pos = batch_size - (self._buf_size - pos)
                remaining = self._buf_size - self._buf_pos

        # Signal background to refill when buffer is half empty
        if remaining < self._buffer_rows // 2:
            self._fill_event.set()

        return torch.from_numpy(batch).float()

    def stop(self):
        self._stop.set()
        self._fill_event.set()  # unblock worker if waiting
        self._store.close()


class TopKSAE(nn.Module):
    def __init__(self, hidden_dim: int, n_features: int, topk: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.topk = topk
        self.encoder = nn.Linear(hidden_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, hidden_dim, bias=True)

    def forward(self, x):
        z = self.encoder(x)
        if self.topk < self.n_features:
            topk_vals, topk_idx = torch.topk(z, self.topk, dim=-1)
            sparse = torch.zeros_like(z)
            sparse.scatter_(1, topk_idx, topk_vals)
            sparse = torch.relu(sparse)
        else:
            sparse = z
        recon = self.decoder(sparse)
        return recon, sparse


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(config.MODELS.keys()))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume-from", dest="resume", help="Alias for --resume")
    parser.add_argument("--resume-step", type=int, default=None, help="Step number of the resume checkpoint")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--run-name", default=None, help="Optional wandb run name override")
    parser.add_argument("--expansion-factor", "--ef", type=int, default=config.SAE_EXPANSION_FACTOR,
                        dest="expansion_factor", help="SAE expansion factor (n_features = hidden_dim × EF)")
    parser.add_argument("--steps", type=int, default=0, help="Override sae_train_steps from config")
    parser.add_argument("--batch-size", type=int, default=0, dest="batch_size", help="Override SAE_BATCH_SIZE from config")
    parser.add_argument("--topk", type=int, default=0, help="Override sae_topk from config")
    parser.add_argument("--filter-sources", dest="filter_sources", default="",
                        help="Comma-separated corpus source names to include (default: all extracted sources)")
    parser.add_argument("--save-every", type=int, default=1000, dest="save_every",
                        help="Save intermediate checkpoint every N training steps (default: 1000)")
    # Advanced loss config
    parser.add_argument("--sparsity-weight", type=float, default=1e-4, dest="sparsity_weight",
                        help="L1 coefficient on feature activations (default: 1e-4)")
    parser.add_argument("--recon-loss", default="mse", dest="recon_loss",
                        choices=["mse", "huber"], help="Reconstruction loss type (default: mse)")
    parser.add_argument("--decoder-norm-weight", type=float, default=0.0, dest="decoder_norm_weight",
                        help="Decoder column norm penalty (default: 0.0)")
    parser.add_argument("--ghost-grads", action="store_true", dest="ghost_grads",
                        help="Enable ghost gradients to revive dead features")
    parser.add_argument("--grad-clip", type=float, default=0.0, dest="grad_clip",
                        help="Max gradient norm; 0 = disabled (default: 0)")
    parser.add_argument("--warmup-steps", type=int, default=0, dest="warmup_steps",
                        help="Linear LR warmup steps (default: 0)")
    parser.add_argument("--run-prefix", default="", dest="run_prefix",
                        help="Prefix for checkpoint filenames (default: none)")
    parser.add_argument("--sae-arch", default="topk", dest="sae_arch",
                        choices=["topk", "jumprelu", "gated", "batchtopk", "swiglu"],
                        help="SAE architecture (default: topk)")
    parser.add_argument("--multi-gpu", action="store_true", dest="multi_gpu",
                        help="Wrap SAE in DataParallel across all visible CUDA devices")
    parser.add_argument("--bf16", action="store_true", dest="bf16",
                        help="Train in bfloat16 (A100/H100 only) — ~2x faster, same quality")
    parser.add_argument("--loss-threshold", type=float, default=None, dest="loss_threshold",
                        help="Early-stop when 20-step rolling avg reconstruction loss drops below this value")
    parser.add_argument("--dead-threshold", type=float, default=None, dest="dead_threshold",
                        help="Early-stop when dead feature %% exceeds this value (e.g. 80 = 80%%)")
    args = parser.parse_args()
    lr = args.lr if args.lr is not None else config.SAE_LR

    load_dotenv()
    if wandb is not None and os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)

    model_name = args.model
    os.makedirs(config.SAE_DIR, exist_ok=True)
    ef = args.expansion_factor
    # Legacy naming for default EF=16 (backward compat with existing checkpoints)
    # New naming includes EF suffix for non-default expansion factors
    prefix = (args.run_prefix.strip("_- ") + "_") if args.run_prefix.strip("_- ") else ""
    ef_tag = "" if ef == config.SAE_EXPANSION_FACTOR else f"_ef{ef}"
    arch_tag = "" if args.sae_arch == "topk" else f"_{args.sae_arch}"
    out_path = os.path.join(config.SAE_DIR, f"{prefix}{model_name}{ef_tag}{arch_tag}_sae.pt")
    cfg_path = os.path.join(config.SAE_DIR, f"{prefix}{model_name}{ef_tag}{arch_tag}_sae_config.json")
    log_jsonl_path = os.path.join(config.SAE_DIR, f"{prefix}{model_name}{ef_tag}{arch_tag}_training_log.jsonl")

    # Idempotence: check both new and legacy paths
    legacy_path = os.path.join(config.SAE_DIR, f"{model_name}_sae.pt")
    already_exists = os.path.exists(out_path) or (ef_tag == "" and os.path.exists(legacy_path))
    if already_exists and not args.force:
        print("SAE already exists. Use --force to retrain.")
        log_run("step3_train_sae.py", start_time, "skipped")
        return 0

    set_seed(42)

    model_cfg = config.MODELS[model_name]
    hidden_dim = model_cfg["hidden_dim"]
    n_features = hidden_dim * ef
    # Use overrides if provided, else fall back to per-model config
    topk = args.topk if args.topk > 0 else model_cfg["sae_topk"]
    n_steps_cfg = args.steps if args.steps > 0 else model_cfg["sae_train_steps"]
    batch_size_cfg = args.batch_size if args.batch_size > 0 else config.SAE_BATCH_SIZE
    # Auto-scale topk when EF differs from default to maintain ~same sparsity ratio
    if args.topk == 0 and ef != config.SAE_EXPANSION_FACTOR:
        base_sparsity = model_cfg["sae_topk"] / (hidden_dim * config.SAE_EXPANSION_FACTOR)
        topk = max(int(n_features * base_sparsity), 32)
        print(f"Auto-scaled topk to {topk} ({base_sparsity*100:.3f}% sparsity) for EF={ef}")

    activations_path = os.path.join(config.ACTIVATIONS_DIR, f"{model_name}_activations_norm.h5")
    if not os.path.exists(activations_path):
        # Try to resolve via artifact_store (downloads from W&B if WANDB_API_KEY is set)
        if _ARTIFACT_STORE_AVAILABLE:
            try:
                print(f"[artifact_store] Combined norm file not found locally — trying W&B...")
                activations_path = _get_activation_path(model_name, "combined")
            except Exception as _art_exc:
                print(f"[artifact_store] W&B resolve failed: {_art_exc}")
        # Fall back: auto-concatenate per-source norm files
        acts_dir = config.ACTIVATIONS_DIR
        prefix = f"{model_name}_"
        suffix = "_activations_norm.h5"
        per_src_files = sorted(
            os.path.join(acts_dir, f) for f in os.listdir(acts_dir)
            if f.startswith(prefix) and f.endswith(suffix)
        )
        if not per_src_files:
            raise FileNotFoundError(
                f"Missing activations at {activations_path} "
                f"(and no per-source files found matching {prefix}*{suffix})"
            )
        activations_path = None  # signal to skip h5 read below — use per_src_files directly

    _cfg_device = model_cfg["device"]
    if _cfg_device == "cuda" and not torch.cuda.is_available():
        print(f"WARNING: CUDA not available on this machine — falling back to CPU for {model_name}. Training will be slower.")
        _cfg_device = "cpu"
    device = torch.device(_cfg_device)

    use_wandb = wandb is not None and (os.getenv("WANDB_PROJECT") or os.getenv("WANDB_API_KEY"))
    # Clean up stale wandb run dirs to prevent disk-full on next init
    _wandb_dir = os.path.join(os.path.dirname(os.path.abspath(config.__file__)), "wandb")
    if os.path.isdir(_wandb_dir):
        import shutil
        _wb_runs = sorted(
            (e for e in os.scandir(_wandb_dir) if e.is_dir() and e.name.startswith("run-")),
            key=lambda e: e.stat().st_mtime,
        )
        # Keep only the 3 most recent runs; delete older ones
        for _old in _wb_runs[:-3]:
            try:
                shutil.rmtree(_old.path)
                print(f"[wandb cleanup] Removed old run dir: {_old.name}")
            except Exception:
                pass
    wandb_run = None
    if use_wandb:
        try:
            wandb_run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "universal_steering"),
                entity=os.getenv("WANDB_ENTITY"),
                name=(args.run_name or f"sae_{model_name}_ef{ef}"),
                config={
                    "model": model_name,
                    "hidden_dim": hidden_dim,
                    "n_features": n_features,
                    "expansion_factor": ef,
                    "topk": topk,
                    "train_steps": n_steps_cfg,
                    "batch_size": batch_size_cfg,
                    "lr": lr,
                },
            )
        except Exception as _wb_err:
            print(f"WARNING: W&B init failed ({_wb_err}) — continuing without W&B logging.")

    sae = TopKSAE(hidden_dim, n_features, topk).to(device)
    use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if use_bf16:
        sae = sae.to(torch.bfloat16)
        print("[bf16] Training in bfloat16")
    n_visible_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"[GPU] Visible GPUs: {n_visible_gpus}")
    if device.type == "cuda":
        for _gi in range(n_visible_gpus):
            _props = torch.cuda.get_device_properties(_gi)
            _free, _total = torch.cuda.mem_get_info(_gi)
            print(f"[GPU {_gi}] {_props.name} — {_props.total_memory // 1024**3} GB total, {_free // 1024**3} GB free")
    use_dp = args.multi_gpu and n_visible_gpus > 1
    if use_dp:
        sae = torch.nn.DataParallel(sae, device_ids=list(range(n_visible_gpus)))
        print(f"[GPU] DataParallel active across {n_visible_gpus} GPUs: {list(range(n_visible_gpus))}")
    else:
        print(f"[GPU] Single-GPU mode on {device}")
    _sae_core = sae.module if use_dp else sae   # unwrapped reference for weight ops
    start_step = 0
    if args.resume:
        ckpt_path = args.resume
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Resume checkpoint not found: {ckpt_path} — starting from scratch.")
        else:
            try:
                state = torch.load(ckpt_path, map_location=device)
                _sae_core.load_state_dict(state)
                if args.resume_step is not None:
                    start_step = args.resume_step
                else:
                    m = re.search(r"step(\d+)", os.path.basename(ckpt_path))
                    if m:
                        start_step = int(m.group(1))
                print(f"Resuming from {ckpt_path} at step {start_step}")
            except Exception as _ckpt_err:
                print(f"WARNING: Checkpoint {ckpt_path} is corrupted ({_ckpt_err}) — deleting and starting from scratch.")
                try:
                    os.remove(ckpt_path)
                except Exception:
                    pass
    # Warmup scheduler
    def _get_lr(step: int) -> float:
        if args.warmup_steps > 0 and step <= args.warmup_steps:
            return lr * step / args.warmup_steps
        return lr

    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    if args.recon_loss == "huber":
        loss_fn = nn.HuberLoss()
    else:
        loss_fn = nn.MSELoss()

    # Build lazy activation store — no RAM allocation, reads batches from disk
    filter_src = args.filter_sources.strip()
    acts_dir = config.ACTIVATIONS_DIR
    if activations_path is None:
        # per-source files found earlier — build store from them
        if filter_src:
            src_keep = {s.strip() for s in filter_src.split(",") if s.strip()}
            store_files = [
                p for p in per_src_files
                if any(src.replace("/","_").replace(" ","_") in os.path.basename(p) for src in src_keep)
            ]
            if not store_files:
                print(f"WARNING: no per-source files matched filter {src_keep!r}, using all")
                store_files = per_src_files
        else:
            store_files = per_src_files
    else:
        # Single combined file
        if filter_src:
            src_keep = {s.strip() for s in filter_src.split(",") if s.strip()}
            store_files = []
            for src in sorted(src_keep):
                safe_src = src.replace("/", "_").replace(" ", "_")
                p = os.path.join(acts_dir, f"{model_name}_{safe_src}_activations_norm.h5")
                if os.path.exists(p):
                    store_files.append(p)
            if not store_files:
                store_files = [activations_path]
        else:
            store_files = [activations_path]

    acts = H5ActivationStore(store_files)
    n_samples = acts.n_total
    print(f"[ActivationStore] {len(store_files)} file(s), {n_samples:,} total rows, dim={acts.shape[1]}")
    if acts.shape[1] != hidden_dim:
        raise ValueError(f"Activation hidden_dim mismatch: {acts.shape[1]} vs expected {hidden_dim}")

    n_steps = n_steps_cfg
    batch_size = batch_size_cfg

    # Prefetcher: reads next batch from disk in background while GPU runs current batch.
    # Uses its own H5ActivationStore (separate file handles — thread safe).
    prefetcher = ShuffleBufferSampler(store_files, n_samples, acts.shape[1], buffer_rows=200_000)
    print(f"[ShuffleBuffer] started (buffer=200k rows ~{200_000 * acts.shape[1] * 4 / 1e9:.1f}GB RAM, sequential NFS reads)")

    feature_active_counts = torch.zeros(n_features, dtype=torch.long, device=device)
    last_loss = None
    _recon_window: collections.deque = collections.deque(maxlen=20)
    _early_stop_reason: str | None = None
    _steps_run: int = start_step

    for step in range(start_step + 1, n_steps + 1):
        batch = prefetcher.next_batch(batch_size).to(device)
        if use_bf16:
            batch = batch.to(torch.bfloat16)
        if batch.shape != (batch_size, hidden_dim):
            raise ValueError(f"Batch shape mismatch: {batch.shape}")

        # Update LR for warmup
        if args.warmup_steps > 0:
            current_lr = _get_lr(step)
            for pg in opt.param_groups:
                pg["lr"] = current_lr

        opt.zero_grad(set_to_none=True)
        recon, sparse = sae(batch)
        recon_loss = loss_fn(recon, batch)
        sparsity_loss = sparse.abs().mean()
        loss = recon_loss + args.sparsity_weight * sparsity_loss
        # Decoder norm penalty
        if args.decoder_norm_weight > 0:
            dec_norm_loss = _sae_core.decoder.weight.norm(dim=0).mean()
            loss = loss + args.decoder_norm_weight * dec_norm_loss
        # Ghost gradients: route grad through dead features
        if args.ghost_grads:
            with torch.no_grad():
                dead_mask = (feature_active_counts == 0).float().to(device)
            if dead_mask.any():
                ghost = _sae_core.encoder(batch)  # raw pre-topk activations
                ghost_loss = (ghost * dead_mask).pow(2).mean()
                loss = loss + 1e-3 * ghost_loss
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        opt.step()
        with torch.no_grad():
            norms = _sae_core.decoder.weight.data.norm(dim=0, keepdim=True)
            _sae_core.decoder.weight.data = _sae_core.decoder.weight.data / norms.clamp(min=1e-8)

        last_loss = loss.item()
        _steps_run = step

        with torch.no_grad():
            active = (sparse != 0).sum(dim=0)
            feature_active_counts += active

        if step % 100 == 0:
            dead_count = int((feature_active_counts == 0).sum().item())
            dead_pct = dead_count / n_features * 100.0
            recon_val = loss_fn(recon.detach(), batch).item()
            sparse_val = sparse.abs().mean().item()
            _recon_window.append(recon_val)
            print(f"step {step}/{n_steps} loss={last_loss:.6f} recon={recon_val:.6f} sparse={sparse_val:.6f} dead={dead_pct:.1f}%")
            log_entry = {
                "step": step,
                "loss": last_loss,
                "recon_loss": recon_val,
                "sparsity_loss": sparse_val,
                "dead_features_pct": dead_pct,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(log_jsonl_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_entry) + "\n")
            if wandb_run is not None:
                wandb_run.log({
                    "loss": last_loss,
                    "recon_loss": recon_val,
                    "sparsity_loss": sparse_val,
                    "step": step,
                }, step=step)
            # Early-stop checks
            if args.loss_threshold is not None and len(_recon_window) == _recon_window.maxlen:
                rolling_avg = sum(_recon_window) / len(_recon_window)
                if rolling_avg < args.loss_threshold:
                    print(f"Early stop at step {step}: rolling recon loss {rolling_avg:.6f} < threshold {args.loss_threshold}")
                    _early_stop_reason = f"loss_threshold={args.loss_threshold} (rolling_avg={rolling_avg:.6f})"
                    ckpt_path = os.path.join(config.SAE_DIR, f"{model_name}{ef_tag}_sae_step{step}.pt")
                    torch.save(_sae_core.state_dict(), ckpt_path)
                    break
            _DEAD_CHECK_MIN_STEP = 5000
            if args.dead_threshold is not None and step >= _DEAD_CHECK_MIN_STEP and dead_pct > args.dead_threshold:
                print(f"Early stop at step {step}: dead features {dead_pct:.1f}% > threshold {args.dead_threshold}%")
                _early_stop_reason = f"dead_threshold={args.dead_threshold}% (actual={dead_pct:.1f}%)"
                ckpt_path = os.path.join(config.SAE_DIR, f"{model_name}{ef_tag}_sae_step{step}.pt")
                torch.save(_sae_core.state_dict(), ckpt_path)
                break

        if step % args.save_every == 0:
            ckpt_path = os.path.join(config.SAE_DIR, f"{model_name}{ef_tag}_sae_step{step}.pt")
            torch.save(_sae_core.state_dict(), ckpt_path)
            # Delete the previous periodic checkpoint to avoid filling disk
            prev_step = step - args.save_every
            if prev_step > 0:
                prev_ckpt = os.path.join(config.SAE_DIR, f"{model_name}{ef_tag}_sae_step{prev_step}.pt")
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)
                    print(f"[ckpt] Removed old checkpoint: {os.path.basename(prev_ckpt)}")

    prefetcher.stop()
    acts.close()

    dead_features = (feature_active_counts == 0).sum().item()
    dead_features_pct = dead_features / n_features * 100.0

    torch.save(_sae_core.state_dict(), out_path)

    cfg = {
        "model": model_name,
        "hidden_dim": hidden_dim,
        "expansion_factor": ef,
        "n_features": n_features,
        "topk": topk,
        "train_steps": n_steps,
        "batch_size": batch_size,
        "lr": lr,
        "final_reconstruction_loss": float(last_loss) if last_loss is not None else None,
        "dead_features_pct": float(dead_features_pct),
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    if wandb_run is not None:
        wandb_run.log({
            "final_reconstruction_loss": float(last_loss) if last_loss is not None else None,
            "dead_features_pct": float(dead_features_pct),
        })
        wandb_run.finish()

    log_run("step3_train_sae.py", start_time, "success")
    print(f"Saved SAE to {out_path}")
    print(f"Config to {cfg_path}")
    # Emit machine-readable summary captured by the job manager → /history
    final_stats = {
        "final_loss": round(float(last_loss), 6) if last_loss is not None else None,
        "dead_pct": round(float(dead_features_pct), 2),
        "steps_run": _steps_run,
        "steps_total": n_steps,
        "early_stop": _early_stop_reason,
        "model": model_name,
        "expansion_factor": ef,
    }
    print(f"FINAL_STATS: {json.dumps(final_stats)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log_run("step3_train_sae.py", time.time(), "error", str(e))
        raise
