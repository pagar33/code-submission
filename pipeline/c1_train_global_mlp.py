#!/usr/bin/env python3
"""
C1 — Train Global Concept Space
================================
Trains a shared encoder-decoder MLP across all registered models, producing
a d_concept-dimensional universal concept space.

Each model gets:
  - encoder:  n_features → d_concept  (SAE features → universal coordinates)
  - decoder:  d_concept → n_features  (concept coordinates → SAE features)

Training signals:
  1. Reconstruction loss (MSE): decode(encode(x)) ≈ x
  2. Contrastive alignment (NT-Xent / SimCLR): for the same passage, encodings
     from different models are pulled together. Different passages are pushed
     apart. This is the honest universality test — no concept labels used.

Multi-GPU: automatically relaunches via torchrun if LOCAL_RANK is not set and
>1 GPU is available. Supports all 8× A100 GPUs simultaneously.

Outputs written to  universal/  directory:
  global_mlp_v{N}.pt          — final checkpoint  (all encoders + decoders)
  global_mlp_v{N}_best.pt     — best-val checkpoint
  global_mlp_v{N}_meta.json   — training config, per-model val losses
  training_log_v{N}.json      — per-epoch loss curves (written every epoch)

  Periodic checkpoints:
  global_mlp_v{N}_ckpt_ep{E}.pt  — every --ckpt-every epochs (default 100)

Usage (example — all 5 models):
  python train_global_mlp.py \\
    --models gpt2-large:64 gemma:64 llama:128 mistral:128 deepseek-llm-7b:128 \\
    --d-concept 512 --epochs 200 --batch-size 256 \\
    --lr 5e-4 --recon-weight 1.0 --contrastive-weight 0.5

C1b ablation (reconstruction only, no contrastive):
  python train_global_mlp.py [same args] --ablation-report
  → forces contrastive-weight=0.0, saves as recon_only tag, writes silhouette stub
"""


import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import gc
import glob as _glob
import json
import os
import re
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
# DataLoader / Dataset / DistributedSampler not needed — data is GPU-resident

import config


# ─────────────────────────────────────────────────────────────────────────────
# GPU inventory
# ─────────────────────────────────────────────────────────────────────────────

def _log_gpu_inventory(rank: int) -> int:
    """
    Print a one-line summary of every visible GPU (name + total VRAM).
    Returns the number of GPUs found.  Always printed — not gated on rank.
    """
    n = torch.cuda.device_count()
    host = socket.gethostname()
    print(f"[C1] Host: {host}  |  CUDA devices visible: {n}", flush=True)
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1024**3
        print(f"[C1]   GPU {i}: {props.name}  {vram_gb:.1f} GB VRAM", flush=True)
    if n == 0:
        print("[C1]   (no CUDA GPUs — running on CPU)", flush=True)
    return n


# ─────────────────────────────────────────────────────────────────────────────
# DDP self-relaunch
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_relaunch_ddp() -> None:
    """
    If multiple GPUs are available and we are NOT already inside a torchrun
    worker (LOCAL_RANK absent), re-exec this script via torchrun so every GPU
    gets its own process with proper NCCL collective ops.
    """
    if "LOCAL_RANK" in os.environ:
        return  # already inside a DDP worker

    n_gpu = _log_gpu_inventory(rank=0)   # always print inventory before relaunch
    if n_gpu <= 1:
        return  # single GPU or CPU — run normally

    import subprocess
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={n_gpu}",
        "--master_port", "29501",
        os.path.abspath(__file__),
    ] + sys.argv[1:]
    print(
        f"[C1] {n_gpu} GPUs detected — relaunching via torchrun "
        f"({n_gpu} processes, effective batch = batch_size × {n_gpu})",
        flush=True,
    )
    ret = subprocess.run(cmd)
    sys.exit(ret.returncode)


# ─────────────────────────────────────────────────────────────────────────────
# Feature file discovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_feature_file(features_dir: Path, model: str, ef: int, suffix: str) -> Path:
    """
    Find the feature activation file for a model, picking the largest top-N
    when multiple files exist (e.g. top382 vs top666 for gpt2-large).

    Naming convention: {model}_ef{EF}_top{N}_{suffix}.npy
    Raises FileNotFoundError if nothing matches.
    """
    pattern = str(features_dir / f"{model}_ef{ef}_top*_{suffix}.npy")
    candidates = _glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"No feature file found matching:\n  {pattern}\n"
            f"Did you run step5_align_features.py (--save-mlp-sidecars) for {model}?"
        )

    def _top_n(path: str) -> int:
        m = re.search(r"_top(\d+)_", os.path.basename(path))
        return int(m.group(1)) if m else 0

    chosen = Path(max(candidates, key=_top_n))
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GpuBatchSampler:
    """
    Yields batches via pure GPU tensor slicing — no CPU involvement during training.
    Tensors are pre-indexed to their split (train or val) and resident on device.
    For DDP: each rank draws an interleaved shard of a shared per-epoch permutation,
    matching DistributedSampler semantics without Python DataLoader overhead.
    """

    def __init__(
        self,
        gpu_tensors: dict,        # {model: Tensor[split_N, d]}  already on device
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        self.gpu_tensors  = gpu_tensors
        self.batch_size   = batch_size
        self.shuffle      = shuffle
        self.drop_last    = drop_last
        self.rank         = rank
        self.world_size   = world_size
        self.seed         = seed
        self._epoch       = 0
        self.N            = next(iter(gpu_tensors.values())).shape[0]
        self.device       = next(iter(gpu_tensors.values())).device

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        n = (self.N + self.world_size - 1) // self.world_size
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed + self._epoch * 31337)  # reproducible, epoch-varying
            perm = torch.randperm(self.N, generator=g, device=self.device)
        else:
            perm = torch.arange(self.N, device=self.device)
        # Each rank takes an interleaved shard (identical to DistributedSampler)
        shard = perm[self.rank :: self.world_size]
        start = 0
        while start < len(shard):
            end = start + self.batch_size
            if end > len(shard):
                if self.drop_last:
                    break
                end = len(shard)
            idx = shard[start:end]
            yield {m: t[idx] for m, t in self.gpu_tensors.items()}
            start = end


# ─────────────────────────────────────────────────────────────────────────────
# Model: per-model encoder + decoder, shared concept space
# ─────────────────────────────────────────────────────────────────────────────

def _make_encoder(n_in: int, d_concept: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, d_concept),
    )


def _make_decoder(d_concept: int, n_out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_concept, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Linear(hidden, n_out),
    )


class GlobalMLP(nn.Module):
    """
    Multi-model encoder-decoder for the universal concept space.

    Each model has its own encoder (n_features → d_concept) and decoder
    (d_concept → n_features). The concept-space dimension d_concept is the
    single shared hyperparameter across all models.

    forward(x_dict) → (z_dict, recon_dict)
      x_dict:     {model_name: Tensor[B, n_features]}
      z_dict:     {model_name: Tensor[B, d_concept]}   — concept coordinates
      recon_dict: {model_name: Tensor[B, n_features]}  — reconstructed features
    """

    def __init__(self, model_n_features: dict, d_concept: int, hidden: int = 2048):
        super().__init__()
        self.model_names = sorted(model_n_features.keys())
        self.d_concept = d_concept

        self.encoders = nn.ModuleDict({
            m: _make_encoder(n, d_concept, hidden)
            for m, n in model_n_features.items()
        })
        self.decoders = nn.ModuleDict({
            m: _make_decoder(d_concept, n, hidden)
            for m, n in model_n_features.items()
        })

    def forward(self, x_dict: dict) -> tuple:
        z_dict, recon_dict = {}, {}
        for m in self.model_names:
            if m not in x_dict:
                continue
            z = self.encoders[m](x_dict[m])       # [B, d_concept]
            recon_dict[m] = self.decoders[m](z)   # [B, n_features]
            z_dict[m] = z
        return z_dict, recon_dict


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    SimCLR NT-Xent loss for one pair of model embeddings.
    Positive pairs are same passage index across the two models.
    z1, z2: [B, d_concept]  (not necessarily pre-normalised)
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.size(0)
    logits = torch.matmul(z1, z2.T) / temperature   # [B, B]
    labels = torch.arange(B, device=z1.device)
    # symmetric: both zA→zB and zB→zA directions
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
    return loss


def compute_loss(
    z_dict: dict,
    recon_dict: dict,
    x_dict: dict,
    recon_weight: float,
    contrastive_weight: float,
    temperature: float = 0.1,
) -> tuple:
    """
    Returns (total_loss, recon_loss, align_loss) — all scalar tensors.

    recon_loss:  mean per-model MSE reconstruction
    align_loss:  mean NT-Xent over all C(M,2) model pairs (0 if contrastive_weight=0)
    """
    model_names = list(z_dict.keys())
    device = list(z_dict.values())[0].device

    # Reconstruction: decode(encode(x)) ≈ x
    recon_terms = [F.mse_loss(recon_dict[m], x_dict[m].float()) for m in model_names]
    recon_loss = sum(recon_terms) / max(len(recon_terms), 1)

    # Contrastive: same passage, different models → close in concept space
    if contrastive_weight > 0.0:
        align_terms = []
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if j <= i:
                    continue
                align_terms.append(_nt_xent(z_dict[m1], z_dict[m2], temperature))
        align_loss = (
            sum(align_terms) / max(len(align_terms), 1)
            if align_terms
            else torch.zeros(1, device=device)
        )
    else:
        align_loss = torch.zeros(1, device=device)

    total = recon_weight * recon_loss + contrastive_weight * align_loss
    return total, recon_loss, align_loss


# ─────────────────────────────────────────────────────────────────────────────
# Versioning
# ─────────────────────────────────────────────────────────────────────────────

def _next_version(universal_dir: Path) -> int:
    """Return the next unused version number for global_mlp_{pooling}_v{N}.pt files."""
    # Match both old-style (global_mlp_vN.pt) and new-style (global_mlp_{pooling}_vN.pt)
    existing = list(universal_dir.glob("global_mlp_*_v*.pt"))
    if not existing:
        return 1
    versions = []
    for f in existing:
        m = re.search(r"_v(\d+)", f.name)
        if m:
            versions.append(int(m.group(1)))
    return max(versions) + 1 if versions else 1


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(f"[C1] {msg}", flush=True)


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _count_dead_neurons(z_dict: dict, threshold: float = 1e-4) -> float:
    """
    Fraction of concept-space dimensions that are near-zero for ≥99% of the batch.
    A dimension is 'dead' if |z| < threshold for every sample in the batch.
    """
    all_z = torch.cat(list(z_dict.values()), dim=0)   # [B*M, d_concept]
    active_per_dim = (all_z.abs() > threshold).float().mean(dim=0)  # [d_concept]
    dead = (active_per_dim < 0.01).float().mean().item()            # fraction dead
    return dead


def _write_artifact_registry(
    registry_path: Path,
    artifact_type: str,
    file_path: str,
    label: str,
    extra: dict,
) -> None:
    """
    Append / update artifacts/registry.json with one artifact record.
    Thread-safe for single-writer (rank 0 only).
    """
    import datetime
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    records: list = []
    if registry_path.exists():
        try:
            with open(registry_path) as f:
                records = json.load(f)
        except (json.JSONDecodeError, OSError):
            records = []
    records.append({
        "artifact_type": artifact_type,
        "file_path":     file_path,
        "label":         label,
        "registered_at": datetime.datetime.utcnow().isoformat() + "Z",
        **extra,
    })
    with open(registry_path, "w") as f:
        json.dump(records, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="C1 — Train Global Concept Space (universal encoder-decoder MLP)"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model names with optional ':EF' suffix, e.g. gpt2-large:64 gemma:64. "
             "Defaults to all models in config.MODELS.",
    )
    parser.add_argument("--d-concept", type=int, default=512,
                        help="Universal concept space dimension (default 512).")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Total training epochs (default 200).")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Peak learning rate with cosine annealing (default 5e-4).")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Per-GPU batch size. Effective batch = batch_size × n_gpus.")
    parser.add_argument("--recon-weight", type=float, default=1.0,
                        help="Weight for reconstruction loss (default 1.0).")
    parser.add_argument("--contrastive-weight", type=float, default=0.5,
                        help="Weight for contrastive alignment loss (default 0.5).")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="NT-Xent contrastive temperature (default 0.1).")
    parser.add_argument("--hidden", type=int, default=2048,
                        help="Encoder/decoder hidden layer width (default 2048).")
    parser.add_argument("--val-split", type=float, default=0.05,
                        help="Fraction of passages held out for validation (default 0.05).")
    parser.add_argument("--loss-threshold", type=float, default=None,
                        help="Early stop when 20-epoch running avg combined val loss < this.")
    parser.add_argument("--ckpt-every", type=int, default=100,
                        help="Save a periodic checkpoint every N epochs (default 100).")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a .pt checkpoint to resume from.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Optional tag suffix for output filenames, e.g. 'recon_only'.")
    parser.add_argument("--ablation-report", action="store_true",
                        help="C1b mode: force contrastive-weight=0, tag as recon_only.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if a final checkpoint already exists.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pooling", type=str, default="mean",
                        help="Pooling strategy inherited from A2 (default: mean). "
                             "Used in output filenames for artifact traceability.")
    parser.add_argument("--checkpoint-pct", type=str, default="100",
                        help="Comma-separated checkpoint percentages for concept-emergence "
                             "tracking (Roadmap R16), e.g. '10,25,50,75,100'. "
                             "Currently recorded in meta; multi-checkpoint sweeps are a "
                             "roadmap feature. Default: '100' (final model only).")
    parser.add_argument("--features-dir", type=str, default=None,
                        help="Override path to features/ directory containing "
                             "{model}_ef{EF}_top{N}_feature_acts.npy files. "
                             "Defaults to {pipeline_dir}/features/. "
                             "Use this to point explicitly at A100 storage.")
    args = parser.parse_args()

    # C1b: force reconstruction-only mode
    if args.ablation_report:
        args.contrastive_weight = 0.0
        if args.run_id is None:
            args.run_id = "recon_only"

    # ── GPU memory allocator — prevents fragmentation OOM on A100 ────────
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ── DDP setup ─────────────────────────────────────────────────────────
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = local_rank
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Print GPU inventory on rank 0 (already printed pre-relaunch, reprint
    # here so it appears in the torchrun log alongside rank assignments)
    if rank == 0:
        _log_gpu_inventory(rank=0)

    _log(rank, f"DDP: world_size={world_size}, rank={rank}, device={device}")

    # ── Paths ──────────────────────────────────────────────────────────────
    pipeline_dir = Path(config._CONFIG_DIR)
    features_dir = Path(args.features_dir) if args.features_dir else pipeline_dir / "features"
    universal_dir = pipeline_dir / "universal"
    if rank == 0:
        universal_dir.mkdir(exist_ok=True)
        _log(rank, f"Pipeline dir : {pipeline_dir.resolve()}")
        _log(rank, f"Features dir : {features_dir.resolve()}")
        _log(rank, f"Universal dir: {universal_dir.resolve()}")

    # Make sure universal_dir exists on all ranks before proceeding
    if is_ddp:
        dist.barrier()

    # ── Resolve model → EF map ─────────────────────────────────────────────
    if args.models:
        model_ef_map: dict = {}
        for spec in args.models:
            if ":" in spec:
                name, ef_str = spec.rsplit(":", 1)
                model_ef_map[name] = int(ef_str)
            else:
                cfg = config.MODELS.get(spec, {})
                model_ef_map[spec] = cfg.get("sae_ef", 64)
    else:
        model_ef_map = {
            m: cfg.get("sae_ef", 64) for m, cfg in config.MODELS.items()
        }

    _log(rank, f"Models & EFs: { {m: f'ef{ef}' for m, ef in model_ef_map.items()} }")

    # ── Load feature activations (rank 0 only, then broadcast) ───────────
    # On single-node DDP all ranks share the host DRAM.  Loading independently
    # on every rank multiplies RAM usage by world_size (8 × ~11.5 GB → OOM).
    # Instead: rank 0 loads from disk, each model tensor is moved to GPU and
    # broadcast via NCCL to every rank's device.  CPU RAM is freed immediately
    # after each broadcast, so peak CPU usage is just one model at a time.
    _log(rank, "Loading feature activations (rank-0 loads → NCCL broadcast)...")
    model_names  = list(model_ef_map.keys())
    n_features_map: dict = {}

    # ── Step 1: rank 0 discovers shapes; broadcast metadata ───────────────
    if rank == 0:
        shape_list = []
        for model_name, ef in model_ef_map.items():
            acts_path = _find_feature_file(features_dir, model_name, ef, "feature_acts")
            arr = np.load(str(acts_path), mmap_mode="r")
            shape_list.append((arr.shape[0], arr.shape[1]))
            del arr
        n_passages_per_model = [s[0] for s in shape_list]
        n_passages = min(n_passages_per_model)
        if len(set(n_passages_per_model)) > 1:
            _log(rank,
                 f"WARNING: passage counts differ {n_passages_per_model}. "
                 f"Truncating all to {n_passages:,}")
        shape_meta = torch.tensor(
            [n_passages] + [s[1] for s in shape_list],
            dtype=torch.long, device=device,
        )
    else:
        shape_meta = torch.zeros(1 + len(model_names), dtype=torch.long, device=device)

    if is_ddp:
        dist.broadcast(shape_meta, src=0)

    n_passages = int(shape_meta[0].item())
    for i, m in enumerate(model_names):
        n_features_map[m] = int(shape_meta[1 + i].item())

    _log(rank, f"Total passages: {n_passages:,}")

    # ── Step 2: for each model, rank 0 loads → GPU → broadcast ───────────
    # After broadcast every rank has the full tensor on its own GPU device.
    # CPU RAM is freed after each model is broadcast (never holds >1 model at once).
    gpu_full: dict = {}
    for model_name, ef in model_ef_map.items():
        n_feat = n_features_map[model_name]
        if rank == 0:
            acts_path = _find_feature_file(features_dir, model_name, ef, "feature_acts")
            arr = np.load(str(acts_path), mmap_mode="r")
            cpu_t = torch.from_numpy(arr[:n_passages].copy()).float()   # [N, F]
            size_mb = cpu_t.nbytes / 1024**2
            del arr; gc.collect()
            t_gpu = cpu_t.to(device)
            del cpu_t; gc.collect()
            _log(rank,
                 f"  {model_name:25s}  {n_passages:>7,} × {n_feat:>4}  "
                 f"({size_mb:.0f} MB)  ← {acts_path}")
        else:
            t_gpu = torch.empty(n_passages, n_feat, dtype=torch.float32, device=device)

        if is_ddp:
            dist.broadcast(t_gpu, src=0)   # NCCL: rank 0 sends, all others receive

        gpu_full[model_name] = t_gpu       # tensor lives on this rank's GPU

    # ── Train / val split ─────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    all_idx = np.arange(n_passages)
    rng.shuffle(all_idx)
    n_val = max(2000, int(n_passages * args.val_split))
    val_idx   = torch.from_numpy(all_idx[:n_val]).to(device)
    train_idx = torch.from_numpy(all_idx[n_val:]).to(device)
    _log(rank, f"Split: {len(train_idx):,} train / {len(val_idx):,} val")

    # ── Index splits on GPU (already there — zero CPU involvement) ────────
    train_gpu = {m: t[train_idx] for m, t in gpu_full.items()}
    val_gpu   = {m: t[val_idx]   for m, t in gpu_full.items()}
    del gpu_full; gc.collect()

    train_loader = GpuBatchSampler(
        train_gpu, args.batch_size, shuffle=True,  drop_last=True,
        rank=rank, world_size=world_size, seed=args.seed,
    )
    val_loader = GpuBatchSampler(
        val_gpu,   args.batch_size * 2, shuffle=False, drop_last=False,
        rank=rank, world_size=world_size, seed=args.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = GlobalMLP(n_features_map, args.d_concept, hidden=args.hidden).to(device)
    _log(rank, f"GlobalMLP: {_param_count(model):,} trainable parameters")
    _log(rank,
         f"  {len(model_ef_map)} encoders + {len(model_ef_map)} decoders"
         f"  |  d_concept={args.d_concept}  hidden={args.hidden}")

    start_epoch    = 0
    best_val_loss  = float("inf")
    training_log   = []
    last_val_recon_per_model: dict = {}

    # Resume from checkpoint?
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.is_absolute():
            ckpt_path = pipeline_dir / ckpt_path
        if ckpt_path.exists():
            _log(rank, f"Resuming from: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch   = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            training_log  = ckpt.get("training_log", [])
            _log(rank, f"Resumed at epoch {start_epoch}")
        else:
            _log(rank, f"WARNING: --resume-from path not found: {ckpt_path}. Starting from scratch.")

    # Wrap in DDP after potential checkpoint load
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5, betas=(0.9, 0.999)
    )
    remaining_epochs = args.epochs - start_epoch
    n_steps = remaining_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(n_steps, 1), eta_min=args.lr * 0.01
    )

    # ── Version + output paths (rank 0 only) ──────────────────────────────
    if rank == 0:
        version = _next_version(universal_dir)
        pooling_tag = args.pooling
        # Spec: global_mlp_{pooling}_v{N}.pt  (with optional run_id suffix)
        base_tag = f"{pooling_tag}_v{version}" + (f"_{args.run_id}" if args.run_id else "")
        tag = base_tag  # used in FINAL_STATS and meta
        ckpt_final_path   = universal_dir / f"global_mlp_{base_tag}.pt"
        ckpt_best_path    = universal_dir / f"global_mlp_{base_tag}_best.pt"
        meta_path         = universal_dir / f"global_mlp_{base_tag}_meta.json"
        training_log_path = universal_dir / f"training_log_{base_tag}.json"
        if args.ablation_report:
            recon_only_path = universal_dir / f"global_mlp_recon_only_{pooling_tag}_v{version}.pt"
        else:
            recon_only_path = None
        _log(rank, f"Output tag: {tag}  (pooling={pooling_tag})")
        _log(rank, f"Checkpoints: {ckpt_final_path.name} | best: {ckpt_best_path.name}")
    else:
        # Non-zero ranks don't need these — set to None to catch accidents
        version = tag = pooling_tag = None
        ckpt_final_path = ckpt_best_path = meta_path = training_log_path = recon_only_path = None

    # ── State-dict helper ─────────────────────────────────────────────────
    def _build_state(epoch: int) -> dict:
        m = model.module if is_ddp else model
        return {
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "training_log": training_log,
            "config": {
                "model_ef_map":       model_ef_map,
                "n_features_map":     n_features_map,
                "d_concept":          args.d_concept,
                "hidden":             args.hidden,
                "recon_weight":       args.recon_weight,
                "contrastive_weight": args.contrastive_weight,
                "temperature":        args.temperature,
                "epochs":             args.epochs,
                "lr":                 args.lr,
                "batch_size":         args.batch_size,
            },
        }

    # ── Training banner ───────────────────────────────────────────────────
    if rank == 0:
        eff_batch = args.batch_size * world_size
        print(f"\n{'='*70}", flush=True)
        print(f"[C1] Training config", flush=True)
        print(f"     Models:            {', '.join(model_ef_map.keys())}", flush=True)
        print(f"     d_concept:         {args.d_concept}", flush=True)
        print(f"     hidden:            {args.hidden}", flush=True)
        print(f"     epochs:            {args.epochs}", flush=True)
        print(f"     lr:                {args.lr}", flush=True)
        print(f"     batch_size/gpu:    {args.batch_size}  (effective: {eff_batch} × {world_size} GPUs)", flush=True)
        print(f"     recon_weight:      {args.recon_weight}", flush=True)
        print(f"     contrastive_weight:{args.contrastive_weight}", flush=True)
        print(f"     temperature:       {args.temperature}", flush=True)
        print(f"     pooling:           {args.pooling}", flush=True)
        print(f"     checkpoint_pct:    {args.checkpoint_pct}", flush=True)
        if args.loss_threshold:
            print(f"     loss_threshold:    {args.loss_threshold} (20-epoch rolling avg)", flush=True)
        print(f"{'='*70}\n", flush=True)

    # ── Training loop ─────────────────────────────────────────────────────
    recent_val_losses: list = []
    last_epoch = start_epoch          # updated each iteration; used by final checkpoint
    dead_pct   = 0.0                  # dead neuron fraction; updated each val pass
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):

        # Advance epoch so GpuBatchSampler uses a fresh per-epoch permutation
        train_loader.set_epoch(epoch)

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        ep_recon = ep_align = ep_total = 0.0
        n_batches = 0

        for batch in train_loader:
            x_dict = batch  # already on device — no transfer needed
            optimizer.zero_grad(set_to_none=True)

            z_dict, recon_dict = model(x_dict)
            loss, r_loss, a_loss = compute_loss(
                z_dict, recon_dict, x_dict,
                args.recon_weight, args.contrastive_weight, args.temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            ep_recon += r_loss.item()
            ep_align += a_loss.item()
            ep_total += loss.item()
            n_batches += 1

        ep_recon /= max(n_batches, 1)
        ep_align /= max(n_batches, 1)
        ep_total /= max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_recon_per_model: dict = defaultdict(float)
        val_recon = val_align = val_total = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x_dict = batch  # already on device
                z_dict, recon_dict = model(x_dict)
                loss, r_loss, a_loss = compute_loss(
                    z_dict, recon_dict, x_dict,
                    args.recon_weight, args.contrastive_weight, args.temperature,
                )
                val_recon += r_loss.item()
                val_align += a_loss.item()
                val_total += loss.item()
                val_batches += 1
                for m in model_ef_map:
                    val_recon_per_model[m] += (
                        F.mse_loss(recon_dict[m], x_dict[m].float()).item()
                    )
                # Track dead neurons on last val batch
                if val_batches == 1:
                    dead_pct = _count_dead_neurons(z_dict)

        val_recon /= max(val_batches, 1)
        val_align /= max(val_batches, 1)
        val_total /= max(val_batches, 1)
        for m in val_recon_per_model:
            val_recon_per_model[m] /= max(val_batches, 1)
        last_val_recon_per_model = dict(val_recon_per_model)

        # Aggregate val_total across DDP ranks (so best-checkpoint logic is consistent)
        if is_ddp:
            vt = torch.tensor(val_total, device=device)
            dist.all_reduce(vt, op=dist.ReduceOp.AVG)
            val_total_global = vt.item()
        else:
            val_total_global = val_total

        # ── Logging & checkpointing (rank 0 only) ─────────────────────────
        if rank == 0:
            cur_lr   = optimizer.param_groups[0]["lr"]
            elapsed  = time.time() - t_start
            ep_done  = epoch - start_epoch + 1
            eta_secs = (elapsed / ep_done) * (args.epochs - epoch - 1)

            per_model_str = "  ".join(
                f"{m}={val_recon_per_model[m]:.4f}" for m in sorted(model_ef_map)
            )
            print(
                f"[C1] ep {epoch+1:4d}/{args.epochs}"
                f"  train[tot={ep_total:.4f} rec={ep_recon:.4f} aln={ep_align:.4f}]"
                f"  val[tot={val_total:.4f} rec={val_recon:.4f} aln={val_align:.4f}]"
                f"  lr={cur_lr:.2e}  ETA={eta_secs/60:.1f}min",
                flush=True,
            )
            print(f"[C1]      per-model val recon: {per_model_str}", flush=True)

            # -- Write training log every epoch so it's readable mid-run --
            log_entry = {
                "epoch":               epoch + 1,
                "train_total":         ep_total,
                "train_recon":         ep_recon,
                "train_align":         ep_align,
                "val_total":           val_total,
                "val_recon":           val_recon,
                "val_align":           val_align,
                "val_recon_per_model": {m: float(v) for m, v in val_recon_per_model.items()},
                "dead_neuron_pct":     float(dead_pct),
                "lr":                  cur_lr,
                "elapsed_s":           elapsed,
            }
            training_log.append(log_entry)
            with open(training_log_path, "w") as f:
                json.dump(training_log, f, indent=2)

            # -- Best checkpoint --
            if val_total_global < best_val_loss:
                best_val_loss = val_total_global
                sd = _build_state(epoch)
                torch.save(sd, str(ckpt_best_path))
                print(
                    f"[C1]   ✓ New best val_loss={best_val_loss:.4f}"
                    f"  → {ckpt_best_path.name}",
                    flush=True,
                )

            # -- Periodic checkpoint --
            if (epoch + 1) % args.ckpt_every == 0:
                ckpt_ep = universal_dir / f"global_mlp_{tag}_ckpt_ep{epoch+1}.pt"
                torch.save(_build_state(epoch), str(ckpt_ep))
                print(f"[C1]   periodic ckpt: {ckpt_ep.name}", flush=True)

        last_epoch = epoch  # track for final checkpoint save

        # ── Early stopping (rolling 20-epoch avg) ─────────────────────────
        recent_val_losses.append(val_total_global)
        if len(recent_val_losses) > 20:
            recent_val_losses.pop(0)
        if args.loss_threshold and len(recent_val_losses) >= 20:
            rolling_avg = sum(recent_val_losses) / len(recent_val_losses)
            if rolling_avg < args.loss_threshold:
                _log(rank,
                     f"Early stop: 20-epoch avg val_total={rolling_avg:.4f}"
                     f" < threshold={args.loss_threshold}")
                break

    # ── Save final checkpoint + metadata (rank 0 only) ────────────────────
    if rank == 0:
        sd = _build_state(last_epoch)
        torch.save(sd, str(ckpt_final_path))
        _log(rank, f"Final checkpoint saved: {ckpt_final_path}")
        # C1b: also save dedicated recon-only checkpoint with spec-compliant name
        if args.ablation_report and recon_only_path is not None:
            torch.save(sd, str(recon_only_path))
            _log(rank, f"Recon-only checkpoint saved: {recon_only_path}")

        meta = {
            "version":                  version,
            "tag":                      tag,
            "run_id":                   args.run_id,
            "ablation_type":            "recon_only" if args.ablation_report else None,
            "active":                   True,
            "pooling":                  pooling_tag,
            "label_supervised_training": False,  # always contrastive (passage); supervised mode is not implemented
            "checkpoint_pct":           args.checkpoint_pct,
            "models":                   model_ef_map,
            "n_features_per_model":     n_features_map,
            "d_concept":                args.d_concept,
            "hidden":                   args.hidden,
            "training_config": {
                "epochs":                args.epochs,
                "epochs_trained":        len(training_log),
                "lr":                    args.lr,
                "batch_size_per_gpu":    args.batch_size,
                "effective_batch_size":  args.batch_size * world_size,
                "world_size":            world_size,
                "recon_weight":          args.recon_weight,
                "contrastive_weight":    args.contrastive_weight,
                "temperature":           args.temperature,
                "loss_threshold":        args.loss_threshold,
                "hidden":                args.hidden,
            },
            "n_passages_train":         int(len(train_idx)),
            "n_passages_val":           int(len(val_idx)),
            "best_val_loss":            float(best_val_loss),
            "final_val_loss":           float(recent_val_losses[-1]) if recent_val_losses else None,
            "val_recon_per_model":      {m: float(v) for m, v in last_val_recon_per_model.items()},
            "checkpoint_final":         ckpt_final_path.name,
            "checkpoint_best":          ckpt_best_path.name,
            "training_log_file":        training_log_path.name,
            "n_parameters":             _param_count(model.module if is_ddp else model),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        _log(rank, f"Meta written: {meta_path}")

        # ── Artifact registry ─────────────────────────────────────────────
        registry_path = pipeline_dir / "artifacts" / "registry.json"
        _artifact_base = {
            "version": version, "pooling": pooling_tag,
            "d_concept": args.d_concept, "n_models": len(model_ef_map),
        }
        _write_artifact_registry(
            registry_path, "global_mlp_weights",
            str(ckpt_final_path.relative_to(pipeline_dir)),
            f"Global MLP v{version} · {len(model_ef_map)} models · d={args.d_concept} · {pooling_tag}",
            _artifact_base,
        )
        _write_artifact_registry(
            registry_path, "global_mlp_meta",
            str(meta_path.relative_to(pipeline_dir)),
            f"Global MLP v{version} · {pooling_tag} · meta",
            _artifact_base,
        )
        _write_artifact_registry(
            registry_path, "global_mlp_training_log",
            str(training_log_path.relative_to(pipeline_dir)),
            f"Global MLP v{version} · {pooling_tag} · training log",
            _artifact_base,
        )
        if args.ablation_report and recon_only_path is not None:
            _write_artifact_registry(
                registry_path, "c1b_recon_only_weights",
                str(recon_only_path.relative_to(pipeline_dir)),
                f"C1b recon-only v{version} · {len(model_ef_map)} models · d={args.d_concept} · {pooling_tag}",
                {**_artifact_base, "ablation_type": "recon_only"},
            )

        # Machine-readable summary line — parsed by the UI server (_persist_job)
        final_entry = training_log[-1] if training_log else {}
        print(
            "FINAL_STATS:" + json.dumps({
                "step":            "C1",
                "version":         version,
                "tag":             tag,
                "val_total":       final_entry.get("val_total"),
                "val_recon":       final_entry.get("val_recon"),
                "val_align":       final_entry.get("val_align"),
                "best_val_loss":   float(best_val_loss),
                "epochs_trained":  len(training_log),
                "models":          list(model_ef_map.keys()),
                "d_concept":       args.d_concept,
            }),
            flush=True,
        )

        # C1b ablation: write skeleton report (full silhouette computed in C2)
        if args.ablation_report:
            report = {
                "ablation_type":        "recon_only",
                "tag":                  tag,
                "pooling":              pooling_tag,
                "val_recon_per_model":  {m: float(v) for m, v in last_val_recon_per_model.items()},
                "best_val_loss":        float(best_val_loss),
                "recon_only_checkpoint": recon_only_path.name if recon_only_path else None,
                "alignment_loss_necessary": None,  # filled by C2 --c1b-compare
                "note": (
                    "Full silhouette comparison (C1 vs C1b) requires C2 clustering. "
                    "Run C2 with --c1b-compare to complete this report."
                ),
            }
            # Spec path: c1b_ablation_{pooling}_v{N}_report.json
            report_path = universal_dir / f"c1b_ablation_{pooling_tag}_v{version}_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            _log(rank, f"C1b ablation report: {report_path}")
            _write_artifact_registry(
                registry_path, "c1b_ablation_report",
                str(report_path.relative_to(pipeline_dir)),
                f"C1b ablation report v{version} · {pooling_tag}",
                {**_artifact_base, "ablation_type": "recon_only"},
            )

    if is_ddp:
        dist.destroy_process_group()

    _log(rank, "Done.")


if __name__ == "__main__":
    _maybe_relaunch_ddp()   # re-execs via torchrun if >1 GPU and not already DDP
    main()
