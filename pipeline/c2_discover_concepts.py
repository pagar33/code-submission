"""C2 – Discover universal concepts across models.

Algorithm
---------
1. Load C1 global_concept_space.pt (per-model encoders).
2. Project all model feature matrices through their respective encoder → d_concept space.
3. Pool all projected vectors and run HDBSCAN clustering (falls back to DBSCAN if hdbscan not installed).
4. Keep clusters that contain features from ≥ min_models distinct models.
5. Optionally name clusters via LLM (OpenAI API if OPENAI_API_KEY is set and --llm is not 'skip').

Outputs
-------
  features/universal_concepts_{run_id}.json
    {
      "universal_concepts": [
        {
          "cluster_id": int,
          "label": str,
          "models_present": [model_name, ...],
          "center": [...],          # d_concept-dim centroid
          "per_model": {
            model_name: {
              "feature_ids": [...],    # top SAE feature ids in cluster for this model
              "activations": [...],    # mean activation per feature
            }
          }
        },
        ...
      ],
      "run_id": str,
      "n_universal_concepts": int,
      "min_models": int,
    }
"""


import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import glob
import json
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

import config


def log_run(script: str, start: float, status: str, error: str = ""):
    entry = {
        "script": script, "start_time": start,
        "end_time": time.time(), "status": status, "error": error,
    }
    with open("run_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / feature-file helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_c1_checkpoint(universal_dir: str, pooling: str) -> str:
    """
    Auto-locate the best C1 checkpoint in universal_dir.
    Prefers {pooling}_best > {pooling}_final > any best > any final.
    """
    patterns = [
        os.path.join(universal_dir, f"global_mlp_{pooling}_v*_best.pt"),
        os.path.join(universal_dir, f"global_mlp_{pooling}_v*.pt"),
        os.path.join(universal_dir, "global_mlp_*_v*_best.pt"),
        os.path.join(universal_dir, "global_mlp_*_v*.pt"),
    ]
    for pat in patterns:
        candidates = sorted(glob.glob(pat))
        # Pick highest version number
        if candidates:
            def _vnum(p):
                import re
                m = re.search(r'_v(\d+)', os.path.basename(p))
                return int(m.group(1)) if m else 0
            return max(candidates, key=_vnum)
    raise FileNotFoundError(
        f"No C1 checkpoint found in {universal_dir}. Run train_global_mlp.py first."
    )


def _find_feature_file(features_dir: str, model: str, ef: int, suffix: str) -> str:
    """
    Find feature file using A100 naming: {model}_ef{EF}_top{N}_{suffix}.npy
    Returns path of the file with the largest top-N.
    Falls back to legacy naming {model}_top{N}_{suffix}.npy.
    """
    # A100-style naming
    pattern = os.path.join(features_dir, f"{model}_ef{ef}_top*_{suffix}.npy")
    candidates = glob.glob(pattern)
    if not candidates:
        # Legacy naming (local dev)
        pattern = os.path.join(features_dir, f"{model}_top*_{suffix}.npy")
        candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No {suffix} file for {model} ef{ef} in {features_dir}")
    import re
    def _topn(p):
        m = re.search(r'_top(\d+)_', os.path.basename(p))
        return int(m.group(1)) if m else 0
    return max(candidates, key=_topn)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder — must exactly match train_global_mlp.py _make_encoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Mirrors GlobalMLP._make_encoder: Linear→LayerNorm→GELU→Dropout→Linear."""
    def __init__(self, in_dim: int, d_concept: int, hidden: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, d_concept),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_encoders_from_checkpoint(ckpt: dict, device: torch.device) -> tuple:
    """
    Extract per-model Encoder instances from a C1 GlobalMLP checkpoint.
    Returns (encoders_dict, d_concept, n_features_map, model_ef_map).
    """
    cfg = ckpt.get("config", {})
    d_concept  = cfg.get("d_concept", 512)
    hidden     = cfg.get("hidden", 2048)
    n_features_map = cfg.get("n_features_map", {})
    model_ef_map   = cfg.get("model_ef_map", {})
    state_dict = ckpt["model_state_dict"]

    encoders: Dict[str, Encoder] = {}
    for model_name, n_feat in n_features_map.items():
        prefix = f"encoders.{model_name}."
        enc_state = {
            k[len(prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        if not enc_state:
            print(f"[C2] WARNING: no encoder keys found for {model_name} in checkpoint")
            continue
        # C1 stores encoders as plain nn.Sequential in ModuleDict,
        # so keys are "0.weight", "1.weight" etc. Remap to "net.0.weight" for Encoder wrapper.
        enc_state_remapped = {f"net.{k}": v for k, v in enc_state.items()}
        enc = Encoder(n_feat, d_concept, hidden)
        enc.load_state_dict(enc_state_remapped)
        enc.eval().to(device)
        encoders[model_name] = enc

    return encoders, d_concept, n_features_map, model_ef_map


# ─────────────────────────────────────────────────────────────────────────────
# Projection
# ─────────────────────────────────────────────────────────────────────────────

def _project_features_gpu(
    feature_mats: Dict[str, np.ndarray],
    encoders: Dict[str, Encoder],
    device: torch.device,
    batch_size: int = 2048,
) -> Dict[str, np.ndarray]:
    """
    Project each model's feature matrix through its encoder on GPU.
    Input:  feature_mats[m] shape (N_passages, k_features)
    Output: projections[m]  shape (N_passages, d_concept)  — numpy, on CPU
    """
    projections: Dict[str, np.ndarray] = {}
    for m, mat in feature_mats.items():
        enc = encoders[m]   # already on device
        n = mat.shape[0]
        out_parts = []
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(mat[i:i + batch_size]).to(device)
            with torch.no_grad():
                z = enc(batch)          # (B, d_concept) on GPU
            out_parts.append(z.cpu().numpy())
        projections[m] = np.concatenate(out_parts, axis=0)
        print(f"[C2]   projected {m:25s}  {n:>7,} passages  →  {projections[m].shape[1]}-dim")
    return projections


def _probe_features_gpu(
    feature_mats: Dict[str, np.ndarray],
    encoders: Dict[str, Encoder],
    device: torch.device,
    top_pct: float = 0.10,
    batch_size: int = 4096,
) -> tuple:
    """
    Build one concept-space point per SAE feature via passage-level projection.

    For each model m and feature k:
      1. Project all passages through encoder → Z_m  shape (N, d_concept)
      2. For feature k, take passages where feature_acts_m[:, k] is in the top
         top_pct percentile (feature is highly active).
      3. concept-space position for feature k = mean(Z_m[top_passages])

    This is in-distribution (encoder sees real dense activation vectors) and
    semantically meaningful: features that activate on the same passages get
    similar concept-space positions across models, because C1 contrastive
    training co-locates same-passage embeddings from different models.

    Returns: (all_vecs [sum_k, d_concept], feature_model_labels, feature_col_labels)
    """
    all_proj_parts = []
    feature_model_labels: List[str] = []
    feature_col_labels:   List[int]  = []

    for m, mat in feature_mats.items():
        N, k = mat.shape
        enc = encoders[m]   # already on device

        # Step 1: project all passages through encoder (batched)
        z_parts = []
        for i in range(0, N, batch_size):
            batch = torch.from_numpy(mat[i:i + batch_size]).to(device)
            with torch.no_grad():
                z_parts.append(enc(batch).cpu().numpy())
        Z = np.concatenate(z_parts, axis=0)   # (N, d_concept)

        # Step 2: per-feature mean over top-activating passages
        threshold = np.quantile(mat, 1.0 - top_pct, axis=0)   # (k,)
        feature_vecs = np.zeros((k, Z.shape[1]), dtype=np.float32)
        for fi in range(k):
            top_mask = mat[:, fi] >= threshold[fi]
            if top_mask.sum() > 0:
                feature_vecs[fi] = Z[top_mask].mean(axis=0)
            else:
                feature_vecs[fi] = Z.mean(axis=0)   # fallback: global mean

        all_proj_parts.append(feature_vecs)
        feature_model_labels.extend([m] * k)
        feature_col_labels.extend(list(range(k)))
        print(f"[C2]   feature-vecs {m:25s}  {k} features  "
              f"(top {int(top_pct*100)}% passages each)  →  {Z.shape[1]}-dim")

    all_vecs = np.concatenate(all_proj_parts, axis=0)   # (sum_k, d_concept)
    return all_vecs, feature_model_labels, feature_col_labels


# ─────────────────────────────────────────────────────────────────────────────
# UMAP reduction (reduces curse of dimensionality before HDBSCAN)
# ─────────────────────────────────────────────────────────────────────────────

def _reduce_umap(vecs: np.ndarray, n_components: int = 30, seed: int = 42) -> np.ndarray:
    """
    Reduce high-dimensional concept vectors to n_components dims via UMAP.
    HDBSCAN with Euclidean distance fails in 512-d (curse of dimensionality) —
    all points appear equidistant, producing ~58% noise. Reducing to 20-50d
    first gives HDBSCAN well-separated density regions to work with.
    Returns float32 array of shape (N, n_components).
    """
    try:
        from umap import UMAP
    except ImportError:
        print("[C2] umap-learn not installed — skipping UMAP reduction (clustering may be poor in high-d).")
        print("[C2]   Install with: pip install umap-learn")
        return vecs
    n_neighbors = max(5, min(50, vecs.shape[0] // 20))
    print(f"[C2] UMAP reduction: {vecs.shape[1]}-d → {n_components}-d  (n_neighbors={n_neighbors}, N={vecs.shape[0]})")
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,       # 0.0 preserves local structure better for clustering
        metric="cosine",    # cosine suits normalised concept vectors better than euclidean
        random_state=seed,
        low_memory=False,
    )
    reduced = reducer.fit_transform(vecs.astype(np.float32))
    print(f"[C2] UMAP done — reduced shape: {reduced.shape}")
    return reduced.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────

def _cluster(
    all_vecs: np.ndarray,
    min_cluster_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    """Run HDBSCAN (with sklearn fallback). Returns label array of shape (N,)."""
    try:
        import hdbscan  # type: ignore
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples or max(1, min_cluster_size // 2),
            metric="euclidean",
        )
        labels = clusterer.fit_predict(all_vecs.astype(np.float64))
    except ImportError:
        print("[C2] hdbscan not found – falling back to sklearn DBSCAN")
        from sklearn.cluster import DBSCAN
        eps = float(np.percentile(np.std(all_vecs, axis=0), 50)) * 0.5 + 1e-4
        clusterer = DBSCAN(eps=eps, min_samples=min_cluster_size, n_jobs=-1)
        labels = clusterer.fit_predict(all_vecs)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# LLM naming
# ─────────────────────────────────────────────────────────────────────────────

# ── Labeling prompt (pinned for NeurIPS reproducibility) ─────────────────────
_LABEL_PROMPT = """\
You are labeling SAE (Sparse Autoencoder) features that cluster together across \
5 different LLMs (GPT-2-large, Gemma, Llama, Mistral, DeepSeek). Features in the \
same cluster activate on the same type of text across ALL models, meaning this \
cluster represents a universal concept shared across architectures.

Below are example passages this cluster activates strongly on:
---
{passages}
---

Give a SHORT 2-4 word label for this universal concept.
Use snake_case. The label should describe what type of text these passages share.
A concept may span multiple domains (e.g. "sports_and_finance", "cooking_and_science").
Reply with ONLY the label, nothing else."""

_CLAUDE_MODEL = "claude-sonnet-4-5"  # pinned — change only with paper revision


def _name_cluster_llm(
    cluster_id: int,
    feature_ids: Dict[str, List[int]],
    api_key: str,
    passages: Optional[List[str]] = None,
) -> str:
    """Ask Claude to name a concept cluster given example passages from the corpus."""
    try:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=api_key)
        passage_block = "\n".join(
            f"[{i+1}] {p[:280].replace(chr(10), ' ')}"
            for i, p in enumerate((passages or [])[:6])
        ) or "(no passages available — infer from cluster id)"
        prompt = _LABEL_PROMPT.format(passages=passage_block)
        resp = client.messages.create(
            model=_CLAUDE_MODEL,
            max_tokens=20,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip().strip('"').replace(' ', '_')
    except Exception as e:
        print(f"  [warn] LLM labeling failed for cluster {cluster_id}: {e}")
        return f"cluster_{cluster_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Discover universal concepts (C2)")
    parser.add_argument("--models", nargs="+", default=list(config.MODELS.keys()))
    parser.add_argument("--global-space-path", default="",
                        help="Path to C1 GlobalMLP checkpoint. Auto-detected if omitted.")
    parser.add_argument("--features-dir", default="",
                        help="Override features directory (default: pipeline features/).")
    parser.add_argument("--pooling", default="mean",
                        help="Pooling tag inherited from A2, used in output filename (default: mean).")
    parser.add_argument("--min-models", type=int, default=2,
                        help="Minimum number of models a cluster must span (default: 2)")
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="HDBSCAN min cluster size (default: 5)")
    parser.add_argument("--umap-dim", type=int, default=30,
                        help="Reduce concept vectors to this many dims via UMAP before HDBSCAN. "
                             "Set to 0 to skip UMAP (not recommended for d_concept>64). Default: 30.")
    parser.add_argument("--llm", default="skip",
                        choices=["skip", "claude"],
                        help="LLM to use for cluster naming (default: skip)")
    parser.add_argument("--anthropic-api-key", default="", dest="anthropic_api_key",
                        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed(42)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[C2] Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    os.makedirs(config.UNIVERSAL_DIR, exist_ok=True)

    # ── Output path (spec: universal/{pooling}_concepts.json) ─────────────
    suffix = f"_{args.run_id}" if args.run_id else ""
    out_path = os.path.join(config.UNIVERSAL_DIR, f"{args.pooling}_concepts{suffix}.json")
    if os.path.exists(out_path) and not args.force:
        print(f"{os.path.basename(out_path)} exists. Use --force to recompute.")
        log_run("universal_discover.py", start, "skipped")
        return 0

    # ── Locate C1 checkpoint ───────────────────────────────────────────────
    ckpt_path = args.global_space_path or _find_c1_checkpoint(config.UNIVERSAL_DIR, args.pooling)
    print(f"[C2] Loading C1 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    encoders, d_concept, n_features_map, model_ef_map = _load_encoders_from_checkpoint(ckpt, device)
    ckpt_models = list(encoders.keys())
    active_models = [m for m in args.models if m in ckpt_models]
    if not active_models:
        raise ValueError(f"No overlap between --models {args.models} and checkpoint models {ckpt_models}")
    print(f"[C2] d_concept={d_concept}  active models: {active_models}")

    # ── Features directory & file loading ─────────────────────────────────
    features_dir = args.features_dir or config.FEATURES_DIR

    # Load feature matrices
    feature_mats: Dict[str, np.ndarray] = {}
    feature_idx_map: Dict[str, np.ndarray] = {}
    for m in active_models:
        ef = model_ef_map.get(m, 64)
        try:
            mat_path = _find_feature_file(features_dir, m, ef, "feature_acts")
            idx_path = _find_feature_file(features_dir, m, ef, "feature_idx")
        except FileNotFoundError as e:
            print(f"[C2] WARNING: {e} — skipping {m}")
            continue
        feature_mats[m] = np.load(mat_path).astype(np.float32)
        feature_idx_map[m] = np.load(idx_path)
        print(f"[C2]   {m:25s}  {feature_mats[m].shape[0]:>7,} × {feature_mats[m].shape[1]:>4}  ← {mat_path}")

    if len(feature_mats) < 2:
        raise RuntimeError(f"Need at least 2 models with feature files, got {list(feature_mats.keys())}")

    # ── Project features into concept space (GPU) ─────────────────────────
    # One point per SAE feature (probe strategy): e_c * mean_activation[c]
    # This clusters features by their position in concept space, not passages.
    print("[C2] Projecting features into concept space...")
    all_vecs, feature_model_labels, feature_col_labels = _probe_features_gpu(
        feature_mats, encoders, device
    )
    print(f"[C2] Total feature vectors: {all_vecs.shape[0]}, d_concept={d_concept}")

    # ── Optionally reduce dimensions before clustering ─────────────────────
    # HDBSCAN in 512-d suffers from the curse of dimensionality — nearly all
    # points appear equidistant, leading to 50-80% noise. UMAP to 20-50d first.
    if args.umap_dim > 0 and all_vecs.shape[1] > args.umap_dim:
        vecs_for_clustering = _reduce_umap(all_vecs, n_components=args.umap_dim)
    else:
        vecs_for_clustering = all_vecs

    # Cluster in (reduced) concept space
    labels_arr = _cluster(vecs_for_clustering, min_cluster_size=args.min_cluster_size)
    unique_labels = set(labels_arr)
    unique_labels.discard(-1)  # -1 = noise

    print(f"[C2] Found {len(unique_labels)} clusters (noise={int((labels_arr == -1).sum())})")

    # Build universal concepts
    universal_concepts = []
    api_key = (
        getattr(args, "anthropic_api_key", "")
        or os.getenv("ANTHROPIC_API_KEY", "")
    )
    use_llm = args.llm == "claude" and bool(api_key)

    for cluster_id in sorted(unique_labels):
        mask = labels_arr == cluster_id
        models_in_cluster = set(feature_model_labels[i] for i in range(len(mask)) if mask[i])
        if len(models_in_cluster) < args.min_models:
            continue

        center = all_vecs[mask].mean(axis=0).tolist()  # always use original 512-d center

        per_model: Dict[str, dict] = {}
        for m in models_in_cluster:
            m_mask = np.array([
                mask[i] and feature_model_labels[i] == m
                for i in range(len(mask))
            ])
            # Global indices → feature column indices for this model
            m_col_start = sum(len(feature_mats[mm]) for mm in feature_mats if mm < m)
            col_indices = [
                feature_col_labels[i]
                for i in range(len(mask))
                if mask[i] and feature_model_labels[i] == m
            ]
            feat_ids = [int(feature_idx_map[m][c]) for c in col_indices]
            mean_acts = [float(feature_mats[m][:, c].mean()) for c in col_indices]
            per_model[m] = {"feature_ids": feat_ids, "activations": mean_acts}

        # Name — passages will be populated by label_universal_concepts.py
        # or a subsequent relabeling pass; here we just assign a placeholder
        if use_llm:
            label = _name_cluster_llm(
                cluster_id,
                {m: v["feature_ids"] for m, v in per_model.items()},
                api_key,
                passages=None,   # no corpus loaded at this stage; relabeling pass adds passages
            )
        else:
            label = f"cluster_{cluster_id}"

        universal_concepts.append({
            "cluster_id": int(cluster_id),
            "label": label,
            "models_present": sorted(models_in_cluster),
            "center": center,
            "per_model": per_model,
        })

    print(f"[C2] Universal concepts (≥{args.min_models} models): {len(universal_concepts)}")

    output = {
        "universal_concepts": universal_concepts,
        "run_id": args.run_id,
        "pooling": args.pooling,
        "n_universal_concepts": len(universal_concepts),
        "min_models": args.min_models,
        "c1_checkpoint": os.path.basename(ckpt_path),
        "d_concept": d_concept,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[C2] Saved {out_path}")
    log_run("universal_discover.py", start, "success")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log_run("universal_discover.py", time.time(), "error", str(e))
        raise
