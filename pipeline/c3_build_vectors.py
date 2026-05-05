#!/usr/bin/env python3
"""
C3 — Build Universal Steering Vectors
=======================================
Three modes, selected via --mode:

  decoder_only  (default)
    Reads C2 concept clusters from ``universal/mean_concepts_clean.json``
    (written by dedup_concepts.py).

  sign_fix
    Post-processing pass on an existing decoder_only output file.
    For each (model, concept) vector, checks dot product with the native sae_vector.
    If negative, flips sign.  No SAE or GlobalMLP needed — pure JSON → JSON.
    Writes: steering/universal_steering_vectors_v1.json  (in-place, atomic replace)
    Logs how many vectors were flipped.

  enc_dec
    Encoder-decoder pipeline:
      guide sae_vector  →  GlobalMLP.encoders[guide]  →  concept-space direction
      →  GlobalMLP.decoders[target]  →  target SAE features  →  SAE decoder
      →  target activation-space steering vector
    Averages over all available guides (excluding the target itself).
    Polarity is correct by construction (inherited from signed native vectors).
    Writes: steering/universal_steering_vectors_enc_dec_v1.json

Usage:
  # Step 1 — already done: decoder_only
  python build_universal_vectors.py --mode decoder_only --run-id v1 ...

  # Step 2 — fix sign of existing file (fast, no GPU needed)
  python build_universal_vectors.py --mode sign_fix --run-id v1

  # Step 3 — build enc-dec variant (needs GlobalMLP checkpoint + SAEs)
  python build_universal_vectors.py --mode enc_dec --run-id v1 ...
"""


import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn

import config


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def log_run(script: str, start_time: float, status: str, error: str = "") -> None:
    entry = {
        "script": script,
        "start_time": start_time,
        "end_time": time.time(),
        "status": status,
        "error": error,
    }
    with open("run_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# GlobalMLP architecture — must mirror train_global_mlp.py exactly
# ---------------------------------------------------------------------------

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
    """Multi-model encoder-decoder for the universal concept space (mirrors C1)."""

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
            z = self.encoders[m](x_dict[m])
            recon_dict[m] = self.decoders[m](z)
            z_dict[m] = z
        return z_dict, recon_dict


# ---------------------------------------------------------------------------
# Concept name mapping (module-level, populated once)
# HDBSCAN canonical concept → A5/B3 native sv key candidates
# ---------------------------------------------------------------------------
_HDBSCAN_TO_NATIVE: Dict[str, List[str]] = {
    "python_code":              ["code_python", "code_instructions", "code_snippets",
                                 "pythoncoding", "coding"],
    "math_problems":            ["math_gsm8k", "math_reasoning", "math_competition",
                                 "math_olympiad"],
    "sql_queries":              ["code_sql"],
    "legal_and_news":           ["legal", "news_reporting"],
    "medical_research":         ["science_biomedical"],
    "academic_scientific":      ["academic_writing"],
    "narrative_fiction":        ["creative_writing"],
    "encyclopedic_historical":  ["news_reporting"],
    "code_and_math":            ["coding", "math_reasoning", "code_python"],
    "customer_reviews":         ["sentiment"],
    "sql_and_medical":          ["code_sql", "science_biomedical"],
}


# ---------------------------------------------------------------------------
# SAE loading helper (mirrors step7)
# ---------------------------------------------------------------------------

def _load_sae(model_name: str, device: torch.device):
    from step3_train_sae import TopKSAE
    sae_dir = config.SAE_DIR
    ef = config.MODELS[model_name]["sae_ef"]
    # Try canonical name first, then glob
    direct = os.path.join(sae_dir, f"{model_name}_ef{ef}_sae.pt")
    if not os.path.exists(direct):
        cands = sorted(glob.glob(os.path.join(sae_dir, f"{model_name}_ef*_sae.pt")))
        if not cands:
            raise FileNotFoundError(f"No SAE found for {model_name} in {sae_dir}")
        direct = cands[-1]
    state = torch.load(direct, map_location=device, weights_only=True)
    hidden_dim = config.MODELS[model_name]["hidden_dim"]
    n_features = hidden_dim * ef
    top_k = state.get("top_k", 64)
    sae = TopKSAE(hidden_dim, n_features, top_k).to(device)
    if "model_state_dict" in state:
        sae.load_state_dict(state["model_state_dict"])
    else:
        sae.load_state_dict(state)
    sae.eval()
    return sae


# ---------------------------------------------------------------------------
# Mode 1: decoder_only  (original pipeline — kept for reference / re-run)
# ---------------------------------------------------------------------------

def _load_feature_idx(model_name: str, k_expected: int) -> Optional[np.ndarray]:
    """Load address-book array mapping column → SAE feature id."""
    features_dir = config.FEATURES_DIR
    k_path = os.path.join(features_dir, f"{model_name}_top{k_expected}_feature_idx.npy")
    if os.path.exists(k_path):
        return np.load(k_path)
    hits = sorted(glob.glob(os.path.join(features_dir, f"{model_name}_top*_feature_idx.npy")))
    if hits:
        print(f"[C3-dec] {model_name}: using fallback feature_idx {os.path.basename(hits[-1])}")
        return np.load(hits[-1])
    return None


def mode_decoder_only(args) -> int:
    """Original C3: centroid → GlobalMLP.decoders[target] → SAE decoder."""
    concepts_path = os.path.join(config.UNIVERSAL_DIR, args.concepts_file)
    ckpt_path = (
        args.checkpoint
        or os.path.join(config.UNIVERSAL_DIR, "global_mlp_last_token_v1_best.pt")
    )
    suffix = f"_v{args.run_id}" if args.run_id else ""
    out_path = os.path.join(config.STEERING_DIR, f"universal_steering_vectors{suffix}.json")

    if not os.path.exists(concepts_path):
        raise FileNotFoundError(f"Missing {concepts_path} — run universal_discover.py first")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing {ckpt_path} — run train_global_mlp.py first")

    with open(concepts_path) as f:
        concepts_data = json.load(f)
    universal_concepts: List[dict] = concepts_data["universal_concepts"]

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c1_cfg = raw["config"]
    d_concept: int = c1_cfg["d_concept"]
    n_features_map: Dict[str, int] = c1_cfg["n_features_map"]
    hidden: int = c1_cfg.get("hidden", 2048)

    global_mlp = GlobalMLP(n_features_map, d_concept, hidden)
    global_mlp.load_state_dict(raw["model_state_dict"])
    global_mlp.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_mlp.to(device)

    models_to_run: List[str] = [
        m for m in sorted(args.models)
        if m in n_features_map and m in config.MODELS
    ]

    labels_ordered  = [c["label"]          for c in universal_concepts]
    cluster_ids     = [c["cluster_id"]     for c in universal_concepts]
    models_present  = [c["models_present"] for c in universal_concepts]
    centers_t = torch.tensor(
        np.stack([c["center"] for c in universal_concepts], axis=0),
        dtype=torch.float32, device=device,
    )

    # Load existing output so incremental runs skip completed models
    existing_output: dict = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing_output = json.load(f)

    results: Dict[str, Dict[str, dict]] = {lbl: {} for lbl in labels_ordered}
    for lbl in labels_ordered:
        results[lbl].update(existing_output.get("universal_steering_vectors", {}).get(lbl, {}))

    def _save():
        n_total = sum(len(v) for v in results.values())
        out = {
            **{k: v for k, v in existing_output.items()
               if k not in ("universal_steering_vectors", "n_total_vectors")},
            "universal_steering_vectors": results,
            "run_id": args.run_id,
            "mode": "decoder_only",
            "n_concepts": len(results),
            "n_total_vectors": n_total,
        }
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, out_path)
        print(f"       [saved] {out_path} ({n_total} vectors)")

    for m in models_to_run:
        already_have = all(
            m in results.get(lbl, {})
            for lbl, mp in zip(labels_ordered, models_present) if m in mp
        )
        if already_have and not args.force:
            print(f"[C3-dec] {m}: already computed — skipping")
            continue

        print(f"[C3-dec] {m}: loading SAE …")
        try:
            sae = _load_sae(m, device)
        except FileNotFoundError as e:
            print(f"[C3-dec] WARNING: {e} — skipping {m}")
            continue

        feat_idx = _load_feature_idx(m, n_features_map[m])
        full_n_features = config.MODELS[m]["hidden_dim"] * config.MODELS[m]["sae_ef"]

        concept_indices = [i for i, mp in enumerate(models_present) if m in mp]
        if not concept_indices:
            del sae; torch.cuda.empty_cache(); continue

        dec = global_mlp.decoders[m]
        batch_centers = centers_t[concept_indices]

        with torch.no_grad():
            feat_vecs = dec(batch_centers)
            n_top = feat_vecs.shape[1]
            full_feats = torch.zeros(len(concept_indices), full_n_features,
                                     dtype=torch.float32, device=device)
            if feat_idx is not None:
                k = min(n_top, len(feat_idx))
                feat_idx_t = torch.from_numpy(feat_idx).long().to(device)
                full_feats[:, feat_idx_t[:k]] = feat_vecs[:, :k]
            else:
                k = min(n_top, full_n_features)
                full_feats[:, :k] = feat_vecs[:, :k]
            steering_vecs = sae.decoder(full_feats)

        steering_np = steering_vecs.cpu().float().numpy()
        for pos, ci in enumerate(concept_indices):
            lbl = labels_ordered[ci]
            results[lbl][m] = {
                "steering_vector": _l2_normalize(steering_np[pos]).tolist(),
                "cluster_id": cluster_ids[ci],
                "models_present": models_present[ci],
                "n_top_features": int(n_top),
                "mode": "decoder_only",
            }

        del sae, full_feats, steering_vecs, feat_vecs
        torch.cuda.empty_cache()
        _save()

    print(f"[C3-dec] Done. Output: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Mode 2: sign_fix  — in-place polarity correction, CPU-only, fast
# ---------------------------------------------------------------------------

def mode_sign_fix(args) -> int:
    """
    Post-processing pass: for each (model, concept) C3 decoder-only vector,
    align sign to the native sae_vector via dot product.

    Justification: the cluster centroid has no inherent polarity (it is a
    position, not a direction). Aligning to the independently-computed native
    direction (pos_activations − neg_activations) is analogous to PCA sign
    disambiguation — it does not use evaluation scores.

    Writes the corrected vectors back to the same file atomically.
    """
    suffix = f"_v{args.run_id}" if args.run_id else ""
    src_path = os.path.join(config.STEERING_DIR, f"universal_steering_vectors{suffix}.json")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Missing {src_path} — run decoder_only mode first")

    with open(src_path) as f:
        data = json.load(f)

    # Load all native steering vectors
    sv: Dict[str, Dict] = {}
    for m in config.MODELS:
        ef = config.MODELS[m]["sae_ef"]
        sv_path = os.path.join(config.STEERING_DIR, f"{m}_ef{ef}_steering_vectors.json")
        if not os.path.exists(sv_path):
            cands = sorted(glob.glob(os.path.join(config.STEERING_DIR,
                                                   f"{m}_ef*_steering_vectors.json")))
            sv_path = cands[-1] if cands else None
        if sv_path and os.path.exists(sv_path):
            all_sv = json.load(open(sv_path))
            sv[m] = all_sv.get(m, all_sv)  # handle model-wrapped or flat

    n_checked = 0
    n_flipped = 0
    n_no_native = 0

    def _find_native_vec(model_name: str, concept: str):
        """Try direct key match first, then HDBSCAN→A5 name mapping.
        If multiple A5 candidates exist, average them (they should be near-parallel).
        """
        m_sv = sv.get(model_name, {})
        # 1. Direct match (e.g. if concept is already an A5 name)
        direct = m_sv.get(concept, {})
        v = direct.get("sae_vector") or direct.get("caa_vector")
        if v is not None:
            return v
        # 2. HDBSCAN → A5 mapping
        candidates = _HDBSCAN_TO_NATIVE.get(concept, [])
        found = []
        for cand in candidates:
            entry_sv = m_sv.get(cand, {})
            v = entry_sv.get("sae_vector") or entry_sv.get("caa_vector")
            if v is not None:
                found.append(np.array(v, dtype=np.float32))
        if not found:
            return None
        # Average L2-normalised candidates → single reference direction
        avg = np.mean([_l2_normalize(f) for f in found], axis=0).astype(np.float32)
        return _l2_normalize(avg).tolist()

    for concept, m_dict in data["universal_steering_vectors"].items():
        for model_name, entry in m_dict.items():
            vec = np.array(entry["steering_vector"], dtype=np.float32)
            native = _find_native_vec(model_name, concept)
            n_checked += 1
            if native is None:
                n_no_native += 1
                continue
            native_np = np.array(native, dtype=np.float32)
            dot = float(np.dot(vec, native_np))
            if dot < 0:
                vec = -vec
                n_flipped += 1
            entry["steering_vector"] = vec.tolist()
            entry["sign_aligned"] = True
            entry["sign_dot_product"] = round(dot, 6)

    data["sign_fix_applied"] = True
    data["sign_fix_n_checked"] = n_checked
    data["sign_fix_n_flipped"] = n_flipped
    data["sign_fix_n_no_native"] = n_no_native

    tmp = src_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, src_path)

    print(f"[C3-fix] Sign alignment complete.")
    print(f"         Checked : {n_checked}")
    print(f"         Flipped : {n_flipped}  ({round(100*n_flipped/max(n_checked,1),1)}%)")
    print(f"         No native: {n_no_native} (kept as-is)")
    print(f"         Written  : {src_path}")
    return 0


# ---------------------------------------------------------------------------
# Mode 3: enc_dec  — full encoder-decoder pipeline, correct polarity
# ---------------------------------------------------------------------------

def mode_enc_dec(args) -> int:
    """
    C3-Enc-Dec: for each target model × concept, encode all guide models'
    native sae_vectors through GlobalMLP.encoders[guide], then decode through
    GlobalMLP.decoders[target], then through target SAE decoder.

    Average the guide-derived vectors (L2-normalised before averaging).
    Polarity is correct by construction — inherits from signed native vectors.

    Key reviewer justification:
      "We feed the guide's native steering direction (computed via CAA/SAE)
       through the guide's GlobalMLP encoder to obtain a direction in
       universal concept space, then decode it for the target model.
       The GlobalMLP encoder was trained with NT-Xent contrastive loss to
       align same-passage representations across architectures, making the
       shared space a valid transport layer for steering directions."

    Writes: steering/universal_steering_vectors_enc_dec_v{run_id}.json

    Concept list is derived from the native steering vector files
    ({model}_ef{N}_steering_vectors.json), NOT from mean_concepts_clean.json.
    This ensures concept names match A5/B3 evaluation exactly.
    """
    ckpt_path = (
        args.checkpoint
        or os.path.join(config.UNIVERSAL_DIR, "global_mlp_last_token_v1_best.pt")
    )
    suffix = f"_v{args.run_id}" if args.run_id else ""
    out_path = os.path.join(config.STEERING_DIR,
                            f"universal_steering_vectors_enc_dec{suffix}.json")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c1_cfg = raw["config"]
    d_concept: int = c1_cfg["d_concept"]
    n_features_map: Dict[str, int] = c1_cfg["n_features_map"]
    hidden: int = c1_cfg.get("hidden", 2048)

    global_mlp = GlobalMLP(n_features_map, d_concept, hidden)
    global_mlp.load_state_dict(raw["model_state_dict"])
    global_mlp.eval()

    mlp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_mlp.to(mlp_device)
    # SAEs are large (up to ~17 GB each for llama/mistral/deepseek on ef=128).
    # enc_dec does single-sample matrix multiplies — CPU is fast enough and avoids
    # VRAM exhaustion when 4+ SAEs are cached simultaneously.
    sae_device = torch.device("cpu")

    # Load all native steering vectors (A5/B3 concept names)
    # File: steering/{model}_ef{N}_steering_vectors.json
    # Structure: { "{model_name}": { "{concept}": { "sae_vector": [...], ... } } }
    sv: Dict[str, Dict] = {}
    for m in config.MODELS:
        ef = config.MODELS[m]["sae_ef"]
        sv_path = os.path.join(config.STEERING_DIR, f"{m}_ef{ef}_steering_vectors.json")
        if not os.path.exists(sv_path):
            cands = sorted(glob.glob(os.path.join(config.STEERING_DIR,
                                                   f"{m}_ef*_steering_vectors.json")))
            sv_path = cands[-1] if cands else None
        if sv_path and os.path.exists(sv_path):
            all_sv = json.load(open(sv_path))
            sv[m] = all_sv.get(m, all_sv)

    # Derive concept list from native sv files (A5/B3 names: code_python, math_gsm8k, ...)
    # A concept is eligible if at least 2 models have a non-null sae_vector for it.
    concept_to_models: Dict[str, List[str]] = {}
    for m, m_concepts in sv.items():
        if m not in n_features_map:
            continue
        for concept, vectors in m_concepts.items():
            if vectors.get("sae_vector") is not None or vectors.get("caa_vector") is not None:
                concept_to_models.setdefault(concept, []).append(m)
    # Keep only concepts with ≥2 models (need at least one guide + one target)
    eligible_concepts = sorted(
        c for c, models in concept_to_models.items() if len(models) >= 2
    )
    print(f"[C3-enc-dec] {len(eligible_concepts)} eligible A5/B3 concepts "
          f"(≥2 models with native vector)")
    for c in eligible_concepts:
        print(f"             {c:35s}  models: {sorted(concept_to_models[c])}")

    models_to_run: List[str] = [
        m for m in sorted(args.models)
        if m in n_features_map and m in config.MODELS
    ]

    # Load existing output for incremental runs
    existing_output: dict = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing_output = json.load(f)

    results: Dict[str, Dict[str, dict]] = {c: {} for c in eligible_concepts}
    for c in eligible_concepts:
        results[c].update(
            existing_output.get("universal_steering_vectors", {}).get(c, {}))

    def _save():
        n_total = sum(len(v) for v in results.values())
        out = {
            "universal_steering_vectors": results,
            "run_id": args.run_id,
            "mode": "enc_dec",
            "concept_source": "native_sv_files",
            "n_concepts": len(results),
            "n_total_vectors": n_total,
        }
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, out_path)
        print(f"       [saved] {out_path} ({n_total} vectors)")

    import gc
    import torch.nn.functional as F

    # Target-model outer loop: load one target SAE at a time.
    # Guide SAEs are processed ONE AT A TIME (load → encode all concepts → free).
    # This caps peak CPU RAM at: target_sae + one_guide_sae ≈ 32 GB for 7B models,
    # vs the old approach (cache all guides) which required ~51 GB for mistral target.

    for tgt in models_to_run:
        if not args.force:
            already = all(
                tgt in results.get(c, {})
                for c in eligible_concepts if tgt in concept_to_models.get(c, [])
            )
            if already:
                print(f"[C3-enc-dec] target={tgt}: already computed — skipping")
                continue

        print(f"[C3-enc-dec] target={tgt}: loading SAE (cpu) …")
        try:
            tgt_sae = _load_sae(tgt, sae_device)
        except FileNotFoundError as e:
            print(f"[C3-enc-dec] WARNING: {e} — skipping target {tgt}")
            continue

        tgt_feat_idx = _load_feature_idx(tgt, n_features_map[tgt])
        full_n_tgt = config.MODELS[tgt]["hidden_dim"] * config.MODELS[tgt]["sae_ef"]

        # --- Guide-first pass: load one guide SAE → encode all concepts → free SAE ---
        # This keeps peak CPU RAM at target_sae + one_guide_sae (≤ 32 GB for 7B targets)
        # instead of target_sae + all_guide_saes_simultaneously (up to 51 GB).
        # guide_encoded[guide][concept] = decoded target-space steering vector (np float32)
        all_guides_needed: List[str] = sorted({
            g
            for lbl in eligible_concepts
            for g in concept_to_models.get(lbl, [])
            if g != tgt and g in sv and g in n_features_map
        })
        guide_encoded: Dict[str, Dict[str, np.ndarray]] = {}

        for guide in all_guides_needed:
            try:
                guide_sae = _load_sae(guide, sae_device)
                print(f"[C3-enc-dec] loaded guide SAE (cpu): {guide}")
            except FileNotFoundError as e:
                print(f"[C3-enc-dec] WARNING: {e} — skipping guide {guide}")
                continue

            guide_feat_idx = _load_feature_idx(guide, n_features_map[guide])
            k_guide = n_features_map[guide]
            guide_encoded[guide] = {}

            with torch.no_grad():
                for lbl in eligible_concepts:
                    # Use sae_vector ONLY — it originates from SAE feature space
                    # (pos_features − neg_features → SAE decoder), so re-encoding
                    # through the SAE encoder is geometrically meaningful.
                    # caa_vector is a pure residual-stream construct that never
                    # touched SAE features — encoding it through SAE encoder is wrong.
                    native_hidden = sv.get(guide, {}).get(lbl, {}).get("sae_vector")
                    if native_hidden is None:
                        continue

                    # Step 1: sae_vector (hidden_dim) → guide SAE encoder (no bias)
                    # No bias: steering vector is a difference direction; adding
                    # the encoder bias would inject a spurious offset (same as B3 TI).
                    h = torch.tensor(native_hidden, dtype=torch.float32).view(1, -1)
                    feat_in_full = F.linear(h, guide_sae.encoder.weight)  # [1, full_n_guide]
                    feat_in_full = torch.relu(feat_in_full)

                    # Step 2: select the top-k features the GlobalMLP was trained on
                    if guide_feat_idx is not None:
                        g_idx_t = torch.from_numpy(guide_feat_idx).long()
                        feat_in = feat_in_full[:, g_idx_t[:k_guide]]     # [1, k_guide]
                    else:
                        feat_in = feat_in_full[:, :k_guide]

                    # Step 3: GlobalMLP.encoders[guide] → universal concept space [1, d_concept]
                    feat_in_dev = feat_in.to(mlp_device)
                    z = global_mlp.encoders[guide](feat_in_dev)

                    # Step 4: GlobalMLP.decoders[target] → target top-k SAE features [1, k_tgt]
                    feat_out = global_mlp.decoders[tgt](z).cpu()

                    # Step 5: scatter target top-k features into full SAE feature space
                    full_feats = torch.zeros(1, full_n_tgt, dtype=torch.float32)
                    if tgt_feat_idx is not None:
                        k = min(feat_out.shape[1], len(tgt_feat_idx))
                        tgt_idx_t = torch.from_numpy(tgt_feat_idx).long()
                        full_feats[:, tgt_idx_t[:k]] = feat_out[:, :k]
                    else:
                        k = min(feat_out.shape[1], full_n_tgt)
                        full_feats[:, :k] = feat_out[:, :k]

                    # Step 6: target SAE decoder → target residual stream [hidden_dim_tgt]
                    decoded = F.linear(full_feats, tgt_sae.decoder.weight.cpu()).squeeze(0)
                    guide_encoded[guide][lbl] = _l2_normalize(decoded.float().numpy())

            # Free guide SAE immediately — don't accumulate all guides in RAM
            del guide_sae
            gc.collect()
            print(f"[C3-enc-dec] guide {guide}: encoded {len(guide_encoded[guide])} concepts, SAE freed")

        # --- Assemble per-concept averaged vectors from guide_encoded ---
        for lbl in eligible_concepts:
            guides = [g for g in concept_to_models.get(lbl, [])
                      if g != tgt and g in guide_encoded and lbl in guide_encoded[g]]
            if not guides:
                print(f"[C3-enc-dec] {tgt}/{lbl}: no guide vectors produced — skipping")
                continue

            guide_decoded_vecs = [guide_encoded[g][lbl] for g in guides]
            avg_vec = _l2_normalize(np.mean(guide_decoded_vecs, axis=0).astype(np.float32))

            results[lbl][tgt] = {
                "steering_vector": avg_vec.tolist(),
                "models_present": sorted(concept_to_models.get(lbl, [])),
                "guides_used": guides,
                "n_guides": len(guide_decoded_vecs),
                "mode": "enc_dec",
            }
            print(f"[C3-enc-dec] {tgt}/{lbl}: built from {len(guide_decoded_vecs)} guides")

        del tgt_sae
        guide_encoded.clear()
        gc.collect()
        torch.cuda.empty_cache()
        _save()

    print(f"[C3-enc-dec] Done. Output: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="C3 universal steering vector builder")
    parser.add_argument("--mode", default="decoder_only",
                        choices=["decoder_only", "sign_fix", "enc_dec"],
                        help=(
                            "decoder_only: original centroid→decoder pipeline. "
                            "sign_fix: post-process existing file to align sign with native vectors. "
                            "enc_dec: full encoder-decoder pipeline, correct polarity."
                        ))
    parser.add_argument("--run-id", default="1", dest="run_id",
                        help="Version suffix for output files (default: 1).")
    parser.add_argument("--concepts-file",
                        default="mean_concepts_clean.json",
                        dest="concepts_file",
                        help="Filename inside UNIVERSAL_DIR with C2 cluster output.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to GlobalMLP .pt checkpoint (overrides default).")
    parser.add_argument("--models", nargs="+",
                        default=list(config.MODELS.keys()),
                        help="Model names to process (default: all in config).")
    parser.add_argument("--force", action="store_true",
                        help="Re-compute even if output already exists.")
    args = parser.parse_args()

    start = time.time()
    try:
        if args.mode == "decoder_only":
            rc = mode_decoder_only(args)
        elif args.mode == "sign_fix":
            rc = mode_sign_fix(args)
        elif args.mode == "enc_dec":
            rc = mode_enc_dec(args)
        else:
            print(f"Unknown mode: {args.mode}")
            rc = 1
        log_run("build_universal_vectors.py", start, "success" if rc == 0 else "error")
        return rc
    except Exception as e:
        log_run("build_universal_vectors.py", start, "error", str(e))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
