
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch

import config
from a3_train_sae import TopKSAE
from a5_build_steering import (
    log_run,
    set_seed,
    _find_latest_checkpoint,
    _load_labels,
    _sample_indices,
    _sentiment_label_split,
    _get_concept_indices,
    _safe_source,
    _sentiment_source_groups,
    _read_sampled_source_rows,
    _l2_normalize,
    _load_feature_labels,
    _select_top_features,
)

def _load_feature_matrix_cross(model_name: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load top-k feature activation matrix and index array. Mirrors step9's load_feature_matrix."""
    import glob as _glob
    feats_dir = config.FEATURES_DIR
    ef = config.MODELS.get(model_name, {}).get("sae_ef", config.SAE_EXPANSION_FACTOR)
    # Try exact path with EF first, then without EF (legacy), then glob by ef+model
    candidates = [
        os.path.join(feats_dir, f"{model_name}_ef{ef}_top{k}_feature_acts.npy"),
        os.path.join(feats_dir, f"{model_name}_top{k}_feature_acts.npy"),
    ]
    acts_path = next((p for p in candidates if os.path.exists(p)), None)
    if acts_path is None:
        # Glob: find any file for this model+ef, pick closest k
        pattern = os.path.join(feats_dir, f"{model_name}_ef{ef}_top*_feature_acts.npy")
        matches = sorted(_glob.glob(pattern))
        if not matches:
            pattern2 = os.path.join(feats_dir, f"{model_name}_top*_feature_acts.npy")
            matches = sorted(_glob.glob(pattern2))
        if not matches:
            raise FileNotFoundError(f"No feature matrix found for {model_name} (k={k}) in {feats_dir}")
        acts_path = matches[-1]  # take the one with most features
    idx_path = acts_path.replace("_feature_acts.npy", "_feature_idx.npy")
    mat = np.load(acts_path)
    idx = np.load(idx_path)
    return mat, idx


def _get_scaler_cross(model_name: str, k: int):
    """Return a fitted StandardScaler and feature index array. Mirrors step9's get_scaler."""
    from sklearn.preprocessing import StandardScaler as _SS
    mat, idx = _load_feature_matrix_cross(model_name, k)
    scaler = _SS()
    scaler.fit(mat)
    return scaler, idx



def main_cross():
    """B3 — Build cross-model steering vectors using the alignment bridge.

    Two modes:
      locator      — bridge locates which target SAE features match the guide
                     concept features, then decodes those into a native target vector.
      translation  — guide A5 vector is encoded to guide SAE features, passed
                     through the MLP bridge, decoded from the target SAE decoder.
      both         — runs both modes and writes separate output files.
    """
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-mode",       default="cross")          # routing sentinel
    parser.add_argument("--guide-model",    required=True,  dest="guide_model")
    parser.add_argument("--target-model",   required=True,  dest="target_model")
    parser.add_argument("--bridge-mode",    default="both",
                        choices=["locator", "translation", "both"])
    parser.add_argument("--method",         default="both",
                        choices=["both", "sae_decoder", "caa_cross"],
                        dest="method",
                        help="Vector construction method for locator mode. "
                             "both: run sae_decoder + caa_cross and store both vectors (default). "
                             "sae_decoder: decode target SAE columns for located features (fast). "
                             "caa_cross: contrastive activation diff on target activations using bridge-matched "
                             "passage pairs (slower, no model inference needed).")
    parser.add_argument("--ef-guide",       type=int, default=0, dest="ef_guide",
                        help="EF override for guide model only (default: from config)")
    parser.add_argument("--ef-target",      type=int, default=0, dest="ef_target",
                        help="EF override for target model only (default: from config)")
    # Legacy single --ef flag: only applied to guide (kept for CLI back-compat)
    parser.add_argument("--ef",             type=int, default=0)
    parser.add_argument("--top-features",   type=int, default=3,  dest="top_features")
    parser.add_argument("--min-confidence", type=float, default=0.0, dest="min_confidence")
    parser.add_argument("--force",          action="store_true")
    args = parser.parse_args()

    guide = args.guide_model
    tgt   = args.target_model

    if guide not in config.MODELS:
        print(f"Unknown guide model '{guide}'. Known: {list(config.MODELS)}")
        return 1
    if tgt not in config.MODELS:
        print(f"Unknown target model '{tgt}'. Known: {list(config.MODELS)}")
        return 1

    set_seed(42)
    os.makedirs(config.STEERING_DIR, exist_ok=True)

    guide_cfg = config.MODELS[guide]
    tgt_cfg   = config.MODELS[tgt]

    # Each model uses its own sae_ef from config — never share a single --ef
    # across both models (they differ: gpt2/gemma=64, llama/mistral=128).
    # --ef-guide / --ef-target allow explicit overrides per model if needed.
    ef_guide = args.ef_guide or args.ef or guide_cfg.get("sae_ef", config.SAE_EXPANSION_FACTOR)
    ef_tgt   = args.ef_target or tgt_cfg.get("sae_ef", config.SAE_EXPANSION_FACTOR)
    print(f"[step7-cross] EF: guide={guide}@ef{ef_guide}, target={tgt}@ef{ef_tgt}", flush=True)

    # Output paths – one per mode
    out_bal = os.path.join(config.STEERING_DIR, "cross_model_steering_vectors_bal.json")
    out_ti  = os.path.join(config.STEERING_DIR, "cross_model_steering_vectors_ti.json")

    import fcntl

    def _locked_read(path: str) -> dict:
        """Read a JSON file under an exclusive lock, return {} on any error."""
        lock_path = path + ".lock"
        with open(lock_path, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception as _e:
                print(f"[step7-cross] WARNING: could not read {path}: {_e}")
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
        return {}

    def _locked_merge_write(path: str, guide_key: str, tgt_key: str, new_data: dict, force: bool):
        """Read-modify-write path under exclusive lock. Merges new_data into [guide_key][tgt_key]."""
        if not new_data:
            return
        lock_path = path + ".lock"
        with open(lock_path, "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                existing = {}
                if os.path.exists(path) and not force:
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            existing = json.load(f)
                    except Exception as _e:
                        print(f"[step7-cross] WARNING: re-read {path} failed ({_e}), starting fresh")
                tgt_dict = existing.setdefault(guide_key, {}).setdefault(tgt_key, {})
                for _concept, _entry in new_data.items():
                    tgt_dict.setdefault(_concept, {}).update(_entry)
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(existing, f)
                os.replace(tmp, path)
                print(f"[step7-cross] wrote {path} ({len(new_data)} concepts for {guide_key}→{tgt_key})")
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    # Check-only read (no lock needed — just for skip detection before heavy work)
    def _load_existing_unsafe(path):
        if os.path.exists(path) and not args.force:
            try:
                return json.load(open(path, "r", encoding="utf-8"))
            except Exception:
                pass
        return {}

    # Skip this pair if already present in output files (enables safe restart)
    if not args.force:
        _bal_snap = _load_existing_unsafe(out_bal) if args.bridge_mode in ("locator", "both") else {}
        _ti_snap  = _load_existing_unsafe(out_ti)  if args.bridge_mode in ("translation", "both") else {}
        _loc_done = (args.bridge_mode not in ("locator", "both")) or bool(_bal_snap.get(guide, {}).get(tgt))
        _ti_done  = (args.bridge_mode not in ("translation", "both")) or bool(_ti_snap.get(guide, {}).get(tgt))
        if _loc_done and _ti_done:
            print(f"[step7-cross] {guide}→{tgt}: already in output files — skipping (use --force to recompute)", flush=True)
            log_run("step7_build_steering.py", start_time, "skipped")
            return 0

    # These dicts accumulate results in-memory; we only write at the end via locked merge
    bal_out: dict = {}
    ti_out:  dict = {}

    # SAEs on GPU — 8× A100 80 GB each. Two SAEs for biggest models ~17 GB; fits easily.
    _dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    guide_sae = TopKSAE(
        guide_cfg["hidden_dim"],
        guide_cfg["hidden_dim"] * ef_guide,
        guide_cfg["sae_topk"],
    ).to(_dev)
    guide_sae.load_state_dict(torch.load(
        _find_latest_checkpoint(guide, config.SAE_DIR, ef=ef_guide), map_location=_dev))
    guide_sae.eval()

    tgt_sae = TopKSAE(
        tgt_cfg["hidden_dim"],
        tgt_cfg["hidden_dim"] * ef_tgt,
        tgt_cfg["sae_topk"],
    ).to(_dev)
    tgt_sae.load_state_dict(torch.load(
        _find_latest_checkpoint(tgt, config.SAE_DIR, ef=ef_tgt), map_location=_dev))
    tgt_sae.eval()

    tgt_n_features = tgt_cfg["hidden_dim"] * ef_tgt

    # Feature labels for guide model (determines which concepts + features to use)
    guide_feat_map = _load_feature_labels(guide, ef=ef_guide)
    concepts_from_labels: set = set()
    for fdata in guide_feat_map.values():
        d = fdata.get("domain")
        if d:
            concepts_from_labels.add(d)
    import re as _re
    # Exclude auto-generated cluster_X concepts — they are model-specific unlabeled
    # clusters that exist only in this model's SAE and cannot be bridged to other models
    # (no shared corpus labels, no aligned pairs, no matching target features).
    concepts_from_labels = {c for c in concepts_from_labels if not _re.match(r'^cluster_\d+$', c)}
    target_concepts = sorted(concepts_from_labels) if concepts_from_labels else list(config.TARGET_CONCEPTS)
    print(f"[step7-cross] {guide}→{tgt}: {len(target_concepts)} concepts (cluster_X excluded)")

    # Feature labels for TARGET model — used as semantic fallback when aligned_pairs
    # doesn't cover a concept's guide features (happens when B2 finds few pairs).
    tgt_feat_map = _load_feature_labels(tgt, ef=ef_tgt)

    # Load aligned_pairs.jsonl once — only rows for this (guide, tgt) pair
    aligned_path = os.path.join(config.ALIGNMENT_DIR, "aligned_pairs.jsonl")
    aligned_pairs: List[Dict] = []
    if os.path.exists(aligned_path):
        with open(aligned_path, "r", encoding="utf-8") as _ap:
            for _line in _ap:
                if not _line.strip():
                    continue
                p = json.loads(_line)
                if p.get("a_model") == guide and p.get("b_model") == tgt and p.get("validation_pass", False):
                    aligned_pairs.append(p)
    print(f"[step7-cross] loaded {len(aligned_pairs)} B2-validated aligned pairs for {guide}→{tgt}")
    # Target activations + corpus labels — needed for caa_cross method
    tgt_acts_cross: np.ndarray = None
    corpus_labels_cross: Dict[int, Dict] = None
    if args.method in ("caa_cross", "both") and args.bridge_mode in ("locator", "both"):
        import glob as _glob
        acts_path = os.path.join(config.ACTIVATIONS_DIR, f"{tgt}_activations_norm.h5")
        if os.path.exists(acts_path):
            with h5py.File(acts_path, "r") as _hf:
                tgt_acts_cross = _hf["activations"][:]
            print(f"[step7-cross] loaded target activations {tgt_acts_cross.shape}")
        else:
            # Try per-domain activation files for this model
            domain_files = sorted(_glob.glob(
                os.path.join(config.ACTIVATIONS_DIR, f"{tgt}_*_activations_norm.h5")
            ))
            if not domain_files:
                raise FileNotFoundError(
                    f"[step7-cross] No target activations found for '{tgt}'.\n"
                    f"  Tried: {acts_path}\n"
                    f"  Glob:  {os.path.join(config.ACTIVATIONS_DIR, tgt + '_*_activations_norm.h5')}\n"
                    f"  Run step2/step3 to generate activations before rebuilding."
                )
            chunks = []
            for _df in domain_files:
                with h5py.File(_df, "r") as _hf:
                    chunks.append(_hf["activations"][:])
            tgt_acts_cross = np.concatenate(chunks, axis=0)
            print(f"[step7-cross] stacked {len(chunks)} domain activation files → {tgt_acts_cross.shape}")
        labels_path = os.path.join(config.DATA_DIR, "corpus_labels.jsonl")
        if os.path.exists(labels_path):
            corpus_labels_cross = {}
            with open(labels_path, "r", encoding="utf-8") as _lf:
                for _i, _line in enumerate(_lf):
                    corpus_labels_cross[_i] = json.loads(_line)
    # MLP bridge + matching scalers/indices (needed for translation mode)
    pair_mlp = None
    src_scaler_xfr = None
    tgt_scaler_xfr = None
    src_idx_xfr = None
    tgt_idx_xfr = None
    if args.bridge_mode in ("translation", "both"):
        mlp_path = os.path.join(config.ALIGNMENT_DIR, f"mlp_{guide}_to_{tgt}.pt")
        if os.path.exists(mlp_path):
            state = torch.load(mlp_path, map_location="cpu")
            w0 = state["0.weight"]
            w_last_key = "3.weight" if "3.weight" in state else "2.weight"
            w_last = state[w_last_key]
            hidden_dim_mlp, in_dim_mlp = w0.shape
            out_dim_mlp, _ = w_last.shape
            if "3.weight" in state:
                pair_mlp = torch.nn.Sequential(
                    torch.nn.Linear(in_dim_mlp, hidden_dim_mlp),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=0.1),
                    torch.nn.Linear(hidden_dim_mlp, out_dim_mlp),
                )
            else:
                pair_mlp = torch.nn.Sequential(
                    torch.nn.Linear(in_dim_mlp, hidden_dim_mlp),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim_mlp, out_dim_mlp),
                )
            pair_mlp.load_state_dict(state)
            pair_mlp.eval()
            pair_mlp = pair_mlp.to(_dev)
            # Load feature indices and normalisation stats saved by step5 alongside
            # the MLP.  These are the EXACT indices the MLP was trained on (top ever-
            # active features from _compute_capped), which may differ from the labeled
            # feature selection stored in the top-K .npy files.
            src_idx_path   = mlp_path.replace(".pt", "_src_idx.npy")
            tgt_idx_path   = mlp_path.replace(".pt", "_tgt_idx.npy")
            src_stats_path = mlp_path.replace(".pt", "_src_stats.npz")
            tgt_stats_path = mlp_path.replace(".pt", "_tgt_stats.npz")
            missing = [p for p in (src_idx_path, tgt_idx_path, src_stats_path, tgt_stats_path) if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(
                    f"[step7-cross] MLP index/stats files missing for {guide}→{tgt}:\n"
                    + "\n".join(f"  {p}" for p in missing)
                    + "\n  Re-run step5 to regenerate MLP + companion files."
                )
            src_idx_xfr = np.load(src_idx_path)
            tgt_idx_xfr = np.load(tgt_idx_path)
            _src_s = np.load(src_stats_path)
            _tgt_s = np.load(tgt_stats_path)
            # Reconstruct sklearn-compatible scalers from saved mean/std
            from sklearn.preprocessing import StandardScaler as _SS
            src_scaler_xfr = _SS(); src_scaler_xfr.mean_ = _src_s["mean"]; src_scaler_xfr.scale_ = _src_s["std"]; src_scaler_xfr.n_features_in_ = len(_src_s["mean"])
            tgt_scaler_xfr = _SS(); tgt_scaler_xfr.mean_ = _tgt_s["mean"]; tgt_scaler_xfr.scale_ = _tgt_s["std"]; tgt_scaler_xfr.n_features_in_ = len(_tgt_s["mean"])
            print(f"[step7-cross] loaded MLP feature indices: src={len(src_idx_xfr)}-d, tgt={len(tgt_idx_xfr)}-d")
        else:
            raise FileNotFoundError(
                f"[step7-cross] No MLP found for {guide}→{tgt} at:\n  {mlp_path}\n"
                f"  Re-run step5 to train the alignment MLP."
            )

    # Guide A5 steering vectors — always loaded (causal_validation propagation + translation mode)
    guide_steering = {}
    sv_path = os.path.join(config.STEERING_DIR, f"{guide}_ef{ef_guide}_steering_vectors.json")
    if not os.path.exists(sv_path):
        sv_path = os.path.join(config.STEERING_DIR, "steering_vectors.json")
    if not os.path.exists(sv_path):
        raise FileNotFoundError(
            f"[step7-cross] Guide A5 steering file not found for '{guide}'.\n"
            f"  Tried: {sv_path}\n"
            f"  Re-run step7 --mode a5 for the guide model first."
        )
    print(f"[step7-cross] loading guide A5 steering from {sv_path}")
    all_sv = json.load(open(sv_path, "r", encoding="utf-8"))
    guide_steering = all_sv.get(guide, {})
    if not guide_steering:
        # Try flat structure: file may be {concept: {...}} without model wrapper
        first_val = next(iter(all_sv.values()), None)
        if isinstance(first_val, dict) and ("sae_vector" in first_val or "caa_vector" in first_val or "vector" in first_val):
            print(f"[step7-cross] steering file is flat (no model wrapper) — using directly")
            guide_steering = all_sv
        else:
            raise KeyError(
                f"[step7-cross] Key '{guide}' not found in {sv_path}.\n"
                f"  Top-level keys: {list(all_sv.keys())[:5]}\n"
                f"  Re-run step7 --mode a5 for guide model '{guide}'."
            )
    if not guide_steering:
        raise ValueError(
            f"[step7-cross] No A5 vectors found for guide '{guide}' in {sv_path}.\n"
            f"  Re-run step7 --mode a5 for the guide model first."
        )

    import torch.nn.functional as _F

    # Feature-selection cache per guide concept
    n_concepts = len(target_concepts)
    n_loc = 0
    n_ti  = 0
    print(f"[step7-cross] {guide}→{tgt}: processing {n_concepts} concepts "
          f"(bridge_mode={args.bridge_mode}, method={args.method})", flush=True)
    for ci, concept in enumerate(target_concepts, 1):
        print(f"[step7-cross] [{ci}/{n_concepts}] {concept}", flush=True)
        top_feats = _select_top_features(guide_feat_map, concept, topn=args.top_features)
        if args.min_confidence > 0:
            top_feats = [(fid, conf) for fid, conf in top_feats if conf >= args.min_confidence]
        if not top_feats:
            print(f"[step7-cross]   skip: no guide features", flush=True)
            continue

        # ── Locator mode ────────────────────────────────────────────────
        if args.bridge_mode in ("locator", "both"):
            guide_feat_weights = {int(fid): float(conf) for fid, conf in top_feats}

            vec_loc: np.ndarray = None
            vec_loc_caa: np.ndarray = None   # secondary vector when method == "both"
            method_used = args.method

            # ── sae_decoder (default) ──────────────────────────────────
            if args.method in ("sae_decoder", "both") or tgt_acts_cross is None:
                if args.method == "caa_cross" and tgt_acts_cross is None:
                    raise RuntimeError(
                        f"[step7-cross] caa_cross requested for '{concept}' but no target activations loaded for '{tgt}'.\n"
                        f"  This should have been caught earlier — check activation loading above."
                    )

                full_tgt_loc = np.zeros(tgt_n_features, dtype=np.float32)
                n_pairs_hit = 0
                for ap in aligned_pairs:
                    a_feat = int(ap["a_feature"])
                    b_feat = int(ap["b_feature"])
                    score  = float(ap.get("score", 0.0))
                    g_weight = guide_feat_weights.get(a_feat, 0.0)
                    if g_weight > 0 and b_feat < tgt_n_features:
                        full_tgt_loc[b_feat] += g_weight * score
                        n_pairs_hit += 1

                if not np.any(full_tgt_loc):
                    raise RuntimeError(
                        f"[step7-cross] sae_decoder: 0 aligned pairs matched guide features for '{concept}' "
                        f"({guide}→{tgt}).\n"
                        f"  Guide feat IDs: {list(guide_feat_weights.keys())}\n"
                        f"  aligned_pairs has {len(aligned_pairs)} entries.\n"
                        f"  Re-run step5 (alignment) to get overlapping features."
                    )

                # Decode without SAE decoder bias — we are decoding a direction
                # vector, not a reconstruction; bias would shift the direction
                _ft_loc = torch.from_numpy(full_tgt_loc).float().view(1, -1).to(_dev)
                with torch.no_grad():
                    decoded_loc = _F.linear(
                        _ft_loc, tgt_sae.decoder.weight
                    ).squeeze(0).cpu().numpy()
                vec_loc = _l2_normalize(decoded_loc)
                if args.method == "both" and tgt_acts_cross is not None:
                    pass  # fall through to also compute caa_cross below

            # ── caa_cross (or second half of 'both') ──────────────────
            if args.method in ("caa_cross", "both") and tgt_acts_cross is not None:
                method_used = "caa_cross"
                # Find corpus indices for this concept using the same logic as A5 CAA
                pos_idx, neg_idx = _get_concept_indices(corpus_labels_cross, concept)
                print(f"[step7-cross]   caa_cross: pos={len(pos_idx)} neg={len(neg_idx)} for '{concept}'", flush=True)

                if pos_idx and neg_idx:
                    # For each bridge-matched pair, collect target passages whose
                    # guide-side feature activation was positive vs negative
                    # Guide concept features set
                    guide_top_fids = set(guide_feat_weights.keys())
                    # Passage-level guide SAE feature activations via the feature matrix
                    try:
                        guide_feat_mat, guide_feat_idx = _load_feature_matrix_cross(guide, len(guide_feat_weights))
                        # Select columns by guide concept feature IDs (not a blind [:k] slice)
                        # guide_feat_idx maps column position → SAE feature id
                        _fid_to_col = {int(guide_feat_idx[c]): c for c in range(len(guide_feat_idx))}
                        _concept_cols = [_fid_to_col[fid] for fid in guide_feat_weights if fid in _fid_to_col]
                        n_tgt = tgt_acts_cross.shape[0]
                        if _concept_cols:
                            guide_top_acts = guide_feat_mat[:, _concept_cols].mean(axis=1)
                            median_act = float(np.median(guide_top_acts))
                            pos_guided = [i for i in range(len(guide_top_acts)) if guide_top_acts[i] > median_act]
                            neg_guided = [i for i in range(len(guide_top_acts)) if guide_top_acts[i] <= median_act]
                            # Clamp to available target activations
                            pos_guided = [i for i in pos_guided if i < n_tgt]
                            neg_guided = [i for i in neg_guided if i < n_tgt]
                            print(f"[step7-cross]   caa_cross: guided split pos={len(pos_guided)} neg={len(neg_guided)}", flush=True)
                        else:
                            # Guide concept features not in feature matrix — fall back to
                            # corpus label split (semantic split, no guide feature info)
                            print(f"[step7-cross]   caa_cross: no concept cols in guide feat matrix — using corpus label split", flush=True)
                            pos_guided = [i for i in pos_idx if i < n_tgt]
                            neg_guided = [i for i in neg_idx if i < n_tgt]
                    except Exception as _feat_exc:
                        raise RuntimeError(
                            f"[step7-cross] Guide feature matrix load failed for '{guide}' "
                            f"({guide}→{tgt}/{concept}): {_feat_exc}\n"
                            f"  Re-run step3/step5 to regenerate feature matrices."
                        ) from _feat_exc

                    if pos_guided and neg_guided:
                        pos_mean = tgt_acts_cross[pos_guided].mean(axis=0)
                        neg_mean = tgt_acts_cross[neg_guided].mean(axis=0)
                        _caa_vec = _l2_normalize((pos_mean - neg_mean).astype(np.float32))
                        if args.method == "both":
                            vec_loc_caa = _caa_vec   # store as secondary; primary = sae_decoder
                        else:
                            vec_loc = _caa_vec
                    elif args.method == "caa_cross":
                        raise RuntimeError(
                            f"[step7-cross] caa_cross '{concept}' ({guide}→{tgt}): insufficient bridge-guided passages "
                            f"(pos_guided={len(pos_guided)}, neg_guided={len(neg_guided)}).\n"
                            f"  Need at least 1 positive and 1 negative passage. Check corpus labels and activation alignment."
                        )
                elif args.method == "caa_cross":
                    raise RuntimeError(
                        f"[step7-cross] caa_cross '{concept}' ({guide}→{tgt}): no concept indices found in corpus labels.\n"
                        f"  corpus_labels_cross has {len(corpus_labels_cross)} entries.\n"
                        f"  Ensure corpus_labels.jsonl contains entries for '{concept}'."
                    )

            if vec_loc is None:
                print(f"[step7-cross] locator {concept}: no vector produced — skipping")
                continue

            # Propagate causal_validation from guide's A5 vector entry
            _guide_cv = guide_steering.get(concept, {}).get("causal_validation")
            _cv_locator = "confirmed" if _guide_cv == "confirmed" else "inherited_unconfirmed"

            # Merge into output dict: bal_out[guide][tgt][concept]
            bal_entry = {
                "method": method_used,
                "guide_model": guide,
                "target_model": tgt,
                "concept": concept,
                "top_features_used": [int(fid) for fid, _ in top_feats],
                "vector_dim": int(tgt_cfg["hidden_dim"]),
                "injection_layer": int(tgt_cfg["target_layer"]),
                "causal_validation": _cv_locator,
            }
            if args.method == "caa_cross":
                bal_entry["caa_cross_vector"] = vec_loc.tolist()
            elif args.method == "both":
                bal_entry["sae_decoder_vector"] = vec_loc.tolist()
                if vec_loc_caa is not None:
                    bal_entry["caa_cross_vector"] = vec_loc_caa.tolist()
            else:
                # sae_decoder only
                bal_entry["sae_decoder_vector"] = vec_loc.tolist()
            bal_out.setdefault(guide, {}).setdefault(tgt, {})[concept] = bal_entry
            n_loc += 1
            print(f"[step7-cross]   loc ok  (loc={n_loc}, ti={n_ti})", flush=True)

        # ── Translation mode ─────────────────────────────────────────────
        if args.bridge_mode in ("translation", "both"):
            if pair_mlp is None:
                print(f"[step7-cross]   ti  skip: no MLP bridge", flush=True)
            elif not guide_steering:
                print(f"[step7-cross]   ti  skip: no guide A5 vectors", flush=True)
        if args.bridge_mode in ("translation", "both") and pair_mlp is not None and guide_steering:
            # Pick best available vector from A5 guide vectors
            concept_entry = guide_steering.get(concept, {})
            src_vec_list = (concept_entry.get("sae_vector")
                            or concept_entry.get("caa_vector")
                            or concept_entry.get("vector"))
            if src_vec_list is None:
                print(f"[step7-cross] translation {guide}→{tgt} {concept}: no A5 guide vector (keys={list(concept_entry.keys())}) — skipping")
                continue
            src_vec = np.array(src_vec_list, dtype=np.float32)

            # Encode guide steering vector into guide SAE feature space.
            # Use weight-only (no bias): a steering vector is a direction/difference,
            # so adding the encoder bias would inject a spurious offset.
            with torch.no_grad():
                src_t = torch.from_numpy(src_vec).float().view(1, -1).to(_dev)
                z = _F.linear(src_t, guide_sae.encoder.weight)  # no encoder bias
                topk_vals, topk_idx = torch.topk(z, guide_sae.topk, dim=-1)
                sparse = torch.zeros_like(z)
                sparse.scatter_(1, topk_idx, topk_vals)
                sparse = torch.relu(sparse)
                guide_feat_acts = sparse.squeeze(0).cpu().numpy()

            # Select the correct non-contiguous feature subset using the same
            # scaler + index as step9 (fixes the contiguous-slice bug)
            src_feat_sub = guide_feat_acts[src_idx_xfr]
            src_feat_s   = src_scaler_xfr.transform(src_feat_sub.reshape(1, -1))

            # Run through MLP bridge (on GPU)
            with torch.no_grad():
                _src_feat_t = torch.from_numpy(src_feat_s).float().to(_dev)
                pred_tgt_s = pair_mlp(_src_feat_t).cpu().numpy()

            # Inverse-scale and scatter into full target feature space at correct indices
            pred_tgt = tgt_scaler_xfr.inverse_transform(pred_tgt_s).squeeze(0)
            full_tgt_ti = np.zeros(tgt_n_features, dtype=np.float32)
            full_tgt_ti[tgt_idx_xfr] = pred_tgt

            # Decode without SAE decoder bias (same reason as locator above)
            _ft_ti = torch.from_numpy(full_tgt_ti).float().view(1, -1).to(_dev)
            with torch.no_grad():
                decoded_ti = _F.linear(
                    _ft_ti, tgt_sae.decoder.weight
                ).squeeze(0).cpu().numpy()
            vec_ti = _l2_normalize(decoded_ti)

            _guide_cv_ti = guide_steering.get(concept, {}).get("causal_validation")
            _cv_ti = "confirmed" if _guide_cv_ti == "confirmed" else "inherited_unconfirmed"

            ti_out.setdefault(guide, {}).setdefault(tgt, {})[concept] = {
                "vector": vec_ti.tolist(),
                "method": "translation_injection",
                "guide_model": guide,
                "target_model": tgt,
                "concept": concept,
                "top_features_used": [int(fid) for fid, _ in top_feats],
                "vector_dim": int(tgt_cfg["hidden_dim"]),
                "injection_layer": int(tgt_cfg["target_layer"]),
                "causal_validation": _cv_ti,
            }
            n_ti += 1
            print(f"[step7-cross]   ti  ok  (loc={n_loc}, ti={n_ti})", flush=True)

    # Summary
    print(f"[step7-cross] DONE {guide}→{tgt}: {n_loc} locator + {n_ti} translation = {n_loc+n_ti} total vectors", flush=True)

    # Write outputs — locked merge so concurrent jobs don't clobber each other
    if args.bridge_mode in ("locator", "both"):
        _locked_merge_write(out_bal, guide, tgt, bal_out.get(guide, {}).get(tgt, {}), args.force)

    if args.bridge_mode in ("translation", "both"):
        _pair_ti = ti_out.get(guide, {}).get(tgt, {})
        if _pair_ti:
            _locked_merge_write(out_ti, guide, tgt, _pair_ti, args.force)
        else:
            print(f"[step7-cross] WARNING: no translation vectors produced — {out_ti} not written")
            print(f"[step7-cross]   pair_mlp loaded={pair_mlp is not None}, guide_steering non-empty={bool(guide_steering)}")

    log_run("step7_build_steering.py", start_time, "success")
    return 0



if __name__ == "__main__":
    try:
        raise SystemExit(main_cross())
    except Exception as e:
        log_run("b3_build_cross_steering.py", time.time(), "error", str(e))
        raise
