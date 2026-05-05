
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import json
import os
import random
import time
from typing import Dict, Tuple

import h5py
import numpy as np
import torch

import config
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
    """Find the latest SAE checkpoint for (model, expansion_factor).

    Naming convention:
      - ef == SAE_EXPANSION_FACTOR (default 16): legacy names {model}_sae_step{N}.pt / {model}_sae.pt
      - ef != default: new names {model}_ef{ef}_sae_step{N}.pt / {model}_ef{ef}_sae.pt
    """
    import re
    ef_tag = "" if ef == config.SAE_EXPANSION_FACTOR else f"_ef{ef}"
    candidates = []
    for fname in os.listdir(sae_dir):
        if not fname.endswith(".pt"):
            continue
        if model_name not in fname:
            continue
        if "step" not in fname:
            continue
        # Must match exact ef_tag prefix
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
    raise FileNotFoundError(f"No SAE checkpoint found for {model_name} ef={ef} in {sae_dir}")


def _load_labels() -> Dict[int, Dict[str, int]]:
    labels_path = os.path.join(config.DATA_DIR, "corpus_labels.jsonl")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels file at {labels_path}")
    labels = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            labels[i] = json.loads(line)
    return labels


def _build_label_indices(labels: Dict[int, Dict[str, int]]) -> Dict[str, np.ndarray]:
    """Build pos/neg index arrays for every concept field found in corpus_labels.

    Concept values may be ±2.0 (positive = >0, negative = <0) or classic 0/1.
    Built-in aliases (sentiment→pos_sent etc.) are kept for backward compat.
    """
    _SKIP = {"id", "source", "text", "formality", "factuality"}

    # Auto-detect all numeric concept fields
    field_vals: Dict[str, set] = {}
    for r in labels.values():
        for k, v in r.items():
            if k in _SKIP:
                continue
            if isinstance(v, (int, float)):
                if k not in field_vals:
                    field_vals[k] = set()
                field_vals[k].add(v)

    result: Dict[str, np.ndarray] = {}
    for field, vals in field_vals.items():
        has_pos = any(v > 0 for v in vals)
        has_neg = any(v < 0 for v in vals)
        if not (has_pos and has_neg):
            continue  # skip one-sided fields
        result[f"pos_{field}"] = np.array(
            [i for i, r in labels.items() if r.get(field, 0) > 0], dtype=np.int64)
        result[f"neg_{field}"] = np.array(
            [i for i, r in labels.items() if r.get(field, 0) < 0], dtype=np.int64)
        print(f"[label] concept '{field}': {len(result[f'pos_{field}'])} pos, {len(result[f'neg_{field}'])} neg")

    # Backward-compat aliases for the three built-in domains
    # (maps to sentiment/formality/certainty if those fields exist, else empty)
    for alias, field in [("sent", "sentiment"), ("form", "formality"), ("fact", "certainty")]:
        result.setdefault(f"pos_{alias}", result.get(f"pos_{field}", np.array([], dtype=np.int64)))
        result.setdefault(f"neg_{alias}", result.get(f"neg_{field}", np.array([], dtype=np.int64)))

    return result


def _load_activations(model_name: str):
    """Return (acts, source_info) where source_info is None for single-file loads,
    or list of (source_tag, n_rows) in concatenation order for per-source loads."""
    import glob as _glob
    acts_dir = config.ACTIVATIONS_DIR
    bare = os.path.join(acts_dir, f"{model_name}_activations_norm.h5")
    if os.path.exists(bare):
        with h5py.File(bare, "r") as h5:
            acts = h5["activations"][:]
        return acts, None
    # Fall back to combining per-source files in memory (no disk write)
    pattern = os.path.join(acts_dir, f"{model_name}_*_activations_norm.h5")
    per_source = sorted(_glob.glob(pattern))
    if not per_source:
        raise FileNotFoundError(
            f"Missing activations at {bare} and no per-source files matching {pattern}")
    chunks = []
    source_info = []
    prefix = model_name + "_"
    suffix = "_activations_norm.h5"
    for p in per_source:
        with h5py.File(p, "r") as h5:
            arr = h5["activations"][:]
        chunks.append(arr)
        fname = os.path.basename(p)
        tag = fname[len(prefix):-len(suffix)] if fname.startswith(prefix) and fname.endswith(suffix) else fname
        source_info.append((tag, arr.shape[0]))
        print(f"[label] Loaded {fname} ({arr.shape[0]} rows)")
    acts = np.concatenate(chunks, axis=0)
    print(f"[label] Combined {len(per_source)} per-source files → {acts.shape[0]} total activations")
    return acts, source_info


def _patch_label_idx_from_sources(label_idx: Dict, source_info, labels: Dict) -> None:
    """Override label_idx for per-source concepts using exact activation row offsets.

    corpus_labels.jsonl may be in a different order than the sorted h5 files.
    For any concept that has a clear source-tag match (e.g. all science_pubmed_50k
    rows are labeled science_biomedical:+2.0), rebuild the pos/neg arrays from
    the actual h5 row offsets so they align with acts rows correctly.
    """
    if not source_info:
        return

    # Build: source_tag → which concepts are positive in that source
    # Scan a sample of corpus_labels rows to find source→concept mapping
    source_concepts: Dict[str, Dict[str, float]] = {}  # source_tag → {concept: value}
    for r in labels.values():
        src = r.get("source", "")
        if src and src not in source_concepts:
            source_concepts[src] = {
                k: v for k, v in r.items()
                if k not in ("id", "source", "text", "formality", "factuality")
                and isinstance(v, (int, float)) and v != 0
            }

    # For each concept, find which source tags are exclusively positive/negative
    # then rebuild arrays using h5 offsets
    concept_to_pos_sources: Dict[str, set] = {}
    concept_to_neg_sources: Dict[str, set] = {}
    for src_tag, concepts in source_concepts.items():
        for concept, val in concepts.items():
            if val > 0:
                concept_to_pos_sources.setdefault(concept, set()).add(src_tag)
            elif val < 0:
                concept_to_neg_sources.setdefault(concept, set()).add(src_tag)

    # Build offset ranges for each source tag in the h5 concatenation order
    offset = 0
    source_ranges: Dict[str, tuple] = {}  # source_tag → (start, end)
    for tag, nrows in source_info:
        source_ranges[tag] = (offset, offset + nrows)
        offset += nrows
    total_rows = offset

    updated = []
    for concept, pos_srcs in concept_to_pos_sources.items():
        neg_srcs = concept_to_neg_sources.get(concept, set())
        if not neg_srcs:
            continue  # still one-sided — nothing to fix
        pos_idx = np.concatenate([
            np.arange(*source_ranges[s], dtype=np.int64)
            for s in pos_srcs if s in source_ranges
        ] or [np.array([], dtype=np.int64)])
        neg_idx = np.concatenate([
            np.arange(*source_ranges[s], dtype=np.int64)
            for s in neg_srcs if s in source_ranges
        ] or [np.array([], dtype=np.int64)])
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        label_idx[f"pos_{concept}"] = pos_idx
        label_idx[f"neg_{concept}"] = neg_idx
        updated.append(f"{concept}(+{len(pos_idx)}/−{len(neg_idx)})")
    if updated:
        print(f"[label] Rebuilt label indices from h5 source offsets: {', '.join(updated)}")


def _compute_feature_stats(model_name: str, acts: np.ndarray, label_idx: Dict[str, np.ndarray],
                           ef: int, custom_concepts: list = None) -> Tuple[np.ndarray, dict]:
    ckpt_path = _find_sae_checkpoint(model_name, config.SAE_DIR, ef)
    hidden_dim = config.MODELS[model_name]["hidden_dim"]
    n_features = hidden_dim * ef
    sae_topk = config.MODELS[model_name].get("sae_topk", 32)
    # Auto-scale topk to maintain same sparsity ratio when EF changes
    if ef != config.SAE_EXPANSION_FACTOR:
        base_sparsity = sae_topk / (hidden_dim * config.SAE_EXPANSION_FACTOR)
        sae_topk = max(int(n_features * base_sparsity), 32)

    # Use GPU if available — 10-50x faster for large SAEs (EF≥32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = TopKSAE(hidden_dim, n_features, sae_topk).to(device)
    sae.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    sae.eval()
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpus > 1:
        sae = torch.nn.DataParallel(sae)
        print(f"[label] Running SAE inference on {n_gpus} GPUs via DataParallel (n_features={n_features})")
    else:
        print(f"[label] Running SAE inference on {device} (n_features={n_features})")

    n = acts.shape[0]
    sum_abs = np.zeros(n_features, dtype=np.float64)
    sum_pos_sent = np.zeros(n_features, dtype=np.float64)
    sum_neg_sent = np.zeros(n_features, dtype=np.float64)
    sum_pos_form = np.zeros(n_features, dtype=np.float64)
    sum_neg_form = np.zeros(n_features, dtype=np.float64)
    sum_pos_fact = np.zeros(n_features, dtype=np.float64)
    sum_neg_fact = np.zeros(n_features, dtype=np.float64)

    # Custom concepts
    if custom_concepts is None:
        custom_concepts = []
    sum_pos_custom: Dict[str, np.ndarray] = {c: np.zeros(n_features, dtype=np.float64) for c in custom_concepts}
    sum_neg_custom: Dict[str, np.ndarray] = {c: np.zeros(n_features, dtype=np.float64) for c in custom_concepts}

    # Build boolean index arrays once (vectorized, no Python loops per batch)
    bool_pos_sent = np.zeros(n, dtype=bool); bool_pos_sent[label_idx["pos_sent"][label_idx["pos_sent"] < n]] = True
    bool_neg_sent = np.zeros(n, dtype=bool); bool_neg_sent[label_idx["neg_sent"][label_idx["neg_sent"] < n]] = True
    bool_pos_form = np.zeros(n, dtype=bool); bool_pos_form[label_idx["pos_form"][label_idx["pos_form"] < n]] = True
    bool_neg_form = np.zeros(n, dtype=bool); bool_neg_form[label_idx["neg_form"][label_idx["neg_form"] < n]] = True
    bool_pos_fact = np.zeros(n, dtype=bool); bool_pos_fact[label_idx["pos_fact"][label_idx["pos_fact"] < n]] = True
    bool_neg_fact = np.zeros(n, dtype=bool); bool_neg_fact[label_idx["neg_fact"][label_idx["neg_fact"] < n]] = True
    bool_pos_custom: Dict[str, np.ndarray] = {}
    bool_neg_custom: Dict[str, np.ndarray] = {}
    for _c in custom_concepts:
        _pi = label_idx.get(f"pos_{_c}", np.array([], dtype=np.int64)); _pi = _pi[_pi < n]
        _ni = label_idx.get(f"neg_{_c}", np.array([], dtype=np.int64)); _ni = _ni[_ni < n]
        bool_pos_custom[_c] = np.zeros(n, dtype=bool); bool_pos_custom[_c][_pi] = True
        bool_neg_custom[_c] = np.zeros(n, dtype=bool); bool_neg_custom[_c][_ni] = True

    # Larger batches on GPU, smaller on CPU; scale with number of GPUs
    batch_size = (4096 * max(n_gpus, 1)) if device.type == "cuda" else 512
    for i in range(0, n, batch_size):
        sl = slice(i, min(i + batch_size, n))
        batch = torch.from_numpy(acts[sl]).float().to(device)
        with torch.no_grad():
            _, sparse = sae(batch)

        sparse_cpu = sparse.float().cpu().numpy()  # [B, n_features]
        sum_abs += np.abs(sparse_cpu).sum(axis=0)

        # Vectorized mask application
        def _add(buf, mask_slice):
            if mask_slice.any():
                buf += sparse_cpu[mask_slice].sum(axis=0)

        _add(sum_pos_sent, bool_pos_sent[sl])
        _add(sum_neg_sent, bool_neg_sent[sl])
        _add(sum_pos_form, bool_pos_form[sl])
        _add(sum_neg_form, bool_neg_form[sl])
        _add(sum_pos_fact, bool_pos_fact[sl])
        _add(sum_neg_fact, bool_neg_fact[sl])
        for _c in custom_concepts:
            _add(sum_pos_custom[_c], bool_pos_custom[_c][sl])
            _add(sum_neg_custom[_c], bool_neg_custom[_c][sl])

        if (i // batch_size) % 20 == 0:
            print(f"[label] batch {i // batch_size + 1}/{(n + batch_size - 1) // batch_size}")

    counts = {
        "sentiment": int(len(label_idx["pos_sent"]) + len(label_idx["neg_sent"])),
        "formality": int(len(label_idx["pos_form"]) + len(label_idx["neg_form"])),
        "certainty": int(len(label_idx["pos_fact"]) + len(label_idx["neg_fact"])),
    }
    for _c in custom_concepts:
        counts[_c] = int(len(label_idx.get(f"pos_{_c}", [])) + len(label_idx.get(f"neg_{_c}", [])))

    stats = {
        "sum_abs": sum_abs,
        "sum_pos_sent": sum_pos_sent,
        "sum_neg_sent": sum_neg_sent,
        "sum_pos_form": sum_pos_form,
        "sum_neg_form": sum_neg_form,
        "sum_pos_fact": sum_pos_fact,
        "sum_neg_fact": sum_neg_fact,
    }
    for _c in custom_concepts:
        stats[f"sum_pos_{_c}"] = sum_pos_custom[_c]
        stats[f"sum_neg_{_c}"] = sum_neg_custom[_c]
    return n_features, stats, counts


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(config.MODELS.keys()))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--expansion-factor", "--ef", type=int, default=config.SAE_EXPANSION_FACTOR,
                        dest="expansion_factor", help="SAE expansion factor used when training the SAE")
    parser.add_argument("--concepts", nargs="*", default=[],
                        help="Extra concept field names in corpus_labels.jsonl to label features for "
                             "(beyond sentiment/formality/certainty). "
                             "e.g. --concepts toxicity empathy")
    parser.add_argument("--min-confidence", type=float, default=0.0, dest="min_confidence",
                        help="Only save features with confidence >= this value (default: 0.0 = keep all)")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(config.FEATURES_DIR, exist_ok=True)

    ef_tag = "" if args.expansion_factor == config.SAE_EXPANSION_FACTOR else f"_ef{args.expansion_factor}"
    out_path = os.path.join(config.FEATURES_DIR, f"{args.model}{ef_tag}_feature_labels.json")
    if os.path.exists(out_path) and not args.force:
        print(f"{out_path} exists. Use --force to recompute.")
        log_run("label_features.py", start_time, "skipped")
        return 0

    labels = _load_labels()
    label_idx = _build_label_indices(labels)

    # Concepts are now auto-detected inside _build_label_indices via pos_*/neg_* keys.
    # If explicit --concepts were passed, honor them; otherwise we use everything detected.
    if args.concepts:
        print(f"[label] Using explicit concepts: {args.concepts}")

    # All concepts (incl. custom --concepts) are already in label_idx from _build_label_indices.
    # Add any explicitly requested concepts that weren't auto-detected (using >0/<0 threshold).
    for concept_name in args.concepts:
        if f"pos_{concept_name}" not in label_idx:
            pos = np.array([i for i, r in labels.items() if r.get(concept_name, 0) > 0], dtype=np.int64)
            neg = np.array([i for i, r in labels.items() if r.get(concept_name, 0) < 0], dtype=np.int64)
            label_idx[f"pos_{concept_name}"] = pos
            label_idx[f"neg_{concept_name}"] = neg
            if len(pos) == 0 and len(neg) == 0:
                print(f"[warn] concept '{concept_name}' has no labeled rows in corpus_labels.jsonl")

    # Derive final concept list from label_idx keys (everything with pos_/neg_ pair)
    _skip_builtin = {"sent", "form", "fact"}
    all_concepts = sorted(
        k[4:] for k in label_idx if k.startswith("pos_") and k[4:] not in _skip_builtin
    )
    if args.concepts:
        # Keep only explicitly requested ones if specified
        all_concepts = [c for c in all_concepts if c in set(args.concepts)]
    print(f"[label] Labelling {len(all_concepts)} concepts: {all_concepts}")

    acts, source_info = _load_activations(args.model)
    # Fix index alignment: rebuild label_idx using actual h5 source row offsets
    _patch_label_idx_from_sources(label_idx, source_info, labels)
    n_features, stats, counts = _compute_feature_stats(
        args.model, acts, label_idx, args.expansion_factor,
        custom_concepts=all_concepts
    )

    # Compute deltas and mean activation
    mean_abs = stats["sum_abs"] / config.N_PASSAGES
    delta_sent = (stats["sum_pos_sent"] / max(len(label_idx["pos_sent"]), 1)) - (stats["sum_neg_sent"] / max(len(label_idx["neg_sent"]), 1))
    delta_form = (stats["sum_pos_form"] / max(len(label_idx["pos_form"]), 1)) - (stats["sum_neg_form"] / max(len(label_idx["neg_form"]), 1))
    delta_fact = (stats["sum_pos_fact"] / max(len(label_idx["pos_fact"]), 1)) - (stats["sum_neg_fact"] / max(len(label_idx["neg_fact"]), 1))

    # Select top per-domain by |delta| with floor
    min_delta_sent = 0.02
    min_delta_other = 0.05
    per_domain = 150
    selected = set()

    def _select_top(delta_arr, floor):
        abs_delta = np.abs(delta_arr)
        idx = np.where(abs_delta >= floor)[0]
        if idx.size == 0:
            return []
        sorted_idx = idx[np.argsort(abs_delta[idx])[::-1]]
        return sorted_idx[:per_domain].tolist()

    sel_sent = _select_top(delta_sent, min_delta_sent)
    sel_form = _select_top(delta_form, min_delta_other)
    sel_fact = _select_top(delta_fact, min_delta_other)

    for s in sel_sent + sel_form + sel_fact:
        selected.add(int(s))

    # Add top 100 by mean activation as catch-all
    top_mean = np.argsort(mean_abs)[-100:][::-1]
    for s in top_mean.tolist():
        selected.add(int(s))

    # Compute deltas for all concepts and include in selection
    delta_all: Dict[str, np.ndarray] = {}
    counts_all: Dict[str, int] = {}
    for concept_name in all_concepts:
        n_pos = max(len(label_idx.get(f"pos_{concept_name}", [])), 1)
        n_neg = max(len(label_idx.get(f"neg_{concept_name}", [])), 1)
        d_c = (stats[f"sum_pos_{concept_name}"] / n_pos) - (stats[f"sum_neg_{concept_name}"] / n_neg)
        delta_all[concept_name] = d_c
        counts_all[concept_name] = int(n_pos + n_neg)
        sel_c = _select_top(d_c, min_delta_other)
        for s in sel_c:
            selected.add(int(s))

    selected_ids = sorted(selected)

    features = []
    domain_counts: Dict[str, int] = {"other": 0}
    confs: Dict[str, list] = {}

    for feat_id in selected_ids:
        # Find best domain across ALL concepts (built-ins + custom)
        best_domain = "other"
        best_delta = 0.0
        all_abs: Dict[str, float] = {}

        # Built-in trio
        for dom, d_arr, floor in [
            ("sentiment", delta_sent, min_delta_sent),
            ("formality", delta_form, min_delta_other),
            ("certainty", delta_fact, min_delta_other),
        ]:
            v = abs(float(d_arr[feat_id]))
            all_abs[dom] = v
            if v > best_delta and v >= floor:
                best_delta = v
                best_domain = dom

        # Custom concepts
        for concept_name in all_concepts:
            v = abs(float(delta_all[concept_name][feat_id]))
            all_abs[concept_name] = v
            if v > best_delta and v >= min_delta_other:
                best_delta = v
                best_domain = concept_name

        sum_delta = sum(all_abs.values())
        confidence = float(best_delta / (sum_delta + 1e-8)) if best_domain != "other" else 0.0

        domain_counts[best_domain] = domain_counts.get(best_domain, 0) + 1
        if best_domain != "other":
            confs.setdefault(best_domain, []).append(confidence)

        features.append({
            "feature_id": int(feat_id),
            "domain": best_domain,
            "confidence": confidence,
            "mean_activation": float(mean_abs[feat_id]),
            **{f"delta_{c}": float(delta_all[c][feat_id]) for c in all_concepts},
            # Keep built-in deltas for backward compat
            "delta_sentiment": float(delta_sent[feat_id]),
            "delta_formality": float(delta_form[feat_id]),
            "delta_certainty": float(delta_fact[feat_id]),
        })

    # Apply min_confidence filter if requested
    if args.min_confidence > 0:
        before = len(features)
        features = [f for f in features if f["confidence"] >= args.min_confidence or f["domain"] == "other"]
        selected_ids = [f["feature_id"] for f in features]
        print(f"[label] min_confidence={args.min_confidence}: kept {len(features)}/{before} features")

    summary = {
        "model": args.model,
        "n_features_total": n_features,
        "n_features_labelled": len(features),
        "selection": {
            "per_domain": per_domain,
            "min_delta_sentiment": min_delta_sent,
            "min_delta_other": min_delta_other,
            "top_mean_activation": 100,
        },
        "domain_counts": domain_counts,
        "mean_confidence": {
            k: (float(np.mean(v)) if v else 0.0) for k, v in confs.items()
        },
        "selected_feature_ids": selected_ids,
        "features": features,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    high_conf = sum(1 for f in features if f["confidence"] >= 0.7)
    print(f"Saved {out_path}")
    print(f"Selected features: {len(selected_ids)}")
    print(f"Domain counts: {domain_counts}")
    print(f"Features with confidence >= 0.7: {high_conf}")

    log_run("label_features.py", start_time, "success")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log_run("label_features.py", time.time(), "error", str(e))
        raise
