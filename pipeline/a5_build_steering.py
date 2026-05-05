
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
from typing import Dict, List, Tuple

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


def _find_latest_checkpoint(model_name: str, sae_dir: str, ef: int = None) -> str:
    import re as _re
    # 1. Try the canonical naming format first: {model}_ef{ef}_sae.pt
    if ef:
        direct = os.path.join(sae_dir, f"{model_name}_ef{ef}_sae.pt")
        if os.path.exists(direct):
            return direct
    # 2. Also try without ef (legacy)
    direct_legacy = os.path.join(sae_dir, f"{model_name}_sae.pt")
    if os.path.exists(direct_legacy):
        return direct_legacy
    # 3. Glob for step-numbered checkpoint or any matching file
    ef_tag = f"ef{ef}" if ef else None
    candidates = []
    for fname in os.listdir(sae_dir):
        if not fname.endswith(".pt"):
            continue
        if model_name not in fname:
            continue
        if ef_tag and ef_tag not in fname:
            continue
        m = _re.search(r"step(\d+)", fname)
        if m:
            candidates.append((int(m.group(1)), fname))
        elif fname.endswith("_sae.pt"):
            candidates.append((999999, fname))
    if candidates:
        step, fname = sorted(candidates, key=lambda x: x[0])[-1]
        return os.path.join(sae_dir, fname)
    raise FileNotFoundError(
        f"No SAE checkpoint found for {model_name} (ef={ef}) in {sae_dir}. "
        f"Expected: {model_name}_ef{ef}_sae.pt. Files present: {[f for f in os.listdir(sae_dir) if f.endswith('.pt')]}"
    )


def _load_labels() -> Dict[int, Dict[str, int]]:
    labels_path = os.path.join(config.DATA_DIR, "corpus_labels.jsonl")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels file at {labels_path}")
    labels = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            labels[i] = json.loads(line)
    return labels


def _sample_indices(idx: List[int], n: int = 50) -> List[int]:
    if len(idx) <= n:
        return idx
    rng = np.random.default_rng(42)
    return rng.choice(idx, size=n, replace=False).tolist()


def _sentiment_label_split(labels: Dict[int, Dict[str, int]]) -> Tuple[List[int], List[int]]:
    """Robust row-level sentiment split for 0/1, -1/+1, or -2/+2 labels."""
    sent_rows = [(i, float(r["sentiment"]))
                 for i, r in labels.items()
                 if r.get("sentiment") is not None]
    if not sent_rows:
        return [], []
    values = [v for _, v in sent_rows]
    median = float(np.median(values))
    pos = [i for i, v in sent_rows if v > median]
    neg = [i for i, v in sent_rows if v < median]
    return pos, neg


def _get_concept_indices(labels: Dict[int, Dict[str, int]], concept: str) -> Tuple[List[int], List[int]]:
    if concept == "sentiment" or concept == "empathy":
        # Positive: sentiment_yelp_50k with sentiment=2.0
        # Negative: prose_openwebtext_50k / prose_wikipedia_50k with sentiment=-2.0
        # Median split also handles 0/1 sentiment encodings.
        pos, neg = _sentiment_label_split(labels)
    elif concept == "formality":
        pos = [i for i, r in labels.items() if r.get("formality") == 1]
        neg = [i for i, r in labels.items() if r.get("formality") == 0]
    elif concept in ("certainty", "hedging"):
        truth_idx = [i for i, r in labels.items() if r.get("source") == "truthfulqa"]
        pos = [i for i in truth_idx if labels[i].get("factuality") == 1]
        neg = [i for i in truth_idx if labels[i].get("factuality") == 0]
        if not neg:
            # Fallback proxy: negative sentiment rows as "uncertain/hedged"
            neg = [i for i, r in labels.items() if r.get("sentiment") is not None and r["sentiment"] < 0]
    else:
        # Explicit mapping from concept name → corpus source prefix(es).
        # Source names in corpus_labels.jsonl don't always match concept names directly.
        _CONCEPT_SOURCE_MAP = {
            "academic_writing":   ["academic_arxiv"],
            "code_instructions":  ["code_python_instructions"],
            "code_python":        ["code_python_50k", "code_python_snippets"],
            "code_snippets":      ["code_python_snippets"],
            "code_sql":           ["code_sql"],
            "creative_writing":   ["creative_writing"],
            "legal":              ["IndustryCorpus_law"],
            "math_competition":   ["math_numina", "math_tiger"],
            "math_gsm8k":         ["math_gsm8k"],
            "math_olympiad":      ["math_numina"],
            "math_reasoning":     ["math_metamath", "math_tiger", "math_numina"],
            "news_reporting":     ["news_ccnews"],
            "question_answering": ["qa_squad"],
            "science_biomedical": ["science_pubmed"],
        }
        # Generic custom concept. corpus_labels.jsonl may encode the concept as:
        #   (a) a named field with values > 0 (pos) / < 0 (neg)  e.g. cluster_0: 1 / -1
        #   (b) a source tag (using explicit map above, then startswith fallback)
        # Try (a) first.
        pos = [i for i, r in labels.items() if (r.get(concept) or 0) > 0]
        neg = [i for i, r in labels.items() if (r.get(concept) or 0) < 0]
        if not pos or not neg:
            # Try (b): use explicit map first, then startswith fallback.
            src_prefixes = _CONCEPT_SOURCE_MAP.get(concept, [concept])
            src_pos = [i for i, r in labels.items()
                       if any(str(r.get("source", "")).startswith(p) for p in src_prefixes)]
            if src_pos:
                pos = src_pos
                neg = [i for i, r in labels.items()
                       if not any(str(r.get("source", "")).startswith(p) for p in src_prefixes)
                       and r.get("source") not in ("sst2", "yelp", "truthfulqa")]

    return _sample_indices(pos, 50), _sample_indices(neg, 50)


def _safe_source(source: str) -> str:
    return source.replace("/", "_").replace(" ", "_")


def _sentiment_source_groups(labels: Dict[int, Dict[str, int]]) -> Tuple[List[str], List[str]]:
    """Infer positive/negative sentiment source groups from label values.

    On the A100 corpus, sentiment is source-level:
      + sentiment_yelp_50k has positive sentiment labels
      + prose_openwebtext_50k / prose_wikipedia_50k have negative sentiment labels

    Those repaired label rows may not map one-to-one onto the extracted 50k
    per-source activation files, so native sentiment CAA should sample directly
    from the source activation files when source groups can be inferred.
    """
    by_source: Dict[str, List[float]] = {}
    for row in labels.values():
        if row.get("sentiment") is None:
            continue
        by_source.setdefault(str(row.get("source", "")), []).append(float(row["sentiment"]))
    if not by_source:
        return [], []

    all_values = [v for values in by_source.values() for v in values]
    median = float(np.median(all_values))
    pos_sources: List[str] = []
    neg_sources: List[str] = []
    for source, values in sorted(by_source.items()):
        if not source:
            continue
        mean_value = float(np.mean(values))
        if mean_value > median:
            pos_sources.append(source)
        elif mean_value < median:
            neg_sources.append(source)
    return pos_sources, neg_sources


def _read_sampled_source_rows(model_name: str, sources: List[str], n: int = 50) -> np.ndarray:
    """Sample rows directly from existing per-source normalized activation files."""
    files = []
    total_rows = 0
    for source in sources:
        path = os.path.join(
            config.ACTIVATIONS_DIR,
            f"{model_name}_{_safe_source(source)}_activations_norm.h5",
        )
        if not os.path.exists(path):
            print(f"[step7] {model_name}: missing activation file for source '{source}', skipping: {path}")
            continue
        with h5py.File(path, "r") as h5:
            n_rows = len(h5["activations"])
        if n_rows <= 0:
            print(f"[step7] {model_name}: empty activation file for source '{source}', skipping: {path}")
            continue
        files.append((source, path, n_rows, total_rows, total_rows + n_rows))
        total_rows += n_rows

    if total_rows < n:
        raise RuntimeError(
            f"{model_name}: not enough activation rows in sources {sources}: "
            f"available={total_rows}, need={n}"
        )

    rng = np.random.default_rng(42)
    flat_indices = sorted(rng.choice(total_rows, size=n, replace=False).tolist())
    grouped = {path: [] for _, path, _, _, _ in files}
    for flat_i in flat_indices:
        for _, path, _, start, end in files:
            if start <= flat_i < end:
                grouped[path].append(flat_i - start)
                break

    rows = []
    files_used = []
    for _, path, _, _, _ in files:
        local_rows = grouped[path]
        if not local_rows:
            continue
        with h5py.File(path, "r") as h5:
            rows.append(h5["activations"][local_rows])
        files_used.append(f"{os.path.basename(path)}[{len(local_rows)}]")

    if not rows:
        raise RuntimeError(f"{model_name}: sampled no activation rows from sources {sources}")
    print(f"[step7] {model_name}: sentiment CAA source activations: {', '.join(files_used)}")
    return np.concatenate(rows, axis=0)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _load_feature_labels(model_name: str, ef: int = None) -> Dict[int, Dict[str, float]]:
    import glob as _glob
    candidates = []
    if ef:
        candidates.append(os.path.join(config.FEATURES_DIR, f"{model_name}_ef{ef}_feature_labels.json"))
    candidates.append(os.path.join(config.FEATURES_DIR, f"{model_name}_feature_labels.json"))
    # Also glob for any ef-tagged variant — pick the richest one if the primary
    # candidates are missing or have very few concepts (legacy 4-concept files).
    # Use a permissive pattern to handle filenames with a space instead of
    # underscore (e.g. "deepseek-llm-7b_ef128 feature_labels.json").
    glob_matches = sorted(_glob.glob(
        os.path.join(config.FEATURES_DIR, f"{model_name}_ef*feature_labels.json")
    ))
    for path in glob_matches:
        if path not in candidates:
            candidates.append(path)

    best_feats: Dict[int, Dict] = {}
    best_path = None
    for path in candidates:
        if not os.path.exists(path):
            continue
        data = json.load(open(path, "r", encoding="utf-8"))
        feats = {int(f["feature_id"]): f for f in data.get("features", [])}
        # Prefer the file with the most unique named (non-cluster) domains
        n_domains = len({f.get("domain") for f in feats.values()
                         if f.get("domain") and not f.get("domain", "").startswith("cluster_")})
        best_n = len({f.get("domain") for f in best_feats.values()
                      if f.get("domain") and not f.get("domain", "").startswith("cluster_")})
        if n_domains > best_n:
            best_feats = feats
            best_path = path

    if not best_feats:
        raise FileNotFoundError(f"Missing labels file: tried {candidates}")
    if best_path != candidates[0]:
        print(f"[step7] NOTE: using {os.path.basename(best_path)} for {model_name} "
              f"(richest available: {len({f.get('domain') for f in best_feats.values() if f.get('domain')})} domains)")
    return best_feats


def _select_top_features(feat_map: Dict[int, Dict[str, float]], concept: str, topn: int = 3) -> List[Tuple[int, float]]:
    domain = concept
    if concept == "empathy":
        domain = "sentiment"
    if concept == "hedging":
        domain = "certainty"

    candidates = [(fid, f["confidence"]) for fid, f in feat_map.items() if f.get("domain") == domain]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:topn]


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", default="", help="Process only this model (default: all)")
    parser.add_argument("--ef", type=int, default=0, help="Expansion factor override (default: config value)")
    parser.add_argument("--mode", default="both", choices=["both", "caa", "sae_decoder", "gradient_cls"],
                        help="Vector construction method (default: both)")
    parser.add_argument("--top-features", type=int, default=3, dest="top_features",
                        help="Top SAE features per concept (default: 3)")
    parser.add_argument("--min-confidence", type=float, default=0.0, dest="min_confidence",
                        help="Minimum feature confidence score (default: 0.0)")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(config.STEERING_DIR, exist_ok=True)

    # Determine which models to process and effective EF
    model_filter = args.model.strip() if args.model else ""
    ef_override = args.ef if args.ef > 0 else 0

    if model_filter:
        ef_val = ef_override if ef_override > 0 else config.SAE_EXPANSION_FACTOR
        out_path = os.path.join(config.STEERING_DIR, f"{model_filter}_ef{ef_val}_steering_vectors.json")
    else:
        out_path = os.path.join(config.STEERING_DIR, "steering_vectors.json")

    if os.path.exists(out_path) and not args.force:
        print(f"{os.path.basename(out_path)} exists. Use --force to recompute.")
        log_run("step7_build_steering.py", start_time, "skipped")
        return 0

    labels = _load_labels()

    models_to_process = (
        {model_filter: config.MODELS[model_filter]}
        if model_filter and model_filter in config.MODELS
        else dict(config.MODELS)
    )
    if model_filter and model_filter not in config.MODELS:
        print(f"Unknown model '{model_filter}'. Known: {list(config.MODELS)}")
        return 1

    output = {}
    for model_name, model_cfg in models_to_process.items():
        hidden_dim = model_cfg["hidden_dim"]
        ef = ef_override if ef_override > 0 else model_cfg.get("sae_ef", config.SAE_EXPANSION_FACTOR)
        n_features = hidden_dim * ef

        ckpt_path = _find_latest_checkpoint(model_name, config.SAE_DIR, ef=ef)
        _sae_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        sae = TopKSAE(hidden_dim, n_features, model_cfg["sae_topk"]).to(_sae_device)
        sae.load_state_dict(torch.load(ckpt_path, map_location=_sae_device))
        sae.eval()

        feat_map = _load_feature_labels(model_name, ef=ef)

        # Load normalized activations for CAA
        acts_path = os.path.join(config.ACTIVATIONS_DIR, f"{model_name}_activations_norm.h5")
        if os.path.exists(acts_path):
            with h5py.File(acts_path, "r") as h5:
                acts = h5["activations"][:]
        else:
            # No merged file — concatenate per-source norm files (same order as step3/MultiSourceActivationStore)
            _prefix = f"{model_name}_"
            _suffix = "_activations_norm.h5"
            _per_src = sorted(
                os.path.join(config.ACTIVATIONS_DIR, f)
                for f in os.listdir(config.ACTIVATIONS_DIR)
                if f.startswith(_prefix) and f.endswith(_suffix)
            )
            if not _per_src:
                raise FileNotFoundError(
                    f"Missing activations for {model_name}: no merged file at {acts_path} "
                    f"and no per-source files matching {_prefix}*{_suffix} in {config.ACTIVATIONS_DIR}"
                )
            print(f"[step7] {model_name}: loading {len(_per_src)} per-source activation files...")
            _parts = []
            for _p in _per_src:
                with h5py.File(_p, "r") as _h5:
                    _parts.append(_h5["activations"][:])
            acts = np.concatenate(_parts, axis=0)
            print(f"[step7] {model_name}: concatenated activations shape: {acts.shape}")

        # Derive concept list dynamically from the feature labels themselves.
        # This respects whatever concepts were discovered in A4 (label_features.py)
        # rather than relying on a hardcoded config list.
        concepts_from_labels: set = set()
        for fdata in feat_map.values():
            d = fdata.get("domain")
            if d:
                concepts_from_labels.add(d)
        # Fall back to config.TARGET_CONCEPTS only if feature labels have no domain info
        if concepts_from_labels:
            target_concepts = sorted(concepts_from_labels)
            print(f"[step7] {model_name}: using {len(target_concepts)} concepts from feature labels: {target_concepts[:8]}{'...' if len(target_concepts) > 8 else ''}")
        else:
            target_concepts = list(config.TARGET_CONCEPTS)
            print(f"[step7] {model_name}: no domain info in feature labels — falling back to config.TARGET_CONCEPTS ({len(target_concepts)} concepts)")

        output[model_name] = {}
        for concept in target_concepts:
            # Method 1: SAE decoder vectors
            top_feats = _select_top_features(feat_map, concept, topn=args.top_features)
            # Filter by min_confidence
            if args.min_confidence > 0:
                top_feats = [(fid, conf) for fid, conf in top_feats if conf >= args.min_confidence]
            method1_vec = None
            top_feat_ids = []
            if args.mode in ("both", "sae_decoder") and top_feats:
                weights = []
                vecs = []
                for fid, conf in top_feats:
                    top_feat_ids.append(int(fid))
                    vec = sae.decoder.weight[:, fid].detach().cpu().numpy()
                    vecs.append(vec)
                    weights.append(conf)
                w = np.array(weights, dtype=np.float32)
                if w.sum() == 0:
                    w = np.ones_like(w)
                w = w / w.sum()
                method1_vec = np.sum([w[i] * vecs[i] for i in range(len(vecs))], axis=0)
                method1_vec = _l2_normalize(method1_vec)

            # Method 2: CAA on raw activations
            method2_vec = None
            if args.mode in ("both", "caa"):
                if concept == "sentiment":
                    pos_sources, neg_sources = _sentiment_source_groups(labels)
                    if pos_sources and neg_sources:
                        pos_acts = _read_sampled_source_rows(model_name, pos_sources, 50)
                        neg_acts = _read_sampled_source_rows(model_name, neg_sources, 50)
                        pos_mean = pos_acts.mean(axis=0)
                        neg_mean = neg_acts.mean(axis=0)
                        method2_vec = _l2_normalize((pos_mean - neg_mean).astype(np.float32))
                    else:
                        pos_idx, neg_idx = _get_concept_indices(labels, concept)
                        if pos_idx and neg_idx:
                            pos_mean = acts[pos_idx].mean(axis=0)
                            neg_mean = acts[neg_idx].mean(axis=0)
                            method2_vec = _l2_normalize((pos_mean - neg_mean).astype(np.float32))
                else:
                    pos_idx, neg_idx = _get_concept_indices(labels, concept)
                    if pos_idx and neg_idx:
                        pos_mean = acts[pos_idx].mean(axis=0)
                        neg_mean = acts[neg_idx].mean(axis=0)
                        method2_vec = _l2_normalize((pos_mean - neg_mean).astype(np.float32))

            cosine = None
            if method1_vec is not None and method2_vec is not None:
                cosine = float(np.dot(method1_vec, method2_vec))

            output[model_name][concept] = {
                "sae_vector": method1_vec.tolist() if method1_vec is not None else None,
                "caa_vector": method2_vec.tolist() if method2_vec is not None else None,
                "cosine_similarity_between_methods": cosine,
                "top_features_used": top_feat_ids,
                "vector_dim": hidden_dim,
                "injection_layer": model_cfg["target_layer"],
            }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f)

    log_run("step7_build_steering.py", start_time, "success")
    return 0



if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log_run("a5_build_steering.py", time.time(), "error", str(e))
        raise
