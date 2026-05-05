"""
C2 manual labeling helper — label_universal_concepts.py
========================================================
For each universal concept cluster, shows top activating passages
from the corpus so you can assign a human label interactively.

Usage (run on A100):
    python label_universal_concepts.py
    python label_universal_concepts.py --concepts-file universal/mean_concepts.json
    python label_universal_concepts.py --passages-per-cluster 5 --skip-labelled

Writes labels back to mean_concepts.json in-place.
To preview without writing: --dry-run
"""


import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
CORPUS_PATH = SCRIPT_DIR / "data" / "corpus.jsonl"
FEATURES_DIR = SCRIPT_DIR / "features"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_corpus(path: Path) -> tuple[list[str], dict[str, list[int]]]:
    """
    Load corpus.jsonl.
    Returns:
      texts:          list of text strings (indexed by position)
      source_index:   {source_name: [list of indices]} for fast domain lookup
    """
    texts: list[str] = []
    source_index: dict[str, list[int]] = {}
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                texts.append(obj.get("text", ""))
                src = obj.get("source", "unknown")
                source_index.setdefault(src, []).append(i)
            except Exception:
                texts.append("")
    return texts, source_index


# ── corpus source → feature label keyword mapping ────────────────────────────
# Maps feature label keywords (from feature_labels.json) to corpus source names.
_LABEL_TO_SOURCES: dict[str, list[str]] = {
    "code_python":        ["code_python_50k", "code_python_instructions_15k", "code_python_snippets_5k"],
    "code_instructions":  ["code_python_instructions_15k", "code_python_50k"],
    "code_snippets":      ["code_python_snippets_5k", "code_python_50k"],
    "code_sql":           ["code_sql_50k"],
    "math_gsm8k":         ["math_gsm8k_8k"],
    "math_metamath":      ["math_metamath_50k"],
    "math_tiger":         ["math_tiger_50k"],
    "math_numina":        ["math_numina_50k"],
    "math_competition":   ["math_numina_50k", "math_tiger_50k"],
    "math_olympiad":      ["math_numina_50k", "math_tiger_50k"],
    "math_reasoning":     ["math_metamath_50k", "math_gsm8k_8k"],
    "math":               ["math_metamath_50k", "math_gsm8k_8k", "math_tiger_50k"],
    "creative_writing":   ["creative_writing_50k"],
    "academic_writing":   ["academic_arxiv_50k"],
    "science_biomedical": ["science_pubmed_50k"],
    "legal":              ["legal_freelaw_50k", "IndustryCorpus_law"],
    "news_reporting":     ["news_ccnews_50k"],
    "question_answering": ["qa_squad_50k"],
    "sentiment":          ["sentiment_yelp_50k"],
    "formality":          ["sentiment_yelp_50k"],
    "certainty":          ["academic_arxiv_50k", "science_pubmed_50k"],
    "prose":              ["prose_openwebtext_50k", "prose_wikipedia_50k"],
    "wikipedia":          ["prose_wikipedia_50k"],
}


def _sample_corpus_by_domain(
    domain_labels: list[str],
    corpus_texts: list[str],
    source_index: dict[str, list[int]],
    n: int,
    seed: int = 0,
) -> list[str]:
    """
    Given a list of feature domain labels (e.g. ['math_gsm8k', 'academic_writing']),
    find matching corpus sources and randomly sample n passages from them.
    Uses seed so different clusters get different samples.
    """
    import random as _random
    rng = _random.Random(seed * 97 + 13)

    candidate_indices: list[int] = []
    for label in domain_labels:
        label_lower = label.lower()
        # Exact match first
        for key, sources in _LABEL_TO_SOURCES.items():
            if key in label_lower or label_lower in key:
                for src in sources:
                    candidate_indices.extend(source_index.get(src, []))
                break

    if not candidate_indices:
        return []

    rng.shuffle(candidate_indices)
    results = []
    for idx in candidate_indices[:n * 10]:   # look at up to 10× candidates to skip empties
        text = corpus_texts[idx].strip()
        if len(text) > 50:
            results.append(text[:300])
        if len(results) >= n:
            break
    return results


# ── domain JSONL files known to exist ───────────────────────────────────────
# Searched in DATA_DIR and common checkpoint paths.
_DOMAIN_FILE_CANDIDATES: list[str] = [
    "code_python_50k.jsonl",
    ".ipynb_checkpoints/creative_writing_50k-checkpoint.jsonl",
    ".ipynb_checkpoints/math_metamath_50k-checkpoint.jsonl",
    ".ipynb_checkpoints/math_tiger_50k-checkpoint.jsonl",
    "corpus_labels.jsonl",  # labels only, no text — skipped if no 'text' key
]
_DOMAIN_KEYWORD_MAP: dict[str, list[str]] = {
    "code":          ["code_python_50k"],
    "python":        ["code_python_50k"],
    "math":          ["math_metamath_50k", "math_tiger_50k", "math_gsm8k"],
    "creative":      ["creative_writing_50k"],
    "writing":       ["creative_writing_50k"],
}


def sample_domain_file(data_dir: Path, domain_hint: str, n: int = 5, seed: int = 0) -> list[str]:
    """Sample n text lines from domain JSONL files matching domain_hint keyword.
    Uses seed (typically cluster_id) to get different samples per cluster."""
    keywords = [k for k, files in _DOMAIN_KEYWORD_MAP.items() if k in domain_hint.lower()]
    target_stems = [f for k in keywords for f in _DOMAIN_KEYWORD_MAP[k]]
    if not target_stems:
        target_stems = ["code_python_50k"]

    for stem in target_stems:
        for candidate in _DOMAIN_FILE_CANDIDATES:
            if stem in candidate:
                fpath = data_dir / candidate
                if fpath.exists():
                    try:
                        # Count lines quickly then pick a random offset
                        import random as _random
                        rng = _random.Random(seed * 97 + 13)
                        all_texts: list[str] = []
                        with open(fpath) as f:
                            for line in f:
                                obj = json.loads(line)
                                t = obj.get("text", "")
                                if t and len(t) > 30:
                                    all_texts.append(t[:280])
                        if all_texts:
                            # Sample n items with deterministic offset from seed
                            start = rng.randint(0, max(0, len(all_texts) - n))
                            return all_texts[start:start + n]
                    except Exception:
                        pass
    return []


def load_autodiscovered(features_dir: Path, model: str, ef: int
                        ) -> tuple[dict[int, list[str]], dict[int, dict]]:
    """
    Load {model}_ef{ef}_autodiscovered.json.
    Returns:
      fid_passages: {feature_id: top_passages_list}
      fid_meta:     {feature_id: {source_distribution, domain, label}}
    Falls back to files without ef tag.
    """
    for fname in [
        f"{model}_ef{ef}_autodiscovered.json",
        f"{model}_autodiscovered.json",
    ]:
        path = features_dir / fname
        if not path.exists():
            # Also try one level up
            path = features_dir.parent / fname
        if path.exists():
            try:
                clusters = json.load(open(path))
                fid_passages: dict[int, list[str]] = {}
                fid_meta:     dict[int, dict]       = {}
                for cl in clusters:
                    passages = cl.get("top_passages") or []
                    src_dist = cl.get("source_distribution") or {}
                    domain   = cl.get("domain", "")
                    label    = cl.get("label", "")
                    meta     = {"source_distribution": src_dist, "domain": domain, "label": label}
                    for fid in cl.get("feature_indices", []):
                        fid_passages[int(fid)] = passages
                        fid_meta[int(fid)]     = meta
                return fid_passages, fid_meta
            except Exception as e:
                print(f"  [warn] could not load autodiscovered for {model}: {e}", file=sys.stderr)
    return {}, {}


_feature_labels_cache: dict[str, dict[int, str]] = {}

def load_feature_labels(features_dir: Path, model: str, ef: int) -> dict[int, str]:
    """
    Load {model}_ef{ef}_feature_labels.json → {feature_id: domain_label}.
    Falls back to any {model}_feature_labels.json.  Cached.
    """
    key = f"{model}:{ef}"
    if key in _feature_labels_cache:
        return _feature_labels_cache[key]

    for fname in [
        f"{model}_ef{ef}_feature_labels.json",
        f"{model}_feature_labels.json",
    ]:
        path = features_dir / fname
        if path.exists():
            try:
                raw = json.load(open(path))
                features = raw.get("features", raw) if isinstance(raw, dict) else raw
                mapping = {}
                for entry in features:
                    fid = entry.get("feature_id")
                    label = (entry.get("label") or entry.get("domain") or
                             entry.get("name") or "").strip()
                    if fid is not None and label:
                        mapping[int(fid)] = label
                _feature_labels_cache[key] = mapping
                return mapping
            except Exception:
                pass

    _feature_labels_cache[key] = {}
    return {}


def top_passages_from_acts(
    features_dir: Path, model: str, ef: int,
    feature_ids: list[int], corpus: list[str],
    n: int = 6,
) -> list[str]:
    """
    Fall back to raw feature_acts.npy: for each feature_id in the cluster,
    find the top-n passages by activation and return unique texts.
    """
    import glob
    pat = str(features_dir / f"{model}_ef{ef}_top*_feature_acts.npy")
    acts_files = sorted(glob.glob(pat))
    idx_pat    = str(features_dir / f"{model}_ef{ef}_top*_feature_idx.npy")
    idx_files  = sorted(glob.glob(idx_pat))
    if not acts_files or not idx_files:
        return []

    # pick the largest (most features)
    acts_path = acts_files[-1]
    idx_path  = idx_files[-1]

    try:
        acts = np.load(acts_path, mmap_mode="r")   # (N_passages, k_features)
        idx  = np.load(idx_path)                   # (k_features,) → actual SAE ids
    except Exception as e:
        print(f"  [warn] could not load acts for {model}: {e}", file=sys.stderr)
        return []

    seen: set[int] = set()
    results: list[str] = []

    for fid in feature_ids:
        col_candidates = np.where(idx == fid)[0]
        if len(col_candidates) == 0:
            continue
        col = int(col_candidates[0])
        act_col = acts[:, col]
        top_rows = np.argsort(act_col)[::-1][:n]
        for row in top_rows:
            if int(row) not in seen and int(row) < len(corpus):
                seen.add(int(row))
                results.append(corpus[int(row)][:300])
        if len(results) >= n:
            break

    return results[:n]


def get_passages_for_cluster(
    cluster: dict,
    corpus: tuple[list[str], dict[str, list[int]]] | None,
    autodiscovered_cache: dict[str, tuple],
    features_dir: Path,
    n: int,
    model_ef_map: dict[str, int],
    data_dir: Path,
) -> tuple[list[str], str, dict]:
    """
    Return (passages, source_model, meta_dict) for a cluster.
    meta_dict may include source_distribution.
    Priority: corpus acts lookup (when corpus loaded) > autodiscovered > domain file samples.
    Autodiscovered passages are pre-baked and may mix unrelated domains (e.g. SQL in math clusters);
    corpus-based lookup uses actual feature activation scores against the full corpus.
    """
    per_model: dict = cluster.get("per_model", {})
    models_ranked = sorted(per_model, key=lambda m: -len(per_model[m].get("feature_ids", [])))

    best_meta: dict = {}

    # Collect autodiscovered meta for all models (for source_distribution etc.)
    # but don't use their pre-baked passages if we have a real corpus.
    for m in models_ranked:
        ef   = model_ef_map.get(m, 64)
        fids: list[int] = per_model[m].get("feature_ids", [])
        if not fids:
            continue
        if m not in autodiscovered_cache:
            autodiscovered_cache[m] = load_autodiscovered(features_dir, m, ef)
        _, fid_meta = autodiscovered_cache[m]
        for fid in fids:
            if fid in fid_meta and not best_meta:
                best_meta = fid_meta[fid]

    # Corpus-based lookup: sample by source matching feature domain labels
    if corpus:
        corpus_texts, source_index = corpus
        # Collect domain labels from all models' features
        domain_labels: list[str] = []
        for m, mv in per_model.items():
            ef = model_ef_map.get(m, 64)
            fmap = load_feature_labels(features_dir, m, ef)
            for fid in mv.get("feature_ids", []):
                lbl = fmap.get(int(fid), "")
                if lbl:
                    domain_labels.append(lbl)

        # Map domain labels to corpus source names
        passages = _sample_corpus_by_domain(
            domain_labels, corpus_texts, source_index, n,
            seed=cluster.get("cluster_id", 0)
        )
        if passages:
            return passages, "corpus", best_meta

    # Autodiscovered fallback (no corpus available)
    for m in models_ranked:
        ef   = model_ef_map.get(m, 64)
        fids = per_model[m].get("feature_ids", [])
        if not fids:
            continue
        fid_passages, fid_meta = autodiscovered_cache.get(m, ({}, {}))
        for fid in fids:
            if fid in fid_passages and fid_passages[fid]:
                meta = fid_meta.get(fid, {})
                snippets = [p[:300].replace("\n", " ") for p in fid_passages[fid][:n]]
                return snippets, m, meta

    # Domain file fallback — use domain hint from feature labels
    domain_hints = []
    for m, mv in per_model.items():
        ef = model_ef_map.get(m, 64)
        fmap = load_feature_labels(features_dir, m, ef)
        for fid in mv.get("feature_ids", []):
            lbl = fmap.get(int(fid), "")
            if lbl:
                domain_hints.append(lbl)
    domain_hint = " ".join(domain_hints)
    cluster_id  = cluster.get("cluster_id", 0)
    sampled = sample_domain_file(data_dir, domain_hint, n, seed=cluster_id)
    if sampled:
        return sampled, f"domain_sample({domain_hint[:30]})", best_meta

    return [], "unknown", best_meta


def get_feature_domain_hints(
    cluster: dict,
    features_dir: Path,
    model_ef_map: dict[str, int],
) -> list[str]:
    """
    For each model in the cluster, look up the domain/label of each feature
    from feature_labels.json.  Returns deduplicated sorted list.
    """
    seen: set[str] = set()
    hints: list[str] = []
    per_model = cluster.get("per_model", {})
    for m, mv in per_model.items():
        ef   = model_ef_map.get(m, 64)
        fmap = load_feature_labels(features_dir, m, ef)
        for fid in mv.get("feature_ids", []):
            lbl = fmap.get(int(fid), "")
            if lbl and lbl not in seen and lbl not in ("other", "unknown"):
                seen.add(lbl)
                hints.append(f"{m}#{fid}: {lbl}")
    return hints


def print_cluster(
    cluster: dict,
    passages: list[str],
    source_model: str,
    domain_hints: list[str],
    meta: dict,
) -> None:
    cid   = cluster["cluster_id"]
    label = cluster["label"]
    models = cluster.get("models_present", [])
    n_models = len(models)
    print()
    print("=" * 72)
    print(f"  Cluster {cid:2d} | current label: {label!r}")
    print(f"  Models ({n_models}/5): {', '.join(models)}")
    per_model = cluster.get("per_model", {})
    for m, v in per_model.items():
        fids = v.get("feature_ids", [])
        print(f"    {m}: {len(fids)} feature(s)  ids={fids[:8]}{'...' if len(fids)>8 else ''}")

    # Source distribution from autodiscovered (% of text from each domain)
    src_dist: dict = meta.get("source_distribution", {})
    if src_dist:
        top_src = sorted(src_dist.items(), key=lambda x: -x[1])[:4]
        print()
        print("  --- Fires mostly on: ---")
        for src, pct in top_src:
            bar = "█" * int(pct * 20)
            print(f"    {bar:<20s} {pct*100:.0f}%  {src}")

    # Feature domain labels (from feature_labels.json)
    if domain_hints:
        print()
        print("  --- Feature label(s): ---")
        for h in domain_hints[:12]:
            print(f"    {h}")
        if len(domain_hints) > 12:
            print(f"    ... (+{len(domain_hints)-12} more)")

    # Passage text
    print()
    if not passages:
        print("  [no passages — use domain hints above to infer the concept]")
    else:
        label_suffix = "" if source_model.startswith("domain_sample") else f"(source: {source_model})"
        print(f"  --- Example passages {label_suffix} ---")
        for i, p in enumerate(passages, 1):
            snippet = p.replace("\n", " ").strip()
            print(f"  [{i}] {snippet[:280]}")
    print("=" * 72)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Label universal concepts interactively")
    parser.add_argument("--concepts-file", default=str(SCRIPT_DIR / "universal" / "mean_concepts.json"),
                        help="Path to mean_concepts.json (default: universal/mean_concepts.json)")
    parser.add_argument("--corpus",         default=str(CORPUS_PATH))
    parser.add_argument("--features-dir",   default=str(FEATURES_DIR))
    parser.add_argument("--passages-per-cluster", type=int, default=6)
    parser.add_argument("--skip-labelled",  action="store_true",
                        help="Skip clusters that already have a human label (non cluster_N)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print clusters but don't save labels")
    args = parser.parse_args()

    concepts_file  = Path(args.concepts_file)
    corpus_path    = Path(args.corpus)
    features_dir   = Path(args.features_dir)

    if not concepts_file.exists():
        sys.exit(f"ERROR: concepts file not found: {concepts_file}")

    print(f"[label] Loading concepts from: {concepts_file}")
    data = json.load(open(concepts_file))
    concepts: list[dict] = data["universal_concepts"]
    total = len(concepts)
    print(f"[label] {total} clusters found")

    # model → ef map (inferred from per_model data or config)
    model_ef_map: dict[str, int] = {}
    try:
        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location("_cfg", SCRIPT_DIR / "config.py")
        cfg  = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)  # type: ignore
        model_ef_map = getattr(cfg, "MODEL_EF_MAP", {})
    except Exception:
        pass
    # Fallback defaults
    for m in ["gpt2-large", "gpt2"]:
        model_ef_map.setdefault(m, 64)
    for m in ["gemma", "llama", "mistral", "deepseek-llm-7b"]:
        model_ef_map.setdefault(m, 128)

    print(f"[label] Loading corpus (this may take a moment)...")
    data_dir = features_dir.parent / "data"
    corpus_texts: list[str] = []
    corpus_source_index: dict[str, list[int]] = {}
    if corpus_path.exists():
        corpus_texts, corpus_source_index = load_corpus(corpus_path)
        print(f"[label] Corpus: {len(corpus_texts):,} passages, {len(corpus_source_index)} sources")
    else:
        # Try common A100 locations
        for candidate in [
            Path("/home/jovyan/steering-v2-a100/universal_steering/data/corpus.jsonl"),
            features_dir.parent / "data" / "corpus.jsonl",
            Path("/data/corpus.jsonl"),
        ]:
            if candidate != corpus_path and candidate.exists():
                corpus_path = candidate
                corpus_texts, corpus_source_index = load_corpus(corpus_path)
                print(f"[label] Corpus found at {corpus_path}: {len(corpus_texts):,} passages")
                break
        else:
            print(f"[label] corpus.jsonl not found — will use autodiscovered passages + domain files")

    # Check which domain files exist and report
    available_domain_files = [
        f for f in _DOMAIN_FILE_CANDIDATES if (data_dir / f).exists()
    ]
    if available_domain_files:
        print(f"[label] Domain files available: {available_domain_files}")

    autodiscovered_cache: dict[str, tuple] = {}
    labelled = 0
    skipped  = 0

    for idx, cluster in enumerate(concepts):
        cid     = cluster["cluster_id"]
        label   = cluster["label"]
        is_auto = label.startswith("cluster_")

        if args.skip_labelled and not is_auto:
            skipped += 1
            continue

        passages, source_model, meta = get_passages_for_cluster(
            cluster,
            (corpus_texts, corpus_source_index) if corpus_texts else None,
            autodiscovered_cache, features_dir,
            args.passages_per_cluster, model_ef_map, data_dir,
        )
        domain_hints = get_feature_domain_hints(cluster, features_dir, model_ef_map)

        # Always store the evidence passages in the cluster (for NeurIPS)
        if passages:
            cluster["evidence_passages"] = passages
            cluster["evidence_source"] = source_model
        if meta.get("source_distribution"):
            cluster["source_distribution"] = meta["source_distribution"]
        cluster["domain_hints"] = domain_hints

        print_cluster(cluster, passages, source_model, domain_hints, meta)
        print(f"  Progress: {idx+1}/{total}  (labelled this session: {labelled})")

        if args.dry_run:
            print("  [dry-run — skipping input]")
            continue

        while True:
            try:
                user_input = input(
                    f"\n  Enter label (or ENTER to keep {label!r}, 's' to skip, 'q' to quit): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[label] Interrupted — saving progress...")
                _save(concepts_file, data)
                sys.exit(0)

            if user_input.lower() == "q":
                print("[label] Quitting — saving progress...")
                _save(concepts_file, data)
                sys.exit(0)
            if user_input.lower() == "s":
                print(f"  Skipped cluster {cid}")
                break
            if user_input == "":
                print(f"  Kept  {label!r}")
                break
            # Confirm non-trivial labels
            confirm = input(f"  Label = {user_input!r}  — confirm? [y/N] ").strip().lower()
            if confirm == "y":
                cluster["label"]        = user_input
                cluster["label_source"] = "human"
                cluster["label_date"]   = __import__("datetime").date.today().isoformat()
                labelled += 1
                print(f"  ✓ Labelled cluster {cid}: {user_input!r}")
                break
            else:
                print("  Re-enter label.")

    if not args.dry_run:
        _save(concepts_file, data)
        print(f"\n[label] Done. Labelled {labelled} clusters (skipped {skipped}). Saved to {concepts_file}")
    else:
        print(f"\n[label] Dry-run complete. {total} clusters shown.")


def _save(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[label] Saved → {path}")


if __name__ == "__main__":
    main()
