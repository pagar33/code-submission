"""
Post-process mean_concepts.json: merge clusters into canonical concepts using a
hand-curated cluster_id → canonical_concept mapping.

This replaces string-exact label matching, which inflates concept count because
Claude produces synonymous labels (e.g. math_word_problems, mathematical_problem_solving,
math_problem_solving are all the same concept).

Strategy per canonical group:
  - models_present: union across all member clusters
  - center:         weighted average of per-cluster centres (weight = n models in cluster)
  - per_model:      union of feature_ids / activations
  - cluster_id:     kept from the cluster with the highest model count (canonical)
  - canonical_concept: added as a new field (used by build_universal_vectors.py)

NOISE_CLUSTER_IDS: excluded from output, reported in paper.

Writes: universal/mean_concepts_clean.json  (does NOT overwrite mean_concepts.json)
"""


import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── Hand-curated canonical mapping (cluster_id → canonical_concept) ──────────
# Derived by human review of all 113 Claude labels (May 2, 2026).
# See experiments.md § C2 Cluster Relabeling for full per-cluster table.
CANONICAL_MAP: dict[int, str] = {
    # python_code — 29 clusters
    11: "python_code", 12: "python_code", 13: "python_code",
    15: "python_code", 17: "python_code", 18: "python_code",
    19: "python_code", 20: "python_code", 23: "python_code",
    27: "python_code", 34: "python_code", 41: "python_code",
    45: "python_code", 48: "python_code", 52: "python_code",
    58: "python_code", 64: "python_code", 65: "python_code",
    67: "python_code", 69: "python_code", 70: "python_code",
    72: "python_code", 79: "python_code", 80: "python_code",
    95: "python_code", 96: "python_code", 104: "python_code",
    105: "python_code", 106: "python_code",
    # math_problems — 20 clusters
    14: "math_problems", 21: "math_problems", 24: "math_problems",
    25: "math_problems", 33: "math_problems", 36: "math_problems",
    38: "math_problems", 49: "math_problems", 54: "math_problems",
    78: "math_problems", 81: "math_problems", 82: "math_problems",
    97: "math_problems", 102: "math_problems", 103: "math_problems",
    117: "math_problems", 121: "math_problems", 122: "math_problems",
    123: "math_problems", 124: "math_problems",
    # sql_queries — 11 clusters
    31: "sql_queries", 40: "sql_queries",  77: "sql_queries",
    100: "sql_queries", 101: "sql_queries", 107: "sql_queries",
    109: "sql_queries", 110: "sql_queries", 116: "sql_queries",
    118: "sql_queries", 119: "sql_queries",
    # legal_and_news — 10 clusters
    35: "legal_and_news",  59: "legal_and_news",  60: "legal_and_news",
    63: "legal_and_news",  66: "legal_and_news",  93: "legal_and_news",
    94: "legal_and_news", 108: "legal_and_news", 111: "legal_and_news",
    115: "legal_and_news",
    # medical_research — 8 clusters
    57: "medical_research",  68: "medical_research",  73: "medical_research",
    76: "medical_research",  85: "medical_research",  87: "medical_research",
    88: "medical_research",  90: "medical_research",
    # academic_scientific — 8 clusters
     32: "academic_scientific",  39: "academic_scientific",  46: "academic_scientific",
     53: "academic_scientific",  56: "academic_scientific",  61: "academic_scientific",
     62: "academic_scientific", 128: "academic_scientific",
    # narrative_fiction — 6 clusters
    43: "narrative_fiction", 44: "narrative_fiction", 51: "narrative_fiction",
    55: "narrative_fiction", 91: "narrative_fiction", 92: "narrative_fiction",
    # encyclopedic_historical — 5 clusters
     42: "encyclopedic_historical",  50: "encyclopedic_historical",
    120: "encyclopedic_historical", 126: "encyclopedic_historical",
    127: "encyclopedic_historical",
    # code_and_math — 4 clusters (compound concept)
    22: "code_and_math", 26: "code_and_math", 37: "code_and_math", 47: "code_and_math",
    # customer_reviews — 3 clusters
     86: "customer_reviews", 112: "customer_reviews", 125: "customer_reviews",
    # sql_and_medical — 3 clusters (compound concept)
    89: "sql_and_medical", 98: "sql_and_medical", 99: "sql_and_medical",
}

# Cluster IDs excluded from output — reported explicitly in paper as boundary noise.
# Reasons:
#   16  multi_domain_text_samples      — passages genuinely mixed, no single concept
#   28  incomplete_text_continuations  — malformed/truncated evidence passages
#   83  code_and_medical_abstracts     — boundary between python_code and medical_research
#   84  technical_formal_prose         — too vague, Claude could not identify domain
#  113  legal_and_mathematical_problems — boundary cluster, no clean semantic core
#  114  truncated_text_passages         — malformed; old label was `news_reporting`news_reporting
NOISE_CLUSTER_IDS: set[int] = {16, 28, 83, 84, 113, 114}

UNIVERSAL_DIR = Path(__file__).parent / "universal"
SRC = UNIVERSAL_DIR / "mean_concepts.json"
DST = UNIVERSAL_DIR / "mean_concepts_clean.json"


def _norm(label: str) -> str:
    """Normalise label to a stable key: strip backticks/quotes, lowercase, underscores."""
    s = label.strip().strip("`\"'").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _clean_label(label: str) -> str:
    """Clean a label: replace embedded backticks/quotes with _, collapse duplicates."""
    import re as _re
    # Replace backtick/quote between word chars with underscore
    s = _re.sub(r'(?<=[a-z0-9])[`\"\' ]+(?=[a-z0-9])', '_', label.strip())
    # Strip any remaining leading/trailing junk
    s = s.strip("`\"' _")
    # Collapse multiple underscores
    s = _re.sub(r'_+', '_', s)
    # Deduplicate repeated token sequences (news_reporting_news_reporting -> news_reporting)
    tokens = s.split('_')
    seen, dedup = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return '_'.join(dedup)


def merge_group(group: list) -> dict:
    """Merge a list of same-label clusters into one canonical entry."""
    # Sort descending by model count so the canonical entry is the richest one
    group = sorted(group, key=lambda c: len(c["models_present"]), reverse=True)
    canonical = group[0]

    all_models = set()
    for c in group:
        all_models.update(c["models_present"])

    # Weighted centre: weight by number of models in each cluster
    centre_vecs = np.array([c["center"] for c in group])
    weights = np.array([len(c["models_present"]) for c in group], dtype=float)
    weights /= weights.sum()
    merged_center = (centre_vecs * weights[:, None]).sum(axis=0).tolist()

    # Per-model: union of feature_ids; average activations for features that appear twice
    per_model: dict = {}
    for c in group:
        for model, info in c.get("per_model", {}).items():
            if model not in per_model:
                per_model[model] = {"feature_ids": list(info["feature_ids"]),
                                    "activations": list(info["activations"])}
            else:
                # Add any new feature_ids not already present
                existing_ids = set(per_model[model]["feature_ids"])
                for fid, act in zip(info["feature_ids"], info["activations"]):
                    if fid not in existing_ids:
                        per_model[model]["feature_ids"].append(fid)
                        per_model[model]["activations"].append(act)
                        existing_ids.add(fid)

    return {
        "cluster_id":     canonical["cluster_id"],
        "label":          _clean_label(canonical["label"]),  # strip stray backticks/quotes anywhere
        "models_present": sorted(all_models),
        "center":         merged_center,
        "per_model":      per_model,
        "merged_from_n_clusters": len(group),
    }


def main():
    with open(SRC) as f:
        data = json.load(f)

    concepts = data["universal_concepts"]
    print(f"Input: {len(concepts)} clusters")

    # Route each cluster to its canonical concept via CANONICAL_MAP
    groups: dict[str, list] = defaultdict(list)
    noise_found = []
    unmapped = []

    for c in concepts:
        cid = c["cluster_id"]
        if cid in NOISE_CLUSTER_IDS:
            noise_found.append(c)
            continue
        canonical = CANONICAL_MAP.get(cid)
        if canonical is None:
            unmapped.append(c)
            print(f"  ⚠️  cluster {cid} ('{c['label']}') not in CANONICAL_MAP — treating as noise")
            continue
        groups[canonical].append(c)

    if noise_found:
        print(f"\nExcluded noise clusters ({len(noise_found)}):")
        for c in sorted(noise_found, key=lambda x: x["cluster_id"]):
            print(f"  cluster {c['cluster_id']:3d}  {c['label']}")

    # Merge each canonical group
    merged = []
    for canonical_concept, group in groups.items():
        entry = merge_group(group)
        entry["canonical_concept"] = canonical_concept
        entry["label"] = canonical_concept   # canonical name is the clean label
        merged.append(entry)

    # Sort: most models first, then alphabetically by canonical concept
    merged.sort(key=lambda c: (-len(c["models_present"]), c["canonical_concept"]))

    n_5 = sum(1 for c in merged if len(c["models_present"]) == 5)
    n_4 = sum(1 for c in merged if len(c["models_present"]) == 4)
    n_3 = sum(1 for c in merged if len(c["models_present"]) == 3)
    n_2 = sum(1 for c in merged if len(c["models_present"]) == 2)

    print(f"\nOutput: {len(merged)} canonical concepts")
    print(f"  5/5 models: {n_5}")
    print(f"  4/5 models: {n_4}")
    print(f"  3/5 models: {n_3}")
    print(f"  2/5 models: {n_2}")
    print()
    print("Canonical concepts:")
    for c in merged:
        n = len(c["models_present"])
        src_n = c["merged_from_n_clusters"]
        print(f"  [{n}/5]  {c['canonical_concept']:30s}  ({src_n} raw clusters)")

    output = {
        **{k: v for k, v in data.items() if k != "universal_concepts"},
        "universal_concepts":   merged,
        "n_universal_concepts": len(merged),
        "n_noise_excluded":     len(noise_found),
        "dedup_note": (
            f"Post-processed from {len(concepts)} raw HDBSCAN clusters → "
            f"{len(merged)} canonical concepts via hand-curated CANONICAL_MAP. "
            f"{len(noise_found)} noise clusters excluded (see NOISE_CLUSTER_IDS). "
            "See experiments.md § C2 Cluster Relabeling for full rationale."
        ),
    }

    with open(DST, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {DST}")


if __name__ == "__main__":
    main()
