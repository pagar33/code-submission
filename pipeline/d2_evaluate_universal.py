"""C4 – Three-tier universality evaluation.

Tier 1 – Geometric: RSA between model concept representations in d_concept space.
              Cross-model cosine similarity of concept centroids.
Tier 2 – Functional: Compare delta_eval across models for same concept.
              Effect size (Cohen's d) and direction agreement.
Tier 3 – Causal (optional, --tier3): Run actual steering intervention
              and measure sentiment/formality delta across models.
              Uses exact same eval pipeline as step8.

Pass criterion (binomial test): at least min_families concepts pass all tiers.

Outputs
-------
  results/universality_evidence_{run_id}.json
    {
      "concept_results": { concept_label: { tier1, tier2, tier3 } },
      "summary": {
        "n_concepts_tested": int,
        "n_passed_tier1": int,
        "n_passed_tier2": int,
        "n_passed_tier3": int,   # if tier3 run
        "n_passed_all": int,
        "binomial_p": float,
        "universal": bool,
      }
    }
"""


import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats

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
# Tier 1: Geometric
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float((a / na) @ (b / nb))


def _tier1_geometric(
    concept: dict,
    all_concepts: List[dict],
    min_cross_model_cos: float = 0.3,
    min_rsa_r: float = 0.2,
) -> dict:
    """Cross-model cosine similarity and RSA check."""
    centers = {
        c["label"]: np.array(c["center"], dtype=np.float32)
        for c in all_concepts if "center" in c
    }
    this_center = np.array(concept["center"], dtype=np.float32)
    models_present = concept.get("models_present", [])

    # Cross-model cosine similarity: compare this concept's center to all other concepts' centers
    # For universality we want this concept center to be distinctive and consistent.
    # Proxy: mean pairwise cosine similarity between per-model feature projections vs random baseline.
    per_model_data = concept.get("per_model", {})

    # Build per-model center estimates from selected feature activations
    model_centers: Dict[str, List[float]] = {}
    for m, data in per_model_data.items():
        acts = data.get("activations", [])
        if acts:
            # Use mean activation value as a scalar proxy — or use the global center
            model_centers[m] = concept["center"]

    if len(model_centers) < 2:
        return {
            "pass": False, "reason": "too_few_models",
            "cross_model_cos": None, "rsa_r": None,
        }

    # Pairwise cosine similarity across model center estimates
    model_list = list(model_centers.keys())
    cos_sims = []
    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            a = np.array(model_centers[model_list[i]])
            b = np.array(model_centers[model_list[j]])
            cos_sims.append(_cosine_sim(a, b))

    mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0

    # RSA: compare this concept center's similarity profile against all other concept centers
    all_centers = [c for lbl, c in centers.items() if lbl != concept["label"]]
    rsa_r: Optional[float] = None
    if len(all_centers) >= 4:
        this_sims = np.array([_cosine_sim(this_center, c) for c in all_centers])
        # Compare to random concept's similarity profile (use first other concept)
        other_sims = np.array([_cosine_sim(all_centers[0], c) for c in all_centers[1:]] + [0.0])
        other_sims = other_sims[:len(this_sims)]
        if this_sims.std() > 0 and other_sims.std() > 0:
            r, _ = scipy_stats.pearsonr(this_sims, other_sims)
            rsa_r = float(r)

    tier1_pass = mean_cos >= min_cross_model_cos
    return {
        "pass": tier1_pass,
        "cross_model_cos": mean_cos,
        "rsa_r": rsa_r,
        "min_cross_model_cos": min_cross_model_cos,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Functional
# ─────────────────────────────────────────────────────────────────────────────

def _tier2_functional(
    concept: dict,
    within_results: dict,
    universal_vectors: dict,
    min_models_positive: int = 2,
) -> dict:
    """Check that the concept drives a positive delta in eval scores across models."""
    label = concept["label"]
    if label not in universal_vectors:
        return {"pass": False, "reason": "no_universal_vectors"}

    # Look up native eval results per model for related base concepts
    # We approximate: check if within_results has any concept related to the cluster's domain
    per_model_delta: Dict[str, float] = {}
    models_vectors = universal_vectors[label]
    for m in models_vectors:
        # Try to find a related concept in within_results (e.g. 'sentiment')
        model_results = within_results.get(m, {})
        for concept_key in model_results:
            max_delta = model_results[concept_key].get("max_score_delta", 0.0)
            per_model_delta[m] = max_delta
            break  # Use first available concept's delta as proxy

    if not per_model_delta:
        return {"pass": False, "reason": "no_within_results"}

    positive_models = [m for m, d in per_model_delta.items() if d > 0]
    direction_agreement = len(positive_models) >= min_models_positive

    # Effect size proxy: mean delta across models
    deltas = list(per_model_delta.values())
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas)) if len(deltas) > 1 else 0.0

    return {
        "pass": direction_agreement,
        "per_model_delta": per_model_delta,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "n_positive_models": len(positive_models),
        "min_models_positive": min_models_positive,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 & 2: B2-reading versions (spec-correct thresholds)
# ─────────────────────────────────────────────────────────────────────────────

def _tier1_from_b2(
    domain: str,
    b2_by_domain: Dict[str, List[dict]],
    min_cos: float = 0.70,
) -> dict:
    """T1 geometric: mean procrustes_cosine_cca across B2 pairs >= 0.70."""
    pairs = b2_by_domain.get(domain, [])
    if not pairs:
        return {"pass": False, "reason": "no_b2_data", "procrustes_cosine_cca": None, "n_pairs": 0}
    cos_vals = [p["procrustes_cosine_cca"] for p in pairs
                if p.get("procrustes_cosine_cca") is not None]
    if not cos_vals:
        return {"pass": False, "reason": "missing_procrustes_cosine_cca",
                "procrustes_cosine_cca": None, "n_pairs": 0}
    mean_cos = float(np.mean(cos_vals))
    return {
        "pass": mean_cos >= min_cos,
        "procrustes_cosine_cca": mean_cos,
        "n_pairs": len(cos_vals),
        "min_cos": min_cos,
    }


def _tier2_from_b2(
    domain: str,
    b2_by_domain: Dict[str, List[dict]],
    min_rho_c: float = 0.60,
) -> dict:
    """T2 functional: mean rho_c_p90_clustered across B2 pairs > 0.60."""
    pairs = b2_by_domain.get(domain, [])
    if not pairs:
        return {"pass": False, "reason": "no_b2_data", "rho_c_p90_clustered": None, "n_pairs": 0}
    rho_vals = [
        p.get("rho_c_p90_clustered") if p.get("rho_c_p90_clustered") is not None
        else p.get("rho_c_p90")
        for p in pairs
    ]
    rho_vals = [v for v in rho_vals if v is not None]
    if not rho_vals:
        return {"pass": False, "reason": "missing_rho_c", "rho_c_p90_clustered": None, "n_pairs": 0}
    mean_rho = float(np.mean(rho_vals))
    cohen_vals = [p["cohen_d"] for p in pairs if p.get("cohen_d") is not None]
    mean_cohen_d = float(np.mean(cohen_vals)) if cohen_vals else None
    return {
        "pass": mean_rho > min_rho_c,
        "rho_c_p90_clustered": mean_rho,
        "cohen_d": mean_cohen_d,
        "n_pairs": len(rho_vals),
        "min_rho_c": min_rho_c,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3: Causal (optional, requires GPU + loaded models)
# ─────────────────────────────────────────────────────────────────────────────

def _tier3_causal(
    concept: dict,
    universal_vectors: dict,
    min_delta: float = 0.05,
) -> dict:
    """Run actual steering interventions on loaded models and measure delta."""
    label = concept["label"]
    if label not in universal_vectors:
        return {"pass": False, "reason": "no_universal_vectors"}

    from dotenv import load_dotenv
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    PROMPTS = [
        "Tell me about your day.",
        "Describe the weather.",
        "Explain how computers work.",
    ]

    def gen_texts(model, tokenizer, vec: np.ndarray, strength: float, layer, prompts):
        v = torch.tensor(vec, dtype=torch.float16, device=model.device).view(1, 1, -1)
        results_out = []
        for p in prompts:
            enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=256)
            ids = enc["input_ids"].to(model.device)
            mask = enc["attention_mask"].to(model.device)
            hook = None
            if strength != 0.0:
                def _hook(module, inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    rest = output[1:] if isinstance(output, tuple) else None
                    h = h + v * strength
                    return (h,) + rest if rest else h
                hook = layer.register_forward_hook(_hook)
            with torch.no_grad():
                out = model.generate(ids, attention_mask=mask, max_new_tokens=30,
                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)
            if hook:
                hook.remove()
            results_out.append(tokenizer.decode(out[0], skip_special_tokens=True))
        return results_out

    sentiment_clf = hf_pipeline(
        "text-classification", model="cardiffnlp/twitter-roberta-base-sentiment",
        device=0, token=hf_token,
    )

    def sent_score(texts):
        scores = []
        for t in texts:
            out = sentiment_clf(t, return_all_scores=True)[0]
            pos = 0.0
            for r in (out if isinstance(out, list) else [out]):
                if r.get("label", "").endswith("2") or r.get("label", "").lower() == "positive":
                    pos = r.get("score", 0.0)
            scores.append(pos)
        return float(np.mean(scores)) if scores else 0.0

    per_model_delta: Dict[str, float] = {}
    models_vectors = universal_vectors[label]
    for m, mdata in models_vectors.items():
        cfg = config.MODELS.get(m)
        if cfg is None:
            continue
        vec = np.array(mdata["steering_vector"], dtype=np.float32)
        scale = np.linalg.norm(vec)
        strength = 1.0  # normalised vector

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(cfg["hf_name"], use_fast=True, token=hf_token)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg["hf_name"], torch_dtype=dtype, device_map=device_map, token=hf_token
        )
        model.eval()

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layer = model.model.layers[cfg["target_layer"]]
        else:
            layer = model.transformer.h[cfg["target_layer"]]

        base_texts = gen_texts(model, tok, vec, 0.0, layer, PROMPTS)
        steered_texts = gen_texts(model, tok, vec, strength, layer, PROMPTS)
        delta = sent_score(steered_texts) - sent_score(base_texts)
        per_model_delta[m] = float(delta)

        del model
        torch.cuda.empty_cache()

    positive = [m for m, d in per_model_delta.items() if d >= min_delta]
    passed = len(positive) >= 2

    return {
        "pass": passed,
        "per_model_delta": per_model_delta,
        "n_positive_models": len(positive),
        "min_delta": min_delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Three-tier universality evaluation (C4)")
    parser.add_argument("--concepts-path", default="",
                        help="Path to universal_concepts.json from C2")
    parser.add_argument("--vectors-path", default="",
                        help="Path to universal_steering_vectors.json from C3")
    parser.add_argument("--within-path", default="",
                        help="Path to within_model_steering.json from step8")
    parser.add_argument("--concepts", nargs="*", default=[],
                        help="Restrict to specific concept labels (default: all)")
    parser.add_argument("--min-families", type=int, default=2,
                        help="Minimum passing concepts for binomial test (default: 2)")
    parser.add_argument("--tier3", action="store_true",
                        help="Run tier-3 causal evaluation (loads models, needs GPU)")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(config.UNIVERSAL_DIR, exist_ok=True)

    suffix = f"_{args.run_id}" if args.run_id else ""
    out_path = os.path.join(config.UNIVERSAL_DIR, f"universality_evidence{suffix}.json")
    if os.path.exists(out_path) and not args.force:
        print(f"{os.path.basename(out_path)} exists. Use --force to recompute.")
        log_run("universality_eval.py", start, "skipped")
        return 0

    # Load inputs
    concepts_path = args.concepts_path or os.path.join(config.UNIVERSAL_DIR, "universal_concepts.json")
    vectors_path = args.vectors_path or os.path.join(config.UNIVERSAL_DIR, "universal_steering_vectors.json")
    within_path = args.within_path or os.path.join(config.RESULTS_DIR, "within_model_steering.json")

    for p in [concepts_path, vectors_path, within_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    with open(concepts_path) as f:
        concepts_data = json.load(f)
    with open(vectors_path) as f:
        vectors_data = json.load(f)
    with open(within_path) as f:
        within_results = json.load(f)

    all_concepts: List[dict] = concepts_data["universal_concepts"]
    universal_vectors: dict = vectors_data.get("universal_steering_vectors", {})

    # Load B2 validation results — required for T1 (procrustes_cosine_cca) and T2 (rho_c_p90_clustered)
    val_results_path = os.path.join(config.ALIGNMENT_DIR, "validation_results.jsonl")
    b2_by_domain: Dict[str, List[dict]] = defaultdict(list)
    if os.path.exists(val_results_path):
        with open(val_results_path) as vf:
            for line in vf:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    dom = rec.get("domain", "")
                    if dom:
                        b2_by_domain[dom].append(rec)
        print(f"[C4] Loaded B2: {sum(len(v) for v in b2_by_domain.values())} pairs across {len(b2_by_domain)} domains")
    else:
        print(f"[C4] WARNING: B2 validation results not found at {val_results_path} — T1/T2 will report no_b2_data")

    # Filter to requested concepts
    if args.concepts:
        all_concepts = [c for c in all_concepts if c["label"] in args.concepts]

    print(f"[C4] Evaluating {len(all_concepts)} universal concepts")

    concept_results: Dict[str, dict] = {}
    n_t1, n_t2, n_t3, n_all = 0, 0, 0, 0

    for concept in all_concepts:
        label = concept["label"]
        t1 = _tier1_from_b2(label, b2_by_domain)
        t2 = _tier2_from_b2(label, b2_by_domain)
        t3 = {"pass": None, "skipped": True}
        if args.tier3:
            t3 = _tier3_causal(concept, universal_vectors)

        all_pass = t1["pass"] and t2["pass"] and (not args.tier3 or t3.get("pass", False))

        if t1["pass"]:
            n_t1 += 1
        if t2["pass"]:
            n_t2 += 1
        if args.tier3 and t3.get("pass"):
            n_t3 += 1
        if all_pass:
            n_all += 1

        concept_results[label] = {"tier1": t1, "tier2": t2, "tier3": t3, "pass_all": all_pass}
        print(f"  {label}: T1={t1['pass']}, T2={t2['pass']}, T3={t3.get('pass', 'skipped')}")

    # Concept-family grouping (1.2 correction: concepts in the same family are not independent)
    CONCEPT_FAMILIES = {
        "sentiment": "sentiment-family",
        "tone": "sentiment-family",
        "valence": "sentiment-family",
        "empathy": "sentiment-family",
        "caution": "caution-family",
        "hedging": "caution-family",
        "uncertainty": "caution-family",
        "refusal": "caution-family",
        "reasoning": "reasoning-family",
        "mathematical": "reasoning-family",
        "step-by-step": "reasoning-family",
        "chain-of-thought": "reasoning-family",
    }

    def _get_family(concept_label: str) -> str:
        """Return the concept family name, or the concept itself if unknown."""
        lbl = concept_label.lower()
        for keyword, family in CONCEPT_FAMILIES.items():
            if keyword in lbl:
                return family
        return f"solo:{concept_label}"

    # Group concepts by family — count one "family pass" per family (majority vote)
    family_results: dict = {}
    for label, res in concept_results.items():
        family = _get_family(label)
        if family not in family_results:
            family_results[family] = {"passes": [], "labels": []}
        family_results[family]["passes"].append(res["pass_all"])
        family_results[family]["labels"].append(label)

    n_family_pass = sum(
        1 for fd in family_results.values() if sum(fd["passes"]) > len(fd["passes"]) / 2
    )
    n_families = len(family_results)

    # Binomial test on concept FAMILIES — H₀: ≤70% pass by chance, p=0.7 reflects informed prior
    from scipy.stats import binomtest as _binomtest
    binom_result = _binomtest(n_family_pass, n_families, p=0.7, alternative="greater").pvalue if n_families > 0 else 1.0

    # Power warning: n_concepts < 15 makes the binomial test underpowered
    power_warning = len(all_concepts) < 15
    if power_warning:
        print(f"[C4] WARNING: n_concepts={len(all_concepts)} < 15 — binomial test is underpowered")

    # Also keep raw concept count for reference
    n_total = len(all_concepts)

    universal = n_family_pass >= args.min_families and float(binom_result) < 0.05

    summary = {
        "n_concepts_tested": n_total,
        "n_passed_tier1": n_t1,
        "n_passed_tier2": n_t2,
        "n_passed_tier3": n_t3 if args.tier3 else None,
        "n_passed_all_concepts": n_all,
        "n_families_tested": n_families,
        "n_families_passed": n_family_pass,
        "family_groupings": {f: fd["labels"] for f, fd in family_results.items()},
        "min_families": args.min_families,
        "binomial_p_family_corrected": float(binom_result),
        "universal": universal,
        "power_warning": power_warning,
        "power_warning_msg": (
            f"n_concepts={len(all_concepts)} < 15 — binomial test underpowered"
            if power_warning else None
        ),
    }

    output = {
        "concept_results": concept_results,
        "summary": summary,
        "run_id": args.run_id,
        "tier3_run": args.tier3,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[C4] Saved {out_path}")
    print(f"[C4] Summary: {n_family_pass}/{n_families} families pass all tiers (concepts: {n_all}/{n_total}), binom_p={binom_result:.4f}, universal={universal}")

    log_run("universality_eval.py", start, "success")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log_run("universality_eval.py", time.time(), "error", str(e))
        raise
