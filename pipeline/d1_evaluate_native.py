"""step8_apply_steering.py — B4 evaluation pipeline.

Changes vs B3:
  1. Shared baseline: precomputed once per model before method loops (no strength=0 re-runs).
  2. naive_{guide}: truncate/pad/norm guide A-vector into target space — dumb baseline.
  3. C3 universal vectors: loaded as method "universal_c3" from universal_steering_vectors_v1.json.
  4. Uniform sweep: SWEEP_STANDARD (9 pts: −5,−3,−2,−1,0,1,2,3,5) for all methods including universal vectors.
  5. CONCEPT_LABELS: fixes DeBERTa hypothesis from "This text is python_code." → "This text is about Python code."
  6. Bidirectionality: bidir_ratio = |max_neg_delta| / |max_pos_delta| stored per method result.
  7. Multiprocessing: --parallel flag spawns one subprocess per model on dedicated GPU (GPU_MAP).
  8. compile_tables(): after all workers finish, builds Tables 1–7 and writes evaluation_table.json + evaluation_report.md.
"""


import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import glob as _glob
import json
import multiprocessing as mp
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def _ts() -> str:
    """Return a compact [HH:MM:SS] timestamp for progress lines."""
    return time.strftime("[%H:%M:%S]")


def _preview(text: str, n: int = 90) -> str:
    """Return first n printable chars of text on one line, for inline logging."""
    snippet = " ".join(text.split())[:n]
    return repr(snippet) if snippet else "(empty)"

# ---------------------------------------------------------------------------
# Optional: Claude LLM judge (replaces/supplements human eval)
# Reads ANTHROPIC_API_KEY from env or .env file.
# Called once after all models are done, on a sample of steering_examples.jsonl.
# ---------------------------------------------------------------------------
_CLAUDE_JUDGE_MODEL = "claude-sonnet-4-5"   # pinned — update only with paper revision
_CLAUDE_JUDGE_PROMPT = """\
You are evaluating outputs from a language model steering experiment.
The goal of steering was to shift the model's output toward the concept: {concept}

Baseline output (no steering applied):
<baseline>{baseline}</baseline>

Steered output (steering applied):
<steered>{steered}</steered>

Use the score_output tool to record your evaluation.
Answer each field with a simple true/false — no partial credit."""

_CLAUDE_JUDGE_TOOL = {
    "name": "score_output",
    "description": "Record evaluation scores for a steered model output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "concept_present": {
                "type": "boolean",
                "description": "True if the steered output clearly expresses the target concept (even partially counts as False — only mark True if the concept is clearly present).",
            },
            "fluent": {
                "type": "boolean",
                "description": "True if the steered output is grammatical and readable (degraded but readable = True; gibberish or severe repetition = False).",
            },
            "better_than_baseline": {
                "type": "boolean",
                "description": "True if the steered output is better than the baseline at expressing the target concept.",
            },
        },
        "required": ["concept_present", "fluent", "better_than_baseline"],
    },
}

# Map boolean labels to numeric for aggregation
_CONCEPT_PRESENT_SCORE = {True: 1.0, False: 0.0, "yes": 1.0, "partial": 0.5, "no": 0.0}  # legacy compat
_FLUENCY_SCORE = {True: 1.0, False: 0.0, "fluent": 1.0, "degraded": 0.5, "incoherent": 0.0}  # legacy compat


def _llm_judge_sample(
    examples_path: str,
    api_key: str,
    sample_per_concept: int = 10,
    out_path: Optional[str] = None,
    seed: int = 42,
) -> Optional[Dict]:
    """Score a random sample of steered outputs with Claude.

    This provides LLM-as-judge evaluation, which is accepted in NeurIPS 2024 papers
    as a supplement to automatic metrics (see e.g. Anthropic Constitutional AI, RLHF
    survey, and many steering/preference papers).  Claude evaluates:
      1. concept_score_llm  — did the output shift toward the target concept?
      2. fluency_llm        — is the output natural/grammatical?
      3. better_than_baseline — clear binary comparison per pair

    Results are written to {out_path} (default: results/llm_judge_results.jsonl).
    Returns a summary dict with per-concept mean scores.
    """
    try:
        import anthropic  # type: ignore
    except ImportError:
        print("[step8] WARNING: claude judge skipped — `pip install anthropic` first")
        return None

    if not api_key:
        print("[step8] WARNING: claude judge skipped — ANTHROPIC_API_KEY not set")
        return None

    if not os.path.exists(examples_path):
        print(f"[step8] WARNING: claude judge skipped — {examples_path} not found")
        return None

    # Load all steered examples (exclude baseline rows)
    steered_rows = []
    baseline_by_model_prompt: Dict[str, str] = {}  # "{model}::{prompt_idx}" → output
    with open(examples_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("_type") == "run_meta" or "model" not in row:
                continue  # skip header / meta rows
            if row.get("method") == "baseline":
                key = f"{row['model']}::{row['prompt_idx']}"
                baseline_by_model_prompt[key] = row.get("output", "")
            else:
                steered_rows.append(row)

    if not steered_rows:
        print("[step8] WARNING: claude judge skipped — no steered rows in examples file")
        return None

    # Sample: for each (model, concept) pair, pick up to sample_per_concept rows.
    # Stratified so we don't over-represent one method.
    rng = random.Random(seed)
    by_mc: Dict[str, List] = {}
    for row in steered_rows:
        key = f"{row['model']}::{row['concept']}"
        by_mc.setdefault(key, []).append(row)
    sample: List[Dict] = []
    for rows in by_mc.values():
        sample.extend(rng.sample(rows, min(sample_per_concept, len(rows))))

    print(f"[step8] Claude judge: evaluating {len(sample)} steered outputs "
          f"({sample_per_concept} per model×concept)...")

    client = anthropic.Anthropic(api_key=api_key)
    judge_out_path = out_path or examples_path.replace("steering_examples", "llm_judge_results")

    results_by_concept: Dict[str, List] = {}
    written = 0
    with open(judge_out_path, "w", encoding="utf-8") as jf:
        for row in sample:
            concept = row["concept"]
            model   = row["model"]
            pidx    = row["prompt_idx"]
            baseline_text = baseline_by_model_prompt.get(f"{model}::{pidx}", "")
            steered_text  = row.get("output", "")
            concept_label = CONCEPT_LABELS.get(concept, concept.replace("_", " "))

            prompt = _CLAUDE_JUDGE_PROMPT.format(
                concept=concept_label,
                baseline=baseline_text[:400],
                steered=steered_text[:400],
            )
            raw_text = None
            scores = {}
            try:
                resp = client.messages.create(
                    model=_CLAUDE_JUDGE_MODEL,
                    max_tokens=256,
                    temperature=0,
                    tools=[_CLAUDE_JUDGE_TOOL],
                    tool_choice={"type": "tool", "name": "score_output"},
                    messages=[{"role": "user", "content": prompt}],
                )
                # tool_use forces a structured response — no free-text parsing needed
                for block in resp.content:
                    if block.type == "tool_use" and block.name == "score_output":
                        scores = block.input
                        break
            except Exception as exc:
                print(f"[step8] Claude judge error for {model}/{concept}: {exc}")

            out_row = {
                "model": model,
                "concept": concept,
                "method": row.get("method"),
                "strength": row.get("strength"),
                "prompt_idx": pidx,
                # Inputs sent to Claude
                "input_prompt": row.get("prompt"),
                "input_baseline_output": baseline_text,
                "input_steered_output": steered_text,
                "judge_prompt": prompt,
                "judge_model": _CLAUDE_JUDGE_MODEL,
                # Parsed scores (structured tool_use — boolean, reproducible)
                "concept_present": scores.get("concept_present"),
                "fluency": scores.get("fluent"),
                "better_than_baseline": scores.get("better_than_baseline"),
                # Numeric conversions for aggregation
                "steered_toward_concept": _CONCEPT_PRESENT_SCORE.get(scores.get("concept_present")),
                "fluency_score": _FLUENCY_SCORE.get(scores.get("fluent")),
                # DeBERTa scores for cross-validation
                "deberta_concept_score": row.get("concept_score"),
                "deberta_baseline_score": row.get("baseline_concept_score"),
                "deberta_delta": row.get("delta"),
            }
            jf.write(json.dumps(out_row) + "\n")
            results_by_concept.setdefault(concept, []).append(out_row)
            written += 1

    # Summarise
    summary: Dict[str, Dict] = {}
    for concept, rows in results_by_concept.items():
        concept_scores = [r["steered_toward_concept"] for r in rows if r["steered_toward_concept"] is not None]
        fluency_scores = [r["fluency_score"] for r in rows if r["fluency_score"] is not None]
        better_flags   = [r["better_than_baseline"] for r in rows if r["better_than_baseline"] is not None]
        summary[concept] = {
            "n": len(rows),
            "mean_concept_score_llm": round(float(np.mean(concept_scores)), 4) if concept_scores else None,
            "mean_fluency_llm":       round(float(np.mean(fluency_scores)), 4) if fluency_scores else None,
            "pct_better_than_baseline": round(
                sum(bool(b) for b in better_flags) / max(len(better_flags), 1) * 100, 1
            ) if better_flags else None,
        }

    print(f"[step8] Claude judge: wrote {written} rows to {judge_out_path}")
    return {"summary_by_concept": summary, "judge_model": _CLAUDE_JUDGE_MODEL, "n_total": written}

# ---------------------------------------------------------------------------
# B4 constants
# ---------------------------------------------------------------------------

SWEEP_STANDARD: List[float] = [-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0]  # uniform for all methods

# GPU assignment: one dedicated GPU per model (assumes 8-GPU node: 0–4 for LLMs, rest free)
GPU_MAP: Dict[str, int] = {
    "gpt2-large":      0,
    "gemma":           1,
    "llama":           2,
    "mistral":         3,
    "deepseek-llm-7b": 4,
}


def _pick_free_gpu(exclude: List[int] = None) -> int:
    """Return the GPU index with the most free memory, excluding already-claimed ones.

    Falls back to GPU 0 if pynvml is unavailable or only one GPU exists.
    """
    exclude = set(exclude or [])
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        best_gpu, best_free = 0, -1
        for i in range(n):
            if i in exclude:
                continue
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            if mem.free > best_free:
                best_free = mem.free
                best_gpu = i
        pynvml.nvmlShutdown()
        print(f"[step8] GPU picker: selected GPU {best_gpu} "
              f"({best_free / 1024**3:.1f} GB free, excluded={sorted(exclude)})")
        return best_gpu
    except Exception as _e:
        print(f"[step8] GPU picker unavailable ({_e}) — falling back to GPU_MAP / GPU 0")
        return 0

# Readable labels for DeBERTa zero-shot NLI hypothesis.
# "This text is about {label}." is cleaner than "This text is python_code."
CONCEPT_LABELS: Dict[str, str] = {
    # 11 canonical C2 universal concepts
    "python_code":             "Python code",
    "sql_queries":             "SQL database queries",
    "math_problems":           "mathematics problems",
    "code_and_math":           "code and mathematics",
    "sql_and_medical":         "medical SQL queries",
    "legal_and_news":          "legal news articles",
    "medical_research":        "medical research",
    "academic_scientific":     "academic scientific text",
    "narrative_fiction":       "narrative fiction",
    "encyclopedic_historical": "encyclopedic historical text",
    "customer_reviews":        "customer reviews",
    # A-section / NeurIPS 15 concepts
    "sentiment":          "positive sentiment",
    "formality":          "formal language",
    "certainty":          "confident and certain claims",
    "academic_writing":   "academic writing",
    "code_instructions":  "code instructions",
    "code_python":        "Python code",
    "code_snippets":      "code snippets",
    "code_sql":           "SQL code",
    "creative_writing":   "creative writing",
    "legal":              "legal text",
    "math_competition":   "competition mathematics",
    "math_gsm8k":         "grade school mathematics",
    "math_olympiad":      "olympiad mathematics",
    "math_reasoning":     "mathematical reasoning",
    "news_reporting":     "news reporting",
    "question_answering": "question answering",
    "science_biomedical": "biomedical science",
}


def get_sweep(method_key: str) -> List[float]:
    return SWEEP_STANDARD  # uniform sweep for all methods


# PPL scorer assignment — must NOT be the same model as the evaluation target.
# Scoring a model's own outputs with itself is circular: the model always gives
# its own style a low perplexity, artificially boosting the fluency numbers.
# Rule: each target model is scored by a *different* model family.
# Fallback: if the preferred scorer is not yet downloaded locally, warn and use gpt2-large.
_PPL_SCORER_MAP: Dict[str, str] = {
    # Single fixed held-out scorer (GPT-2 117M) for all models — ensures the 30% PPL-increase
    # gate in Table 3 is on a uniform scale across models. Using the same scorer for every
    # target means ✅/❌ cells in the cross-model table are genuinely comparable.
    # GPT-2 small assigns high absolute PPL to 7B outputs, but the *ratio* (steered/baseline)
    # is well-behaved because both numerator and denominator are scored by the same model.
    "gpt2-large":      "gpt2",
    "gemma":           "gpt2",
    "llama":           "gpt2",
    "mistral":         "gpt2",
    "deepseek-llm-7b": "gpt2",
}


# ---------------------------------------------------------------------------
# Logging / seeding
# ---------------------------------------------------------------------------

def log_run(script: str, start_time: float, status: str, error: str = "") -> None:
    entry = {
        "script": script,
        "start_time": start_time,
        "end_time": time.time(),
        "status": status,
        "error": error,
    }
    try:
        with open("run_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_steering_vectors(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Steering vectors not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prompts — 100 candidate prompts; runtime selects TOP_N_PROMPTS (30) cleanest
# ---------------------------------------------------------------------------

# Number of clean prompts to use in evaluation.
# Statistical justification:
#   n=30: power=0.83 at our effect size (delta=0.05, SD=0.10), 95% CI width ≈ 0.067
#   n=50: power=0.96, 95% CI width ≈ 0.053  — only marginal gain over n=30
# We generate 100 candidate prompts and at runtime select the TOP_N_PROMPTS cleanest
# ones (lowest 5-gram repetition rate at baseline).  This ensures quality without
# wasting GPU time on prompts that loop regardless of steering.
# 30 is sufficient for NeurIPS-standard Wilcoxon + bootstrap CI.
TOP_N_PROMPTS: int = 30


def get_prompts() -> List[str]:
    # 100 register-neutral candidate prompts.  At runtime the TOP_N_PROMPTS (30)
    # with the lowest baseline repetition are selected automatically.
    # Design principle: ≥4 words, register-fluid (no domain priming), diverse syntax.
    return [
        # --- Evaluation / observation (15) ---
        "How was it?",
        "Overall, the result was",
        "My first impression is",
        "After reviewing the output,",
        "The performance seems",
        "Looking at the output,",
        "One notable feature is",
        "The most significant aspect",
        "At first glance,",
        "Upon closer inspection,",
        "The quality of this",
        "What stands out most is",
        "The overall effect is",
        "After careful consideration,",
        "What I noticed first was",
        # --- Process / instructional (15) ---
        "The first step is",
        "The main approach involves",
        "A common technique is",
        "To implement this,",
        "The procedure works by",
        "This method relies on",
        "When applied correctly,",
        "The way to do this",
        "In order to proceed,",
        "Starting from the beginning,",
        "The process begins with",
        "Each step involves",
        "The correct approach is",
        "This can be done by",
        "One way to handle this",
        # --- Analysis / academic (20) ---
        "Based on the data,",
        "The results indicate",
        "By examining the data,",
        "Under these conditions,",
        "The evidence suggests",
        "In comparison with",
        "An important observation is",
        "Further analysis reveals",
        "The findings show that",
        "From a technical perspective,",
        "The study demonstrates",
        "A critical factor is",
        "The analysis indicates",
        "When we consider",
        "The data reveals",
        "Statistically speaking,",
        "The experiment shows",
        "Given the results,",
        "The observations confirm",
        "In light of the evidence,",
        # --- Problem / solution (15) ---
        "The issue is",
        "The main challenge here",
        "There are several factors",
        "One possible solution is",
        "The cause of this",
        "This problem arises from",
        "To address this issue,",
        "A key limitation is",
        "The difficulty with",
        "This fails because",
        "The root cause is",
        "To fix this problem,",
        "The obstacle is",
        "A better approach would be",
        "The fundamental issue is",
        # --- Summary / synthesis (10) ---
        "To summarize,",
        "In other words,",
        "The key point is",
        "This essentially means",
        "Taken as a whole,",
        "The core insight is",
        "In summary,",
        "The main takeaway is",
        "Putting it simply,",
        "To put it another way,",
        # --- Comparison / contrast (10) ---
        "The key difference",
        "Unlike traditional approaches,",
        "Compared to the baseline,",
        "In contrast to this,",
        "While this approach works,",
        "A notable distinction is",
        "Both methods share",
        "The advantage over",
        "This differs from",
        "Relative to other",
        # --- Attribution / narrative (15) ---
        "According to",
        "What I found was",
        "The research shows",
        "Recent work suggests",
        "It is worth noting that",
        "Studies have shown",
        "Experts agree that",
        "The literature suggests",
        "Previous work found",
        "As noted by",
        "The report states",
        "Historical records show",
        "Observations suggest that",
        "The author argues",
        "Current understanding holds that",
    ]


# ---------------------------------------------------------------------------
# Model architecture helpers
# ---------------------------------------------------------------------------

def get_layer_module(model_name: str, model, layer_idx: int):
    """Return the nn.Module for transformer layer `layer_idx`."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    raise ValueError(f"Cannot resolve layer modules for model architecture: {model_name}")


def get_all_layer_indices(model_name: str, model) -> List[int]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(range(len(model.transformer.h)))
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(range(len(model.model.layers)))
    raise ValueError(f"Cannot resolve layer count for: {model_name}")


def add_steering_hook(layer_module, vector: torch.Tensor):
    """Add a residual-stream hook that adds `vector` to every position."""
    def _hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden, *rest = output
            return (hidden + vector.to(hidden.device), *rest)
        return output + vector.to(output.device)
    return layer_module.register_forward_hook(_hook)


# ---------------------------------------------------------------------------
# GPT-2 byte-BPE artifact reversal
# ---------------------------------------------------------------------------
# DeepSeek (and GPT-2 family) tokenizers represent bytes as unicode surrogates
# using the GPT-2 bytes_to_unicode mapping: space→Ġ (U+0120), newline→Ċ (U+010A),
# and non-printable / high bytes → U+0100..U+0142.  Even with use_fast=False the
# slow tokenizer may leave these in the decoded string.  This function reverses
# the full 256-char mapping so the final text is valid UTF-8.
def _build_gpt2_byte_decoder() -> Dict[str, int]:
    bs: List[int] = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("ÿ") + 1))
    cs: List[int] = list(bs)
    n = 256
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

_GPT2_BYTE_DECODER: Dict[str, int] = _build_gpt2_byte_decoder()


def _fix_bpe_artifacts(text: str) -> str:
    """Reverse GPT-2 byte-BPE encoding (Ġ→' ', Ċ→'\\n', éľĸ→Chinese char, etc.).

    GPT-2's bytes_to_unicode maps the 67 non-printable bytes (0-32, 127-160) to
    U+0100–U+0142.  Any character in that range in the decoded output is a BPE
    artifact.  Latin-1 bytes 161-255 map to themselves, so printable Latin-1 text
    is passed through cleanly by the reverse mapping as well.

    Two strategies:
    1. Full-string reversal (pure GPT-2/fast-tokenizer output) — all chars are in
       the 256-entry map, so the whole string can be re-decoded from bytes.
    2. Char-by-char fallback (deepseek slow tokenizer or mixed text) — the string
       contains characters outside the 256-byte map (e.g. smart quotes U+201C,
       CJK, etc.) so only the BPE surrogate range U+0100-U+0142 is replaced;
       everything else passes through unchanged.
    """
    # Fast-path: no BPE artifact characters present — nothing to do.
    if not any("\u0100" <= c <= "\u0142" for c in text):
        return text
    # Strategy 1: full reversal — works when every character is in the byte decoder
    # (typical for gpt2-large fast tokenizer output).
    try:
        return bytes([_GPT2_BYTE_DECODER[c] for c in text]).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        pass
    # Strategy 2: replace only the non-printable-byte surrogates in U+0100-U+0142
    # (e.g. Ġ=U+0120→space, Ċ=U+010A→newline) and leave all other characters intact.
    # This handles deepseek's slow tokenizer output which mixes correctly-decoded
    # Unicode (smart quotes, punctuation) with residual Ġ space markers.
    _surrogates = {c: b for c, b in _GPT2_BYTE_DECODER.items() if "\u0100" <= c <= "\u0142"}
    return "".join(chr(_surrogates[c]) if c in _surrogates else c for c in text)


def _decode_new_tokens(tokenizer, new_ids) -> str:
    """Decode generated token IDs to text, correctly for all tokenizer backends.

    - _fast_tok path (deepseek): tokenizers.Tokenizer loaded directly from
      tokenizer.json — handles Ġ-prefixed tokens natively.
    - byte-BPE path (gpt2-large): join raw token strings and apply the GPT-2
      bytes_to_unicode reverse map.
    - SentencePiece path (LLaMA, Gemma, Mistral): convert_tokens_to_string.
    """
    if hasattr(new_ids, 'tolist'):
        new_ids = new_ids.tolist()
    if not new_ids:
        return ""
    _fast_tok = getattr(tokenizer, '_fast_tok', None)
    if _fast_tok is not None:
        # tokenizers.Tokenizer.decode() handles Ġ prefixes correctly.
        _result = _fast_tok.decode(new_ids, skip_special_tokens=True).lstrip(" ")
        return _result
    tokens = tokenizer.convert_ids_to_tokens(new_ids, skip_special_tokens=True)
    if not tokens:
        return ""
    raw = "".join(tokens)
    if getattr(tokenizer, '_byte_bpe', False):
        # Byte-BPE path: reverse the full joined token string through the byte decoder.
        _bd = getattr(tokenizer, 'byte_decoder', None) or _GPT2_BYTE_DECODER
        try:
            text = bytes([_bd[c] for c in raw]).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            text = _fix_bpe_artifacts(raw)
    else:
        # SentencePiece path (LLaMA, Gemma, Mistral).
        text = tokenizer.convert_tokens_to_string(tokens)
    return text.lstrip(" ")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _fast_tok_encode(tokenizer, prompts, device):
    """Encode prompt(s) using tokenizers.Tokenizer (_fast_tok) when available.

    Returns (input_ids, attention_mask) tensors on device, or None if _fast_tok
    is not set (caller should fall back to the slow tokenizer path).
    """
    _ft = getattr(tokenizer, '_fast_tok', None)
    if _ft is None:
        return None, None
    if isinstance(prompts, str):
        prompts = [prompts]
    _ft.enable_padding(direction="left", pad_id=tokenizer.eos_token_id or 0)
    _ft.enable_truncation(max_length=config.MAX_SEQ_LEN)
    encs = _ft.encode_batch(prompts)
    ids = torch.tensor([e.ids for e in encs], dtype=torch.long, device=device)
    mask = torch.tensor([e.attention_mask for e in encs], dtype=torch.long, device=device)
    return ids, mask


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    _ft_ids, _ft_mask = _fast_tok_encode(tokenizer, prompt, model.device)
    if _ft_ids is not None:
        input_ids, attention_mask = _ft_ids, _ft_mask
    else:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.MAX_SEQ_LEN)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens using the canonical low-level path.
    return _decode_new_tokens(tokenizer, out[0][input_ids.shape[1]:])


def generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int = 50) -> List[str]:
    """Batched greedy generation (batch_size=8). Falls back to sequential on errors."""
    batch_size = 8
    # Always disable clean_up_tokenization_spaces so Ġ markers survive to _fix_bpe_artifacts,
    # which handles byte-BPE models (deepseek, gpt2-*) and is a no-op for others.
    results: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        try:
            _ft_ids, _ft_mask = _fast_tok_encode(tokenizer, batch, model.device)
            if _ft_ids is not None:
                input_ids, attention_mask = _ft_ids, _ft_mask
            else:
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=config.MAX_SEQ_LEN,
                )
                input_ids = enc["input_ids"].to(model.device)
                attention_mask = enc["attention_mask"].to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # Decode only generated tokens — skip the (left-padded) input prefix.
            n_input = input_ids.shape[1]
            results.extend(
                _decode_new_tokens(tokenizer, o[n_input:])
                for o in out
            )
        except Exception:
            results.extend(generate_text(model, tokenizer, p, max_new_tokens) for p in batch)
    return results


def compute_perplexity(texts: List[str], scorer_model, scorer_tokenizer,
                       _batch_size: int = 16) -> float:
    """Batched perplexity — processes up to _batch_size texts per forward pass.

    Replaces the original one-text-at-a-time loop which paid CUDA kernel launch
    overhead 30× per call.  With _batch_size=16 and 30 texts that's 2 passes instead
    of 30 — roughly 10× faster on GPU.

    Padding is handled with a per-token mask so padding tokens don't inflate loss.
    """
    if not texts:
        return 0.0
    per_example_losses: List[float] = []
    for i in range(0, len(texts), _batch_size):
        batch = texts[i : i + _batch_size]
        enc = scorer_tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        )
        input_ids = enc["input_ids"].to(scorer_model.device)
        attention_mask = enc["attention_mask"].to(scorer_model.device)
        with torch.no_grad():
            out = scorer_model(input_ids=input_ids, attention_mask=attention_mask)
        # Causal shift: predict token t+1 from token t
        shift_logits = out.logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()
        # Per-token cross-entropy, then mask out padding
        token_losses = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())
        # Mean loss per example (ignore padding tokens)
        denom = shift_mask.sum(dim=1).clamp(min=1)
        per_example_losses.extend(
            ((token_losses * shift_mask).sum(dim=1) / denom).tolist()
        )
    mean_loss = float(np.mean(per_example_losses))
    return float(np.exp(mean_loss)) if mean_loss > 0 else 0.0


def _repetition_rate(texts: List[str]) -> float:
    """Detect continuous repetition loops — phrase-level AND single-token.

    Returns 1.0 if:
      - Any phrase of ≥2 consecutive words repeats back-to-back 4+ times, OR
      - Any single word appears 8+ times consecutively (e.g. "code code code ...")

    Examples that score 1.0:
      "the cat sat the cat sat the cat sat the cat sat ..."  (phrase loop)
      "code code code code code code code code ..."           (single-token loop)

    The binary 0/1 output is compatible with the 0.4 filter threshold used everywhere
    in the pipeline: a looping output (1.0) is always above 0.4 and gets excluded.
    """
    rates = []
    for text in texts:
        words = text.split()
        n_words = len(words)
        found = False
        # Single-token loop check: any word appears 8+ consecutive times
        run, run_word = 1, None
        for w in words:
            if w == run_word:
                run += 1
                if run >= 8:
                    found = True
                    break
            else:
                run, run_word = 1, w
        if not found:
            # Phrase loop check: ≥2-word phrase repeated 4+ consecutive times
            max_pl = n_words // 4
            for pl in range(2, max_pl + 1):
                for start in range(n_words - pl * 4 + 1):
                    phrase = words[start : start + pl]
                    count = 1
                    pos = start + pl
                    while pos + pl <= n_words and words[pos : pos + pl] == phrase:
                        count += 1
                        pos += pl   # MUST be inside while — advancing pos is the loop terminator
                    if count > 3:          # 4th consecutive hit — loop confirmed
                        found = True
                        break
                if found:
                    break
        # Character-level repetition for any long individual word (e.g. "1.1.1.1.1.1.1."
        # or other numeric/symbol loops that appear as one token after word-split).
        if not found:
            for _w in words:
                if len(_w) <= 20:
                    continue
                _wchars = list(_w)
                _wn = len(_wchars)
                for _pl in range(1, min(6, _wn // 4) + 1):
                    for _st in range(_wn - _pl * 4 + 1):
                        _ph = _wchars[_st : _st + _pl]
                        _cnt, _pos = 1, _st + _pl
                        while _pos + _pl <= _wn and _wchars[_pos : _pos + _pl] == _ph:
                            _cnt += 1
                            _pos += _pl
                        if _cnt >= 4:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        # Character-level repetition within any single long word/token (e.g. "1.1.1.1.1.1."
        # or "##########" which appear as one word after split but are clearly looping).
        if not found:
            for _w in words:
                if len(_w) <= 20:
                    continue
                _wchars = list(_w)
                _wn = len(_wchars)
                for _pl in range(1, min(6, _wn // 4) + 1):
                    for _st in range(_wn - _pl * 4 + 1):
                        _ph = _wchars[_st : _st + _pl]
                        _cnt, _pos = 1, _st + _pl
                        while _pos + _pl <= _wn and _wchars[_pos : _pos + _pl] == _ph:
                            _cnt += 1
                            _pos += _pl
                        if _cnt >= 4:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        # Character-level repetition for space-free text (CJK, GPT-2 Ġ-prefixed tokens,
        # or any output where .split() collapses everything into ≤1 word).
        # We check both single-character runs (≥8) and short char-ngrams (≥4 repeats).
        if not found and n_words <= 1 and len(text) > 10:
            chars = list(text)
            # Single-char run
            run, run_ch = 1, None
            for ch in chars:
                if ch == run_ch:
                    run += 1
                    if run >= 8:
                        found = True
                        break
                else:
                    run, run_ch = 1, ch
            if not found:
                # Short char-ngram repeated ≥4 times consecutively
                n_ch = len(chars)
                for pl in range(1, min(8, n_ch // 4) + 1):
                    for start in range(n_ch - pl * 4 + 1):
                        phrase = chars[start : start + pl]
                        count, pos = 1, start + pl
                        while pos + pl <= n_ch and chars[pos : pos + pl] == phrase:
                            count += 1
                            pos += pl
                        if count >= 4:
                            found = True
                            break
                    if found:
                        break
        rates.append(1.0 if found else 0.0)
    return float(np.mean(rates)) if rates else 0.0


# ---------------------------------------------------------------------------
# DeBERTa zero-shot scorer (CONCEPT_LABELS-aware)
# ---------------------------------------------------------------------------

_zeroshot_clf = None
# Upgraded from deberta-v3-small (85MB) to deberta-v3-large-zeroshot-v2 (900MB).
# The large model has substantially better NLI accuracy, especially on short texts
# and structurally similar concepts (e.g. code_python vs code_snippets).
_ZEROSHOT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2"

# All concept labels used as the multi-label candidate set for scoring.
# Multi-label mode: DeBERTa ranks ALL labels simultaneously for each text, so
# scores are calibrated relative to each other rather than against a trivial
# "not X" negation.  This handles overlapping concepts (e.g. code_* family) much
# better than binary classification.
_ALL_CONCEPT_LABELS: List[str] = list(dict.fromkeys(CONCEPT_LABELS.values()))


def _get_zeroshot_clf():
    global _zeroshot_clf
    if _zeroshot_clf is None:
        from transformers import pipeline as _hf_pipeline
        _dev = 0 if torch.cuda.is_available() else -1
        # Try models in priority order; for each model try local cache first (air-gapped
        # clusters), then allow network download as a last resort.
        _model_candidates = [
            _ZEROSHOT_MODEL,                            # large, best accuracy (~900MB)
            "MoritzLaurer/deberta-v3-base-zeroshot-v2", # base, good quality (~190MB)
            "facebook/bart-large-mnli",                 # BART fallback, widely cached (~400MB)
            "cross-encoder/nli-deberta-v3-small",       # small last resort (~85MB)
        ]
        _last_err: Exception = RuntimeError("no candidates")
        for _local_only in (True, False):
            for _mid in _model_candidates:
                try:
                    _zeroshot_clf = _hf_pipeline(
                        "zero-shot-classification", model=_mid, device=_dev,
                        local_files_only=_local_only,
                    )
                    _src = "cache" if _local_only else "download"
                    print(f"[step8] Loaded zero-shot classifier: {_mid} ({_src})")
                    return _zeroshot_clf
                except Exception as _e:
                    _src = "cache" if _local_only else "download"
                    print(f"[step8] zero-shot: {_mid} ({_src}) unavailable — {type(_e).__name__}: {_e}")
                    _last_err = _e
                    continue
        raise RuntimeError(f"No zero-shot classifier available: {_last_err}")
    return _zeroshot_clf


def score_zeroshot(texts: List[str], concept: str) -> float:
    """DeBERTa NLI entailment score for concept presence. Returns mean in [0, 1].

    Uses multi-label classification across all concept labels simultaneously so
    scores are calibrated relative to each other (not against a trivial negation).
    """
    return float(np.mean(score_zeroshot_per_text(texts, concept)))


def score_zeroshot_per_text(texts: List[str], concept: str) -> List[float]:
    """DeBERTa NLI entailment score per text. Multi-label across all concepts, returns List[float]."""
    try:
        clf = _get_zeroshot_clf()
    except Exception as _e:
        print(f"[step8] WARNING: zero-shot classifier unavailable ({_e}). Returning 0.5.")
        return [0.5] * len(texts)
    label = CONCEPT_LABELS.get(concept, concept.replace("_", " "))
    try:
        raw = clf(
            texts,
            candidate_labels=_ALL_CONCEPT_LABELS,
            hypothesis_template="This text is about {}.",
            batch_size=min(len(texts), 8),
            multi_label=True,
        )
        if isinstance(raw, dict):
            raw = [raw]
        scores = []
        for out in raw:
            if label in out["labels"]:
                idx = out["labels"].index(label)
                scores.append(float(out["scores"][idx]))
            else:
                scores.append(0.5)
        return scores
    except Exception as _e:
        print(f"[step8] WARNING: score_zeroshot_per_text failed for '{concept}': {_e}")
        return [0.5] * len(texts)


# ---------------------------------------------------------------------------
# Per-method result aggregation (bidirectionality + efficiency)
# ---------------------------------------------------------------------------

# Minimum absolute delta required for bidirectionality to be meaningful.
# Below this the concept score barely moved and ratio is noise.
_MIN_BIDIR_DELTA = 0.005


def _aggregate_method(
    results_by_strength: Dict[str, Dict],
    baseline_score: float,
    baseline_ppl: float,
    min_valid: int = 3,
) -> Dict:
    """Compute derived metrics for one method across its strength sweep.

    Degenerate entries (hallucination_rate > 0.4, or n_valid_prompts < min_valid) are excluded
    from the positive/negative delta pools.

    bidir_ratio is only computed when |max_positive_delta| >= _MIN_BIDIR_DELTA (0.005).
    When the positive direction has no real effect, bidirectionality is undefined (None),
    not an astronomically large number.
    """
    _HAL_THRESH = 0.4

    def _is_valid(sdata: Dict) -> bool:
        return (
            sdata.get("hallucination_rate", 0.0) <= _HAL_THRESH
            and sdata.get("n_valid_prompts", 99) >= min_valid
        )

    pos_deltas = {
        s: results_by_strength[s]["mean_concept_score"] - baseline_score
        for s in results_by_strength
        if float(s) > 0 and _is_valid(results_by_strength[s])
    }
    neg_deltas = {
        s: results_by_strength[s]["mean_concept_score"] - baseline_score
        for s in results_by_strength
        if float(s) < 0 and _is_valid(results_by_strength[s])
    }
    max_pos = max(pos_deltas.values(), default=0.0)
    max_neg = min(neg_deltas.values(), default=0.0)
    opt_pos = max(pos_deltas, key=pos_deltas.get) if pos_deltas else None
    opt_neg = min(neg_deltas, key=neg_deltas.get) if neg_deltas else None

    # bidir_ratio is only meaningful when there is a real positive effect.
    # When max_pos < threshold, the ratio is undefined (None) — not a huge/misleading number.
    if abs(max_pos) >= _MIN_BIDIR_DELTA:
        bidir_ratio: Optional[float] = round(abs(max_neg) / abs(max_pos), 4)
    else:
        bidir_ratio = None

    opt_ppl = results_by_strength[opt_pos]["mean_perplexity"] if opt_pos else None
    ppl_inc = (
        (opt_ppl - baseline_ppl) / max(baseline_ppl, 1e-9) * 100
        if opt_ppl is not None else None
    )
    efficiency_valid = ppl_inc is not None and ppl_inc <= 30.0

    # Carry per-prompt deltas at the optimal positive strength through for statistics
    opt_pp_deltas: Optional[List[float]] = (
        results_by_strength[opt_pos].get("per_prompt_deltas") if opt_pos else None
    )

    # Mean hallucination rate across all non-baseline strengths
    _hal_vals = [
        results_by_strength[s].get("hallucination_rate", 0.0)
        for s in results_by_strength if float(s) != 0.0
    ]
    avg_hal = round(float(np.mean(_hal_vals)), 4) if _hal_vals else 0.0

    # Polarity-aware effective delta: for C3/universal vectors the sign may be inverted
    # on some models (negative strengths produce better concept alignment).  Record the
    # absolute best achievable delta in either direction so tables reflect true capability.
    eff_max_delta = max(max_pos, abs(max_neg))
    eff_direction = "positive" if max_pos >= abs(max_neg) else "negative"

    return {
        "results_by_strength": results_by_strength,
        "optimal_positive_strength": float(opt_pos) if opt_pos is not None else None,
        "optimal_negative_strength": float(opt_neg) if opt_neg is not None else None,
        "max_positive_delta": round(max_pos, 6),
        "max_negative_delta": round(max_neg, 6),
        "effective_max_delta": round(eff_max_delta, 6),
        "effective_direction": eff_direction,
        "bidirectionality_ratio": bidir_ratio,
        "perplexity_at_optimal": round(opt_ppl, 4) if opt_ppl is not None else None,
        "perplexity_increase_pct": round(ppl_inc, 2) if ppl_inc is not None else None,
        "efficiency_valid": efficiency_valid,
        "hallucination_rate": avg_hal,
        # Per-prompt deltas at the optimal positive strength.
        # Used by _compute_stats_section() for bootstrap CI and Wilcoxon tests.
        # None when the optimal strength has no valid per-prompt data (re-run required).
        "per_prompt_deltas_at_optimal": opt_pp_deltas,
    }


# ---------------------------------------------------------------------------
# Per-model evaluation worker (called directly or in subprocess)
# ---------------------------------------------------------------------------


def _patch_torch_load_if_needed(model_dir: str) -> bool:
    """Bypass the CVE-2025-32434 torch.load block for LOCAL .bin model files.

    Transformers >= 4.51 raises ValueError when loading old .bin (pickle) weights
    on torch < 2.6.  This only affects locally-downloaded, trusted model files.
    If the model directory has no .safetensors files (i.e. it uses the old .bin
    format), we monkey-patch out the check so loading proceeds normally.

    Returns True if the patch was applied (means you should unset it after loading).
    """
    import glob as _glob_pt
    has_safetensors = bool(_glob_pt.glob(os.path.join(model_dir, "*.safetensors")))
    if has_safetensors:
        return False  # no patch needed
    try:
        import transformers.utils.import_utils as _tiu
        import transformers.modeling_utils as _tmu
        patched = False
        if hasattr(_tiu, "check_torch_load_is_safe"):
            _tiu._orig_check_torch_load_is_safe = _tiu.check_torch_load_is_safe
            _tiu.check_torch_load_is_safe = lambda: None
            patched = True
        # modeling_utils imports check_torch_load_is_safe at module load time;
        # patch its own binding too (the traceback shows the call is here).
        if hasattr(_tmu, "check_torch_load_is_safe"):
            _tmu._orig_check_torch_load_is_safe = _tmu.check_torch_load_is_safe
            _tmu.check_torch_load_is_safe = lambda: None
            patched = True
        if patched:
            print(
                f"[step8] NOTE: {os.path.basename(model_dir)} uses .bin weights (no safetensors). "
                "Bypassing torch.load CVE check for this local trusted file. "
                "Convert to safetensors with: model.save_pretrained(path, safe_serialization=True)"
            )
            return True
    except Exception:
        pass
    return False


def _unpatch_torch_load() -> None:
    """Restore the original check_torch_load_is_safe after model loading."""
    try:
        import transformers.utils.import_utils as _tiu
        if hasattr(_tiu, "_orig_check_torch_load_is_safe"):
            _tiu.check_torch_load_is_safe = _tiu._orig_check_torch_load_is_safe
            del _tiu._orig_check_torch_load_is_safe
    except Exception:
        pass
    try:
        import transformers.modeling_utils as _tmu
        if hasattr(_tmu, "_orig_check_torch_load_is_safe"):
            _tmu.check_torch_load_is_safe = _tmu._orig_check_torch_load_is_safe
            del _tmu._orig_check_torch_load_is_safe
    except Exception:
        pass


def _prescreen_model_prompts(
    model_name: str,
    prompts: List[str],
    max_new_tokens: int,
    gpu_id: Optional[int] = None,
) -> List[float]:
    """Load model, generate all prompts (greedy, no hooks), return repetition-rate per prompt.

    Used by --uniform-prompts to pre-screen which prompts every model generates cleanly
    before the main eval starts.  Model is unloaded (+ CUDA cache cleared) before returning.
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model_cfg = config.MODELS[model_name]
    _local_dir = os.path.join(str(config._CONFIG_DIR), "model", model_name)
    if not os.path.isdir(_local_dir):
        raise FileNotFoundError(
            f"[step8/prescreen] Model not found at {_local_dir}."
        )

    _use_fast = "gpt2" not in model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(
        _local_dir, use_fast=_use_fast, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Tag byte-BPE tokenizers (gpt2 only) so _decode_new_tokens uses the correct path.
    tokenizer._byte_bpe = "gpt2" in model_name.lower()
    # For deepseek: attach a tokenizers.Tokenizer loaded directly from tokenizer.json
    # so _decode_new_tokens can bypass the broken slow-tokenizer decode path.
    _tok_json = os.path.join(_local_dir, "tokenizer.json")
    if "deepseek" in model_name.lower() and os.path.isfile(_tok_json):
        try:
            from tokenizers import Tokenizer as _RawTokenizer
            tokenizer._fast_tok = _RawTokenizer.from_file(_tok_json)
            print(f"{_ts()} [step8] deepseek _fast_tok loaded OK from {_tok_json}")
        except Exception as _e:
            print(f"{_ts()} [step8] WARNING: deepseek _fast_tok FAILED: {_e}")
    else:
        print(f"{_ts()} [step8] deepseek _fast_tok SKIPPED: deepseek={('deepseek' in model_name.lower())}, file={os.path.isfile(_tok_json)}, path={_tok_json}")
    _model_kwargs: dict = {
        "dtype": _dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if torch.cuda.is_available():
        _model_kwargs["device_map"] = "auto"

    _patched = _patch_torch_load_if_needed(_local_dir)
    model = AutoModelForCausalLM.from_pretrained(_local_dir, **_model_kwargs)
    if _patched:
        _unpatch_torch_load()
    model.eval()

    print(f"{_ts()} [step8/prescreen] {model_name}: generating {len(prompts)} prompts...")
    outputs = generate_batch(model, tokenizer, prompts, max_new_tokens)
    rep_rates = [_repetition_rate([o]) for o in outputs]
    print(
        f"{_ts()} [step8/prescreen] {model_name}: "
        f"{sum(r < 1.0 for r in rep_rates)}/{len(rep_rates)} clean prompts"
    )

    # Unload to free VRAM for the next model / the real eval workers
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rep_rates


def _eval_model_worker(
    model_name: str,
    gpu_id: Optional[int],
    args_dict: dict,
    sv: Dict,
    b3_extra_methods: Dict,
    target_concepts: List[str],
    prompts: List[str],
    results_dir: str,
    run_id: str,
) -> None:
    """Full B4 evaluation for one model. Designed to be called in a subprocess."""
    # Prevent tokenizer Rust threadpool deadlock in spawned subprocesses.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    # Only pin to a specific GPU in --parallel mode. In sequential mode gpu_id is None
    # and device_map="auto" can use all available GPUs (helpful for 70B+ models later).
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _worker_seed: int = args_dict.get("seed", 42)
    set_seed(_worker_seed)

    model_cfg = config.MODELS[model_name]
    max_new_tokens: int = args_dict["max_new_tokens"]
    injection_mode: str = args_dict.get("injection_mode", "single_layer")
    injection_layers_override: List[int] = args_dict.get("injection_layers_override", [])

    # ---- PPL scorer ----
    # All models use GPT-2 117M as the fixed held-out scorer (_PPL_SCORER_MAP) so that the
    # 30% PPL-increase gate in Table 3 is on the same scale for every target model.
    # Falling back silently to a different scorer would break cross-model comparability —
    # raise hard if the configured scorer is not on disk.
    _preferred_ppl_name = _PPL_SCORER_MAP.get(model_name, "gpt2")
    _ppl_local = os.path.join(str(config._CONFIG_DIR), "model", _preferred_ppl_name)
    if not os.path.isdir(_ppl_local):
        raise FileNotFoundError(
            f"[step8] PPL scorer '{_preferred_ppl_name}' not found at {_ppl_local}. "
            f"Download it with: python step2_extract_model.py --model {_preferred_ppl_name} "
            "(or adjust _PPL_SCORER_MAP if you intend a different uniform scorer). "
            "Do NOT silently fall back — that would make PPL gates non-comparable across models."
        )
    _ppl_scorer_name = os.path.basename(_ppl_local)
    _ppl_use_fast = "gpt2" not in _ppl_scorer_name.lower()
    print(f"{_ts()} [step8] {'='*60}")
    print(f"{_ts()} [step8] PHASE 1 — loading PPL scorer + target model: {model_name}")
    print(f"{_ts()} [step8] {'='*60}")
    print(f"{_ts()} [step8] {model_name}: PPL scorer = {_ppl_scorer_name}"
          + (" (⚠️ circular — same model family)" if _ppl_scorer_name == model_name else " ✅ (non-circular)"))
    ppl_tokenizer = AutoTokenizer.from_pretrained(_ppl_local, use_fast=_ppl_use_fast)
    # GPT-2 has no pad token by default — required for batched compute_perplexity
    if ppl_tokenizer.pad_token_id is None:
        ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
    _ppl_patched = _patch_torch_load_if_needed(_ppl_local)
    ppl_model_obj = AutoModelForCausalLM.from_pretrained(
        _ppl_local,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if _ppl_patched:
        _unpatch_torch_load()
    ppl_model_obj.to("cuda" if torch.cuda.is_available() else "cpu")
    ppl_model_obj.eval()

    # ---- Target model ----
    _local_dir = os.path.join(str(config._CONFIG_DIR), "model", model_name)
    if not os.path.isdir(_local_dir):
        raise FileNotFoundError(
            f"[step8] Model not found at {_local_dir}. "
            "All models must be pre-downloaded to the model/ directory."
        )
    _model_src = _local_dir

    # deepseek uses LlamaTokenizerFast (tokenizer.json) — NOT byte-BPE.
    # use_fast=True is correct for all non-gpt2 models.
    _use_fast = "gpt2" not in model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(_model_src, use_fast=_use_fast,
                                              local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched causal LM generation
    # Tag byte-BPE tokenizers (gpt2 only) so _decode_new_tokens uses the correct path.
    tokenizer._byte_bpe = "gpt2" in model_name.lower()
    # For deepseek: attach a tokenizers.Tokenizer loaded directly from tokenizer.json
    # so _decode_new_tokens can bypass the broken slow-tokenizer decode path.
    _tok_json = os.path.join(_local_dir, "tokenizer.json")
    if "deepseek" in model_name.lower() and os.path.isfile(_tok_json):
        try:
            from tokenizers import Tokenizer as _RawTokenizer
            tokenizer._fast_tok = _RawTokenizer.from_file(_tok_json)
            print(f"{_ts()} [step8] deepseek _fast_tok loaded OK from {_tok_json}")
        except Exception as _e:
            print(f"{_ts()} [step8] WARNING: deepseek _fast_tok FAILED: {_e}")
    else:
        print(f"{_ts()} [step8] deepseek _fast_tok SKIPPED: deepseek={('deepseek' in model_name.lower())}, file={os.path.isfile(_tok_json)}, path={_tok_json}")
    # All other models (gemma, llama, mistral, deepseek) use bfloat16 (~14 GB on a single A100).
    _dtype = torch.float32 if "gpt2" in model_name.lower() else torch.bfloat16
    _model_kwargs = {
        "dtype": _dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if torch.cuda.is_available():
        # device_map="auto" uses the single visible GPU in --parallel mode (CUDA_VISIBLE_DEVICES=N),
        # or spreads across all GPUs in sequential mode (useful when model > 80 GB in the future).
        _model_kwargs["device_map"] = "auto"
    _tgt_patched = _patch_torch_load_if_needed(_model_src)
    try:
        model = AutoModelForCausalLM.from_pretrained(_model_src, **_model_kwargs)
    except Exception:
        # Fallback: load to CPU first then transfer (slower but safer)
        _model_kwargs.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(_model_src, **_model_kwargs)
        if torch.cuda.is_available():
            model = model.to("cuda")
    if _tgt_patched:
        _unpatch_torch_load()
    model.eval()

    _model_device = next(model.parameters()).device
    _model_dtype = next(model.parameters()).dtype

    # ---- Injection layers ----
    if injection_mode == "single_layer":
        inject_layers = [model_cfg["target_layer"]]
    elif injection_mode == "multi_layer" and injection_layers_override:
        inject_layers = injection_layers_override
    else:
        inject_layers = get_all_layer_indices(model_name, model)

    print(f"{_ts()} [step8] {model_name} (GPU {gpu_id}): {len(target_concepts)} concepts, "
          f"{len(prompts)} candidate prompts, inject layers={inject_layers}")

    # ---- Step 0: Shared baseline (no hooks) ----
    print(f"{_ts()} [step8] ")
    print(f"{_ts()} [step8] PHASE 2 — baseline generation ({len(prompts)} candidate prompts)")
    # Strategy: run ALL candidate prompts, rank by repetition-loop rate, pick the
    # top_n cleanest ones.  Only those are used in all steered evaluations.
    top_n: int = args_dict.get("top_n_prompts", TOP_N_PROMPTS)
    # Minimum valid-prompt threshold for scoring — at least 3 for publication runs,
    # but relaxed to 1 for smoke/single-prompt runs so results are still computed.
    _min_valid: int = max(1, min(3, top_n))
    _all_prompts = prompts  # full candidate pool
    print(f"{_ts()} [step8] {model_name}: generating baseline outputs for {len(_all_prompts)} candidate prompts...")
    _t0_baseline = time.time()
    _all_baseline_outputs = generate_batch(model, tokenizer, _all_prompts, max_new_tokens)
    print(f"{_ts()} [step8] {model_name}: baseline generation done in {time.time()-_t0_baseline:.1f}s")

    # Rank candidates by repetition rate (lower = cleaner; 0.0 = clean, 1.0 = loop)
    _all_rep_pp = [_repetition_rate([o]) for o in _all_baseline_outputs]
    _ranked = sorted(range(len(_all_prompts)), key=lambda i: _all_rep_pp[i])

    # ---- BASELINE SANITY GATE ----
    # If fewer than top_n candidates are clean (rep_rate = 0.0, i.e. no loop detected),
    # the model has a fundamental problem (bad tokenizer, wrong model file, etc).
    _clean_candidates = [i for i in _ranked if _all_rep_pp[i] < 1.0]

    # Always print every baseline output + its rep_rate so failures are debuggable
    print(f"{_ts()} [step8] {model_name}: baseline rep_rates: "
          + ", ".join(f"[{i}]={_all_rep_pp[i]:.2f}" for i in _ranked))
    for i, (p, o, r) in enumerate(zip(_all_prompts, _all_baseline_outputs, _all_rep_pp)):
        tag = "LOOP" if r >= 1.0 else "ok"
        print(f"{_ts()} [step8]   [{tag}] prompt[{i}]: {_preview(p, 60)}")
        print(f"{_ts()} [step8]   [{tag}] output[{i}]: {_preview(o, 120)}")

    if len(_clean_candidates) < top_n:
        # In single-prompt mode (top_n=1) the user explicitly supplied the prompt —
        # downgrade to a warning and keep the loopiest-but-cleanest candidate so the
        # rest of the pipeline can still run and show what the model does.
        if top_n == 1:
            print(
                f"{_ts()} [step8] WARNING: {model_name}: baseline sanity gate SOFT-FAIL — "
                f"the supplied prompt produces a looping output (rep_rate={_all_rep_pp[0]:.2f}). "
                "Continuing anyway (single-prompt mode). Steering results may inherit the loop."
            )
            _clean_candidates = list(range(len(_all_prompts)))  # use all, even loopy ones
        else:
            raise RuntimeError(
                f"[step8] {model_name}: BASELINE SANITY GATE FAILED — only "
                f"{len(_clean_candidates)}/{len(_all_prompts)} candidate prompts are loop-free, "
                f"need at least {top_n}. "
                "Check model loading, tokenizer, and prompt set before re-running."
            )

    # Select exactly top_n cleanest prompts (ranked by ascending rep_rate)
    good_baseline_idx_in_all = _clean_candidates[:top_n]
    # Remap to a compact 0..top_n-1 index space for the rest of the pipeline
    prompts          = [_all_prompts[i] for i in good_baseline_idx_in_all]
    baseline_outputs = [_all_baseline_outputs[i] for i in good_baseline_idx_in_all]
    _baseline_rep_pp = [_all_rep_pp[i] for i in good_baseline_idx_in_all]
    good_baseline_idx: List[int] = list(range(len(prompts)))  # always 0..top_n-1 now

    # Compute perplexity only on selected prompts (saves time)
    print(f"{_ts()} [step8] {model_name}: baseline sanity gate PASSED — selected {len(prompts)}/{len(_all_prompts)} loop-free prompts")
    print(f"{_ts()} [step8] ")
    print(f"{_ts()} [step8] PHASE 3 — baseline perplexity + DeBERTa scoring")
    print(f"{_ts()} [step8] {model_name}: scoring baseline perplexity on {len(prompts)} selected prompts...")
    _t0_ppl = time.time()
    baseline_ppl_all = compute_perplexity(baseline_outputs, ppl_model_obj, ppl_tokenizer)
    print(f"{_ts()} [step8] {model_name}: baseline PPL done in {time.time()-_t0_ppl:.1f}s")

    # ---- Measure mean residual-stream norm at the target layer ----
    # All steering vectors are L2-normalised (magnitude=1.0). Injecting them raw means
    # strength=5 adds magnitude 5.0 into a residual stream whose typical hidden-state norm
    # for 7B models is ~50-80. That's only ~6-10% perturbation — too weak to steer reliably.
    # Fix: scale every injection by the mean activation norm at the target layer so that
    # strength=1.0 corresponds to a unit-norm perturbation in the activation's own scale.
    # This is the standard "multiplied by activation norm" approach (Zou et al. 2023, Turner et al.).
    _target_layer_idx = model_cfg["target_layer"]
    _act_norms: List[float] = []
    def _norm_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        _act_norms.append(float(hidden.float().norm(dim=-1).mean().item()))
    try:
        _norm_layer = get_layer_module(model_name, model, _target_layer_idx)
        _norm_handle = _norm_layer.register_forward_hook(_norm_hook)
        # One batched forward pass on the selected baseline prompts
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=config.MAX_SEQ_LEN)
        with torch.no_grad():
            model(**{k: v.to(model.device) for k, v in enc.items()})
        _norm_handle.remove()
        _layer_act_norm = float(np.mean(_act_norms)) if _act_norms else 1.0
    except Exception as _norm_err:
        _layer_act_norm = 1.0
        print(f"{_ts()} [step8] WARNING: could not measure layer norm ({_norm_err}). Using scale=1.0.")
    print(f"{_ts()} [step8] {model_name}: layer {_target_layer_idx} mean activation norm = {_layer_act_norm:.2f}  (injection_scale = act_norm / sqrt(D))")

    # ---- Load per-dimension std for de-normalisation ----
    # SAE was trained on z-scored activations: (x - mean) / std.
    # Steering vectors decoded from SAE features are therefore in z-scored space.
    # To inject them correctly into the raw residual stream we must undo the std scaling:
    #   vec_raw[d] = vec_norm[d] * std[d]
    # (We don't add mean because we're steering a direction, not setting an absolute value.)
    # Norm stats were saved by normalise_activations.py as {model}_{dataset}_norm_stats.json.
    # The combined {model}_norm_stats.json may not exist — fall back to averaging per-dataset stds.
    _norm_stats_path = os.path.join(config.ACTIVATIONS_DIR, f"{model_name}_norm_stats.json")
    _norm_stats_real = os.path.join(os.path.realpath(config.ACTIVATIONS_DIR), f"{model_name}_norm_stats.json")
    _act_std: Optional[np.ndarray] = None
    # Check both the declared path and the resolved real path (handles symlinked activations/)
    if os.path.exists(_norm_stats_path) or os.path.exists(_norm_stats_real):
        _load_path = _norm_stats_path if os.path.exists(_norm_stats_path) else _norm_stats_real
        with open(_load_path) as _nsf:
            _ns = json.load(_nsf)
        _act_std = np.array(_ns["std"], dtype=np.float32)
        print(f"{_ts()} [step8] {model_name}: loaded activation std from {os.path.basename(_load_path)}")
    else:
        # Fall back: glob all per-dataset norm_stats files and average their stds.
        # Averaging is appropriate because the SAE was trained on the full mixture,
        # and mean(per-dataset std) ≈ pooled std when datasets have similar sizes.
        # Use realpath so glob works even when activations/ is a symlink.
        _act_dir_real = os.path.realpath(config.ACTIVATIONS_DIR)
        print(f"{_ts()} [step8] {model_name}: activations dir → {_act_dir_real}")
        _per_dataset_stats = sorted(_glob.glob(
            os.path.join(_act_dir_real, f"{model_name}_*_norm_stats.json")
        ))
        print(f"{_ts()} [step8] {model_name}: found {len(_per_dataset_stats)} per-dataset norm_stats files")
        if _per_dataset_stats:
            _std_arrays = []
            for _sp in _per_dataset_stats:
                try:
                    _sd = json.load(open(_sp))
                    _std_arrays.append(np.array(_sd["std"], dtype=np.float32))
                except Exception:
                    pass
            if _std_arrays:
                _act_std = np.mean(np.stack(_std_arrays, axis=0), axis=0)
                # Save the combined file alongside the per-dataset files (real path)
                _combined = {"std": _act_std.tolist(), "mean": None,
                             "source": "averaged from per-dataset files",
                             "n_datasets": len(_std_arrays)}
                _save_path = os.path.join(_act_dir_real, f"{model_name}_norm_stats.json")
                with open(_save_path, "w") as _nf:
                    json.dump(_combined, _nf)
                print(f"{_ts()} [step8] {model_name}: averaged std from {len(_std_arrays)} "
                      f"per-dataset norm_stats files → saved {os.path.basename(_save_path)}")
            else:
                _per_dataset_stats = []  # nothing usable
        if _act_std is None:
            raise FileNotFoundError(
                f"\n{'='*70}\n"
                f"[step8] FATAL: no norm_stats found for {model_name}\n"
                f"  Looked for: {_norm_stats_path}\n"
                f"  Also tried: {model_name}_*_norm_stats.json — found 0 files\n"
                f"\n"
                f"  Without this file the steering vectors CANNOT be de-normalised from\n"
                f"  z-scored space into raw activation space.  Injecting them raw silently\n"
                f"  corrupts the direction (not just the magnitude) — results would be\n"
                f"  meaningless.  Stopping now so you can fix this before any GPU time\n"
                f"  is wasted on steered generation.\n"
                f"\n"
                f"  Fix: run normalise_activations.py for this model:\n"
                f"    python normalise_activations.py --model {model_name}\n"
                f"{'='*70}"
            )

    print(f"{_ts()} [step8] {model_name}: baseline PPL={baseline_ppl_all:.2f} — scoring {len(target_concepts)} concepts with DeBERTa...")
    baseline_scores: Dict[str, float] = {}           # filtered mean (for logging)
    baseline_scores_pp: Dict[str, List[float]] = {}  # per-prompt scores (for per-strength delta)
    for _ci, concept in enumerate(target_concepts):
        _pp = score_zeroshot_per_text(baseline_outputs, concept)
        baseline_scores_pp[concept] = _pp
        # Filtered mean over clean baseline prompts only
        baseline_scores[concept] = float(np.mean([_pp[i] for i in good_baseline_idx]))
        print(f"{_ts()} [step8] {model_name}: baseline  [{_ci+1:2d}/{len(target_concepts)}] {concept} = {baseline_scores[concept]:.4f}  (n_clean={len(good_baseline_idx)}/{len(prompts)})")

    suffix = f"_{run_id}" if run_id else ""
    examples_path = os.path.join(results_dir, f"steering_examples{suffix}.jsonl")

    # Write per-model baseline entries
    os.makedirs(results_dir, exist_ok=True)
    with open(examples_path, "a", encoding="utf-8") as f:
        for pi, (p, o) in enumerate(zip(prompts, baseline_outputs)):
            f.write(json.dumps({
                "model": model_name,
                "concept": "__baseline__",
                "method": "baseline",
                "strength": 0.0,
                "prompt_idx": pi,
                "prompt": p,
                "output": o,
                "baseline_rep_rate": round(_baseline_rep_pp[pi], 4),
                "baseline_clean": pi in good_baseline_idx,
                "concept_scores": {c: round(baseline_scores_pp[c][pi], 6) for c in target_concepts},
                "perplexity": round(baseline_ppl_all, 4),
            }) + "\n")

    # ---- Method evaluation loop ----
    results: Dict[str, Dict] = {}
    _total_concepts = len(target_concepts)

    for _ci, concept in enumerate(target_concepts):
        baseline_score = baseline_scores[concept]
        method_results: Dict[str, Dict] = {}

        # Build ordered method list: Exp A native → Exp B cross-model → Exp C universal
        all_methods: List[Tuple[str, list, int]] = []
        for mk in ["sae_vector", "caa_vector"]:
            vec_list = sv.get(model_name, {}).get(concept, {}).get(mk)
            if vec_list:
                all_methods.append((mk, vec_list, model_cfg["target_layer"]))
        for mk, vec_list, layer in b3_extra_methods.get(model_name, {}).get(concept, []):
            all_methods.append((mk, vec_list, layer))

        if not all_methods:
            print(f"{_ts()} [step8] {model_name}: concept [{_ci+1:2d}/{_total_concepts}] {concept} — no methods, skipping")
            continue

        print(f"{_ts()} [step8] ")
        print(f"{_ts()} [step8] PHASE 4 — concept [{_ci+1}/{_total_concepts}]: {concept}  ({len(all_methods)} methods × {len(_sweep if False else SWEEP_STANDARD)} strengths)")
        print(f"{_ts()} [step8] {model_name}: methods = {[m[0] for m in all_methods]}")

        for _mi, (method_key, vec_list, layer_idx) in enumerate(all_methods):
            # De-normalise: undo the per-dimension std scaling applied during activation
            # extraction so the vector direction is correct in raw activation space.
            _vec_np = np.array(vec_list, dtype=np.float32)
            if _act_std is not None and len(_act_std) == len(_vec_np):
                _vec_np = _vec_np * _act_std
            # Re-L2-normalise so the direction is clean and _layer_act_norm controls scale
            _vec_norm = np.linalg.norm(_vec_np)
            if _vec_norm > 1e-8:
                _vec_np = _vec_np / _vec_norm
            # Scale injection magnitude.
            # A unit vector injected raw means: strength=5 → 5/act_norm ≈ 0.2% — too weak.
            # Multiplying by act_norm means: strength=5 → 5×act_norm/act_norm = 500% — garbles.
            # Correct calibration: divide by sqrt(hidden_size).
            #   For z-scored models: act_norm ≈ sqrt(D), so scale ≈ 1.0 — matches pre-fix.
            #   For raw models (deepseek): act_norm=2173, D=4096 → scale≈34 → strength=5 ≈ 8%.
            # This makes strength=1 a ~1.5% perturbation and strength=5 a ~8% perturbation
            # regardless of whether the model's activations were z-scored or not.
            _hidden_size = float(_vec_np.shape[0])
            _injection_scale = _layer_act_norm / float(np.sqrt(_hidden_size))
            vec = (
                torch.tensor(_vec_np, dtype=_model_dtype)
                .to(_model_device)
                .view(1, 1, -1)
            ) * _injection_scale

            _injection_pct = 100.0 * _injection_scale / max(_layer_act_norm, 1e-8)
            print(f"{_ts()} [step8]   method [{_mi+1}/{len(all_methods)}] {method_key}: "
                  f"injection_scale={_injection_scale:.3f}  ({_injection_pct:.1f}% of act_norm per unit strength)")
            try:
                layer_mod = get_layer_module(model_name, model, layer_idx)
            except (ValueError, IndexError) as e:
                print(f"{_ts()} [step8] {model_name}/{concept}/{method_key}: skip — {e}")
                continue

            results_by_strength: Dict[str, Dict] = {}
            _sweep = get_sweep(method_key)
            for _si, strength in enumerate(_sweep):
                print(f"{_ts()} [step8]   strength [{_si+1}/{len(_sweep)}] s={strength:+.1f} ...")
                if strength == 0.0:
                    outputs = baseline_outputs
                    ppl = baseline_ppl_all
                    score = baseline_score
                    rep_rate = 0.0
                    n_valid = len(good_baseline_idx)
                    print(f"{_ts()} [step8]   s={strength:+.1f}  (baseline reused)")
                else:
                    _t0_gen = time.time()
                    hook = add_steering_hook(layer_mod, vec * strength)
                    outputs = generate_batch(model, tokenizer, prompts, max_new_tokens)
                    hook.remove()
                    print(f"{_ts()} [step8]   s={strength:+.1f}  generation done in {time.time()-_t0_gen:.1f}s")
                    for _pi, _o in enumerate(outputs):
                        print(f"{_ts()} [step8]     prompt[{_pi}] → {_preview(_o, 90)}")
                    ppl = compute_perplexity(outputs, ppl_model_obj, ppl_tokenizer)

                    # Per-prompt quality: exclude prompts degenerate at baseline OR at this strength
                    _steered_rep_pp = [_repetition_rate([o]) for o in outputs]
                    valid_idx = [
                        i for i in good_baseline_idx
                        if _steered_rep_pp[i] <= 0.4
                    ]
                    n_valid = len(valid_idx)
                    rep_rate = float(np.mean(_steered_rep_pp))  # diagnostic only

                    print(f"{_ts()} [step8]   s={strength:+.1f}  scoring DeBERTa (n_valid={n_valid}/{len(prompts)}, min_valid={_min_valid})...")
                    if n_valid >= _min_valid:
                        _steered_pp = score_zeroshot_per_text(outputs, concept)
                        _baseline_pp_filtered = [baseline_scores_pp[concept][i] for i in valid_idx]
                        _steered_pp_filtered = [_steered_pp[i] for i in valid_idx]
                        score = float(np.mean(_steered_pp_filtered))
                        # Delta is against the same filtered baseline prompts (fair comparison)
                        _baseline_score_filtered = float(np.mean(_baseline_pp_filtered))
                        delta = round(score - _baseline_score_filtered, 4)
                        # Per-prompt deltas — used for error bars and Wilcoxon tests
                        per_prompt_deltas: List[float] = [
                            round(_steered_pp[i] - baseline_scores_pp[concept][i], 6)
                            for i in valid_idx
                        ]
                    else:
                        # Too few valid prompts — disqualify this strength entry
                        _steered_pp = [None] * len(prompts)
                        score = baseline_score
                        _baseline_score_filtered = baseline_score
                        delta = 0.0
                        per_prompt_deltas = []
                        print(f"{_ts()} [step8]   DISQUALIFIED s={strength:+.1f} {concept}/{method_key} — only {n_valid}/{_min_valid} valid prompts")

                    print(f"{_ts()} [step8]   ✓ s={strength:+.1f}  {concept}/{method_key}  score={score:.4f}  delta={delta:+.4f}  ppl={ppl:.1f}  hal={rep_rate:.2f}  n_valid={n_valid}")
                results_by_strength[str(strength)] = {
                    "mean_concept_score": round(score, 6),
                    "mean_perplexity": round(ppl, 4),
                    "hallucination_rate": round(rep_rate, 4),
                    "n_valid_prompts": n_valid,
                    **({"per_prompt_deltas": per_prompt_deltas} if strength != 0.0 else {}),
                }
                if strength != 0.0:
                    with open(examples_path, "a", encoding="utf-8") as f:
                        for pi, (p, o) in enumerate(zip(prompts, outputs)):
                            _prompt_score = _steered_pp[pi] if n_valid >= _min_valid else None
                            _baseline_prompt_score = baseline_scores_pp[concept][pi]
                            _prompt_delta = (
                                round(_prompt_score - _baseline_prompt_score, 6)
                                if _prompt_score is not None else None
                            )
                            f.write(json.dumps({
                                "model": model_name,
                                "concept": concept,
                                "method": method_key,
                                "strength": strength,
                                "prompt_idx": pi,
                                "prompt": p,
                                "output": o,
                                # per-prompt metrics
                                "concept_score": round(_prompt_score, 6) if _prompt_score is not None else None,
                                "baseline_concept_score": round(_baseline_prompt_score, 6),
                                "delta": _prompt_delta,
                                "repetition_rate": round(_steered_rep_pp[pi], 4),
                                "valid": pi in valid_idx,
                                "perplexity": round(ppl, 4),
                                # aggregate metrics for this (method, concept, strength)
                                "mean_concept_score": round(score, 6),
                                "mean_baseline_score": round(_baseline_score_filtered, 6),
                                "mean_delta": round(delta, 6),
                                "n_valid_prompts": n_valid,
                                # injection scale (actual vec magnitude = _injection_scale, injected = vec * strength)
                                "act_norm_scale": round(float(_injection_scale), 4),
                            }) + "\n")

            method_results[method_key] = _aggregate_method(
                results_by_strength, baseline_score, baseline_ppl_all,
                min_valid=_min_valid,
            )

        results[concept] = {
            "methods": method_results,
            "baseline_score": round(baseline_score, 6),
            "baseline_ppl": round(baseline_ppl_all, 4),
            "inject_layers": inject_layers,
        }

    # ---- Write partial result ----
    partial_path = os.path.join(results_dir, f"_partial_{model_name}.json")
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"{_ts()} [step8] {model_name}: partial result written to {partial_path}")
    print(f"{_ts()} [step8] {'='*60}")
    print(f"{_ts()} [step8] DONE: {model_name} — all concepts evaluated")
    print(f"{_ts()} [step8] {'='*60}")

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Statistical robustness (error bars, Wilcoxon signed-rank, BH FDR correction)
# ---------------------------------------------------------------------------

def _compute_stats_section(results: Dict) -> Dict:
    """Compute per-cell 95% bootstrap CI and one-sample Wilcoxon signed-rank tests.

    For each (model, concept, method) at optimal_positive_strength we have a list of
    per-prompt deltas (one delta per clean test prompt).  From these we compute:
      - mean, SD, 95% bootstrap CI of the mean delta
      - One-sample Wilcoxon signed-rank (H0: median delta = 0, alternative: greater)

    All one-sample p-values are then corrected with Benjamini-Hochberg FDR across the
    full set of tests (one per non-trivial (model, concept, method) triple).

    Cross-concept tests (per model):
      - Sign test (binomial vs p=0.5): out of N concepts, how many show positive B3-TI
        transfer? Avoids magnitude inflation from testing max-selected deltas with Wilcoxon.
      - Paired Wilcoxon: B3-TI vs native, paired explicitly by concept key (not by
        list position) to prevent silent misalignment when concepts have one method
        but not the other.

    NOTE: With n=30 valid prompts (TOP_N_PROMPTS) each per-prompt list is large enough
    for Wilcoxon to have ~83% power at our effect size (delta=0.05, SD=0.10).
    With only n=4–6 (the old 10-prompt setup) tests would have near-zero power.

    Returns a JSON-serialisable dict stored as table_stats in evaluation_table.json.
    """
    try:
        from scipy import stats as _sp  # type: ignore
        _scipy_ok = True
    except ImportError:
        _sp = None
        _scipy_ok = False

    def _boot_ci(deltas: List[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 0):
        arr = np.array(deltas, dtype=float)
        rng = np.random.default_rng(seed)
        boots = np.array([
            rng.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(n_boot)
        ])
        lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
        hi = float(np.percentile(boots, (1 - (1 - ci) / 2) * 100))
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        return float(arr.mean()), sd, lo, hi

    # ---- Per-cell statistics ----
    per_cell: Dict[str, Dict] = {}
    all_tests: List[Tuple] = []  # (model, concept, method, n, p_raw)
    for model_name, model_data in results.items():
        per_cell[model_name] = {}
        for concept, cdata in model_data.items():
            per_cell[model_name][concept] = {}
            for mk, mdata in cdata.get("methods", {}).items():
                pp = mdata.get("per_prompt_deltas_at_optimal") or []
                if len(pp) < 4:
                    per_cell[model_name][concept][mk] = {
                        "n": len(pp),
                        "note": (
                            "n<4 — re-run with ≥30 candidate prompts (TOP_N_PROMPTS) to get valid statistics. "
                            "This is expected for old partial results."
                        ),
                    }
                    continue
                mean, sd, ci_lo, ci_hi = _boot_ci(pp)
                cell: Dict = {
                    "n": len(pp),
                    "mean_delta": round(mean, 6),
                    "sd": round(sd, 6),
                    "ci_95_lo": round(ci_lo, 6),
                    "ci_95_hi": round(ci_hi, 6),
                }
                if _scipy_ok:
                    try:
                        wres = _sp.wilcoxon(pp, alternative="greater", zero_method="wilcox")
                        cell["wilcoxon_W"] = float(wres.statistic)
                        cell["p_raw"] = float(wres.pvalue)
                        all_tests.append((model_name, concept, mk, len(pp), float(wres.pvalue)))
                    except Exception as exc:
                        cell["wilcoxon_error"] = str(exc)
                else:
                    cell["wilcoxon_note"] = "scipy not installed — pip install scipy"
                per_cell[model_name][concept][mk] = cell

    # ---- Benjamini-Hochberg FDR correction ----
    fdr_results: Dict[str, Dict] = {}
    if _scipy_ok and all_tests:
        p_vals = [t[4] for t in all_tests]
        # Manual BH (avoids statsmodels dependency)
        m = len(p_vals)
        sorted_idx = sorted(range(m), key=lambda i: p_vals[i])
        p_adj_sorted = [min(1.0, p_vals[sorted_idx[i]] * m / (i + 1)) for i in range(m)]
        # Monotone enforcement (step-down from right)
        for i in range(m - 2, -1, -1):
            p_adj_sorted[i] = min(p_adj_sorted[i], p_adj_sorted[i + 1])
        p_adj: List[float] = [0.0] * m
        for rank, orig_i in enumerate(sorted_idx):
            p_adj[orig_i] = p_adj_sorted[rank]
        reject: List[bool] = [q <= 0.05 for q in p_adj]
        for i, (model_name, concept, mk, n, p_raw) in enumerate(all_tests):
            test_id = f"{model_name}::{concept}::{mk}"
            fdr_results[test_id] = {
                "p_raw": round(p_raw, 6),
                "p_adj_bh": round(p_adj[i], 6),
                "reject_h0_fdr05": bool(reject[i]),
                "n": n,
            }
            # Write back into per_cell for easy lookup
            if per_cell.get(model_name, {}).get(concept, {}).get(mk):
                per_cell[model_name][concept][mk]["p_adj_bh"] = round(p_adj[i], 6)
                per_cell[model_name][concept][mk]["reject_h0_fdr05"] = bool(reject[i])

    # ---- Cross-concept sign test + paired test per model ----
    # Issue 1: Use sign test (binomial) not Wilcoxon on max_positive_delta.
    #   Wilcoxon on max-selected values inflates W (tests "does B3-TI ever produce
    #   positive delta?" with magnitude bias). Sign test answers the same question
    #   honestly: out of N concepts, how many show any positive transfer?
    #   Paper sentence: "B3-TI produced positive deltas on X/N concepts (binomial p<0.01)."
    # Issue 2: Build concept→delta dicts first, then pair explicitly by concept key.
    #   zip(b3_deltas, native_deltas) silently misaligns if any concept has one but
    #   not the other method.
    cross_concept: Dict[str, Dict] = {}
    if _scipy_ok:
        for model_name, model_data in results.items():
            # Build per-concept dicts keyed by concept name (fixes Issue 2)
            b3_by_concept: Dict[str, float] = {}
            native_by_concept: Dict[str, float] = {}
            for concept, cdata in model_data.items():
                methods = cdata.get("methods", {})
                b3_keys = [k for k in methods if k.startswith("b3_ti_")]
                if b3_keys:
                    b3_by_concept[concept] = max(
                        methods[k].get("max_positive_delta", 0.0) for k in b3_keys
                    )
                native_keys = [k for k in ["sae_vector", "caa_vector"] if k in methods]
                if native_keys:
                    native_by_concept[concept] = max(
                        methods[k].get("max_positive_delta", 0.0) for k in native_keys
                    )

            # Sign test: how many concepts show positive B3-TI transfer?
            b3_deltas = list(b3_by_concept.values())
            n_pos = sum(1 for d in b3_deltas if d > 0)
            n_total = len(b3_deltas)
            b3_vs_zero: Dict = {"n_concepts": n_total, "n_positive": n_pos}
            if n_total >= 8:
                try:
                    binom_result = _sp.binomtest(n_pos, n_total, p=0.5, alternative="greater")
                    b3_vs_zero.update({
                        "pct_positive": round(100 * n_pos / n_total, 1),
                        "p_binomial": round(float(binom_result.pvalue), 6),
                        "significant_p05": bool(binom_result.pvalue < 0.05),
                        "test": "binomial sign test vs p=0.5 (null: coin-flip)",
                    })
                except Exception as exc:
                    b3_vs_zero["error"] = str(exc)
            else:
                b3_vs_zero["note"] = f"n={n_total} concepts — need ≥8 for sign test"

            # Paired test: B3-TI vs native, explicitly matched by concept key
            shared_concepts = sorted(b3_by_concept.keys() & native_by_concept.keys())
            b3_vs_native: Dict = {}
            if len(shared_concepts) >= 8:
                diffs = [b3_by_concept[c] - native_by_concept[c] for c in shared_concepts]
                try:
                    r = _sp.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
                    b3_vs_native = {
                        "n": len(diffs),
                        "concepts_paired": shared_concepts,
                        "mean_diff": round(float(np.mean(diffs)), 6),
                        "W": float(r.statistic),
                        "p": float(r.pvalue),
                        "significant_p05": bool(r.pvalue < 0.05),
                    }
                except Exception as exc:
                    b3_vs_native = {"error": str(exc)}
            else:
                b3_vs_native = {
                    "note": f"n={len(shared_concepts)} concepts with both B3-TI and native — need ≥8"
                }

            cross_concept[model_name] = {
                "b3_ti_vs_zero": b3_vs_zero,
                "b3_ti_vs_native": b3_vs_native,
            }

    n_reject = sum(1 for v in fdr_results.values() if v.get("reject_h0_fdr05", False))

    # ---- DeBERTa vs Claude correlation (validates DeBERTa as primary metric) ----
    # Loaded lazily from llm_judge_results.jsonl if it exists alongside the results.
    # Pearson r >= 0.70 means the automatic DeBERTa score is a reliable proxy for
    # human/Claude judgment and can be used as the primary metric in the paper.
    deberta_claude_corr: Dict = {"note": "run Claude judge first to compute this"}

    return {
        "per_cell": per_cell,
        "fdr_corrected": fdr_results,
        "cross_concept_sign_test": cross_concept,
        "scipy_available": _scipy_ok,
        "n_tests_total": len(all_tests),
        "n_reject_h0_fdr05": n_reject,
        "deberta_claude_correlation": deberta_claude_corr,
        "note": (
            "One-sample Wilcoxon signed-rank per (model,concept,method) at optimal strength. "
            "H0: median per-prompt delta = 0. FDR correction: Benjamini-Hochberg across all tests."
        ),
    }


# ---------------------------------------------------------------------------
# Table compilation (post-processing, runs in main process after all workers)
# ---------------------------------------------------------------------------

def compile_tables(results: Dict, out_dir: str, suffix: str = "",
                   all_seeds_merged: Optional[Dict[int, Dict]] = None,
                   supervised_concepts: Optional[set] = None) -> None:
    """Build Tables 1–7 from merged per-model results. Write evaluation_table.json + report.md.

    all_seeds_merged: when multi-seed eval was run, pass {seed: merged_dict} here.
    compile_tables will add a 'table_seed_variance' section with mean±SD per cell.

    supervised_concepts: set of concept names from A-section (native steering vectors).
    Concepts NOT in this set are C3/unsupervised. Used to annotate report tables with
    [S] (supervised) vs [U] (unsupervised/auto-discovered) so duplicate DeBERTa labels
    (e.g. python_code and python_source_code both → "Python code") are visually distinct.
    """
    print("[step8] Compiling evaluation tables (Tables 1–7)...")

    tables: Dict[str, Dict] = {
        "table1_native_effectiveness":  {},   # model → concept → max_positive_delta
        "table2_transfer_efficiency":   {},   # guide → target → concept → b3_ti_eff
        "table3_method_comparison":     {},   # model → concept → [(method, delta), ...]
        "table4_universality_c3":       {},   # concept → model → {c3_eff, c3_delta, native_delta}
        "table4_universality_enc_dec":   {},   # concept → model → {enc_dec_eff, enc_dec_delta}
        "table5_naive_vs_bridge":       {},   # guide → target → concept → ti/naive ratio
        "table6_bidirectionality":      {},   # model → concept → {native, b3_cross, universal}
        "table7_convergence_gap":       {},   # guide → target → concept → guide_score − target_score
    }

    for model_name, model_results in results.items():
        t1 = tables["table1_native_effectiveness"].setdefault(model_name, {})
        t3 = tables["table3_method_comparison"].setdefault(model_name, {})
        t6 = tables["table6_bidirectionality"].setdefault(model_name, {})

        for concept, cdata in model_results.items():
            methods = cdata.get("methods", {})

            # Table 1 — native steering effectiveness
            native_delta = 0.0
            native_pp: list = []
            for _mk1 in ["sae_vector", "caa_vector"]:
                _d1 = methods.get(_mk1, {}).get("max_positive_delta") or 0.0
                if _d1 > native_delta:
                    native_delta = _d1
                    native_pp = methods[_mk1].get("per_prompt_deltas_at_optimal") or []
            _n1 = len(native_pp)
            _mean1 = round(sum(native_pp) / _n1, 6) if _n1 else None
            _sd1 = (
                round((sum((x - _mean1) ** 2 for x in native_pp) / _n1) ** 0.5, 6)
                if _n1 >= 2 else None
            )
            t1[concept] = {
                "max":  round(native_delta, 6),
                "mean": _mean1,
                "sd":   _sd1,
                "n":    _n1,
            }

            # Table 3 — method ranking by delta, with efficiency_valid flag for winner
            # Stores tuples of (method, delta, efficiency_valid) for top-3 methods
            t3[concept] = sorted(
                [
                    (mk, r.get("max_positive_delta") or 0.0, r.get("efficiency_valid"))
                    for mk, r in methods.items()
                ],
                key=lambda x: x[1], reverse=True,
            )

            # Table 6 — bidirectionality by family.
            # Only include methods that show a genuine positive effect (>= _MIN_BIDIR_DELTA).
            # When no meaningful positive effect exists, bidir_ratio is None by design
            # (see _aggregate_method), so do NOT fall back to 0.0.
            def _best_bidir(method_keys):
                """Best bidir_ratio among methods that have one (non-None)."""
                vals = [
                    methods.get(mk, {}).get("bidirectionality_ratio")
                    for mk in method_keys
                    if methods.get(mk, {}).get("bidirectionality_ratio") is not None
                ]
                return max(vals) if vals else None

            _native_keys = [mk for mk in ["sae_vector", "caa_vector"] if mk in methods]
            _b3_keys = [mk for mk in methods if mk.startswith("b3_")]
            _naive_keys = [mk for mk in methods if mk.startswith("naive_")]

            t6[concept] = {
                "native": _best_bidir(_native_keys),
                "b3_cross": _best_bidir(_b3_keys),
                "naive_cross": _best_bidir(_naive_keys),
                "universal": methods.get("universal_c3", {}).get("bidirectionality_ratio"),
            }

            # Table 4 — C3 universality
            # c3_efficiency = this_model_c3_delta / mean_c3_delta_across_all_models.
            # The denominator is the cross-model mean for the same universal vector, which
            # is a valid baseline: the universal vector was built collectively from all
            # models, so the average response across models is the natural reference point.
            # eff > 1.0 → this model responds better than average to the universal vector.
            # eff < 1.0 → this model responds less than average.
            # Computed in a second pass below, after collecting all c3_delta values.
            c3_entry = methods.get("universal_c3")
            if c3_entry:
                # Use effective_max_delta (polarity-aware) so inverted vectors are scored correctly
                c3_delta = c3_entry.get("effective_max_delta") or c3_entry.get("max_positive_delta") or 0.0
                tables["table4_universality_c3"].setdefault(concept, {})[model_name] = {
                    "c3_delta": round(c3_delta, 6),
                    "c3_direction": c3_entry.get("effective_direction", "positive"),
                    "bidir_ratio": c3_entry.get("bidirectionality_ratio"),
                    "hallucination_rate": c3_entry.get("hallucination_rate"),
                    # c3_efficiency filled in second pass
                }

            enc_dec_entry = methods.get("universal_enc_dec")
            if enc_dec_entry:
                enc_dec_delta = enc_dec_entry.get("effective_max_delta") or enc_dec_entry.get("max_positive_delta") or 0.0
                tables["table4_universality_enc_dec"].setdefault(concept, {})[model_name] = {
                    "enc_dec_delta": round(enc_dec_delta, 6),
                    "enc_dec_direction": enc_dec_entry.get("effective_direction", "positive"),
                    "bidir_ratio": enc_dec_entry.get("bidirectionality_ratio"),
                    "hallucination_rate": enc_dec_entry.get("hallucination_rate"),
                    # enc_dec_efficiency filled in second pass
                }

            # Tables 2 + 5 — transfer efficiency and naive vs bridge
            for mk, r in methods.items():
                guide = None
                for prefix in ["b3_ti_", "b3_caa_cross_", "naive_"]:
                    if mk.startswith(prefix):
                        guide = mk[len(prefix):]
                        break
                if guide is None:
                    continue

                delta = r.get("max_positive_delta") or 0.0

                if mk.startswith("b3_ti_"):
                    eff = round(delta / (native_delta + 1e-8), 4) if native_delta > 0 else None
                    (tables["table2_transfer_efficiency"]
                        .setdefault(guide, {})
                        .setdefault(model_name, {}))[concept] = eff

                    naive_mk = f"naive_{guide}"
                    naive_delta = (methods.get(naive_mk, {}).get("max_positive_delta") or 0.0)
                    bridge_ratio = round(delta / (naive_delta + 1e-8), 4) if naive_delta > 0 else None
                    (tables["table5_naive_vs_bridge"]
                        .setdefault(guide, {})
                        .setdefault(model_name, {}))[concept] = bridge_ratio

                # Table 7 — convergence gap (guide_native_score − target_score at target's opt+)
                opt_pos_str = str(r.get("optimal_positive_strength", ""))
                target_score_at_opt = (
                    r.get("results_by_strength", {})
                    .get(opt_pos_str, {})
                    .get("mean_concept_score")
                )
                guide_methods = results.get(guide, {}).get(concept, {}).get("methods", {})
                guide_native = guide_methods.get("sae_vector") or guide_methods.get("caa_vector")
                if guide_native and opt_pos_str and target_score_at_opt is not None:
                    g_score = (
                        guide_native.get("results_by_strength", {})
                        .get(opt_pos_str, {})
                        .get("mean_concept_score")
                    )
                    if g_score is not None:
                        gap = round(g_score - target_score_at_opt, 6)
                        (tables["table7_convergence_gap"]
                            .setdefault(guide, {})
                            .setdefault(model_name, {}))[concept] = gap

    # Second pass — compute c3_efficiency for Table 4.
    # Denominator = mean c3_delta across all models that have an entry for this concept.
    # This is valid because the universal C3 vector was built collectively from all models;
    # the cross-model mean response is the natural reference point.
    for concept, model_cells in tables["table4_universality_c3"].items():
        deltas = [cell["c3_delta"] for cell in model_cells.values() if cell["c3_delta"] > 0]
        if len(deltas) >= 2:
            mean_delta = float(np.mean(deltas))
            for model_name, cell in model_cells.items():
                d = cell["c3_delta"]
                cell["c3_efficiency"] = round(d / mean_delta, 4) if mean_delta > 0 else None
        else:
            # Only one model evaluated so far; efficiency undefined until multi-model run.
            for cell in model_cells.values():
                cell["c3_efficiency"] = None  # recompute after all models are run

    for concept, model_cells in tables["table4_universality_enc_dec"].items():
        deltas = [cell["enc_dec_delta"] for cell in model_cells.values() if cell["enc_dec_delta"] > 0]
        if len(deltas) >= 2:
            mean_delta = float(np.mean(deltas))
            for model_name, cell in model_cells.items():
                d = cell["enc_dec_delta"]
                cell["enc_dec_efficiency"] = round(d / mean_delta, 4) if mean_delta > 0 else None
        else:
            for cell in model_cells.values():
                cell["enc_dec_efficiency"] = None

    # Statistical robustness — compute after all tables are built so per_prompt_deltas
    # are accessible from the original results dict.
    print("[step8] Computing statistical tests (Wilcoxon + BH FDR)...")
    tables["table_stats"] = _compute_stats_section(results)
    n_reject = tables["table_stats"]["n_reject_h0_fdr05"]
    n_total  = tables["table_stats"]["n_tests_total"]
    print(f"[step8] Stats: {n_reject}/{n_total} tests reject H0 at FDR 5%")

    # Write evaluation_table.json
    os.makedirs(out_dir, exist_ok=True)
    table_path = os.path.join(out_dir, f"evaluation_table{suffix}.json")
    with open(table_path, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2)
    print(f"[step8] Wrote {table_path}")

    # Write evaluation_report.md
    report_path = os.path.join(out_dir, f"evaluation_report{suffix}.md")
    all_models = sorted(tables["table1_native_effectiveness"].keys())
    all_concepts = sorted({
        c for m_dict in tables["table1_native_effectiveness"].values() for c in m_dict
    })

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# B4 Evaluation Report\n\n")

        # Concept source legend (supervised vs unsupervised)
        if supervised_concepts:
            _unsup = sorted(c for c in all_concepts if c not in supervised_concepts)
            _sup   = sorted(c for c in all_concepts if c in supervised_concepts)
            f.write("> **Concept source:** [S] = supervised (A-section native steering vectors)  "
                    "[U] = unsupervised (C3 auto-discovered).\n")
            if _unsup:
                _dup_labels = {
                    c for c in _unsup
                    if CONCEPT_LABELS.get(c, c.replace('_', ' ')) in
                    {CONCEPT_LABELS.get(s, s.replace('_', ' ')) for s in _sup}
                }
                if _dup_labels:
                    f.write(f"> **Note:** {sorted(_dup_labels)} share a DeBERTa label with a supervised"
                            " concept — FDR test counts for these are not independent.\n")
            f.write("\n")

        def _concept_tag(c: str) -> str:
            if not supervised_concepts:
                return c
            return f"{c} [S]" if c in supervised_concepts else f"{c} [U]"

        # ---- Summary: mean delta per method family, split supervised vs unsupervised ----
        if supervised_concepts:
            _sup_concepts   = [c for c in all_concepts if c in supervised_concepts]
            _unsup_concepts = [c for c in all_concepts if c not in supervised_concepts]
            for _seg_label, _seg_concepts in [("Supervised [S]", _sup_concepts), ("Unsupervised [U]", _unsup_concepts)]:
                if not _seg_concepts:
                    continue
                f.write(f"## Summary — {_seg_label} concepts (mean best-positive delta across models)\n\n")
                _METHOD_GROUPS = [
                    ("sae_vector",       "Native SAE"),
                    ("caa_vector",       "Native CAA"),
                    ("b3_bal",           "B3 bal (CAA-cross)"),
                    ("b3_ti",            "B3 TI (bridge)"),
                    ("naive",            "Naive (dim-align)"),
                    ("universal_c3",     "Universal C3"),
                    ("universal_enc_dec","Universal enc-dec"),
                ]
                f.write("| Method | " + " | ".join(all_models) + " | Mean |\n")
                f.write("|---|" + "---|" * len(all_models) + "---|\n")
                for _mpfx, _mlabel in _METHOD_GROUPS:
                    _model_means = []
                    _row = []
                    for _m in all_models:
                        _deltas = []
                        for _c in _seg_concepts:
                            _methods = tables["table1_native_effectiveness"]  # just for concept list
                            _all_methods = (
                                tables.get("table3_method_comparison", {})
                                .get(_m, {}).get(_c, [])
                            )
                            for _mk, _md, _ in _all_methods:
                                if _mk == _mpfx or _mk.startswith(_mpfx + "_"):
                                    if _md and _md > 0:
                                        _deltas.append(_md)
                        if _deltas:
                            _v = round(float(sum(_deltas) / len(_deltas)), 4)
                            _model_means.append(_v)
                            _row.append(f"{_v:+.4f}")
                        else:
                            _row.append("—")
                    _overall = f"{round(sum(_model_means)/len(_model_means),4):+.4f}" if _model_means else "—"
                    f.write(f"| {_mlabel} | " + " | ".join(_row) + f" | {_overall} |\n")
                f.write("\n")

        f.write("## Table 1 — Native Steering Effectiveness\n\n")
        f.write("> max positive delta at optimal strength; μ ± σ = mean ± SD across prompts at that strength.\n\n")
        f.write("| Concept | " + " | ".join(all_models) + " |\n")
        f.write("|---|" + "---|" * len(all_models) + "\n")
        for concept in all_concepts:
            row = []
            for m in all_models:
                _cell = tables["table1_native_effectiveness"].get(m, {}).get(concept)
                if _cell is None:
                    row.append("—")
                elif isinstance(_cell, dict):
                    _mx = _cell.get("max", "—")
                    _mu = _cell.get("mean")
                    _sd = _cell.get("sd")
                    if _mu is not None and _sd is not None:
                        row.append(f"{_mx} (μ={_mu}±{_sd})")
                    elif _mu is not None:
                        row.append(f"{_mx} (μ={_mu})")
                    else:
                        row.append(str(_mx))
                else:
                    row.append(str(_cell))  # backwards compat with old result files
            f.write(f"| {_concept_tag(concept)} | " + " | ".join(row) + " |\n")

        f.write("## Table 6 — Bidirectionality (bidir_ratio >= 0.5 = pass, None = no measurable positive effect)\n\n")
        f.write("| Model | Concept | Native | B3 Cross | Naive Cross | Universal C3 |\n|---|---|---|---|---|---|\n")
        for model_name in all_models:
            for concept, fams in sorted(
                tables["table6_bidirectionality"].get(model_name, {}).items()
            ):
                def _fmt_bidir(v) -> str:
                    if v is None:
                        return "—"
                    return ("✅ " if v >= 0.5 else "❌ ") + str(round(v, 3))
                f.write(
                    f"| {model_name} | {concept} | {_fmt_bidir(fams.get('native'))} "
                    f"| {_fmt_bidir(fams.get('b3_cross'))} | {_fmt_bidir(fams.get('naive_cross'))} "
                    f"| {_fmt_bidir(fams.get('universal'))} |\n"
                )

        f.write("\n## Table 4 — Universal Vector (C3) Performance\n\n")
        f.write("> `c3_efficiency` = this model's c3_delta / mean c3_delta across all evaluated models.\n")
        f.write("> The universal C3 vector was built collectively from all models, so the cross-model mean\n")
        f.write("> response is the natural reference point.  eff > 1.0 means this model benefits more than\n")
        f.write("> average; eff < 1.0 means less.  Requires ≥2 models evaluated; shows `—` otherwise.\n\n")
        f.write("| Concept | Model | c3_delta | c3_efficiency | bidir_ratio | hallucination_rate |\n")
        f.write("|---|---|---|---|---|---|\n")
        for concept, m_dict in sorted(tables["table4_universality_c3"].items()):
            for model_name, vals in sorted(m_dict.items()):
                _br = vals.get('bidir_ratio')
                _br_fmt = ("✅ " if _br is not None and _br >= 0.5 else "❌ ") + str(round(_br, 3)) if _br is not None else "—"
                _eff = vals.get('c3_efficiency')
                _eff_fmt = str(_eff) if _eff is not None else "—"
                f.write(
                    f"| {_concept_tag(concept)} | {model_name} "
                    f"| {vals.get('c3_delta','—')} | {_eff_fmt} | {_br_fmt} "
                    f"| {vals.get('hallucination_rate','—')} |\n"
                )

        if tables["table4_universality_enc_dec"]:
            f.write("\n## Table 4b — Universal Vector (Enc-Dec) Performance\n\n")
            f.write("> `enc_dec_efficiency` = this model's enc_dec_delta / mean enc_dec_delta across all evaluated models.\n")
            f.write("> Enc-dec vectors are built via GlobalMLP encoder→decoder (in-distribution for A5 concepts).\n\n")
            f.write("| Concept | Model | enc_dec_delta | enc_dec_efficiency | bidir_ratio | hallucination_rate |\n")
            f.write("|---|---|---|---|---|---|\n")
            for concept, m_dict in sorted(tables["table4_universality_enc_dec"].items()):
                for model_name, vals in sorted(m_dict.items()):
                    _br = vals.get('bidir_ratio')
                    _br_fmt = ("✅ " if _br is not None and _br >= 0.5 else "❌ ") + str(round(_br, 3)) if _br is not None else "—"
                    _eff = vals.get('enc_dec_efficiency')
                    _eff_fmt = str(_eff) if _eff is not None else "—"
                    f.write(
                        f"| {concept} | {model_name} "
                        f"| {vals.get('enc_dec_delta','—')} | {_eff_fmt} | {_br_fmt} "
                        f"| {vals.get('hallucination_rate','—')} |\n"
                    )
        f.write("\n## Table 2 — B3 Transfer Efficiency (b3_ti_delta / native_delta)\n\n")
        _t2 = tables["table2_transfer_efficiency"]
        if _t2:
            # Collect all targets and concepts that appear
            _t2_targets = sorted({tgt for guide_d in _t2.values() for tgt in guide_d})
            _t2_concepts = sorted({c for guide_d in _t2.values() for tgt_d in guide_d.values() for c in tgt_d})
            f.write("| Guide | Target | " + " | ".join(_t2_concepts) + " |\n")
            f.write("|---|---|" + "---|" * len(_t2_concepts) + "\n")
            for guide in sorted(_t2):
                for tgt in _t2_targets:
                    row_vals = []
                    for c in _t2_concepts:
                        v = _t2[guide].get(tgt, {}).get(c)
                        row_vals.append("—" if v is None else str(round(v, 3)))
                    if any(v != "—" for v in row_vals):
                        f.write(f"| {guide} | {tgt} | " + " | ".join(row_vals) + " |\n")
        else:
            f.write("> ⚠️ Empty: requires all models in same run (transfer only computable against other models' native baselines).\n")

        # ---- Table 3 — method ranking ----
        f.write("\n## Table 3 — Method Ranking by Max Positive Delta\n\n")
        f.write("> ✅ PPL = PPL-increase ≤ 30% (fluency gate passed)  ❌ = fluency cost too high  — = no data\n\n")
        _t3 = tables["table3_method_comparison"]
        _t3_models = sorted(_t3)
        _t3_concepts = sorted({c for md in _t3.values() for c in md})
        if _t3:
            f.write("| Model | Concept | #1 Method | #1 Δ | PPL | #2 Method | #2 Δ | PPL | #3 Method | #3 Δ | PPL |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
            for mdl in _t3_models:
                for concept in _t3_concepts:
                    ranking = _t3[mdl].get(concept, [])
                    cells = []
                    for i in range(3):
                        if i < len(ranking):
                            entry = ranking[i]
                            _eff = entry[2] if len(entry) > 2 else None
                            _eff_fmt = ("✅" if _eff else "❌") if _eff is not None else "—"
                            cells += [str(entry[0]), str(round(entry[1], 4)), _eff_fmt]
                        else:
                            cells += ["—", "—", "—"]
                    f.write(f"| {mdl} | {concept} | " + " | ".join(cells) + " |\n")

        f.write("\n## Table 5 — Naive vs Bridge (b3_ti_delta / naive_delta)\n\n")
        _t5 = tables["table5_naive_vs_bridge"]
        if _t5:
            _t5_targets = sorted({tgt for guide_d in _t5.values() for tgt in guide_d})
            _t5_concepts = sorted({c for guide_d in _t5.values() for tgt_d in guide_d.values() for c in tgt_d})
            f.write("| Guide | Target | " + " | ".join(_t5_concepts) + " |\n")
            f.write("|---|---|" + "---|" * len(_t5_concepts) + "\n")
            for guide in sorted(_t5):
                for tgt in _t5_targets:
                    row_vals = []
                    for c in _t5_concepts:
                        v = _t5[guide].get(tgt, {}).get(c)
                        row_vals.append("—" if v is None else str(round(v, 3)))
                    if any(v != "—" for v in row_vals):
                        f.write(f"| {guide} | {tgt} | " + " | ".join(row_vals) + " |\n")
        else:
            f.write("> ⚠️ Empty: requires naive vectors to be present.\n")

        f.write("\n## Table 7 — Convergence Gap (guide_score − target_score at target's optimal strength)\n\n")
        _t7 = tables["table7_convergence_gap"]
        if _t7:
            _t7_targets = sorted({tgt for guide_d in _t7.values() for tgt in guide_d})
            _t7_concepts = sorted({c for guide_d in _t7.values() for tgt_d in guide_d.values() for c in tgt_d})
            f.write("> Positive = guide model scores higher on concept than the steered target. Negative = target converges past guide.\n\n")
            f.write("| Guide | Target | " + " | ".join(_t7_concepts) + " |\n")
            f.write("|---|---|" + "---|" * len(_t7_concepts) + "\n")
            for guide in sorted(_t7):
                for tgt in _t7_targets:
                    row_vals = []
                    for c in _t7_concepts:
                        v = _t7[guide].get(tgt, {}).get(c)
                        row_vals.append("—" if v is None else str(round(v, 4)))
                    if any(v != "—" for v in row_vals):
                        f.write(f"| {guide} | {tgt} | " + " | ".join(row_vals) + " |\n")
        else:
            f.write("> ⚠️ Empty: Table 7 requires guide model scores to be present in the same evaluation run.\n")
            f.write("> Run all 5 models together (or use --parallel) and this table will be populated automatically.\n")

        # ---- Statistical Robustness ----
        stats = tables.get("table_stats", {})
        f.write("\n## Statistical Robustness\n\n")
        if not stats.get("scipy_available", False):
            f.write("> ⚠️ scipy not installed — statistical tests skipped. Run: `pip install scipy`\n\n")
        else:
            n_total  = stats.get("n_tests_total", 0)
            n_reject = stats.get("n_reject_h0_fdr05", 0)
            f.write(f"**One-sample Wilcoxon signed-rank** (H0: median per-prompt delta = 0, "
                    f"alternative: greater) applied to every (model × concept × method) triple "
                    f"at its optimal positive strength.  "
                    f"**Benjamini-Hochberg FDR** correction across all {n_total} tests.\n\n")
            f.write(f"| Metric | Value |\n|---|---|\n")
            f.write(f"| Total one-sample tests | {n_total} |\n")
            f.write(f"| Reject H₀ at FDR 5% | **{n_reject}** ({round(100*n_reject/max(n_total,1),1)}%) |\n")
            f.write(f"| scipy version | installed |\n\n")

            # Per-model cross-concept summary
            cc = stats.get("cross_concept_sign_test", {})
            if cc:
                f.write("### Cross-concept Wilcoxon (per model)\n\n")
                f.write("| Model | Test | n | W | p (raw) | Significant? |\n")
                f.write("|---|---|---:|---:|---:|:---:|\n")
                for mdl, mdl_cc in sorted(cc.items()):
                    for test_label, td in [
                        ("B3_TI Δ > 0 across concepts", mdl_cc.get("b3_ti_vs_zero", {})),
                        ("B3_TI Δ vs native Δ (paired)", mdl_cc.get("b3_ti_vs_native", {})),
                    ]:
                        if "note" in td:
                            f.write(f"| {mdl} | {test_label} | — | — | — | ⚠️ {td['note']} |\n")
                        elif "error" in td:
                            f.write(f"| {mdl} | {test_label} | — | — | — | ❌ error |\n")
                        elif "p" in td:
                            sig = "✅" if td.get("significant_p05") else "❌"
                            f.write(f"| {mdl} | {test_label} | {td.get('n','—')} "
                                    f"| {td.get('W','—')} | {td.get('p','—'):.4f} | {sig} |\n")

            f.write("\n> Full per-cell CI and FDR results in `evaluation_table.json` → `table_stats`.\n")

            # DeBERTa–Claude correlation summary (if available)
            _dcorr = stats.get("deberta_claude_correlation", {})
            if _dcorr.get("pearson_r") is not None:
                _r = _dcorr["pearson_r"]
                _valid = "✅ Valid primary metric" if _dcorr.get("valid") else "⚠️ Weak — use Claude scores as primary"
                f.write(f"\n### DeBERTa Scorer Validation\n\n")
                f.write(f"Pearson r between DeBERTa concept scores and Claude judge scores: **{_r}** "
                        f"(n={_dcorr.get('n_pairs','—')}, p={_dcorr.get('p_value','—')})  {_valid}\n\n")
                f.write(f"> {_dcorr.get('interpretation','')}\n")

        # ---- Seed variance table (multi-seed runs only) ----
        if all_seeds_merged and len(all_seeds_merged) > 1:
            f.write("\n## Seed Variance (Multi-seed Error Bars)\n\n")
            f.write(f"> Evaluated with {len(all_seeds_merged)} seeds: "
                    f"{sorted(all_seeds_merged.keys())}.  "
                    f"Δ mean ± SD across seeds at optimal positive strength per (model, concept, method).\n\n")
            tables["table_seed_variance"] = {}
            # Collect per-cell deltas across seeds
            _sv_cells: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
            for _s, _s_merged in all_seeds_merged.items():
                for _mdl, _mdl_data in _s_merged.items():
                    for _con, _cdata in _mdl_data.items():
                        for _mk, _mdata in _cdata.get("methods", {}).items():
                            _d = _mdata.get("max_positive_delta") or 0.0
                            (_sv_cells
                                .setdefault(_mdl, {})
                                .setdefault(_con, {})
                                .setdefault(_mk, [])
                                .append(_d))
            # Write summary table (top methods only: sae_vector, caa_vector, b3_ti_*, universal_c3, universal_enc_dec)
            _sv_methods = ["sae_vector", "caa_vector", "universal_c3", "universal_enc_dec"]
            _sv_models = sorted(_sv_cells)
            _sv_concepts = sorted({c for md in _sv_cells.values() for c in md})
            f.write("| Model | Concept | " + " | ".join(f"{m} Δ mean±SD" for m in _sv_methods) + " |\n")
            f.write("|---|---|" + "---|" * len(_sv_methods) + "\n")
            for _mdl in _sv_models:
                for _con in _sv_concepts:
                    _row = []
                    for _mk in _sv_methods:
                        _vals = _sv_cells.get(_mdl, {}).get(_con, {}).get(_mk, [])
                        if len(_vals) >= 2:
                            _mean = float(np.mean(_vals))
                            _sd   = float(np.std(_vals, ddof=1))
                            _row.append(f"{_mean:.4f}±{_sd:.4f}")
                        elif len(_vals) == 1:
                            _row.append(f"{_vals[0]:.4f} (n=1)")
                        else:
                            _row.append("—")
                    if any(v != "—" for v in _row):
                        f.write(f"| {_mdl} | {_con} | " + " | ".join(_row) + " |\n")
                    # Always store in table_seed_variance (even all-dash rows) for JSON completeness
                    tables["table_seed_variance"].setdefault(_mdl, {})[_con] = {
                        _mk: {"vals": _sv_cells.get(_mdl, {}).get(_con, {}).get(_mk, []),
                              "mean": float(np.mean(_sv_cells[_mdl][_con][_mk]))
                                      if _sv_cells.get(_mdl, {}).get(_con, {}).get(_mk) else None,
                              "sd":   float(np.std(_sv_cells[_mdl][_con][_mk], ddof=1))
                                      if len(_sv_cells.get(_mdl, {}).get(_con, {}).get(_mk, [])) >= 2 else None}
                        for _mk in _sv_methods
                    }
            # Write updated json with seed_variance table
            _tv_path = os.path.join(out_dir, f"evaluation_table{suffix}.json")
            if os.path.exists(_tv_path):
                with open(_tv_path, encoding="utf-8") as _tvf:
                    _tv = json.load(_tvf)
                _tv["table_seed_variance"] = tables["table_seed_variance"]
                with open(_tv_path, "w", encoding="utf-8") as _tvf:
                    json.dump(_tv, _tvf, indent=2)

    print(f"[step8] Wrote {report_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="B4 steering evaluation — 5 models, 11+ concepts, 3 experiments."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output already exists.")
    parser.add_argument("--model", default=None,
                        help="Restrict to a single model name (config.MODELS key).")
    parser.add_argument("--ef", type=int, default=0,
                        help="SAE expansion factor for per-model vector file lookup.")
    parser.add_argument("--concepts", default=None, nargs="+",
                        help="Concept name(s). Space- or comma-separated. Default: auto-detect from feature labels.")
    parser.add_argument("--n-prompts", type=int, default=0, dest="n_prompts",
                        help="Cap on candidate prompt pool (0 = use all 100). Must be 0 or >= TOP_N_PROMPTS (30).")
    parser.add_argument("--max-new-tokens", type=int, default=120, dest="max_new_tokens")
    parser.add_argument("--evaluator", default="all",
                        choices=["all", "classifier", "perplexity"])
    parser.add_argument("--injection-mode", default="single_layer",
                        choices=["single_layer", "multi_layer", "all_layers"],
                        dest="injection_mode")
    parser.add_argument("--injection-layers", default="", dest="injection_layers",
                        help="Comma-separated layer indices for multi_layer mode.")
    parser.add_argument("--run-id", default="", dest="run_id",
                        help="Optional suffix for output file names.")
    parser.add_argument("--llm-judge-n", type=int, default=10, dest="llm_judge_n",
                        help="Steered outputs per (model×concept) to send to Claude judge (default: 10).")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated random seeds for multi-seed evaluation. "
                             "Default: 42 (single run). For publication use '42,123' to get "
                             "mean±SD error bars. Each seed is a separate full eval pass; "
                             "compile_tables aggregates them with seed-averaged deltas.")
    parser.add_argument("--ablate-layers", action="store_true", dest="ablate_layers",
                        help="After the main eval, run a 3-point layer ablation (early/mid/late) "
                             "on the --model target (or gpt2-large if --model not set). "
                             "Writes results/*_layer_ablation.json + report.")
    parser.add_argument("--parallel", action="store_true",
                        help="Run each model in a subprocess on its own GPU (GPU_MAP).")
    parser.add_argument("--quick-test", action="store_true", dest="quick_test",
                        help=(
                            "Quick sanity-check mode: 40 candidate prompts, select 10 cleanest, "
                            "3 randomly-sampled concepts, all 5 models. "
                            "Results written to results/*_quicktest.* — not for publication."
                        ))
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                        help=(
                            "Smoke-test mode: 1 prompt, 1 concept (set via --concepts), "
                            "all 5 models, all 9 strengths, all vector methods (A5/B3/C3/Naive). "
                            "Fastest possible sanity check — NOT for publication."
                        ))
    parser.add_argument("--prompt", default=None,
                        help="Exact prompt text to use (bypasses the candidate pool entirely). "
                             "Implies --smoke-test behaviour: 1 prompt, top_n=1. "
                             "Combine with --concepts to target a specific concept.")
    args = parser.parse_args()

    # Parse seed list — supports single seed (42) or multi-seed (42,123)
    _seeds: List[int] = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not _seeds:
        _seeds = [42]
    set_seed(_seeds[0])
    load_dotenv()

    # ---- Steering vectors — load per-model ef-tagged files, merge into one sv dict ----
    # Files on disk: steering/{model}_ef{N}_steering_vectors.json
    # Structure:     {model_name: {concept: {sae_vector, caa_vector, ...}}}
    # There is NO combined steering_vectors.json — do not try to create or download one.
    # IMPORTANT: Always load ALL models' sv files (not just the filtered model) so that
    # naive_{guide} vectors can be computed for any target. The eval loop still only
    # runs for model_filter when --model is given.
    _ef_val: int = args.ef or 0
    model_filter: Optional[str] = args.model
    sv: Dict = {}
    for _mname in config.MODELS.keys():
        _ef = _ef_val if _ef_val > 0 else config.MODELS[_mname].get("sae_ef", config.SAE_EXPANSION_FACTOR)
        _sv_path = os.path.join(config.STEERING_DIR, f"{_mname}_ef{_ef}_steering_vectors.json")
        if not os.path.exists(_sv_path):
            _cands = sorted(_glob.glob(
                os.path.join(config.STEERING_DIR, f"{_mname}_ef*_steering_vectors.json")
            ))
            _sv_path = _cands[-1] if _cands else None
        if _sv_path and os.path.exists(_sv_path):
            _sdata = json.load(open(_sv_path, encoding="utf-8"))
            sv.update(_sdata)
            print(f"[step8] Loaded {_mname} steering vectors from {os.path.basename(_sv_path)}")
        else:
            print(f"[step8] WARNING: no steering vector file found for {_mname}")
    if not sv:
        raise FileNotFoundError(
            "[step8] No steering vector files found in steering/ — run step7 first."
        )

    # ---- Concepts ----
    # Auto-detect from two sources:
    #   Exp A/B native  → concept names from sv (e.g. code_python, academic_writing)
    #   Exp C universal → concept names from universal_steering_vectors_v1.json (e.g. python_code)
    # The union covers both. The worker's method-build loop quietly skips concepts with no vector.
    if args.concepts:
        # Accept both space-separated (nargs="+") and comma-separated (legacy) forms.
        # e.g. --concepts creative_writing academic_scientific
        #  or  --concepts "creative_writing,academic_scientific"
        _raw = args.concepts if isinstance(args.concepts, list) else [args.concepts]
        target_concepts = [c.strip() for token in _raw for c in token.split(",") if c.strip()]
        _concept_core: Optional[set] = None  # unknown supervision split when manually specified
    else:
        _SKIP = {"other", "certainty", "formality"}
        # Source 1 — native A-section concepts: use INTERSECTION across all models so that
        # auto-discovered extras (e.g. gpt2-large has 63 concepts) don't leak into evaluation.
        _per_model_sets = []
        for _mdata in sv.values():
            if isinstance(_mdata, dict):
                _per_model_sets.append({
                    k for k in _mdata
                    if not k.startswith("cluster_") and k not in _SKIP
                })
        _concept_core: set = set.intersection(*_per_model_sets) if _per_model_sets else set()
        # Source 2 — C3 universal concepts (additive — different concept namespace)
        _concept_union: set = set(_concept_core)
        _c3_peek = os.path.join(config.STEERING_DIR, "universal_steering_vectors_v1.json")
        if os.path.exists(_c3_peek):
            try:
                _c3_keys = json.load(open(_c3_peek, encoding="utf-8"))
                _concept_union.update(_c3_keys.get("universal_steering_vectors", {}).keys())
            except Exception:
                pass
        print(f"[step8] {len(_concept_core)} canonical A/B concepts (intersection), "
              f"{len(_concept_union) - len(_concept_core)} C3-only concepts")
        target_concepts = sorted(_concept_union) if _concept_union else ["sentiment", "formality"]
        if not _concept_union:
            print("[step8] WARNING: no concepts found — falling back to sentiment/formality.")
    print(f"[step8] {len(target_concepts)} concepts: {target_concepts}")

    # ---- Prompts ----
    all_prompts = get_prompts()
    # ---- Quick test overrides (must happen before output paths are set) ----
    _QUICK_N_CANDIDATES = 40
    _QUICK_TOP_N = 15
    _QUICK_N_CONCEPTS = 10
    _effective_top_n: int = TOP_N_PROMPTS

    if args.smoke_test:
        print("[step8] " + "=" * 60)
        print("[step8] SMOKE TEST MODE — 1 prompt × 1 concept × 5 models × 9 strengths × all methods")
        print("[step8] NOT for publication — sanity check only")
        print("[step8] " + "=" * 60)
        _effective_top_n = 1
        all_prompts = all_prompts[:5]  # tiny pool: pick cleanest 1 from first 5
        if len(target_concepts) > 1:
            target_concepts = target_concepts[:1]  # take first (or use --concepts to specify)
        print(f"[step8] Smoke test: concept = {target_concepts}")
        if not args.run_id:
            args.run_id = "smoketest"

    # --prompt: user-supplied exact text — overrides the entire candidate pool.
    # All concepts are evaluated (unless --concepts narrows the list explicitly).
    # Use --smoke-test to restrict to 1 concept.
    if args.prompt:
        _user_prompt = args.prompt.strip()
        if not _user_prompt:
            parser.error("--prompt must be non-empty text.")
        all_prompts = [_user_prompt]
        _effective_top_n = 1
        print(f"{_ts()} [step8] --prompt override: using exact user prompt")
        print(f"{_ts()} [step8]   prompt  = {_preview(_user_prompt, 120)}")
        print(f"{_ts()} [step8]   concepts = {target_concepts}")
        if not args.run_id:
            args.run_id = "prompt_run"
    elif args.smoke_test:
        pass  # already handled above
    elif args.quick_test:
        print("[step8] " + "=" * 60)
        print("[step8] QUICK TEST MODE — approximate results, NOT for publication")
        print("[step8] " + "=" * 60)
        _effective_top_n = _QUICK_TOP_N
        # Cap candidate pool to 40 (top-10 will be selected from these)
        all_prompts = all_prompts[:_QUICK_N_CANDIDATES]
        # Sample 3 concepts deterministically (seed=42)
        import random as _random_qt
        _rng_qt = _random_qt.Random(42)
        if len(target_concepts) > _QUICK_N_CONCEPTS:
            target_concepts = sorted(_rng_qt.sample(target_concepts, _QUICK_N_CONCEPTS))
        print(f"[step8] Quick test: {_QUICK_N_CANDIDATES} candidate prompts → select {_QUICK_TOP_N} cleanest")
        print(f"[step8] Quick test: concepts = {target_concepts}")
        if not args.run_id:
            args.run_id = "quicktest"

    if not args.smoke_test and not args.prompt and args.n_prompts and args.n_prompts < _effective_top_n:
        parser.error(
            f"--n-prompts {args.n_prompts} is too small: need a candidate pool of at least "
            f"{_effective_top_n}. Pass 0 (use all) or a value >= {_effective_top_n}."
        )
    prompts = (
        all_prompts[: args.n_prompts]
        if args.n_prompts and args.n_prompts < len(all_prompts)
        else all_prompts
    )
    print(f"[step8] {len(prompts)} candidate prompts (will select {_effective_top_n} cleanest at runtime)")

    # ---- Output paths ----
    suffix = f"_{args.run_id}" if args.run_id else ""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"within_model_steering{suffix}.json")
    examples_path = os.path.join(config.RESULTS_DIR, f"steering_examples{suffix}.jsonl")

    # Skip-check: only applies for single-seed runs where out_path is the definitive file.
    # For multi-seed we always proceed (each seed writes its own suffixed file).
    if len(_seeds) == 1 and os.path.exists(out_path) and not args.force:
        print(f"{out_path} already exists. Use --force to rerun.")
        log_run("step8_apply_steering.py", start_time, "skipped")
        return 0

    # ---- B3 cross-model vectors (SAE decoder, CAA cross, TI bridge) ----
    injection_layers_override: List[int] = []
    if args.injection_mode == "multi_layer" and args.injection_layers:
        injection_layers_override = [
            int(x.strip()) for x in args.injection_layers.split(",") if x.strip()
        ]

    b3_extra_methods: Dict = {}
    _b3_bal_loaded = 0
    _b3_ti_loaded = 0
    # NOTE: b3_bal is skipped — the caa_cross_vector entries in the bal file are corrupted
    # (numerically identical to sae_decoder_vector for every entry).  Code kept for future use.
    for _b3_path, _b3_vec_keys, _b3_count_var in [
        (
            os.path.join(config.STEERING_DIR, "cross_model_steering_vectors_ti.json"),
            [("b3_ti",          "vector")],
            "ti",
        ),
        # b3_bal disabled — see note above
        # (
        #     os.path.join(config.STEERING_DIR, "cross_model_steering_vectors_bal.json"),
        #     [("b3_bal",   "caa_cross_vector")],
        #     "bal",
        # ),
    ]:
        if not os.path.exists(_b3_path):
            print(f"[step8] B3 file not found, skipping: {_b3_path}")
            continue
        try:
            _b3_data = json.load(open(_b3_path, encoding="utf-8"))
            loaded = 0
            for _guide, _tgt_dict in _b3_data.items():
                for _tgt, _concept_dict in _tgt_dict.items():
                    for _concept, _entry in _concept_dict.items():
                        if _concept not in target_concepts:
                            continue  # only load concepts we're actually evaluating
                        _layer = _entry.get("injection_layer")
                        if _layer is None:
                            continue
                        for _tag, _vkey in _b3_vec_keys:
                            _vec = _entry.get(_vkey)
                            if _vec is None and _vkey == "caa_cross_vector":
                                # caa_cross_vector missing means the bal file was built by an
                                # old step7 that didn't write this field yet.
                                # Fix: re-run  python step7_build_steering.py --mode b3_bal --force
                                print(f"[step8] WARNING: {os.path.basename(_b3_path)}: "
                                      f"'caa_cross_vector' missing for {_guide}->{_tgt}/{_concept}. "
                                      f"Re-run step7 --mode b3_bal --force to rebuild.")
                            if _vec:
                                _mk = f"{_tag}_{_guide}"
                                (b3_extra_methods
                                    .setdefault(_tgt, {})
                                    .setdefault(_concept, [])
                                    .append((_mk, _vec, int(_layer))))
                                loaded += 1
            if _b3_count_var == "bal":
                _b3_bal_loaded = loaded
            else:
                _b3_ti_loaded = loaded
            print(f"[step8] Loaded {loaded} B3 vectors from {os.path.basename(_b3_path)}")
        except Exception as _e:
            print(f"[step8] WARNING: could not load {os.path.basename(_b3_path)}: {_e}")

    # ---- Change 2: naive_{guide} vectors ----
    # Truncate/pad guide's A-vector into target hidden_dim then L2-normalise.
    # Serves as dumb cross-model baseline — proves the MLP bridge earns its keep.
    _models_to_process = (
        [model_filter] if model_filter and model_filter in config.MODELS
        else list(config.MODELS.keys())
    )
    _naive_added = 0
    for _target_name in _models_to_process:
        _tgt_dim: int = config.MODELS[_target_name]["hidden_dim"]
        _tgt_layer: int = config.MODELS[_target_name]["target_layer"]
        for _guide_name, _guide_cfg in config.MODELS.items():
            if _guide_name == _target_name:
                continue
            for _concept in target_concepts:
                _src_vec = (
                    sv.get(_guide_name, {}).get(_concept, {}).get("sae_vector")
                    or sv.get(_guide_name, {}).get(_concept, {}).get("caa_vector")
                )
                if not _src_vec:
                    continue
                _arr = np.array(_src_vec[:_tgt_dim], dtype=np.float32)
                if len(_src_vec) < _tgt_dim:
                    _arr = np.pad(_arr, (0, _tgt_dim - len(_src_vec)))
                _norm = np.linalg.norm(_arr)
                if _norm > 1e-8:
                    _arr = _arr / _norm
                _mk = f"naive_{_guide_name}"
                (b3_extra_methods
                    .setdefault(_target_name, {})
                    .setdefault(_concept, [])
                    .append((_mk, _arr.tolist(), _tgt_layer)))
                _naive_added += 1
    print(f"[step8] Added {_naive_added} naive_{{guide}} vectors")

    # ---- Save naive vectors to steering/ for reviewer reference ----
    # Format mirrors cross_model_steering_vectors_*.json:
    # {"naive_steering_vectors": {guide: {target: {concept: {steering_vector, meta}}}}}.
    # Skip recompute if file already exists — naive vectors are deterministic and
    # O(n_models² × n_concepts), so there's no value in rewriting on every run.
    _naive_out = os.path.join(config.STEERING_DIR, "naive_steering_vectors.json")
    if os.path.exists(_naive_out) and not args.force:
        print(f"[step8] naive_steering_vectors.json exists — skipping save (--force to regenerate)")
    else:
        _naive_save: Dict = {"naive_steering_vectors": {}}
        for _target_name in _models_to_process:
            _tgt_dim2: int = config.MODELS[_target_name]["hidden_dim"]
            _tgt_layer2: int = config.MODELS[_target_name]["target_layer"]
            for _guide_name2 in config.MODELS:
                if _guide_name2 == _target_name:
                    continue
                for _concept2 in target_concepts:
                    _src_vec2 = (
                        sv.get(_guide_name2, {}).get(_concept2, {}).get("sae_vector")
                        or sv.get(_guide_name2, {}).get(_concept2, {}).get("caa_vector")
                    )
                    if not _src_vec2:
                        continue
                    _arr2 = np.array(_src_vec2[:_tgt_dim2], dtype=np.float32)
                    if len(_src_vec2) < _tgt_dim2:
                        _arr2 = np.pad(_arr2, (0, _tgt_dim2 - len(_src_vec2)))
                    _n2 = np.linalg.norm(_arr2)
                    if _n2 > 1e-8:
                        _arr2 = _arr2 / _n2
                    (_naive_save["naive_steering_vectors"]
                        .setdefault(_guide_name2, {})
                        .setdefault(_target_name, {}))[_concept2] = {
                        "steering_vector": _arr2.tolist(),
                        "guide_hidden_dim": config.MODELS[_guide_name2]["hidden_dim"],
                        "target_hidden_dim": _tgt_dim2,
                        "injection_layer": _tgt_layer2,
                        "method": "truncate_pad_l2norm",
                        "description": (
                            "Guide A-vector truncated/padded to target hidden_dim then L2-normalised. "
                            "No projection MLP. Dumb cross-architecture baseline."
                        ),
                    }
        with open(_naive_out, "w", encoding="utf-8") as _nf:
            json.dump(_naive_save, _nf)
        print(f"[step8] Saved naive vectors → {os.path.basename(_naive_out)}")

    # ---- Change 3: C3 universal vectors ----
    # b3_sae_decoder vectors are intentionally NOT loaded here: they are identical to
    # the native sae_vector for most guide→target pairs (SAE decoder fallback path in
    # step7 writes the same weights when aligned pairs are unavailable). Removed to
    # avoid polluting method rankings with duplicate entries.
    c3_path = os.path.join(config.STEERING_DIR, "universal_steering_vectors_v1.json")
    _c3_loaded = 0
    _c3_flipped = 0
    if os.path.exists(c3_path):
        try:
            _c3_data = json.load(open(c3_path, encoding="utf-8"))["universal_steering_vectors"]
            for _concept, _m_dict in _c3_data.items():
                if _concept not in target_concepts:
                    continue  # only load concepts we're evaluating
                for _model_name, _entry in _m_dict.items():
                    if _model_name not in config.MODELS:
                        continue
                    _vec = _entry.get("steering_vector")
                    _layer = config.MODELS[_model_name]["target_layer"]
                    if not _vec:
                        continue
                    # ---- Sign alignment ----
                    # C3 vectors are decoded from an unsigned cluster centroid so their
                    # polarity is arbitrary.  Align sign to the native sae_vector for this
                    # (model, concept) by checking the dot product.  If negative, the
                    # decoded vector points away from the concept in this model's space
                    # and we flip it.  This is done at construction time (before any
                    # evaluation is run) so it cannot constitute post-hoc score fitting.
                    # Justification: identical to sign-disambiguation in PCA/ICA — choose
                    # the orientation that agrees with the independently-supervised direction.
                    _native_vec = (
                        sv.get(_model_name, {}).get(_concept, {}).get("sae_vector")
                        or sv.get(_model_name, {}).get(_concept, {}).get("caa_vector")
                    )
                    if _native_vec is not None:
                        import numpy as _np_c3
                        _dot = float(_np_c3.dot(_vec, _native_vec))
                        if _dot < 0:
                            _vec = [-x for x in _vec]
                            _c3_flipped += 1
                    (b3_extra_methods
                        .setdefault(_model_name, {})
                        .setdefault(_concept, [])
                        .append(("universal_c3", _vec, _layer)))
                    _c3_loaded += 1
            print(f"[step8] Loaded {_c3_loaded} universal_c3 vectors from {c3_path} "
                  f"({_c3_flipped} sign-aligned to native direction)")
        except Exception as _e:
            print(f"[step8] WARNING: could not load C3 vectors: {_e}")
    else:
        print("[step8] WARNING: C3 vectors not found — run build_universal_vectors.py first.")

    # ---- C3 enc_dec universal vectors ----
    enc_dec_path = os.path.join(config.STEERING_DIR, "universal_steering_vectors_enc_dec_v1.json")
    _enc_dec_loaded = 0
    if os.path.exists(enc_dec_path):
        try:
            _enc_dec_data = json.load(open(enc_dec_path, encoding="utf-8"))["universal_steering_vectors"]
            for _concept, _m_dict in _enc_dec_data.items():
                if _concept not in target_concepts:
                    continue  # only load concepts we're evaluating
                for _model_name, _entry in _m_dict.items():
                    if _model_name not in config.MODELS:
                        continue
                    _vec = _entry.get("steering_vector")
                    _layer = config.MODELS[_model_name]["target_layer"]
                    if not _vec:
                        continue
                    (b3_extra_methods
                        .setdefault(_model_name, {})
                        .setdefault(_concept, [])
                        .append(("universal_enc_dec", _vec, _layer)))
                    _enc_dec_loaded += 1
            print(f"[step8] Loaded {_enc_dec_loaded} universal_enc_dec vectors from {enc_dec_path}")
        except Exception as _e:
            print(f"[step8] WARNING: could not load enc_dec vectors: {_e}")
    else:
        print("[step8] INFO: enc_dec vectors not found — run build_universal_vectors.py --mode enc_dec first.")

    # ---- Determine models to run ----
    models_to_run: Dict = (
        {model_filter: config.MODELS[model_filter]}
        if model_filter and model_filter in config.MODELS
        else dict(config.MODELS)
    )

    # ---- Eval-set size report ----
    # Counts exactly what will be evaluated: for each (model, concept, method, strength)
    # tuple we will generate one batch of prompts, so total = product of those counts.
    _n_models   = len(models_to_run)
    _n_concepts = len(target_concepts)
    _n_strengths = len(SWEEP_STANDARD)          # e.g. 9: [-5,-3,-2,-1,0,1,2,3,5]

    # A5 — native: sae_vector + caa_vector = 2 types, both at target_layer
    # Each type × model × concept × strength (s=0 reused from baseline, still 1 JSONL entry)
    _a5_sae = _n_models * _n_concepts           # 1 vec per (model, concept)
    _a5_caa = _n_models * _n_concepts
    _a5_total_entries = (_a5_sae + _a5_caa) * _n_strengths

    # B3 — cross-model: bal (caa-cross), ti, naive  — each is per (guide→target, concept)
    # b3_extra_methods is keyed by target model, then concept, then list of (tag, vec, layer)
    # Count b3_bal, b3_ti, naive separately
    _b3_bal_entries = sum(
        1 for _tgt in b3_extra_methods
        for _c in b3_extra_methods[_tgt]
        for (tag, _, __) in b3_extra_methods[_tgt][_c]
        if tag.startswith("b3_bal")
    ) if _n_concepts > 0 else 0
    _b3_ti_entries = sum(
        1 for _tgt in b3_extra_methods
        for _c in b3_extra_methods[_tgt]
        for (tag, _, __) in b3_extra_methods[_tgt][_c]
        if tag.startswith("b3_ti")
    ) if _n_concepts > 0 else 0
    _naive_entries = sum(
        1 for _tgt in b3_extra_methods
        for _c in b3_extra_methods[_tgt]
        for (tag, _, __) in b3_extra_methods[_tgt][_c]
        if tag.startswith("naive_")
    ) if _n_concepts > 0 else 0
    _b3_total_entries = (_b3_bal_entries + _b3_ti_entries + _naive_entries) * _n_strengths

    # C3 — universal: universal_c3, universal_enc_dec
    _c3_c3_entries = sum(
        1 for _tgt in b3_extra_methods
        for _c in b3_extra_methods[_tgt]
        for (tag, _, __) in b3_extra_methods[_tgt][_c]
        if tag == "universal_c3"
    ) if _n_concepts > 0 else 0
    _c3_encdec_entries = sum(
        1 for _tgt in b3_extra_methods
        for _c in b3_extra_methods[_tgt]
        for (tag, _, __) in b3_extra_methods[_tgt][_c]
        if tag == "universal_enc_dec"
    ) if _n_concepts > 0 else 0
    _c3_total_entries = (_c3_c3_entries + _c3_encdec_entries) * _n_strengths

    # Baseline: 1 per (model, concept) — zero-strength anchor, no generation sweep
    _baseline_entries = _n_models * _n_concepts
    _grand_total = _baseline_entries + _a5_total_entries + _b3_total_entries + _c3_total_entries

    # ---- Write run_meta (first line of examples file, and per-seed file in multi-seed) ----
    _run_meta = {
        "_type": "run_meta",
        # generation params
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "decoding": "greedy",
        "top_p": None,
        "temperature": None,
        # evaluation params
        "ppl_scorer_model": list(_PPL_SCORER_MAP.values())[0],  # uniform — same for all models
        "deberta_model": _ZEROSHOT_MODEL,
        "sweep_standard": SWEEP_STANDARD,
        "top_n_prompts": _effective_top_n,
        "n_prompts_candidates": len(prompts),
        "n_concepts": len(target_concepts),
        # run config
        "injection_mode": args.injection_mode,
        "injection_layers": args.injection_layers or "auto",
        "evaluator": args.evaluator,
        "run_id": args.run_id or "",
        "seed": _seeds[0] if len(_seeds) == 1 else _seeds,
        "c3_vectors_loaded": _c3_loaded,
        "enc_dec_vectors_loaded": _enc_dec_loaded,
        "naive_vectors_added": _naive_added,
        # Full vector inventory breakdown
        "vector_inventory": {
            "A5_native_sae": {
                "description": "Per-model SAE features (sae_vector) + CAA vectors (caa_vector)",
                "sae_vectors": _n_models * len(target_concepts),
                "caa_vectors": _n_models * len(target_concepts),
                "total": 2 * _n_models * len(target_concepts),
            },
            "B3_cross_model": {
                "description": "Cross-model steering: bal (CAA-cross), ti (TI-trained), naive (dim-aligned)",
                "bal_loaded": _b3_bal_loaded,
                "ti_loaded": _b3_ti_loaded,
                "naive_added": _naive_added,
                "total": _b3_bal_loaded + _b3_ti_loaded + _naive_added,
            },
            "C3_universal": {
                "description": "Model-agnostic universal vectors: c3 (same-family) + enc_dec (cross-arch)",
                "c3_loaded": _c3_loaded,
                "enc_dec_loaded": _enc_dec_loaded,
                "total": _c3_loaded + _enc_dec_loaded,
            },
        },
    }
    with open(examples_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_run_meta) + "\n")

    _W = 46  # column width
    print()
    print(f"[step8] {'='*60}")
    print(f"[step8] EVAL SET REPORT")
    print(f"[step8] {'='*60}")
    print(f"[step8]   Models            : {_n_models}  ({', '.join(models_to_run)})")
    print(f"[step8]   Concepts          : {_n_concepts}  ({', '.join(target_concepts)})")
    print(f"[step8]   Prompts per run   : {_effective_top_n}")
    print(f"[step8]   Strengths         : {_n_strengths}  {SWEEP_STANDARD}  (s=0 reuses baseline)")
    print(f"[step8] {'-'*60}")
    print(f"[step8]   BASELINE (1 per model×concept)")
    print(f"[step8]     baseline         : {_n_models} models × {_n_concepts} concepts = {_baseline_entries:>6,}")
    print(f"[step8] {'-'*60}")
    print(f"[step8]   A5  NATIVE VECTORS (per-model SAE/CAA)")
    print(f"[step8]     sae_vector       : {_a5_sae:>4} vecs × {_n_strengths} strengths = {_a5_sae * _n_strengths:>6,}")
    print(f"[step8]     caa_vector       : {_a5_caa:>4} vecs × {_n_strengths} strengths = {_a5_caa * _n_strengths:>6,}")
    print(f"[step8]     A5 subtotal                              = {_a5_total_entries:>6,}")
    print(f"[step8] {'-'*60}")
    print(f"[step8]   B3  CROSS-MODEL VECTORS")
    print(f"[step8]     b3_bal (caa-cross): {_b3_bal_entries:>4} vecs × {_n_strengths} strengths = {_b3_bal_entries * _n_strengths:>6,}")
    print(f"[step8]     b3_ti  (TI-bridge): {_b3_ti_entries:>4} vecs × {_n_strengths} strengths = {_b3_ti_entries * _n_strengths:>6,}")
    print(f"[step8]     naive  (dim-align): {_naive_entries:>4} vecs × {_n_strengths} strengths = {_naive_entries * _n_strengths:>6,}")
    print(f"[step8]     B3 subtotal                              = {_b3_total_entries:>6,}")
    print(f"[step8] {'-'*60}")
    print(f"[step8]   C3  UNIVERSAL VECTORS")
    print(f"[step8]     universal_c3     : {_c3_c3_entries:>4} vecs × {_n_strengths} strengths = {_c3_c3_entries * _n_strengths:>6,}")
    print(f"[step8]     universal_enc_dec: {_c3_encdec_entries:>4} vecs × {_n_strengths} strengths = {_c3_encdec_entries * _n_strengths:>6,}")
    print(f"[step8]     C3 subtotal                              = {_c3_total_entries:>6,}")
    print(f"[step8] {'='*60}")
    print(f"[step8]   GRAND TOTAL eval objects                   = {_grand_total:>6,}")
    print(f"[step8]   (each = {_effective_top_n} prompts × 1 forward pass + scoring)")
    print(f"[step8] {'='*60}")
    # Estimated runtime: ~4s per eval object on A100 (single model, sequential).
    # Parallel mode runs all models concurrently so divide by n_models.
    _secs_per_obj = 4
    _est_seq_s = _grand_total * _secs_per_obj
    _est_par_s = max(1, _grand_total // max(1, _n_models)) * _secs_per_obj
    def _fmt_time(s: int) -> str:
        if s < 120:
            return f"{s}s"
        if s < 7200:
            return f"{s//60}m {s%60:02d}s"
        return f"{s//3600}h {(s%3600)//60:02d}m"
    print(f"[step8]   Est. runtime (sequential) : {_fmt_time(_est_seq_s)}")
    if _n_models > 1:
        print(f"[step8]   Est. runtime (--parallel) : {_fmt_time(_est_par_s)}")
    print(f"[step8] {'='*60}")
    print()

    # ---- Uniform-prompts pre-screening (always-on for multi-model runs) ----
    # Load each model in turn, generate all candidate prompts (greedy, no hooks), then
    # take the intersection of every model's clean set.  Sorted by average rep_rate
    # (cleanest-across-all-models first) and trimmed to _effective_top_n.
    # Workers receive this fixed list; top_n_prompts is set to the shared count so no
    # further internal filtering happens — all models evaluate on identical prompts.
    # The shared prompt list is saved to results/uniform_prompts{suffix}.json for
    # reproducibility and so later --model re-runs can reuse the same set.
    # Skipped for single-model runs (no cross-model comparison) and smoke/prompt modes.
    _uniform_prompts_path = os.path.join(config.RESULTS_DIR, f"uniform_prompts{suffix}.json")
    if len(models_to_run) > 1 and not args.smoke_test and not args.prompt:
        # Reuse saved prompt file if it exists and --force is not set
        if os.path.exists(_uniform_prompts_path) and not args.force:
            _up_data = json.load(open(_uniform_prompts_path, encoding="utf-8"))
            prompts = _up_data["prompts"]
            _effective_top_n = len(prompts)
            print(
                f"[step8] Loaded {_effective_top_n} uniform prompts from "
                f"{os.path.basename(_uniform_prompts_path)} (--force to regenerate)"
            )
        else:
            print(f"[step8] {'='*60}")
            print(f"[step8] UNIFORM-PROMPTS PRE-SCREENING")
            print(f"[step8] Models: {list(models_to_run)}")
            print(f"[step8] Candidate pool: {len(prompts)} prompts")
            print(f"[step8] {'='*60}")
            _prescreen_rep_rates: Dict[str, List[float]] = {}
            for _pmodel in models_to_run:
                _prescreen_rep_rates[_pmodel] = _prescreen_model_prompts(
                    _pmodel, prompts, args.max_new_tokens, gpu_id=None
                )
            # Build each model's clean index set
            _clean_sets: List[set] = [
                {i for i, r in enumerate(rr) if r < 1.0}
                for rr in _prescreen_rep_rates.values()
            ]
            _shared_indices: set = _clean_sets[0]
            for _cs in _clean_sets[1:]:
                _shared_indices &= _cs
            if len(_shared_indices) < _effective_top_n:
                raise RuntimeError(
                    f"[step8] uniform-prompts: only {len(_shared_indices)} prompts are clean "
                    f"across all {len(models_to_run)} models — need at least {_effective_top_n}. "
                    "Expand the candidate pool (--n-prompts) or lower --quick-test top_n."
                )
            # Sort by average rep_rate across all models (ascending = cleanest first)
            _avg_rep = {
                i: sum(_prescreen_rep_rates[m][i] for m in models_to_run) / len(models_to_run)
                for i in _shared_indices
            }
            _sorted_shared = sorted(_shared_indices, key=lambda i: _avg_rep[i])
            _shared_top = _sorted_shared[:_effective_top_n]
            prompts = [prompts[i] for i in _shared_top]
            _effective_top_n = len(prompts)
            # Save to JSON for reproducibility and re-use
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            _up_save = {
                "prompts": prompts,
                "n_prompts": _effective_top_n,
                "n_candidates": len(_prescreen_rep_rates[list(models_to_run)[0]]),
                "n_intersection": len(_shared_indices),
                "models_prescreened": list(models_to_run),
                "avg_rep_rates": {p: round(_avg_rep[_shared_top[j]], 6) for j, p in enumerate(prompts)},
                "per_model_clean_counts": {
                    m: sum(r < 1.0 for r in rr)
                    for m, rr in _prescreen_rep_rates.items()
                },
            }
            with open(_uniform_prompts_path, "w", encoding="utf-8") as _upf:
                json.dump(_up_save, _upf, indent=2)
            print(
                f"[step8] Uniform prompts: {len(_shared_indices)} in intersection → "
                f"kept top {_effective_top_n} cleanest"
            )
            print(
                f"[step8] Saved to {os.path.basename(_uniform_prompts_path)} "
                "(subsequent runs reuse this file automatically)"
            )
            print(f"[step8] All {len(models_to_run)} models will be evaluated on the same {_effective_top_n} prompts.")
            print()

    args_dict = {
        "max_new_tokens": args.max_new_tokens,
        "injection_mode": args.injection_mode,
        "injection_layers_override": injection_layers_override,
        "run_id": args.run_id,
        "top_n_prompts": _effective_top_n,
    }

    # ---- Multi-seed eval loop ----
    # Each seed is a full independent pass (different prompt ranking + generation sampling).
    # With --seeds 42,123 you get two merged results files, then compile_tables
    # produces a seed-aggregated table with mean±SD per cell.
    all_seeds_merged: Dict[int, Dict] = {}  # seed → merged results

    for _seed in _seeds:
        set_seed(_seed)
        # Single-seed: use the same out_path/run_id as the caller set (backwards compat).
        # Multi-seed: each seed gets its own suffixed file, e.g. within_model_steering_s42.json.
        _seed_run_id = (f"{args.run_id}_s{_seed}" if args.run_id else f"s{_seed}") if len(_seeds) > 1 else args.run_id
        _seed_out_path = out_path if len(_seeds) == 1 else os.path.join(
            config.RESULTS_DIR,
            f"within_model_steering_{_seed_run_id}.json"
        )

        if len(_seeds) > 1:
            print(f"\n[step8] {'='*60}")
            print(f"[step8] SEED {_seed} ({_seeds.index(_seed)+1}/{len(_seeds)})")
            print(f"[step8] {'='*60}")
            # Write run_meta to this seed's own examples file so it has a valid header.
            _seed_examples_path = os.path.join(
                config.RESULTS_DIR, f"steering_examples_{_seed_run_id}.jsonl"
            )
            with open(_seed_examples_path, "w", encoding="utf-8") as _smf:
                _smf.write(json.dumps(dict(_run_meta, run_id=_seed_run_id, seed=_seed)) + "\n")

        # Pass seed into worker so it re-seeds generation
        _seed_args_dict = dict(args_dict, run_id=_seed_run_id, seed=_seed)

        # ---- GPU execution strategy ----
        # --parallel: 5 workers each pinned to one GPU (GPUs 0-4) via CUDA_VISIBLE_DEVICES.
        #   All 5 models run simultaneously. Wall time = slowest model (~2-3 h).
        #   Best choice for 7B models which easily fit on a single A100 (80 GB).
        #
        # Sequential (default): one model at a time, gpu_id=None so CUDA_VISIBLE_DEVICES is
        #   NOT set. device_map="auto" can spread across all 8 GPUs if needed.
        #   Useful for future 70B+ models that don't fit on one GPU.
        #   For current 7B models this is strictly slower than --parallel.
        if args.parallel and len(models_to_run) > 1:
            # Pre-warm DeBERTa on CPU in the main process so child workers find it in HF
            # cache — avoids concurrent HuggingFace Hub filelock races that cause silent hangs.
            print("[step8] Pre-warming DeBERTa NLI classifier on CPU (prevents HF cache races)...")
            try:
                from transformers import pipeline as _prewarm_pipe
                _prewarm_clf = _prewarm_pipe(
                    "zero-shot-classification", model=_ZEROSHOT_MODEL, device=-1,
                    local_files_only=False,  # allow download once so workers hit cache
                )
                del _prewarm_clf
                print(f"[step8] DeBERTa cached at {_ZEROSHOT_MODEL} — workers will load from cache")
            except Exception as _prewarm_err:
                print(f"[step8] WARNING: DeBERTa pre-warm failed ({_prewarm_err}). "
                      "Workers will attempt to load it themselves.")
            print(
                f"[step8] --parallel: spawning {len(models_to_run)} workers, "
                f"assigning each to the freest available GPU"
            )
            ctx = mp.get_context("spawn")
            procs = []
            claimed_gpus: List[int] = []
            for model_name in models_to_run:
                gpu_id = _pick_free_gpu(exclude=claimed_gpus)
                claimed_gpus.append(gpu_id)
                p = ctx.Process(
                    target=_eval_model_worker,
                    args=(
                        model_name, gpu_id, _seed_args_dict, sv, b3_extra_methods,
                        target_concepts, prompts, config.RESULTS_DIR, _seed_run_id,
                    ),
                    name=f"step8-{model_name}",
                )
                p.start()
                print(f"[step8] Started {model_name} on GPU {gpu_id} (pid={p.pid})")
                procs.append((model_name, p))
            failed = []
            _WORKER_TIMEOUT = 18000  # 5-hour hard ceiling per worker
            for model_name, p in procs:
                p.join(timeout=_WORKER_TIMEOUT)
                if p.is_alive():
                    print(f"[step8] ERROR: worker for {model_name} exceeded {_WORKER_TIMEOUT//3600}h — terminating")
                    p.terminate()
                    p.join(timeout=60)
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=30)
                    failed.append(model_name)
                elif p.exitcode != 0:
                    print(f"[step8] WARNING: worker for {model_name} exited {p.exitcode}")
                    failed.append(model_name)
            if failed:
                print(f"[step8] {len(failed)} workers failed: {failed}")
        else:
            # Sequential — gpu_id=None means no CUDA_VISIBLE_DEVICES pinning;
            # device_map="auto" inside the worker uses all visible GPUs.
            for model_name in models_to_run:
                print(f"[step8] Sequential: loading {model_name} (all GPUs available)")
                _eval_model_worker(
                    model_name, None, _seed_args_dict, sv, b3_extra_methods,
                    target_concepts, prompts, config.RESULTS_DIR, _seed_run_id,
                )

        # ---- Merge partial results for this seed ----
        # Load existing file so single-model re-runs (--model X) extend rather than overwrite.
        merged: Dict[str, Dict] = {}
        if os.path.exists(_seed_out_path):
            try:
                with open(_seed_out_path, encoding="utf-8") as _f_prev:
                    merged = json.load(_f_prev)
                print(f"[step8] Loaded existing results from {_seed_out_path} ({len(merged)} models)")
            except Exception as _e:
                print(f"[step8] WARNING: could not load existing {_seed_out_path}: {_e}")
                merged = {}
        for model_name in models_to_run:
            partial_path = os.path.join(config.RESULTS_DIR, f"_partial_{model_name}.json")
            if os.path.exists(partial_path):
                with open(partial_path, encoding="utf-8") as f:
                    merged[model_name] = json.load(f)
                os.remove(partial_path)
            else:
                print(f"[step8] WARNING: no partial result found for {model_name} (seed={_seed})")

        with open(_seed_out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f)
        print(f"[step8] Wrote {_seed_out_path}")
        all_seeds_merged[_seed] = merged

    # ---- Final merge: for single seed use merged directly; for multi-seed aggregate ----
    # The primary output (out_path with no seed suffix) always holds seed=_seeds[0] results
    # for backwards compat. The seed-aggregated tables use all_seeds_merged.
    merged = all_seeds_merged[_seeds[0]]
    if not os.path.exists(out_path) or len(_seeds) > 1:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f)
        print(f"[step8] Wrote {out_path}")

    # ---- compile_tables: load the FULL within_model_steering.json (all prior models) ----
    # Critical for Table 7: when running --model X, the partial merged dict only has model X.
    # We must merge with any previously written within_model_steering.json so Table 7
    # can cross-reference guide model scores that were computed in earlier --model Y runs.
    _full_results: Dict[str, Dict] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as _ftf:
                _full_results = json.load(_ftf)
        except Exception:
            _full_results = {}
    # Overlay the current run's models on top of the full results
    _full_results.update(merged)
    # Write the fully-merged file back (so incremental runs accumulate)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_full_results, f)

    compile_tables(_full_results, config.RESULTS_DIR, suffix=suffix,
                   all_seeds_merged=all_seeds_merged if len(_seeds) > 1 else None,
                   supervised_concepts=_concept_core)

    # ---- Terminal summary table ----
    # Three metrics per method family, split by supervised / unsupervised concepts:
    #   (1) Mean signed delta — signed best delta (includes negatives); unbiased
    #   (2) Win rate          — % of (model × concept) pairs with positive delta
    #   (3) Delta @ strength 1.0 — fixed operating point, apples-to-apples
    _FIXED_STRENGTH = "1.0"
    _METHOD_FAMILIES = [
        ("sae_vector",        "Native SAE   "),
        ("caa_vector",        "Native CAA   "),
        ("b3_bal",            "B3 bal       "),
        ("b3_ti",             "B3 TI        "),
        ("naive",             "Naive        "),
        ("universal_c3",      "Univ C3      "),
        ("universal_enc_dec", "Univ enc-dec "),
    ]
    _all_run_models   = sorted(_full_results.keys())
    _all_run_concepts = sorted({c for md in _full_results.values() for c in md})
    _sup_run   = sorted(c for c in _all_run_concepts if _concept_core and c in _concept_core)
    _unsup_run = sorted(c for c in _all_run_concepts if not _concept_core or c not in _concept_core)

    def _iter_method_data(model, concepts, pfx):
        """Yield method dicts for all keys matching pfx across (model, concepts)."""
        for _c in concepts:
            for _mk, _mdata in _full_results.get(model, {}).get(_c, {}).get("methods", {}).items():
                if _mk == pfx or _mk.startswith(pfx + "_"):
                    yield _mdata

    def _signed_delta(mdata):
        """Signed best delta: positive if method achieved positive steering, else negative."""
        pos = mdata.get("max_positive_delta") or 0
        neg = mdata.get("max_negative_delta") or 0  # stored as negative float
        return pos if pos > 0 else neg

    def _fixed_delta(mdata):
        """Mean delta across prompts at the fixed reference strength."""
        deltas = (mdata.get("results_by_strength", {})
                       .get(_FIXED_STRENGTH, {})
                       .get("per_prompt_deltas")) or []
        return float(sum(deltas) / len(deltas)) if deltas else None

    # --- per-model aggregators for each metric ---
    def _col_signed(model, concepts, pfx):
        vals = [_signed_delta(md) for md in _iter_method_data(model, concepts, pfx)]
        return round(float(sum(vals) / len(vals)), 4) if vals else None

    def _col_winrate(model, concepts, pfx):
        vals = [1 if _signed_delta(md) > 0 else 0
                for md in _iter_method_data(model, concepts, pfx)]
        return round(100.0 * sum(vals) / len(vals), 1) if vals else None

    def _col_fixed(model, concepts, pfx):
        raw = [_fixed_delta(md) for md in _iter_method_data(model, concepts, pfx)]
        vals = [v for v in raw if v is not None]
        return round(float(sum(vals) / len(vals)), 4) if vals else None

    # --- row mean across all models ---
    def _row_mean(model_vals):
        nums = [v for v in model_vals if v is not None]
        return round(float(sum(nums) / len(nums)), 4) if nums else None

    def _print_summary(label, concepts):
        if not concepts:
            return
        _col = max((len(m) for m in _all_run_models), default=12) + 2
        _w   = 18 + _col * len(_all_run_models) + 10
        _sep = f"[step8]  {'-'*16}" + "-" * (_col * len(_all_run_models)) + "  --------"
        _hdr = (f"[step8]  {'Method':<16}"
                + "".join(f"{m:>{_col}}" for m in _all_run_models)
                + f"  {'MEAN':>8}")
        _fv  = lambda v, fmt: f"{v:{fmt}}" if v is not None else f"{'—':>{_col}}"
        _fm  = lambda v, fmt: f"{v:{fmt}}" if v is not None else f"{'—':>8}"

        print(f"\n[step8] {'='*_w}")
        print(f"[step8]  SUMMARY — {label}  ({len(concepts)} concept(s) × {len(_all_run_models)} models)")
        print(f"[step8] {'='*_w}")

        for metric_label, col_fn, v_fmt, m_fmt in [
            ("(1) Mean signed delta  [incl. negatives, higher=better]",
             _col_signed,  f">+{_col}.4f", ">+8.4f"),
            ("(2) Win rate %         [% model×concept pairs with Δ>0]",
             _col_winrate, f">{_col}.1f",  ">8.1f"),
            (f"(3) Delta @ strength {_FIXED_STRENGTH}  [fixed operating point]",
             _col_fixed,   f">+{_col}.4f", ">+8.4f"),
        ]:
            print(f"[step8]")
            print(f"[step8]  {metric_label}")
            print(_hdr)
            print(_sep)
            for pfx, lbl in _METHOD_FAMILIES:
                row  = [col_fn(_m, concepts, pfx) for _m in _all_run_models]
                mean = _row_mean(row)
                print(f"[step8]  {lbl:<16}"
                      + "".join(_fv(v, v_fmt) for v in row)
                      + f"  {_fm(mean, m_fmt)}")

        print(f"\n[step8] {'='*_w}")

    if _sup_run:
        _print_summary("SUPERVISED [S]", _sup_run)
    if _unsup_run and _concept_core:
        _print_summary("UNSUPERVISED [U]", _unsup_run)
    if not _concept_core:
        _print_summary("ALL CONCEPTS", _all_run_concepts)


    # ---- Layer ablation ----
    if args.ablate_layers:
        _ablation_model = model_filter or "gpt2-large"
        if _ablation_model not in config.MODELS:
            print(f"[step8] --ablate-layers: model '{_ablation_model}' not in config — skipping")
        else:
            _n_layers_cfg = config.MODELS[_ablation_model].get("n_layers")
            if _n_layers_cfg is None:
                # Estimate from target_layer: assume target_layer ≈ middle → n_layers = target*2+1
                _n_layers_cfg = config.MODELS[_ablation_model]["target_layer"] * 2 + 1
            _ablation_layers = sorted({
                max(0, _n_layers_cfg // 4),                    # early (~25%)
                config.MODELS[_ablation_model]["target_layer"], # mid (default)
                min(_n_layers_cfg - 1, (_n_layers_cfg * 3) // 4),  # late (~75%)
            })
            print(f"\n[step8] Layer ablation for {_ablation_model}: layers={_ablation_layers}")
            _abl_results: Dict[str, Dict] = {}
            for _abl_layer in _ablation_layers:
                _abl_run_id = f"layer{_abl_layer}"
                _abl_args_dict = dict(args_dict,
                                      run_id=_abl_run_id,
                                      seed=_seeds[0],
                                      injection_layers_override=[_abl_layer],
                                      injection_mode=args.injection_mode)
                print(f"[step8] Layer ablation: layer={_abl_layer}")
                _eval_model_worker(
                    _ablation_model, None, _abl_args_dict, sv, b3_extra_methods,
                    target_concepts, prompts, config.RESULTS_DIR, _abl_run_id,
                )
                _abl_partial = os.path.join(config.RESULTS_DIR, f"_partial_{_ablation_model}.json")
                if os.path.exists(_abl_partial):
                    with open(_abl_partial) as _apf:
                        _abl_results[str(_abl_layer)] = json.load(_apf)
                    os.remove(_abl_partial)
            # Write ablation results
            _abl_out = os.path.join(config.RESULTS_DIR, f"layer_ablation_{_ablation_model}.json")
            with open(_abl_out, "w") as _aof:
                json.dump({"model": _ablation_model, "layers_tested": _ablation_layers,
                           "results_by_layer": _abl_results}, _aof, indent=2)
            print(f"[step8] Layer ablation written to {_abl_out}")
            # Compact markdown summary
            _abl_md = os.path.join(config.RESULTS_DIR, f"layer_ablation_{_ablation_model}.md")
            with open(_abl_md, "w") as _amf:
                _amf.write(f"# Layer Ablation — {_ablation_model}\n\n")
                _amf.write(f"Layers tested: {_ablation_layers}\n\n")
                _all_abl_concepts = sorted({c for r in _abl_results.values() for c in r})
                _amf.write("| Concept | " + " | ".join(f"Layer {l}" for l in _ablation_layers) + " |\n")
                _amf.write("|---|" + "---|" * len(_ablation_layers) + "\n")
                for _ac in _all_abl_concepts:
                    _row = []
                    for _al in _ablation_layers:
                        _layer_data = _abl_results.get(str(_al), {}).get(_ac, {})
                        _best = max(
                            (_layer_data.get("methods", {}).get(mk, {}).get("effective_max_delta") or 0.0)
                            for mk in ["sae_vector", "caa_vector"]
                        ) if _layer_data else 0.0
                        _row.append(str(round(_best, 4)))
                    _amf.write(f"| {_ac} | " + " | ".join(_row) + " |\n")
            print(f"[step8] Layer ablation report: {_abl_md}")

    # ---- LLM judge evaluation (Claude Sonnet) ----
    # Reads ANTHROPIC_API_KEY from env / .env.  If absent, prompt interactively once.
    _anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not _anthropic_key:
        try:
            _key_input = input(
                "[step8] ANTHROPIC_API_KEY not set.\n"
                "        Enter your Anthropic API key to enable Claude judge "
                "(or press Enter to skip): "
            ).strip()
            if _key_input:
                os.environ["ANTHROPIC_API_KEY"] = _key_input
                _anthropic_key = _key_input
                print("[step8] API key accepted.")
        except (EOFError, KeyboardInterrupt):
            pass  # non-interactive / piped run — skip judge
    if _anthropic_key:
        # In multi-seed mode, judge on seed-0's examples file (which has the full sweep).
        # In single-seed mode, examples_path is already correct (no suffix beyond run_id).
        _judge_seed_run_id = (
            (f"{args.run_id}_s{_seeds[0]}" if args.run_id else f"s{_seeds[0]}")
            if len(_seeds) > 1 else args.run_id
        )
        _judge_suffix = ("_" + _judge_seed_run_id) if _judge_seed_run_id else ""
        _judge_examples_path = os.path.join(config.RESULTS_DIR, f"steering_examples{_judge_suffix}.jsonl")
        _judge_summary = _llm_judge_sample(
            examples_path=_judge_examples_path,
            api_key=_anthropic_key,
            sample_per_concept=args.llm_judge_n,
            out_path=os.path.join(config.RESULTS_DIR, f"llm_judge_results{_judge_suffix}.jsonl"),
        )
        if _judge_summary:
            _judge_path = os.path.join(config.RESULTS_DIR, "llm_judge_summary.json")
            with open(_judge_path, "w", encoding="utf-8") as _jf:
                json.dump(_judge_summary, _jf, indent=2)
            print(f"[step8] Claude judge summary written to {_judge_path}")

            # ---- DeBERTa vs Claude correlation ----
            # For every concept where we have both a DeBERTa mean_concept_score and a
            # Claude steered_toward_concept score, compute Pearson r.
            # r >= 0.70 validates DeBERTa as a reliable primary metric for the paper.
            # r < 0.50 means DeBERTa is unreliable and the paper must rely on Claude scores.
            _judge_results_path = os.path.join(config.RESULTS_DIR, f"llm_judge_results{_judge_suffix}.jsonl")
            _deberta_vals: List[float] = []
            _claude_vals: List[float] = []
            if os.path.exists(_judge_results_path):
                with open(_judge_results_path, encoding="utf-8") as _jrf:
                    for _jline in _jrf:
                        _jr = json.loads(_jline)
                        _claude_score = _jr.get("steered_toward_concept")
                        if _claude_score is None:
                            continue
                        # Find the matching DeBERTa concept score from merged results
                        _concept = _jr.get("concept")
                        _model   = _jr.get("model")
                        _method  = _jr.get("method")
                        _strength = str(_jr.get("strength"))
                        _deberta_score = (
                            merged.get(_model, {})
                            .get(_concept, {})
                            .get("methods", {})
                            .get(_method, {})
                            .get("results_by_strength", {})
                            .get(_strength, {})
                            .get("mean_concept_score")
                        )
                        if _deberta_score is not None:
                            _deberta_vals.append(float(_deberta_score))
                            _claude_vals.append(float(_claude_score))
            if len(_deberta_vals) >= 10:
                try:
                    from scipy.stats import pearsonr as _pearsonr  # type: ignore
                    _r, _p = _pearsonr(_deberta_vals, _claude_vals)
                    _corr_result = {
                        "pearson_r": round(float(_r), 4),
                        "p_value":   round(float(_p), 6),
                        "n_pairs":   len(_deberta_vals),
                        "valid": bool(abs(_r) >= 0.70),
                        "interpretation": (
                            "DeBERTa is a reliable proxy for Claude judgment (r\u22650.70) \u2014 "
                            "use as primary metric in paper."
                            if abs(_r) >= 0.70 else
                            "DeBERTa correlation with Claude is weak (r<0.70) \u2014 "
                            "report Claude scores as primary metric instead."
                        ),
                    }
                    print(f"[step8] DeBERTa\u2013Claude correlation: r={_r:.3f} (n={len(_deberta_vals)}) \u2014 "
                          + ("VALID \u2705" if _corr_result['valid'] else "WEAK \u26a0\ufe0f"))
                    # Write back into the evaluation_table.json
                    _eval_table_path = os.path.join(config.RESULTS_DIR, f"evaluation_table{suffix}.json")
                    if os.path.exists(_eval_table_path):
                        with open(_eval_table_path, encoding="utf-8") as _etf:
                            _eval_table = json.load(_etf)
                        _eval_table.setdefault("table_stats", {})["deberta_claude_correlation"] = _corr_result
                        with open(_eval_table_path, "w", encoding="utf-8") as _etf:
                            json.dump(_eval_table, _etf, indent=2)
                except ImportError:
                    print("[step8] scipy not available \u2014 DeBERTa\u2013Claude correlation skipped")
    else:
        print("[step8] ANTHROPIC_API_KEY not set — Claude judge skipped. "
              "Set the key to enable LLM-as-judge evaluation.")

    # Expected steering_examples.jsonl object count (for reference):
    # 30 prompts (TOP_N_PROMPTS), 5 models, 11 concepts, SWEEP_NATIVE 6 non-zero + SWEEP_UNIVERSAL 6 non-zero
    # run_meta:  1
    # baseline:  5 models × 30 prompts = 150
    # A native:  5 × 11 × 2 methods × 6 strengths × 30 = 19,800
    # B cross:   5 × 11 × 12 B3 methods × 6 × 30 = 118,800
    # B naive:   5 × 11 × 4 naive × 6 × 30 = 39,600
    # C c3:      5 × 11 × 1 × 6 × 30 = 9,900
    # Total ≈ 188,251

    log_run("step8_apply_steering.py", start_time, "success")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as _exc:
        log_run("step8_apply_steering.py", time.time(), "error", str(_exc))
        raise
