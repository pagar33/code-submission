# config.py
import os
from pathlib import Path

# Get the directory where this config.py is located
_CONFIG_DIR = Path(__file__).parent

# --- Model settings ---
MODELS = {
    "gpt2-large": {
        "hf_name": "gpt2-large",
        "hidden_dim": 1280,
        "target_layer": 19,
        "n_layers": 36,
        "device": "cuda",
        "extract_batch_size": 128,
        "dtype": "bf16",
        "extract_backend": "transformers",
        "hook_output_index": 0,
        "sae_topk": 64,
        "sae_ef": 64,
        "sae_train_steps": 100_000,
    },
    "gemma": {
        "hf_name": "google/gemma-2-2b",
        "hidden_dim": 2304,
        "target_layer": 13,
        "n_layers": 26,
        "device": "cuda",
        "extract_batch_size": 64,
        "dtype": "bf16",
        "extract_backend": "transformers",
        "hook_output_index": 0,
        "sae_topk": 100,
        "sae_ef": 64,
        "sae_train_steps": 100_000,
    },
    "llama": {
        "hf_name": "NousResearch/Hermes-3-Llama-3.1-8B",
        "hidden_dim": 4096,
        "target_layer": 16,
        "n_layers": 32,
        "device": "cuda",
        "extract_batch_size": 64,
        "dtype": "bf16",
        "extract_backend": "transformers",
        "hook_output_index": 0,
        "sae_topk": 200,
        "sae_ef": 128,
        "sae_train_steps": 200_000,
    },
    "mistral": {
        "hf_name": "mistralai/Mistral-7B-v0.3",
        "hidden_dim": 4096,
        "target_layer": 16,
        "n_layers": 32,
        "device": "cuda",
        "extract_batch_size": 32,
        "dtype": "bf16",
        "extract_backend": "transformers",
        "hook_output_index": 0,
        "sae_topk": 200,
        "sae_ef": 128,
        "sae_train_steps": 200_000,
    },
    "deepseek-llm-7b": {
        "hf_name": "deepseek-ai/deepseek-llm-7b-base",
        "hidden_dim": 4096,
        "target_layer": 15,
        "n_layers": 30,
        "device": "cuda",
        "extract_batch_size": 32,
        "dtype": "bf16",
        "extract_backend": "transformers",
        "hook_output_index": 0,
        "sae_topk": 200,
        "sae_ef": 128,
        "sae_train_steps": 200_000,
    },
}

# --- SAE settings ---
SAE_EXPANSION_FACTOR = 16        # hidden_dim * 16 = number of features
# TopK is now per-model — use MODELS[model]["sae_topk"]
SAE_BATCH_SIZE = 4096
SAE_LR = 1e-4
# NOTE: train steps are per-model — use MODELS[model]["sae_train_steps"]

# --- Dataset sources — all freely available on HuggingFace ---
# Production corpus: 394,508 passages across 17 domain-diverse datasets.
# Each entry specifies how to obtain and process text from the source dataset.
# Fields:
#   hf_path / hf_config / split / n (0 = all rows)
#   text_mode: "single" (one field) or "concat" (join two fields with sep)
#   text_field: field name (single) or [field1, field2] (concat)
#   sep: separator for concat mode
#   min_tok / max_tok: approximate token-count filter (word-count proxy)
#   strip_html, remove_urls, preserve_code: text-cleaning flags
#   strip_regex: regex to remove from text before filtering
DATASET_SOURCES = {
    "code_python_50k": {
        "hf_path": "codeparrot/codeparrot-clean", "hf_config": None, "split": "train",
        "n": 50_000, "text_mode": "single", "text_field": "content",
        "min_tok": 20, "max_tok": 512, "preserve_code": True,
    },
    "code_python_instructions_15k": {
        "hf_path": "iamtarun/python_code_instructions_18k_alpaca", "hf_config": None, "split": "train",
        "n": 15_000, "text_mode": "single", "text_field": "output",
        "min_tok": 20, "max_tok": 512, "preserve_code": True,
    },
    "code_python_snippets_5k": {
        "hf_path": "flytech/python-codes-25k", "hf_config": None, "split": "train",
        "n": 5_000, "text_mode": "single", "text_field": "output",
        "min_tok": 20, "max_tok": 512, "preserve_code": True,
    },
    "code_sql_50k": {
        "hf_path": "b-mc2/sql-create-context", "hf_config": None, "split": "train",
        "n": 50_000, "text_mode": "concat", "text_field": ["question", "answer"], "sep": "\n",
        "min_tok": 20, "max_tok": 512, "preserve_code": True,
    },
    "math_gsm8k_8k": {
        "hf_path": "openai/gsm8k", "hf_config": "main", "split": "train",
        "n": 0, "text_mode": "concat", "text_field": ["question", "answer"], "sep": "\n",
        "strip_regex": r"\n?#+.*$",
        "min_tok": 20, "max_tok": 512,
    },
    "math_metamath_50k": {
        "hf_path": "meta-math/MetaMathQA", "hf_config": None, "split": "train",
        "n": 50_000, "text_mode": "concat", "text_field": ["query", "response"], "sep": "\n",
        "min_tok": 20, "max_tok": 512,
    },
    "math_tiger_50k": {
        "hf_path": "TIGER-Lab/MATH-plus", "hf_config": None, "split": "train",
        "n": 50_000, "text_mode": "concat", "text_field": ["problem", "solution"], "sep": "\n",
        "min_tok": 20, "max_tok": 512,
    },
    "math_numina_50k": {
        "hf_path": "AI-MO/NuminaMath-CoT", "hf_config": None, "split": "train",
        "n": 50_000, "text_mode": "concat", "text_field": ["problem", "solution"], "sep": "\n",
        "min_tok": 20, "max_tok": 512,
    },
    "sentiment_yelp_50k": {
        "hf_path": "fancyzhx/yelp_polarity", "hf_config": None, "split": "train",
        "n": 50_000, "text_mode": "single", "text_field": "text",
        "min_tok": 20, "max_tok": 512, "strip_html": True, "remove_urls": True,
    },
    "creative_writing_50k": {
        "hf_path": "euclaise/writingprompts", "hf_config": None, "split": "train",
        "n": 50_000, "text_mode": "single", "text_field": "story",
        "min_tok": 20, "max_tok": 512, "remove_urls": True,
    },
    "academic_arxiv_50k": {
        "hf_path": "ccdv/arxiv-summarization", "hf_config": "document", "split": "train",
        "n": 50_000, "text_mode": "single", "text_field": "abstract",
        "min_tok": 20, "max_tok": 512, "remove_urls": True,
    },
    "science_pubmed_50k": {
        "hf_path": "qiaojin/PubMedQA", "hf_config": "pqa_unlabeled", "split": "train",
        "n": 50_000, "text_mode": "single", "text_field": "long_answer",
        "min_tok": 20, "max_tok": 512, "remove_urls": True,
    },
    "legal_freelaw_50k": {
        "hf_path": "pile-of-law/pile-of-law", "hf_config": "freelaw", "split": "train",
        "n": 50_000, "text_mode": "single", "text_field": "text",
        "min_tok": 20, "max_tok": 512, "strip_html": True, "remove_urls": True,
    },
    "news_ccnews_50k": {
        "hf_path": "cc_news", "hf_config": None, "split": "train",
        "n": 50_000, "text_mode": "single", "text_field": "text",
        "min_tok": 20, "max_tok": 512, "strip_html": True, "remove_urls": True,
    },
    "qa_squad_50k": {
        "hf_path": "rajpurkar/squad", "hf_config": None, "split": "train",
        "n": 50_000, "text_mode": "single", "text_field": "context",
        "min_tok": 20, "max_tok": 512, "remove_urls": True,
    },
    "prose_openwebtext_50k": {
        "hf_path": "Skylion007/openwebtext", "hf_config": "plain_text", "split": "train",
        "n": 50_000, "text_mode": "single", "text_field": "text",
        "min_tok": 20, "max_tok": 512, "strip_html": True, "remove_urls": True,
    },
    "prose_wikipedia_50k": {
        "hf_path": "wikimedia/wikipedia", "hf_config": "20231101.en", "split": "train",
        "n": 50_000, "text_mode": "single", "text_field": "text",
        "min_tok": 20, "max_tok": 512, "strip_html": True, "remove_urls": True,
    },
}

N_PASSAGES = 394_508   # total passages after min/max token filtering
MAX_SEQ_LEN = 512      # max tokens per passage for activation extraction

# --- Alignment ---
ALIGNMENT_CONFIDENCE_THRESHOLD = 0.7
TOP_FEATURES_FOR_ALIGNMENT = 500

# --- Steering ---
STEERING_STRENGTHS = [0.5, 1.0, 1.5, 2.0, 3.0]
TARGET_CONCEPTS = ['formality', 'sentiment', 'certainty', 'empathy', 'hedging', 'coding', 'pythoncoding', 'legal', 'code_python', 'code_instructions', 'code_snippets', 'code_sql', 'math_gsm8k', 'math_reasoning', 'math_competition', 'math_olympiad', 'creative_writing', 'academic_writing', 'science_biomedical', 'news_reporting', 'question_answering']

# --- Paths ---
DATA_DIR = str(_CONFIG_DIR / "data")
ACTIVATIONS_DIR = str(_CONFIG_DIR / "activations")
SAE_DIR = str(_CONFIG_DIR / "saes")
FEATURES_DIR = str(_CONFIG_DIR / "features")
ALIGNMENT_DIR = str(_CONFIG_DIR / "alignment")
UNIVERSAL_DIR = str(_CONFIG_DIR / "universal")   # Section C outputs (C2/C3/C4)
STEERING_DIR = str(_CONFIG_DIR / "steering")
RESULTS_DIR = str(_CONFIG_DIR / "results")

# --- HuggingFace hub (for data/checkpoint storage — all large files live here) ---
# Code lives on GitHub. Data, activations, SAEs, results live on HF Hub.
HF_REPO_ID = "nips348734/submission-artifacts"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # set in .env file, never hardcode

# Polarity corrections for cross-model transfer
# These pairs have inverted sign conventions between independently trained SAEs
# Value -1 means negate the transferred vector before applying
POLARITY_CORRECTIONS = {
    ("gemma", "gpt2-large"): -1,
    ("llama", "gpt2-large"): -1,
}

