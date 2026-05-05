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

# --- L4 GPU memory management ---
# CRITICAL: only one GPU model loaded at a time. Between Gemma and Llama jobs:
#   del model
#   torch.cuda.empty_cache()
#   import gc; gc.collect()
# Also set this env var before loading Gemma:
#   os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# --- SAE settings ---
SAE_EXPANSION_FACTOR = 16        # hidden_dim * 16 = number of features
# TopK is now per-model — use MODELS[model]["sae_topk"]
SAE_BATCH_SIZE = 4096
SAE_LR = 1e-4
# NOTE: train steps are per-model — use MODELS[model]["sae_train_steps"]

# --- Dataset sources — verified row counts from HuggingFace ---
# All freely available, no license approval needed
DATASET_SOURCES = {
    "sst2": {
        "hf_path": "nyu-mll/glue", "config": "sst2",
        "split": "train",        # MUST be train (67,349 rows). validation only has 872.
        "text_field": "sentence", "label_field": "label",
        "n": 5000,
        "hf_url": "https://huggingface.co/datasets/nyu-mll/glue"
    },
    "yelp": {
        "hf_path": "fancyzhx/yelp_polarity", "config": None,
        "split": "train",        # 560,000 rows
        "text_field": "text", "label_field": "label",
        "n": 3000,
        "hf_url": "https://huggingface.co/datasets/fancyzhx/yelp_polarity"
    },
    "truthfulqa": {
        "hf_path": "truthfulqa/truthful_qa", "config": "generation",
        "split": "validation",   # ONLY split available — 817 rows total, use ALL
        "text_field": "question", "label_field": "correct_answers",
        "n": 817,                # use every single row
        "hf_url": "https://huggingface.co/datasets/truthfulqa/truthful_qa"
    },
    "formality": {
        "hf_path": "osyvokon/pavlick-formality-scores", "config": None,
        "split": "train",        # 9,270 rows — sample from this
        "text_field": "sentence", "label_field": "avg_score",
        # Score range: -3.0 to +3.0 (NOT 1-7)
        # Binarize: formality=1 if avg_score >= 1.0, formality=0 if avg_score <= -1.0
        # Skip rows with -1.0 < avg_score < 1.0
        "n": 1183,
        "hf_url": "https://huggingface.co/datasets/osyvokon/pavlick-formality-scores"
    },
}
# Total: 5000 + 3000 + 817 + 1183 = 10,000 exactly

N_PASSAGES = 10_000
MAX_SEQ_LEN = 256  # increased from 128 — needed to capture full code function bodies

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
HF_REPO_ID = "YOUR_HF_USERNAME/universal-steering-data"  # CHANGE THIS — private dataset repo
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # set in .env file, never hardcode

# Polarity corrections for cross-model transfer
# These pairs have inverted sign conventions between independently trained SAEs
# Value -1 means negate the transferred vector before applying
POLARITY_CORRECTIONS = {
    ("gemma", "gpt2-large"): -1,
    ("llama", "gpt2-large"): -1,
}

