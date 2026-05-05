# Universal Concept Representations Across Language Models

---

## Repository layout

```
.
├── config.py                          # all paths, model configs, hyperparameters
├── pyproject.toml
├── requirements.txt
│
├── pipeline/
│   ├── a1_download_data.py            # download corpus + model weights
│   ├── a2_extract_activations.py      # extract residual-stream activations
│   ├── a3_train_sae.py                # train sparse autoencoder per model
│   ├── a4_normalise_activations.py    # normalise activations across domains
│   ├── a4b_label_features.py          # label SAE features via LLM
│   ├── a5_build_steering.py           # build per-model (native) steering vectors
│   │
│   ├── b1_align_features.py           # learn cross-model MLP alignment bridges
│   ├── b2_validate_alignment.py       # validate alignment quality
│   ├── b3_build_cross_steering.py     # build cross-model steering vectors
│   │
│   ├── c1_train_global_mlp.py         # train global concept MLP
│   ├── c2_discover_concepts.py        # discover universal concept clusters
│   ├── c2b_auto_discover.py           # auto-discover concept axes
│   ├── c2c_label_concepts.py          # label universal concepts via LLM
│   ├── c2d_dedup_concepts.py          # deduplicate concept atlas
│   ├── c3_build_vectors.py            # project concepts → per-model vectors
│   │
│   └── d1_evaluate_native.py          # evaluate steering (native + universal)
│
├── activations/                       # ← download from HF (see below)
├── alignment/                         # ← download from HF
├── data/                              # ← download from HF
├── features/                          # SAE feature label JSONs (in git)
├── logs/
├── model/                             # ← download from HF
├── results/
├── saes/                              # ← download from HF
├── steering/                          # ← download from HF
└── universal/                         # ← download from HF
```

---

## Setup

```bash
git clone https://github.com/pagar33/code-submission.git && cd code-submission
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # add HF_TOKEN and WANDB_API_KEY
```

---

## Downloading assets from Hugging Face

Large binary assets (activations, SAE checkpoints, alignment bridges, steering vectors, corpus) are hosted on Hugging Face and **not stored in git**.

```bash
python pipeline/a1_download_data.py
```

This populates `data/`, `model/`, `saes/`, `activations/`, `alignment/`, `steering/`, and `universal/` from the dataset repo specified in `config.py`.

---

## Models

| Key in `config.py` | HuggingFace ID | Hidden dim |
|--------------------|----------------|------------|
| `gpt2-large` | `gpt2-large` | 1280 |
| `gemma` | `google/gemma-2-2b` | 2304 |
| `llama` | `NousResearch/Hermes-3-Llama-3.1-8B` | 4096 |
| `mistral` | `mistralai/Mistral-7B-v0.3` | 4096 |
| `deepseek-llm-7b` | `deepseek-ai/deepseek-llm-7b-base` | 4096 |

---

## Running the pipeline

Run scripts in stage order. Each script reads paths and hyperparameters from `config.py`.

```bash
# A-track: per-model
python pipeline/a1_download_data.py
python pipeline/a2_extract_activations.py --model gpt2-large
python pipeline/a3_train_sae.py           --model gpt2-large
python pipeline/a4_normalise_activations.py --model gpt2-large
python pipeline/a4b_label_features.py     --model gpt2-large
python pipeline/a5_build_steering.py      --model gpt2-large

# B-track: cross-model alignment
python pipeline/b1_align_features.py --model-a gpt2-large --model-b llama
python pipeline/b2_validate_alignment.py
python pipeline/b3_build_cross_steering.py --guide-model gpt2-large --target-model llama

# C-track: universal concept discovery
python pipeline/c1_train_global_mlp.py
python pipeline/c2_discover_concepts.py
python pipeline/c3_build_vectors.py

# D-track: evaluation
python pipeline/d1_evaluate_native.py
```
