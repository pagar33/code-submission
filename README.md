# Universal Concept Representations Across Language Models

Replication code for the NeurIPS 2026 submission.

---

## Overview

This repository contains the full pipeline for discovering and transferring **universal concept representations** — steering vectors that generalise across model families without model-specific fine-tuning.

The pipeline is organised into four tracks:

| Track | Stage | Script | Description |
|-------|-------|--------|-------------|
| **A** | A1 | `pipeline/a1_download_data.py` | Download corpus and model weights |
| | A2 | `pipeline/a2_extract_activations.py` | Extract residual-stream activations |
| | A3 | `pipeline/a3_train_sae.py` | Train sparse autoencoder per model |
| | A4 | `pipeline/a4_normalise_activations.py` | Normalise activations across domains |
| | A4b | `pipeline/a4b_label_features.py` | Label SAE features via LLM |
| | A5 | `pipeline/a5_build_steering.py` | Build per-model steering vectors |
| **B** | B1 | `pipeline/b1_align_features.py` | Learn cross-model MLP alignment bridges |
| | B2 | `pipeline/b2_validate_alignment.py` | Validate alignment quality |
| **C** | C1 | `pipeline/c1_train_global_mlp.py` | Train global concept MLP |
| | C2 | `pipeline/c2_discover_concepts.py` | Discover universal concept clusters |
| | C2b | `pipeline/c2b_auto_discover.py` | Auto-discover concept axes |
| | C2c | `pipeline/c2c_label_concepts.py` | Label universal concepts via LLM |
| | C2d | `pipeline/c2d_dedup_concepts.py` | Deduplicate concept atlas |
| | C3 | `pipeline/c3_build_vectors.py` | Project concepts → per-model vectors |
| **D** | D1 | `pipeline/d1_evaluate_native.py` | Evaluate native (per-model) steering |
| | D2 | `pipeline/d2_evaluate_universal.py` | Evaluate universal steering transfer |

---

## Setup

```bash
# 1. Clone the repo
git clone <repo_url> && cd concept-universality

# 2. Create environment
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. Set credentials
cp .env.example .env
# Edit .env: add your HF_TOKEN and WANDB_API_KEY
```

---

## Asset Directories

Large binary assets are **not stored in git**. Download them from our Hugging Face dataset repo:

```bash
python pipeline/a1_download_data.py   # downloads corpus → data/
# Model weights and SAE checkpoints are downloaded automatically by each script
```

The directory layout mirrors the structure described in the paper:

```
activations/   # per-model, per-domain activation h5 files
alignment/     # MLP bridge checkpoints (.pt) and aligned pair metadata
data/          # raw corpus (corpus.jsonl, ~2 GB)
features/      # SAE feature label JSONs (in git)
logs/          # run logs
model/         # LLM weights (downloaded from HF)
results/       # evaluation outputs
saes/          # trained SAE checkpoints
steering/      # per-model and universal steering vectors
universal/     # global MLP and concept atlas
```

---

## Configuration

All paths, model names, and hyperparameters are in `config.py`. Override any value via environment variables (see `.env.example`) or by editing the file directly.

---

## Models

Experiments use the following five models (auto-downloaded via `huggingface_hub`):

- GPT-2 (124M)
- GPT-2-Large (774M)
- LLaMA-3-8B
- Mistral-7B-v0.3
- DeepSeek-LLM-7B

---

## Citation

```bibtex
@inproceedings{anonymous2026universal,
  title     = {Universal Concept Representations Across Language Models},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```
