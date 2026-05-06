# Methodology

End-to-end pipeline for universal steering vector discovery and evaluation across five LLMs.  
Tracks data → activations → SAEs → labelling → cross-model alignment → universal concept space → steering → evaluation.

---

## Models

| Key | HuggingFace ID | Hidden dim | Target layer (depth %) | SAE ef | SAE top-k (eff.) |
|-----|----------------|-----------|----------------------|--------|------------------|
| `gpt2-large` | `gpt2-large` | 1 280 | 19 / 36 (53%) | 64 | 256 |
| `gemma` | `google/gemma-2-2b` | 2 304 | 13 / 26 (50%) | 64 | 400 |
| `llama` | `NousResearch/Hermes-3-Llama-3.1-8B` | 4 096 | 16 / 32 (50%) | 128 | 1 600 |
| `mistral` | `mistralai/Mistral-7B-v0.3` | 4 096 | 16 / 32 (50%) | 128 | 1 600 |
| `deepseek-llm-7b` | `deepseek-ai/deepseek-llm-7b-base` | 4 096 | 15 / 30 (50%) | 128 | 1 600 |

---

## A1 — Data Download (`a1_download_data.py`)

Assembles a corpus of **394,508 passages** from 17 HuggingFace datasets covering 15 domains.

| Source tag | HuggingFace path | Config | Split | Rows |
|------------|-----------------|--------|-------|------|
| `code_python_50k` | `codeparrot/codeparrot-clean` | — | train | 60,292 |
| `code_python_instructions_15k` | `iamtarun/python_code_instructions_18k_alpaca` | — | train | 16,667 |
| `code_python_snippets_5k` | `flytech/python-codes-25k` | — | train | 5,029 |
| `code_sql_50k` | `b-mc2/sql-create-context` | — | train | 50,047 |
| `math_gsm8k_8k` | `openai/gsm8k` | main | train | 7,473 |
| `math_metamath_50k` | `meta-math/MetaMathQA` | — | train | 50,000 |
| `math_tiger_50k` | `TIGER-Lab/MATH-plus` | — | train | 50,000 |
| `math_numina_50k` | `AI-MO/NuminaMath-CoT` | — | train | 50,000 |
| `sentiment_yelp_50k` | `fancyzhx/yelp_polarity` | — | train | 50,000 |
| `creative_writing_50k` | `euclaise/writingprompts` | — | train | 50,000 |
| `academic_arxiv_50k` | `ccdv/arxiv-summarization` | document | train | 50,000 |
| `science_pubmed_50k` | `qiaojin/PubMedQA` | pqa_unlabeled | train | 50,000 |
| `legal_freelaw_50k` | `pile-of-law/pile-of-law` | freelaw | train | 50,000 |
| `news_ccnews_50k` | `cc_news` | — | train | 50,000 |
| `qa_squad_50k` | `rajpurkar/squad` | — | train | 50,000 |
| `prose_openwebtext_50k` | `Skylion007/openwebtext` | plain_text | train | 50,000 |
| `prose_wikipedia_50k` | `wikimedia/wikipedia` | 20231101.en | train | 50,000 |

**Common pre-processing (all datasets):** normalise whitespace; strip HTML and URLs; code datasets preserve code blocks; token length filter `min=10–50`, `max=256–512` tokens per passage (per-dataset).

**Text construction:**
- Multi-field datasets (SQL, math, QA): concatenate `question + "\n" + answer` (or `problem + "\n" + solution`).
- Single-field datasets: use the designated text column directly.
- `math_gsm8k_8k`: strip computation markers (`\n*####.*` regex after concatenation).

**Output:** `data/corpus.jsonl`, one passage per line with `{text, source, domain}`.

---

## A2 — Activation Extraction (`a2_extract_activations.py`)

Runs every passage through each model and stores the residual-stream representation at the target layer.

**Model loading**  
- `bfloat16` on CUDA.  
- `gpt2-large`: [TransformerLens](https://github.com/neelnanda-io/TransformerLens) `HookedTransformer`; hook at `blocks.{layer}.hook_resid_post`.  
- All others: HuggingFace `AutoModelForCausalLM` with `output_hidden_states=True`; takes `hidden_states[target_layer]`.

**Pooling:** mean pool over non-padding token positions (attention-mask weighted sum ÷ number of valid tokens). This is the prevailing approach in the literature for extracting passage-level representations from causal LMs and is used as the industry standard baseline. Future work should systematically compare last-token, mean-pool, and max-pool strategies and investigate how pooling choice interacts with SAE feature semantics and downstream steering effectiveness.

**Layer hook:** residual-stream post-MLP (`hook_resid_post` / `hidden_states[target_layer]`). Middle-layer residual streams carry maximal semantic content — early layers are primarily syntactic, late layers specialise toward output token prediction (Geva et al. 2021). All target layers are at ≈50% network depth. Future work should study how layer selection and pooling strategy jointly affect SAE feature quality and transfer.

**Tokenisation:** `padding=True`, `truncation=True`, `max_length=256`.

| Model | Batch size |
|-------|-----------|
| `gpt2-large` | 128 |
| `gemma` | 64 |
| `llama` / `mistral` | 64 |
| `deepseek-llm-7b` | 32 |

Activations saved as `float32` in HDF5.

**Output:** `activations/{model}_{source}_activations.h5`  — shape `(10 000, hidden_dim)`.

---

## A3 — Sparse Autoencoder Training (`a3_train_sae.py`)

Trains a TopK Sparse Autoencoder (SAE) per model to decompose residual-stream activations into an over-complete feature dictionary.

### Architecture — `TopKSAE`

```
x  →  encoder: Linear(hidden_dim, n_features, bias=True)
   →  TopK: keep top-k values, zero the rest, apply ReLU
   →  decoder: Linear(n_features, hidden_dim, bias=True)
   →  x̂
```

`n_features = hidden_dim × sae_ef` (expansion factor):

| Model | `hidden_dim` | `sae_ef` | `n_features` | `top_k` (effective) | Train steps | Batch | Ghost grads | Final recon loss |
|-------|-------------|---------|-------------|---------------------|------------|-------|-------------|------------------|
| `gpt2-large` | 1 280 | 64 | 81 920 | 256 | 91 000 | 4 096 | off | 0.065 |
| `gemma` | 2 304 | 64 | 147 456 | 400 | 100 000 | 4 096 | on | 0.147 |
| `llama` | 4 096 | 128 | 524 288 | 1 600 | 200 000 | 2 048 | on | 0.250 |
| `mistral` | 4 096 | 128 | 524 288 | 1 600 | 200 000 | 4 096 | on | 0.076 |
| `deepseek-llm-7b` | 4 096 | 128 | 524 288 | 1 600 | 200 000 | 4 096 | on | 0.223 |

`top_k` is auto-scaled from the base sparsity ratio: $k_\text{eff} = \lfloor n_\text{features} \times (k_\text{base} / (d_\text{hidden} \times \text{EF}_\text{default})) \rfloor$, where $\text{EF}_\text{default}=16$.

Elevated recon loss on llama (0.250) and deepseek (0.223) is consistent with the larger 524K-feature space; both converged to 0% dead features.

### Training details

- **Loss:** $\mathcal{L} = \text{MSE}(\hat{x}, x) + \lambda_{\text{sp}} \cdot \|z\|_1$, with $\lambda_{\text{sp}} = 10^{-4}$.  
- **Optimizer:** Adam, `lr = 1e-4`, `warmup = 2 000 steps`.  
- **Batch size:** per-model (see table); `ShuffleBufferSampler` with 200 000-row RAM buffer.  
- **Decoder normalisation:** after every optimiser step, each decoder column $d_j$ is L2-renormalised: $d_j \leftarrow d_j / \max(\|d_j\|_2, 10^{-8})$.  
- **Ghost gradients:** enabled for all 7B models (prevents dead-neuron accumulation in large SAEs); disabled for `gpt2-large` (sufficient gradient flow at smaller scale).  
- **Early stopping threshold:** $5 \times 10^{-5}$ (gpt2-large, llama); $10^{-4}$ (gemma).  
- **Checkpointing:** every 1 000 steps.

**Output:** `saes/{model}_ef{ef}_sae.pt`

---

## A4 — Activation Normalisation (`a4_normalise_activations.py`)

Z-scores every feature dimension independently:

```
x_norm[d] = (x[d] - mean[d]) / std[d],   std floored at 1e-8
```

Statistics are computed per source file separately, then saved alongside the normalised activations.

**Outputs:** `activations/{model}_{source}_activations_norm.h5`, `activations/{model}_{source}_norm_stats.json`

---

## A4b — Feature Labelling (`a4b_label_features.py`)

Assigns each SAE feature to a semantic concept by measuring its differential activation.

### Algorithm

1. Run all normalised activations through the trained SAE in batches (`batch_size=4096` on GPU, `512` on CPU) → sparse feature activation matrix $F$ of shape $(N, n_\text{features})$.  
2. For each concept $c$ and feature $f$, compute:

$$\delta_c[f] = \bar{F}_{\text{pos},c}[f] - \bar{F}_{\text{neg},c}[f]$$

   All 15 supervised domain concepts: split by corpus source tag (`domain` field in `corpus.jsonl`). There is no core/domain distinction — all concepts are treated uniformly.

3. **Feature selection:**
   - Threshold: `min_delta_sentiment = 0.02`, `min_delta_other = 0.05`
   - Keep top `per_domain = 150` features per concept by $|\delta_c[f]|$
   - Also keep top-`100` features by mean absolute activation
   - **Min feature confidence: `0.0`** — no confidence floor; all features meeting the delta threshold are retained

4. **Domain assignment:** $\text{domain}(f) = \arg\max_c |\delta_c[f]|$

5. **Confidence score:** $\text{conf}(f) = \delta_{\text{best}}(f) \;/\; (\textstyle\sum_c |\delta_c[f]| + 10^{-8})$

### Results

| Model | SAE `n_features` | EF | Features selected |
|-------|-----------------|-----|------------------|
| `gpt2-large` | 81 920 | 64 | 666 (supervised, used in B1/C1) + 816 auto-discovered (A4b-auto; see below) |
| `gemma` | 147 456 | 64 | 865 |
| `llama` | 524 288 | 128 | 871 |
| `mistral` | 524 288 | 128 | 883 |
| `deepseek-llm-7b` | 524 288 | 128 | 815 |

**`gpt2-large` note:** supervised labelling produced 666 features — below the 800–880 range of the 7B models, reflecting gpt2-large's narrower representational diversity. The auto-discovery step (below) was run exclusively for `gpt2-large`, producing an additional 816 features across 48 clusters (stored separately; not included in B1 alignment). Only the 666 supervised features were used in B1/C1.

**Output:** `features/{model}_ef{ef}_feature_labels.json`

---

## A4b-auto — gpt2-large Auto-Discovery (`c2b_auto_discover.py`, gpt2-large only)

Co-activation clustering to surface latent concepts not captured by the supervised delta method. **Run only for `gpt2-large`** (other models had sufficient supervised features).

| Parameter | Value |
|-----------|-------|
| Model | `gpt2-large` |
| EF | 64 |
| Min activation variance (`min_var`) | 0.01 |
| Min cluster size (`min_cluster`) | 5 |
| Cluster merge threshold (`merge_thresh`) | 0.85 |
| Max passages | 10 000 |
| Max features to cluster | 8 000 |
| LLM for naming | `claude-sonnet-4-5` |

**Result:** 48 auto-discovered concept clusters; 816 additional features across those clusters. These are stored separately in the feature label file and are not included in B1 alignment (which uses the 666 supervised features only).

**Output:** `features/gpt2-large_ef64_autodiscovered.json`

---

## A5 — Native Steering Vectors (`a5_build_steering.py`)

Builds per-model, per-concept steering vectors using both methods simultaneously.

| Parameter | Value |
|-----------|-------|
| Mode | Both (CAA + SAE decoder) |
| Top features per concept | 3 |
| Min feature confidence | 0.0 (no threshold — use all features meeting A4b delta criteria) |
| Passages sampled per class (CAA) | 50 |

### Method 1 — SAE decoder (feature-weighted)

For each concept, take the top-3 SAE features by $|\delta_c[f]|$. Build the steering vector as:

$$\vec{v}_{\text{SAE}} = \text{L2-norm}\!\left(\sum_{i=1}^{3} \text{conf}_i \cdot D_{:,f_i}\right)$$

where $D_{:,f_i}$ is the $i$-th SAE decoder column (a direction in hidden-dim space).

### Method 2 — Contrastive Activation Addition (CAA)

Random sample of 50 positive and 50 negative passages per concept:

$$\vec{v}_{\text{CAA}} = \text{L2-norm}\!\left(\frac{1}{50}\sum_{p \in \text{pos}} h_p - \frac{1}{50}\sum_{n \in \text{neg}} h_n\right)$$

where $h_p, h_n \in \mathbb{R}^{d_\text{hidden}}$ are the A2 mean-pooled activations.

Both vectors and their mutual cosine similarity are stored per concept.

**Output:** `steering/{model}_ef{ef}_steering_vectors.json`

---

## B1 — Cross-Model Feature Alignment (`b1_align_features.py`)

Aligns the SAE feature spaces of every model pair so same-concept features can be matched.

### Input

Top-`K = 500` (`TOP_FEATURES_FOR_ALIGNMENT`) features per model, stored as `float16` memory-mapped arrays on `/tmp` (avoids OOM). Up to `max_passages = 100 000` passages used.

### Alignment methods and design rationale

Four scoring methods are computed per model pair and combined into a composite score. Pair extraction uses MNN-first with Hungarian assignment as fallback.

| Method | Key parameters | Role |
|--------|---------------|------|
| **CCA** (GPU) | 128 components, regularisation `1e-6`; whitened SVD cross-covariance | Produces CCA loading space used by MNN |
| **SVCCA** | SVD to top-64 singular vectors → CCA (Raghu et al. 2017) | Supplemental composite signal |
| **Procrustes** | `scipy.linalg.orthogonal_procrustes` | Global orthogonal alignment score |
| **MLP bridge** (3-layer) | `Linear(M_src, 8192) → ReLU → Dropout(0.1) → Linear(8192, M_tgt)` where `M = min(4096, ever-active features)`; Adam `lr=1e-3`, `weight_decay=1e-5`, 100 epochs, `batch=256`; cosine annealing `lr → lr/100`; Pearson penalty `weight=0.1`; trains on **full SAE activations** (not top-K) | Translation bridge for B3 — excluded from composite score when using full-dim inputs |
| **Mutual nearest neighbours (MNN)** | Bidirectional argmax in CCA loading space; keeps only pairs where `bwd[fwd[i]] == i` (Conneau et al. 2018) | Primary pair-extraction method |

**Method evolution:** CCA, Procrustes, SVCCA, and MLP were all prototyped for composite scoring. MNN emerged as the dominant pair-extraction strategy because it enforces bidirectional preference — features must mutually prefer each other in CCA loading space — eliminating hub contamination where a single popular feature gets spuriously matched to many targets. Hungarian assignment maximises global total score and forces 1:1 coverage even for unmatched features, diluting output with weak pairs. The final pipeline uses MNN-first extraction, falling back to Hungarian only when MNN produces no pairs.

**MLP training details:** both forward (guide → target) and reverse (target → guide) bridges are trained per pair. The MLP trains on full SAE activation matrices — up to `max_mlp_features = 4096` ever-active feature columns selected by cumulative absolute activation — not the top-K label set. The top-K matrices are still used for CCA/Procrustes/SVCCA/MNN scoring.

**Composite score:** `s_comp = mean({s_cca, s_svcca, s_procrustes, s_mnn})`. The MLP score is excluded from the composite when trained on full-dim inputs (different output space from the top-K feature space). Pairs are written to `aligned_pairs.jsonl` when `s_comp ≥ confidence = 0.7` (`ALIGNMENT_CONFIDENCE_THRESHOLD`).

**Outputs:** `alignment/aligned_pairs.jsonl`, `alignment/mlp_{guide}_to_{target}.pt`, companion `_src_idx.npy`, `_tgt_idx.npy`, `_src_stats.npz`, `_tgt_stats.npz`

### Run 2 results (definitive — 20 bidirectional pairs)

MNN extraction on all 5-model directional pairs yielded **3,308 validated pairs** across 20 guide→target directions.

| Guide → Target | Pairs |
|---|---|
| gpt2-large → gemma | 36 |
| gpt2-large → llama | 70 |
| gpt2-large → mistral | 54 |
| gpt2-large → deepseek | 85 |
| gemma → llama | 188 |
| gemma → mistral | 171 |
| gemma → deepseek | 182 |
| llama → mistral | 309 |
| llama → deepseek | 285 |
| mistral → deepseek | 274 |
| (+ 10 reverse directions, symmetric counts) | |

MLP training detail (gpt2-large → gemma, only fully instrumented pair): forward train_loss=0.3081, val_mse=0.5372, val_r=0.6538; reverse 0.2802/0.5238/0.5960. All other pairs converged comparably. (`b2_validate_alignment.py`)

Validates each aligned feature pair in `aligned_pairs.jsonl` using multiple statistical tests.

| Metric | Threshold / detail |
|--------|-------------------|
| Pearson *r* | `validation_pass` if `r ≥ 0.6` |
| Spearman *ρ* | reported |
| Lin's concordance CCC | reported |
| Permutation null test | 1 000 shuffles; Benjamini–Hochberg FDR correction |
| Co-activation rate | at P90 activation threshold |
| Neutral deconfound | `k = 3` nearest neutral-concept neighbours regressed out before scoring |
| Random-init baseline | enabled — shuffled-activation calibration for `threshold_status` |

Auto-generated cluster labels matching `^(concept|cluster|topic|unknown)[_\s]*\d` are excluded from domain scoring.

**Outputs:** `alignment/validation_results.jsonl`, `alignment/b2_results.json`

### Results (Run 2 — 3,308 pairs, 20 bidirectional)

**⚠️ rho_c naming note:** `rho_c_p90` is a sparse co-activation rate, **not** Lin's CCC. At ~5% SAE activation density the independence null is $0.05 \times 0.05 = 0.0025$; absolute values of 0.05–0.06 represent the physically attainable maximum. All results use `rho_c_fold_enrichment_p90` (vs permutation null). Lin's CCC is stored separately in the `ccc` field.

**Scale tier breakdown (do not headline the 45.5% overall — it mixes three scientifically distinct populations):**

| Scale tier | n | Pass% | Notes |
|---|---|---|---|
| 7B↔7B (llama/mistral/deepseek) | 1,736 | **48.9%** | Core universality finding |
| 1.7B↔7B (gemma↔7B) | 1,082 | **46.9%** | Only 2pp below same-scale; nearly indistinguishable |
| 0.8B↔1.7B (gpt2↔gemma) | 72 | **33.3%** | |
| 0.8B↔7B (gpt2↔7B) | 418 | **29.9%** | Representational capacity cliff below ~1.7B |
| **Overall** | **3,308** | **45.5%** | Mixture statistic |

**Direction symmetry:** fwd vs rev pass rates differ by ≤1.4pp for all 10 model pairs — strong methodological validity signal.

**Key metric distributions (3,308 pairs):**

| Metric | mean | median | p25 | p75 |
|---|---|---|---|---|
| procrustes_cosine_cca | 0.641 | 0.827 | 0.510 | 0.963 |
| saefree_procrustes_cosine | 0.925 | 0.937 | 0.908 | 0.955 |
| rho_c_fold_enrichment_p90 | 30.5× | 28.6× | 19.2× | 37.7× |
| pearson_r | 0.512 | 0.545 | 0.274 | 0.787 |
| ccc (Lin's) | 0.461 | 0.480 | 0.221 | 0.720 |
| rsa_score | 0.537 | 0.609 | 0.423 | 0.655 |
| cohen_d | 0.423 | 0.063 | −0.074 | 0.826 |

**NeurIPS claim (≥1.7B models):** "47–49% of MNN-extracted feature pairs validate cross-model (Pearson r ≥ 0.6, rho_c fold enrichment median 28.6× above permutation null, SAE-free procrustes cosine 0.895–0.956), with direction symmetry fwd/rev Δ ≤ 1.4pp across all pairs."

**Results by model pair (Run 2, both directions):**

| Guide → Target | n | Pass% | proc_cos | rho_c_clust | cohen_d | RSA | saefree |
|---|---|---|---|---|---|---|---|
| mistral → deepseek | 274 | 54.4% | 0.823 | 0.058 | 0.481 | 0.677 | 0.917 |
| deepseek → mistral | 274 | 53.3% | 0.823 | 0.058 | 0.464 | 0.677 | 0.895 |
| gemma → mistral | 171 | 52.0% | 0.757 | 0.056 | 0.571 | 0.513 | 0.946 |
| mistral → gemma | 171 | 52.0% | 0.757 | 0.056 | 0.592 | 0.513 | 0.928 |
| llama → deepseek | 285 | 47.7% | 0.663 | 0.057 | 0.432 | 0.655 | 0.922 |
| deepseek → llama | 285 | 46.3% | 0.663 | 0.057 | 0.396 | 0.655 | 0.931 |
| deepseek → gemma | 182 | 46.7% | 0.598 | 0.055 | 0.461 | 0.489 | 0.919 |
| gemma → deepseek | 182 | 45.1% | 0.598 | 0.055 | 0.483 | 0.489 | 0.895 |
| llama → mistral | 309 | 46.9% | 0.773 | 0.061 | 0.439 | 0.609 | 0.956 |
| mistral → llama | 309 | 45.6% | 0.773 | 0.061 | 0.442 | 0.609 | 0.940 |
| gemma → llama | 188 | 43.6% | 0.423 | 0.053 | 0.405 | 0.423 | 0.930 |
| llama → gemma | 188 | 42.6% | 0.423 | 0.053 | 0.422 | 0.423 | 0.936 |
| gpt2-large → deepseek | 85 | 34.1% | 0.392 | 0.038 | 0.221 | 0.335 | 0.877 |
| deepseek → gpt2-large | 85 | 34.1% | 0.392 | 0.038 | 0.223 | 0.335 | 0.904 |
| gpt2-large → gemma | 36 | 33.3% | 0.570 | 0.035 | 0.200 | 0.253 | 0.917 |
| gemma → gpt2-large | 36 | 33.3% | 0.570 | 0.035 | 0.214 | 0.253 | 0.937 |
| gpt2-large → mistral | 54 | 29.6% | 0.530 | 0.035 | 0.127 | 0.308 | 0.922 |
| mistral → gpt2-large | 54 | 29.6% | 0.530 | 0.035 | 0.158 | 0.308 | 0.891 |
| gpt2-large → llama | 70 | 25.7% | 0.102 | 0.035 | 0.290 | 0.239 | 0.930 |
| llama → gpt2-large | 70 | 24.3% | 0.102 | 0.036 | 0.277 | 0.239 | 0.935 |

**Results by domain (Run 2, NeurIPS claim status):**

| Domain | n | Pass% | proc_cos | cohen_d | NeurIPS |
|---|---|---|---|---|---|
| math_olympiad | 57 | 96.5% | 0.489 | 1.510 | ✅ |
| code_instructions | 368 | 86.7% | 0.604 | 0.954 | ✅ |
| academic_writing | 115 | 76.5% | 0.359 | 1.091 | ✅ |
| math_gsm8k | 250 | 60.0% | 0.534 | 0.398 | ✅ |
| creative_writing | 52 | 67.3% | 0.480 | 1.142 | ⚠️ small n |
| math_competition | 331 | 55.9% | 0.512 | 0.534 | ✅ |
| news_reporting | 116 | 50.9% | 0.542 | 0.516 | ✅ promoted from Run 1 |
| legal | 207 | 51.2% | 0.630 | 1.276 | ⚠️ borderline |
| code_python | 327 | 49.2% | 0.711 | 0.706 | ⚠️ borderline |
| math_reasoning | 172 | 37.2% | 0.866 | 0.234 | ❌ weak effect |
| code_sql | 233 | 52.8% | 0.630 | −0.262 | ❌ negative cohen_d |
| code_snippets | 120 | 25.8% | 0.595 | −0.045 | ❌ negative cohen_d |
| sentiment | 108 | 24.1% | 0.834 | 0.106 | ❌ weak effect |
| science_biomedical | 612 | 13.7% | 0.679 | −0.057 | ❌ negative cohen_d |
| question_answering | 240 | 7.9% | 0.847 | −0.011 | ❌ negative cohen_d |

**SAE-free cosines by pair (no SAE artefact risk):**

| Pair | saefree_cos |
|---|---|
| llama–mistral | 0.956 |
| gemma–mistral | 0.935 |
| gpt2-large–mistral | 0.929 |
| gpt2-large–llama | 0.925 |
| gemma–llama | 0.922 |
| llama–deepseek | 0.915 |
| gpt2-large–gemma | 0.911 |
| mistral–deepseek | 0.906 |
| gemma–deepseek | 0.886 |
| gpt2-large–deepseek | 0.839 |

**Scaling law finding:** gemma-2-2b (**1.7B**, not 7B) aligns at 46.9% — nearly indistinguishable from 7B↔7B (48.9%, Δ=2pp). The universality cliff is between 0.8B and 1.7B. gpt2-large (0.8B) passes at only 29.9–33.3% despite SAE-free cosines remaining 0.877–0.937, indicating base geometric structure persists even below the threshold for validated feature-level universality. — Cross-Model Steering Vectors (`b3_build_cross_steering.py`)

Uses B2-validated alignments to transfer steering vectors between model pairs in two modes.

| Parameter | Value |
|-----------|-------|
| Top features per concept (guide) | 3 |
| Min feature confidence | 0.3 |

### Vector types

**Locator — `sae_decoder` (primary):** For each guide concept, select top-3 guide SAE features by confidence. For every guide feature $a$, look up matching target features $b$ from `aligned_pairs.jsonl`. Accumulate weighted target feature activations:

$$v_{\text{tgt}}[b] = \sum_{(a,b)\in\text{pairs}} w_a \cdot s_{ab}$$

where $w_a$ is guide feature confidence and $s_{ab}$ is the B1 composite alignment score. Decode without bias: $\vec{d} = W_\text{dec}^\top v_{\text{tgt}}$; L2-normalise.

**Locator — `caa_cross` (secondary, stored alongside when target activations are available):** For each guide concept, split corpus passages by whether guide-side feature activations are above/below the median. Compute the mean target residual-stream activations for each split:

$$\vec{v}_{\text{caa\_cross}} = \text{L2-norm}\!\left(\bar{h}_{\text{pos}} - \bar{h}_{\text{neg}}\right)$$

where the positive/negative split is determined via the guide feature matrix (guided CAA), falling back to corpus label split if guide features are unavailable.

**Translation — `translation_injection`:** Encode the guide's A5 `sae_vector` through the guide SAE encoder (weight-only, no bias) → sparse TopK feature activations. Select the $M_\text{src}$-dimensional ever-active feature subset that the B1 MLP was trained on. Apply saved `src_scaler` normalisation → run through forward MLP bridge → inverse-transform with `tgt_scaler`. Scatter into full target feature space at saved `tgt_idx`. Decode without bias through target SAE decoder → L2-normalise.

All modes use weight-only linear operations (no decoder/encoder bias) since they operate on directions, not reconstructions. Concurrent-safe file writes via `fcntl.flock`.

**Outputs:** `steering/cross_model_steering_vectors_bal.json` (locator), `steering/cross_model_steering_vectors_ti.json` (translation)

---

## C1 — Global MLP Concept Space (`c1_train_global_mlp.py`)

Trains a shared multi-model encoder-decoder that maps each model's SAE feature space into a common `d_concept`-dimensional semantic space.

### Architecture

- **Encoder** (per model): `Linear(n_features, 2048) → LayerNorm(2048) → GELU → Dropout(0.1) → Linear(2048, 512)`
- **Decoder** (per model): `Linear(512, 2048) → LayerNorm(2048) → GELU → Linear(2048, n_features)`
- `d_concept = 512`, `hidden = 2048`.

### Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | `5e-4` |
| Epochs | 200 |
| Batch size | 256 |
| Validation split | 5% |
| Checkpoint interval | 100 epochs |
| Multi-GPU | DDP via `torchrun` |

**Loss = reconstruction + contrastive:**

1. **Reconstruction loss** (weight `1.0`): `MSE(decode(encode(x)), x)` per model.  
2. **Contrastive loss** (weight `0.5`): NT-Xent / SimCLR — same passage seen across different models is the positive pair; temperature `τ = 0.1`. This enforces that same-passage embeddings from different architectures are co-located in concept space.

**Output:** `universal/global_mlp_v{N}.pt`, `universal/global_mlp_v{N}_meta.json`

### Training results (Run 1 — 8× A100, 200 epochs, ~66 min)

| Metric | Value |
|---|---|
| train_total | 0.06654 |
| train_recon | 0.02219 |
| train_align | 0.08870 |
| val_total | 0.08544 |
| val_recon | 0.01329 |
| val_align | 0.14431 |
| dead_neuron_pct | **0%** — all 512 concept dimensions active throughout |

**Per-model val_recon at epoch 200:**

| Model | val_recon | Note |
|---|---|---|
| gpt2-large | 0.00202 | Fewest features (666); easiest to compress |
| gemma-2-2b | 0.00540 | |
| deepseek-llm-7b | 0.01654 | |
| llama | 0.01999 | Dense 524K-feature space |
| mistral | 0.02249 | Dense 524K-feature space | (`c2_discover_concepts.py`)

Clusters SAE features across all five models in the shared concept space to find universal concepts.

### Algorithm

1. Load C1 encoders and each model's top-`K` feature activation matrix (`{model}_ef{ef}_top*_feature_acts.npy`).  
2. **Feature-probe projection:** for each SAE feature `f` in model `m`:  
   - Find passages where `feature_acts[:, f]` is in the top-`10%` percentile.  
   - Encode those passages through the C1 encoder (batch `4096`).  
   - Feature `f`'s concept-space position = mean of those encodings.  
   This yields one point per SAE feature in `d_concept`-dimensional space.  
3. **UMAP reduction:** `512 → 30` dimensions before clustering (`umap_dim=30`; `metric="cosine"`, `min_dist=0.0`, `n_neighbors=max(5, min(50, N//20))`, `random_state=42`). Skipping UMAP in 512-d causes ≈50–80% noise due to the curse of dimensionality.  
4. **HDBSCAN clustering** (`metric="euclidean"`, `min_cluster_size=5`; fallback to DBSCAN with `eps=median_std × 0.5`).  
5. Keep clusters spanning `≥ min_models = 2` distinct architectures.  
6. Optional cluster naming: Claude (`claude-sonnet-4-5`, `max_tokens=20`, `temperature=0`) prompted with up to 6 top-activating passages; default is `cluster_{id}`.

**Output:** `universal/{pooling}_concepts.json`

---

## C2b — Per-Model Auto-Discovery (`c2b_auto_discover.py`)

Single-model cross-architecture alternative to C2: clusters SAE features by co-activation pattern. In the production run this step was used at **A4b-auto** (gpt2-large feature augmentation) rather than here. The C2b script is available for per-model concept discovery on any single model.

### Algorithm

1. Run SAE on up to `max_passages=10 000` passages → feature activation matrix $F$ of shape $(n_\text{sam}, n_\text{features})$.  
2. **Variance filter:** discard features with $\text{var}(F_{:,f}) < \texttt{min\_var}=0.01$.  
3. If surviving features exceed `max_features=8 000`, keep top-8 000 by mean absolute activation.  
4. Compute pairwise cosine distance $d_{ij} = 1 - \cos(F_{:,i}, F_{:,j})$ over surviving features.  
5. **HDBSCAN** (`metric="precomputed"`, `min_cluster_size=5`; fallback DBSCAN with `eps=10th percentile of distances`).  
6. **Cluster merging:** greedily merge cluster pairs whose centroid cosine similarity exceeds `merge_thresh=0.85` (union-find, single-linkage).  
7. Per cluster: rank passages by sum of member feature activations; take top/bottom `top_k=20`; compute CAA vector $= \text{mean}(h_{\text{top}}) - \text{mean}(h_{\text{bottom}})$; confidence = mean intra-cluster cosine similarity.

**Output:** `features/{model}_ef{ef}_autodiscovered.json`

---

## C2c — Concept Labelling UI (`c2c_label_concepts.py`)

Interactive human-labelling helper. For each concept cluster, retrieves top-activating passages from `corpus.jsonl` (or domain JSONL files), displays them, and writes the human-assigned label back to the concepts JSON in-place. No model inference; pure annotation tool.

**Output:** updated `universal/mean_concepts.json`

---

## C2d — Concept Deduplication (`c2d_dedup_concepts.py`)

Collapses 113 raw HDBSCAN clusters down to 11 canonical concepts via a hand-curated `CANONICAL_MAP` (derived by human review of all cluster labels).

| Canonical concept | Raw clusters merged |
|-------------------|-------------------|
| `python_code` | 29 |
| `math_problems` | 20 |
| `sql_queries` | 11 |
| `legal_and_news` | 10 |
| `medical_research` | 8 |
| `academic_scientific` | 8 |
| `narrative_fiction` | 6 |
| `encyclopedic_historical` | 5 |
| `code_and_math` | 4 |
| `customer_reviews` | 3 |
| `sql_and_medical` | 3 |

**6 noise clusters excluded** (IDs 16, 28, 83, 84, 113, 114): boundary cases with no clean semantic core.

**Merging strategy:** union of `models_present`; weighted centroid (weight = model count per raw cluster); union of per-model `feature_ids`.

**Output:** `universal/mean_concepts_clean.json`

---

## C3 — Universal Steering Vectors (`c3_build_vectors.py`)

Produces per-(model, concept) steering vectors in the raw activation space using three modes:

### Mode 1 — `decoder_only` (default)

```
cluster centroid (d_concept=512)
  → GlobalMLP.decoders[target]      # d_concept → n_features
  → scatter into full feature space  # top-K features addressed by feature_idx.npy
  → SAE.decoder                     # n_features → hidden_dim
  → L2-normalise
```

The cluster centroid is a position in concept space (no polarity); sign is resolved in Mode 2.

### Mode 2 — `sign_fix` (post-processing, CPU-only)

For each `(model, concept)` vector, compute dot product with the corresponding native A5 `sae_vector`. If `dot < 0`, flip sign. Uses `_HDBSCAN_TO_NATIVE` mapping for concepts whose names differ between C2 and A5. Writes corrected vectors back atomically.

### Mode 3 — `enc_dec`

```
guide native sae_vector
  → GlobalMLP.encoders[guide]   # n_features_guide → d_concept
  → GlobalMLP.decoders[target]  # d_concept → n_features_target
  → SAE.decoder[target]         # n_features_target → hidden_dim
  → L2-normalise
```

Averaged over all available guide models (excluding the target itself). Polarity is correct by construction since it is inherited from the signed native direction.

**Outputs:** `steering/universal_steering_vectors_v1.json` (Modes 1 + 2), `steering/universal_steering_vectors_enc_dec_v1.json` (Mode 3)

---

## D1 — Evaluation (`d1_evaluate_native.py`)

Measures the effectiveness of every steering method across all concepts and models.

**Global seed:** `set_seed(42)` — seeds `random`, `numpy.random`, and `torch.manual_seed` at the start of each per-model worker.

### Methods evaluated (ordered)

Each model × concept combination is evaluated for all available methods in order:

| Tag | Source | Description |
|-----|--------|-------------|
| `sae_vector` | A5 | Native SAE-decoder steering vector |
| `caa_vector` | A5 | Native CAA contrastive steering vector |
| `b3_ti_{guide}` | B3 | Translation-injection cross-model vector (MLP bridge) |
| `naive_{guide}` | D1 | Dumb baseline: guide A-vector truncated/zero-padded to target `hidden_dim`, L2-normalised. No projection. |
| `universal_c3` | C3 | Universal decoder-only vector |
| `universal_enc_dec` | C3 | Universal encoder→decoder vector |

**B3 BAL excluded:** Both the `sae_decoder_vector` and `caa_cross_vector` fields from `cross_model_steering_vectors_bal.json` were found to be numerically identical to each other and to the native A5 `sae_vector` for every guide→target pair. The root cause is twofold: (1) the `caa_cross` path in B3 falls back to corpus-label split when guide feature columns are absent from the feature matrix, producing the same directional signal as the SAE decoder averaging; (2) the `sae_decoder` path in step7 uses a fallback that writes the guide's own SAE decoder weights when no B2-validated aligned pairs exist for the guide's top features, recovering the same vector as the A5 computation. The BAL file is retained for future reprocessing but is excluded from evaluation to avoid duplicate entries in method rankings.

### Prompt selection

140 register-neutral candidate prompts spanning six registers (evaluation/observation, process/instructional, analysis/academic, problem/solution, opinion/sentiment, conversational). At runtime the `TOP_N_PROMPTS = 30` prompts with the lowest baseline repetition rate are selected (sorted ascending by per-prompt repetition score). Baseline sanity gate: at least 25 loop-free candidates must exist or evaluation aborts.

**Statistical justification:** $n=30$ prompts yield power $\approx 0.83$ at our effect size ($\delta = 0.05$, $\mathrm{SD} = 0.10$), 95% CI width $\approx 0.067$. $n=50$ gives only marginal gain (power $\approx 0.96$) at substantially higher GPU cost.

### Shared baseline

Baseline generation (all 140 candidates, strength 0, no hooks) runs **once per model** before any method or concept loop. The `TOP_N_PROMPTS = 30` cleanest outputs are selected and reused as the baseline for every `(concept, method, strength)` cell. This eliminates redundant zero-strength forward passes and ensures the baseline distribution is identical across all method comparisons.

### Steering injection

For each steering vector, the injection proceeds as follows:

**Step 1 — denormalise.** Steering vectors are in z-scored activation space. Recover the raw activation direction:

$$\tilde{v}[d] = \hat{v}[d] \cdot \sigma[d]$$

where $\sigma \in \mathbb{R}^D$ is the per-dimension activation standard deviation from A4 `norm_stats.json`.

**Step 2 — re-normalise.** L2-normalise $\tilde{v}$ to get a unit-norm direction in raw activation space.

**Step 3 — scale.** Compute the injection scale relative to the stream's own magnitude:

$$\alpha = \frac{\bar{\|h_l\|}}{\sqrt{D}}$$

where $\bar{\|h_l\|}$ is the mean L2 norm of layer `target_layer` hidden states measured on one unhooked baseline forward pass, and $D$ is `hidden_dim`. For z-scored models $\bar{\|h_l\|} \approx \sqrt{D}$ so $\alpha \approx 1$; for raw-activation models (e.g. deepseek, $\bar{\|h_l\|} \approx 2173$, $D = 4096$) $\alpha \approx 34$.

**Step 4 — inject** via `register_forward_hook` at `target_layer`:

$$h_l^{(t)} \leftarrow h_l^{(t)} + s \cdot \alpha \cdot \hat{v}_{\mathrm{raw}}$$

where $s$ is the signed strength scalar from the sweep. This makes `strength = 1` correspond to a unit-norm perturbation scaled to the stream's own magnitude, and `strength = 5` to a $\approx 8\%$ perturbation of the stream norm, regardless of whether activations were z-scored during training.

- **Injection mode:** `single_layer` (default — injects only at `target_layer`); `multi_layer`; `all_layers`.

### Strength sweep

$$\text{SWEEP\_STANDARD} = [-5, -3, -2, -1, 0, 1, 2, 3, 5]$$

Applied uniformly to all methods (native, cross-model, universal).

### Generation

- `max_new_tokens = 50`, greedy decoding (`do_sample=False`).  
- Batched in groups of 8 prompts; falls back to sequential on error.

### Metrics

#### Concept score (DeBERTa NLI)

Model: `MoritzLaurer/deberta-v3-large-zeroshot-v2`. Zero-shot multi-label NLI across all $N=27$ concept labels simultaneously, hypothesis template `"This text is about {label}."`:

$$\text{score}(y, c) = P_{\text{NLI}}(\text{entailment} \mid y,\ \texttt{"This text is about\ } c\texttt{."})$$

Multi-label mode ensures scores are calibrated relative to all concepts, not just a binary entailment/contradiction split. Batch size 8.

#### Concept delta

For each method $m$ at strength $s$:

$$\Delta_{m,s} = \frac{1}{n} \sum_{i=1}^{n} \text{score}(y_{m,s,i},\, c) - \frac{1}{n} \sum_{i=1}^{n} \text{score}(y_{\text{base},i},\, c)$$

where $y_{m,s,i}$ is the steered output for prompt $i$ and $y_{\text{base},i}$ is the unsteered baseline. Only prompts with repetition rate $< 0.4$ and $n_\text{valid} \geq 3$ enter the sum.

$$\Delta_{m}^{+} = \max_{s > 0} \Delta_{m,s}, \qquad \Delta_{m}^{-} = \min_{s < 0} \Delta_{m,s}$$

#### Perplexity

Fixed scorer: GPT-2 117M (`gpt2`) for **all five** target models — ensures the 30% gate is on a uniform scale across architectures. For a text $w_{1:T}$:

$$\text{PPL}(w_{1:T}) = \exp\!\left(\frac{1}{T-1} \sum_{t=1}^{T-1} \ell(w_{t+1} \mid w_{1:t})\right)$$

where $\ell$ is the per-token cross-entropy loss. Computed in batches of 16, `max_length=256`.

#### Repetition rate

Binary indicator $r \in \{0, 1\}$:

$$r = 1 \iff \begin{cases} \text{any phrase of } \geq 2 \text{ words repeats } \geq 4 \text{ consecutive times, OR} \\ \text{any single word repeats } \geq 8 \text{ consecutive times} \end{cases}$$

Also catches character-level loops (surrogates, CJK) and single-token numeric loops.

#### Bidirectionality ratio

$$r_{\text{bidir}} = \frac{|\Delta_{m}^{-}|}{|\Delta_{m}^{+}|}$$

Only reported when $|\Delta_{m}^{+}| \geq 0.005$ (below this threshold the positive direction has no real effect and the ratio is undefined/noise).

#### PPL efficiency

Let $\text{PPL}^* = \text{PPL at optimal positive strength}$ (the $s^* = \arg\max_{s>0} \Delta_{m,s}$):

$$\text{efficiency\_valid} = \mathbb{1}\left[\frac{\text{PPL}^* - \text{PPL}_{\text{base}}}{\text{PPL}_{\text{base}}} \leq 0.30\right]$$

A vector passes the efficiency gate if it steers the model without increasing its output perplexity by more than 30%.

### Filtering

- Outputs with repetition rate $r > 0.4$ (hallucination gate) are excluded from all delta pools.  
- A strength entry requires $n_\text{valid} \geq 3$ clean prompts to contribute to $\Delta_{m,s}$.

### LLM judge (optional)

Claude (`claude-sonnet-4-5`) evaluates a stratified sample of `10` steered outputs per `(model, concept)` pair using `tool_use` for structured responses. Reports: `concept_present`, `fluent`, `better_than_baseline`.

### Parallelism

`--parallel` spawns one subprocess per model on a dedicated GPU (`GPU_MAP`: gpt2-large→0, gemma→1, llama→2, mistral→3, deepseek→4). After all workers finish, `compile_tables()` builds Tables 1–7 and writes `evaluation_table.json` + `evaluation_report.md`.

### Statistical robustness

Per `(model, concept, method)` cell at the optimal positive strength, the list of 30 per-prompt deltas is used to compute:

- **95% bootstrap CI** (2 000 resamples, `numpy.random.default_rng(seed=0)`): lower/upper percentiles of resample means.
- **One-sample Wilcoxon signed-rank** ($H_0$: median per-prompt $\delta = 0$, one-sided alternative: greater).
- **Benjamini–Hochberg FDR correction** across all tests (one per `(model, concept, method)` triple). Cells with `p_adj_bh ≤ 0.05` are marked `reject_h0_fdr05 = True`.

Cross-concept tests per model:
- **Sign test (binomial, $p = 0.5$):** out of $N$ concepts, how many show positive B3-TI delta? Avoids magnitude inflation of Wilcoxon on max-selected deltas.
- **Paired Wilcoxon (two-sided):** B3-TI vs native, matched explicitly by concept key (avoids silent misalignment from positional zip).

Results written to `table_stats` in `evaluation_table.json`.

**Outputs:** `results/evaluation_results_{model}.json`, `results/steering_examples.jsonl`, `results/evaluation_table.json`, `results/evaluation_report.md`
