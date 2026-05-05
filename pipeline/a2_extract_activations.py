
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import json
import os
import random
import time

import h5py
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_TRANSFORMER_LENS_IMPORT_ERROR = None
try:
    from transformer_lens import HookedTransformer
except Exception as _e:
    HookedTransformer = None
    _TRANSFORMER_LENS_IMPORT_ERROR = _e

try:
    import wandb
except Exception:
    wandb = None

import config


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


def load_corpus(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = hidden * mask.unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return masked.sum(dim=1) / denom.unsqueeze(-1)


def _get_torch_device(model_cfg: dict) -> torch.device:
    if model_cfg.get("device") == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_tl_model(model_name: str, model_cfg: dict):
    """Load a TransformerLens model and return (model, hook_name, pad_id)."""
    if HookedTransformer is None:
        raise ImportError(
            f"transformer_lens failed to import. "
            f"Underlying error: {_TRANSFORMER_LENS_IMPORT_ERROR}"
        )
    _tl_device = model_cfg["device"] if (model_cfg.get("device") != "cuda" or torch.cuda.is_available()) else "cpu"
    # Determine dtype: UI/CLI override → config entry → bfloat16 on CUDA
    _dtype_str = model_cfg.get("dtype", "")
    if _dtype_str in ("fp16",):
        _dtype = torch.float16
    elif _dtype_str in ("fp32",):
        _dtype = torch.float32
    else:  # bf16, bfloat16, or unset → use bfloat16 on A100/H100, float16 elsewhere
        _dtype = torch.bfloat16 if _tl_device == "cuda" else torch.float32
    model = HookedTransformer.from_pretrained(model_cfg["hf_name"], device=_tl_device, dtype=_dtype)
    if model_cfg["target_layer"] >= model.cfg.n_layers:
        raise ValueError(
            f"target_layer={model_cfg['target_layer']} exceeds model n_layers={model.cfg.n_layers}."
        )
    hook_name = f"blocks.{model_cfg['target_layer']}.hook_resid_post"
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    return model, hook_name, pad_id


def _run_tl_extraction(tl_model, hook_name, pad_id, model_name: str, texts, out_path: str, model_cfg: dict, wandb_run, source_tag: str = ""):
    """Extract activations using a pre-loaded TransformerLens model."""
    batch_size = model_cfg["extract_batch_size"]
    hidden_dim = model_cfg["hidden_dim"]
    desc = f"extract_{model_name}" + (f":{source_tag}" if source_tag else "")
    if os.path.exists(out_path):
        os.remove(out_path)
    with h5py.File(out_path, "w", locking=False) as h5:
        dset = h5.create_dataset("activations", shape=(len(texts), hidden_dim), dtype="float32")
        idx = 0
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, total=total_batches):
            batch_texts = texts[i : i + batch_size]
            tokens = tl_model.to_tokens(batch_texts, prepend_bos=True)
            if tokens.shape[1] > config.MAX_SEQ_LEN:
                tokens = tokens[:, : config.MAX_SEQ_LEN]
            tokens = tokens.to(next(tl_model.parameters()).device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=tokens.is_cuda):
                _, cache = tl_model.run_with_cache(tokens, names_filter=[hook_name])
                hidden = cache[hook_name]
            if hidden.shape[-1] != hidden_dim:
                raise ValueError(f"Hidden dim mismatch: {hidden.shape[-1]} vs {hidden_dim}")
            mask = (tokens != pad_id).float()
            pooled = mean_pool(hidden, mask)
            if pooled.shape != (tokens.shape[0], hidden_dim):
                raise ValueError(f"Pooled shape mismatch: {pooled.shape}")
            pooled_np = pooled.float().cpu().numpy().astype("float32")
            dset[idx : idx + pooled_np.shape[0]] = pooled_np
            idx += pooled_np.shape[0]
            if wandb_run is not None and idx % (batch_size * 50) == 0:
                wandb_run.log({"examples_processed": idx})


def _extract_gpt2_transformerlens(model_name: str, texts, out_path: str, model_cfg: dict, wandb_run):
    tl_model, hook_name, pad_id = _load_tl_model(model_name, model_cfg)
    _run_tl_extraction(tl_model, hook_name, pad_id, model_name, texts, out_path, model_cfg, wandb_run)


def _load_hf_model(model_name: str, model_cfg: dict):
    """Load a HuggingFace causal LM and return (model, tokenizer)."""
    if model_name == "gemma":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    device = _get_torch_device(model_cfg)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or True
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_name"], use_fast=True, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": dtype,
        "output_hidden_states": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "token": hf_token,
        "use_safetensors": True,  # prefer safetensors to avoid torch.load CVE-2025-32434 check
    }
    if device.type == "cuda":
        model_kwargs["device_map"] = "cuda"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_name"], **model_kwargs)
    except Exception:
        # Fall back without use_safetensors if the model doesn't have them
        model_kwargs.pop("use_safetensors", None)
        model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_name"], **model_kwargs)
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def _run_hf_extraction(hf_model, tokenizer, model_name: str, texts, out_path: str, model_cfg: dict, wandb_run, source_tag: str = ""):
    """Extract activations using a pre-loaded HuggingFace model."""
    batch_size = model_cfg["extract_batch_size"]
    hidden_dim = model_cfg["hidden_dim"]
    target_layer = model_cfg["target_layer"]
    desc = f"extract_{model_name}" + (f":{source_tag}" if source_tag else "")
    if os.path.exists(out_path):
        os.remove(out_path)
    with h5py.File(out_path, "w", locking=False) as h5:
        dset = h5.create_dataset("activations", shape=(len(texts), hidden_dim), dtype="float32")
        idx = 0
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), desc=desc, total=total_batches):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.MAX_SEQ_LEN,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(hf_model.device)
            attn_mask = enc["attention_mask"].to(hf_model.device)
            with torch.no_grad():
                outputs = hf_model(input_ids=input_ids, attention_mask=attn_mask)
                hidden_states = outputs.hidden_states
                if target_layer >= len(hidden_states):
                    raise ValueError(f"target_layer={target_layer} out of range for hidden_states len={len(hidden_states)}")
                hidden = hidden_states[target_layer]
            if hidden.shape[-1] != hidden_dim:
                raise ValueError(f"Hidden dim mismatch: {hidden.shape[-1]} vs {hidden_dim}")
            pooled = mean_pool(hidden, attn_mask.float())
            if pooled.shape != (input_ids.shape[0], hidden_dim):
                raise ValueError(f"Pooled shape mismatch: {pooled.shape}")
            pooled_np = pooled.float().cpu().numpy().astype("float32")
            dset[idx : idx + pooled_np.shape[0]] = pooled_np
            idx += pooled_np.shape[0]
            if wandb_run is not None and idx % (batch_size * 20) == 0:
                wandb_run.log({"examples_processed": idx})


def _extract_transformers(model_name: str, texts, out_path: str, model_cfg: dict, wandb_run):
    hf_model, tokenizer = _load_hf_model(model_name, model_cfg)
    _run_hf_extraction(hf_model, tokenizer, model_name, texts, out_path, model_cfg, wandb_run)


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(config.MODELS.keys()))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--source",
        default=None,
        help="Filter corpus to this source name and save as {model}_{source}_activations.h5.",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated list of source names. Loads the model ONCE and extracts all sources "
             "sequentially, skipping any whose output .h5 already exists (unless --force).",
    )
    parser.add_argument(
        "--corpus-file",
        default=None,
        help="Path to an alternative corpus JSONL file. When provided the output "
             "file is named {model}_activations_{stem}.h5 so the standard "
             "activations are not overwritten.",
    )
    parser.add_argument(
        "--pooling",
        default="last_token",
        help="Pooling strategy for token activations (default: last_token). "
             "Stored in metadata; full multi-strategy support is progressive.",
    )
    parser.add_argument(
        "--layer-hook",
        default="resid_post",
        help="Hook point within the transformer block (default: resid_post). "
             "Use mlp_in only for Transcoder/MOLT SAEs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override extract_batch_size from config (e.g. set via UI).",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Override dtype for model loading: fp16, bf16, fp32 (default: from config).",
    )
    args = parser.parse_args()

    load_dotenv()
    if wandb is not None and os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)

    set_seed(42)

    model_name = args.model
    model_cfg = dict(config.MODELS[model_name])  # copy so we can mutate

    # Apply UI / CLI overrides on top of config.MODELS defaults
    if args.batch_size is not None:
        model_cfg["extract_batch_size"] = args.batch_size
    if args.dtype is not None:
        model_cfg["dtype"] = args.dtype

    os.makedirs(config.ACTIVATIONS_DIR, exist_ok=True)

    # ── Multi-source mode: load model once, extract each source sequentially ──
    if args.sources:
        source_list = [s.strip() for s in args.sources.split(",") if s.strip()]
        corpus_path = os.path.join(config.DATA_DIR, "corpus.jsonl")
        # Load corpus rows once
        with open(corpus_path, encoding="utf-8") as _cf:
            all_rows = [json.loads(l) for l in _cf if l.strip()]
        # Determine which sources actually need extraction
        pending = []  # list of (source, texts, out_path)
        for src in source_list:
            safe_src = src.replace("/", "_").replace(" ", "_")
            out_path = os.path.join(config.ACTIVATIONS_DIR, f"{model_name}_{safe_src}_activations.h5")
            if os.path.exists(out_path) and not args.force:
                print(f"[skip] {src}: activations already exist at {out_path}")
                continue
            texts = [r["text"] for r in all_rows if r.get("source") == src]
            if not texts:
                print(f"[warn] {src}: no corpus rows found, skipping.")
                continue
            pending.append((src, texts, out_path))
        if not pending:
            print("All sources already extracted. Nothing to do.")
            log_run("step2_extract_model.py", start_time, "skipped")
            return 0

        extract_backend = model_cfg.get("extract_backend") or (
            "transformer_lens" if model_name == "gpt2" else "transformers"
        )
        use_wandb = wandb is not None and (os.getenv("WANDB_PROJECT") or os.getenv("WANDB_API_KEY"))
        wandb_run = None
        if use_wandb:
            try:
                wandb_run = wandb.init(
                    project=os.getenv("WANDB_PROJECT", "universal_steering"),
                    entity=os.getenv("WANDB_ENTITY"),
                    name=f"extract_{model_name}_multi",
                    config={"model": model_name, "sources": source_list},
                    settings=wandb.Settings(start_method="fork"),
                )
            except Exception as e:
                print(f"[wandb] init failed ({e}); continuing without wandb.")

        src_names = ", ".join(s for s, _, _ in pending)
        print(f"Loading model '{model_name}' once for {len(pending)} source(s): {src_names}")
        if extract_backend == "transformer_lens":
            tl_model, hook_name, pad_id = _load_tl_model(model_name, model_cfg)
            for src, texts, out_path in pending:
                print(f"\n[{src}] {len(texts)} passages → {out_path}")
                _run_tl_extraction(tl_model, hook_name, pad_id, model_name, texts, out_path, model_cfg, wandb_run, source_tag=src)
                print(f"[{src}] Saved.")
        elif extract_backend == "transformers":
            hf_model, tokenizer = _load_hf_model(model_name, model_cfg)
            for src, texts, out_path in pending:
                print(f"\n[{src}] {len(texts)} passages → {out_path}")
                _run_hf_extraction(hf_model, tokenizer, model_name, texts, out_path, model_cfg, wandb_run, source_tag=src)
                print(f"[{src}] Saved.")
        else:
            raise ValueError(f"Unsupported extract_backend '{extract_backend}'")

        if wandb_run is not None:
            wandb_run.finish()
        log_run("step2_extract_model.py", start_time, "success")
        print(f"Multi-source extraction complete for model '{model_name}'.")
        return 0

    # ── Single-source / full-corpus mode ──
    if args.source:
        # Per-source extraction: {model}_{source}_activations.h5
        safe_source = args.source.replace("/", "_").replace(" ", "_")
        corpus_path = os.path.join(config.DATA_DIR, "corpus.jsonl")
        out_path = os.path.join(config.ACTIVATIONS_DIR, f"{model_name}_{safe_source}_activations.h5")
    elif args.corpus_file:
        corpus_path = args.corpus_file
        stem = os.path.splitext(os.path.basename(corpus_path))[0]
        out_path = os.path.join(config.ACTIVATIONS_DIR, f"{model_name}_activations_{stem}.h5")
    else:
        corpus_path = os.path.join(config.DATA_DIR, "corpus.jsonl")
        out_path = os.path.join(config.ACTIVATIONS_DIR, f"{model_name}_activations.h5")

    if os.path.exists(out_path) and not args.force:
        print(f"Activations already exist at {out_path}. Use --force to regenerate.")
        log_run("step2_extract_model.py", start_time, "skipped")
        return 0

    all_texts = load_corpus(corpus_path)
    if args.source:
        # Filter to the requested source
        with open(corpus_path, encoding="utf-8") as _cf:
            rows = [json.loads(l) for l in _cf if l.strip()]
        texts = [r["text"] for r in rows if r.get("source") == args.source]
        if not texts:
            raise ValueError(f"No corpus rows found for source='{args.source}'. Check corpus.jsonl.")
        print(f"Loaded {len(texts)} passages for source='{args.source}'")
    else:
        texts = all_texts
    print(f"Extracting {len(texts)} passages → {out_path}")

    use_wandb = wandb is not None and (os.getenv("WANDB_PROJECT") or os.getenv("WANDB_API_KEY"))
    wandb_run = None
    if use_wandb:
        try:
            wandb_run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "universal_steering"),
                entity=os.getenv("WANDB_ENTITY"),
                name=f"extract_{model_name}",
                config={
                    "model": model_name,
                    "target_layer": model_cfg["target_layer"],
                    "batch_size": model_cfg["extract_batch_size"],
                    "max_seq_len": config.MAX_SEQ_LEN,
                },
                settings=wandb.Settings(start_method="fork"),
            )
        except Exception as e:
            print(f"[wandb] init failed ({e}); continuing without wandb tracking.")
            wandb_run = None

    extract_backend = model_cfg.get("extract_backend")
    if extract_backend is None:
        extract_backend = "transformer_lens" if model_name == "gpt2" else "transformers"

    if extract_backend == "transformer_lens":
        _extract_gpt2_transformerlens(model_name, texts, out_path, model_cfg, wandb_run)
    elif extract_backend == "transformers":
        _extract_transformers(model_name, texts, out_path, model_cfg, wandb_run)
    else:
        raise ValueError(f"Unsupported extract_backend '{extract_backend}' for model {model_name}")

    if wandb_run is not None:
        wandb_run.finish()

    log_run("step2_extract_model.py", start_time, "success")
    print(f"Saved {model_name} activations to {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log_run("step2_extract_model.py", time.time(), "error", str(e))
        raise
