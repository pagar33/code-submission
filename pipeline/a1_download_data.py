
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import html
import json
import os
import re
import shutil
import time
from typing import List, Dict

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


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def _remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean(text: str, cfg: dict) -> str:
    if cfg.get("strip_html"):
        text = _strip_html(text)
    if cfg.get("remove_urls"):
        text = _remove_urls(text)
    strip_regex = cfg.get("strip_regex")
    if strip_regex:
        text = re.sub(strip_regex, "", text, flags=re.MULTILINE)
    if not cfg.get("preserve_code"):
        text = _normalise_whitespace(text)
    else:
        text = text.strip()
    return text


def _approx_tokens(text: str) -> int:
    """Word-count proxy for token length (fast; ~0.75 tokens/word ratio is fine for filtering)."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Download prebuilt corpus from HuggingFace artifacts repo (primary path)
# ---------------------------------------------------------------------------

def _download_from_hub(corpus_path: str, labels_path: str) -> int:
    """Pull corpus.jsonl and corpus_labels.jsonl from nips348734/submission-artifacts."""
    from huggingface_hub import hf_hub_download

    repo_id = config.HF_REPO_ID
    print(f"Downloading corpus from {repo_id} ...")

    for fname, dest in [("data/corpus.jsonl", corpus_path),
                        ("data/corpus_labels.jsonl", labels_path)]:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset",
            token=config.HF_TOKEN or None,
        )
        shutil.copy(local, dest)
        print(f"  Saved {fname} -> {dest}")

    with open(corpus_path, encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return n


# ---------------------------------------------------------------------------
# Rebuild corpus from source datasets (fallback / --rebuild path)
# ---------------------------------------------------------------------------

def _rebuild_corpus(corpus_path: str, labels_path: str) -> int:
    """Re-download and process all 17 DATASET_SOURCES entries."""
    from datasets import load_dataset

    records: List[Dict] = []
    labels: List[Dict] = []
    global_id = 0

    for source_tag, cfg in config.DATASET_SOURCES.items():
        print(f"  Loading {source_tag} ...")

        hf_config = cfg.get("hf_config")
        ds = load_dataset(
            cfg["hf_path"],
            hf_config,
            split=cfg["split"],
            trust_remote_code=True,
        )

        n = cfg.get("n", 0)
        if n and n < len(ds):
            ds = ds.shuffle(seed=42).select(range(n))

        min_tok = cfg.get("min_tok", 1)
        max_tok = cfg.get("max_tok", 512)
        text_mode = cfg.get("text_mode", "single")
        text_field = cfg["text_field"]
        sep = cfg.get("sep", "\n")

        accepted = 0
        for row in ds:
            if text_mode == "concat":
                raw = row[text_field[0]] + sep + row[text_field[1]]
            else:
                raw = row[text_field]

            if not isinstance(raw, str):
                raw = str(raw) if raw is not None else ""

            text = _clean(raw, cfg)
            tok_len = _approx_tokens(text)
            if tok_len < min_tok or tok_len > max_tok:
                continue

            records.append({"id": global_id, "source": source_tag, "text": text})
            labels.append({"id": global_id, "source": source_tag})
            global_id += 1
            accepted += 1

        print(f"    {accepted} passages accepted from {source_tag}")

    with open(corpus_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with open(labels_path, "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r) + "\n")

    return len(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Obtain the text corpus used for activation extraction."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing corpus files.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Re-download all 17 source datasets and rebuild corpus from scratch "
             "instead of pulling the prebuilt corpus from HuggingFace.",
    )
    args = parser.parse_args()

    os.makedirs(config.DATA_DIR, exist_ok=True)
    corpus_path = os.path.join(config.DATA_DIR, "corpus.jsonl")
    labels_path = os.path.join(config.DATA_DIR, "corpus_labels.jsonl")

    if (os.path.exists(corpus_path) or os.path.exists(labels_path)) and not args.force:
        print("Corpus files already exist. Use --force to overwrite.")
        log_run("a1_download_data.py", start_time, "skipped")
        return 0

    if args.rebuild:
        print("Rebuilding corpus from source datasets ...")
        n = _rebuild_corpus(corpus_path, labels_path)
    else:
        n = _download_from_hub(corpus_path, labels_path)

    print(f"Corpus ready: {n:,} passages  ->  {corpus_path}")
    log_run("a1_download_data.py", start_time, "success")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log_run("a1_download_data.py", time.time(), "error", str(e))
        raise
