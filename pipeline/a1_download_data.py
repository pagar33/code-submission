
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root (for config)
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # pipeline/ (for sibling scripts)

import argparse
import json
import os
import random
import re
import time
import html
from typing import List, Dict

from datasets import load_dataset

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


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sample_dataset(ds, n: int, seed: int) -> List[Dict]:
    if len(ds) <= n:
        return list(ds)
    ds = ds.shuffle(seed=seed)
    return list(ds.select(range(n)))


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.makedirs(config.DATA_DIR, exist_ok=True)
    corpus_path = os.path.join(config.DATA_DIR, "corpus.jsonl")
    labels_path = os.path.join(config.DATA_DIR, "corpus_labels.jsonl")

    if (os.path.exists(corpus_path) or os.path.exists(labels_path)) and not args.force:
        print("Data files already exist. Use --force to regenerate.")
        log_run("step1_download_data.py", start_time, "skipped")
        return 0

    rng = random.Random(42)

    records = []
    labels = []

    # SST-2
    sst2_cfg = config.DATASET_SOURCES["sst2"]
    sst2 = load_dataset(sst2_cfg["hf_path"], sst2_cfg["config"], split=sst2_cfg["split"])
    sst2_samples = sample_dataset(sst2, sst2_cfg["n"], 42)
    for row in sst2_samples:
        text = clean_text(row[sst2_cfg["text_field"]])
        records.append({"source": "sst2", "text": text})
        labels.append({"source": "sst2", "sentiment": int(row[sst2_cfg["label_field"]]), "formality": None, "factuality": None})

    # Yelp
    yelp_cfg = config.DATASET_SOURCES["yelp"]
    yelp = load_dataset(yelp_cfg["hf_path"], split=yelp_cfg["split"])
    yelp_samples = sample_dataset(yelp, yelp_cfg["n"], 42)
    for row in yelp_samples:
        text = clean_text(row[yelp_cfg["text_field"]])
        records.append({"source": "yelp", "text": text})
        labels.append({"source": "yelp", "sentiment": int(row[yelp_cfg["label_field"]]), "formality": None, "factuality": None})

    # TruthfulQA
    tq_cfg = config.DATASET_SOURCES["truthfulqa"]
    truthful = load_dataset(tq_cfg["hf_path"], tq_cfg["config"], split=tq_cfg["split"])
    for row in truthful:
        text = clean_text(row[tq_cfg["text_field"]])
        correct = row[tq_cfg["label_field"]]
        factuality = 1 if correct and len(correct) > 0 else 0
        records.append({"source": "truthfulqa", "text": text})
        labels.append({"source": "truthfulqa", "sentiment": None, "formality": None, "factuality": factuality})

    # Formality
    fm_cfg = config.DATASET_SOURCES["formality"]
    formality = load_dataset(fm_cfg["hf_path"], split=fm_cfg["split"])
    filtered = []
    for row in formality:
        score = float(row[fm_cfg["label_field"]])
        if score >= 1.0:
            lbl = 1
        elif score <= -1.0:
            lbl = 0
        else:
            continue
        filtered.append((row, lbl))

    rng.shuffle(filtered)
    filtered = filtered[: fm_cfg["n"]]
    for row, lbl in filtered:
        text = clean_text(row[fm_cfg["text_field"]])
        records.append({"source": "formality", "text": text})
        labels.append({"source": "formality", "sentiment": None, "formality": int(lbl), "factuality": None})

    if len(records) != config.N_PASSAGES:
        raise ValueError(f"Expected {config.N_PASSAGES} records, got {len(records)}")

    # Assign IDs
    for i in range(len(records)):
        records[i]["id"] = i
        labels[i]["id"] = i

    with open(corpus_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with open(labels_path, "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} records to {corpus_path}")
    print(f"Wrote {len(labels)} labels to {labels_path}")

    log_run("step1_download_data.py", start_time, "success")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log_run("step1_download_data.py", time.time(), "error", str(e))
        raise
