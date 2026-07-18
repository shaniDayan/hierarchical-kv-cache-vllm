#!/usr/bin/env python3

import csv
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path.home() / "kv_cache_project"

DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "longbench"
    / "data"
    / "qasper_e.jsonl"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "original_vllm"
    / "longbench"
)

MODEL_NAME = "Qwen/Qwen3-0.6B"

LENGTHS_CSV = RESULTS_DIR / "qasper_e_token_lengths.csv"
SELECTED_JSON = RESULTS_DIR / "qasper_e_selected_examples.json"


def load_dataset(path):
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def build_prompt(tokenizer, row):
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using only the information in the "
                "provided document. Give a concise final answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Document:\n{row['context']}\n\n"
                f"Question:\n{row['input']}"
            ),
        },
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def get_bucket(token_count):
    if token_count <= 2048:
        return "0-2K"
    if token_count <= 4096:
        return "2K-4K"
    if token_count <= 8192:
        return "4K-8K"
    if token_count <= 16384:
        return "8K-16K"
    if token_count <= 32768:
        return "16K-32K"
    return ">32K"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    rows = load_dataset(DATASET_PATH)
    records = []

    for index, row in enumerate(rows):
        prompt = build_prompt(tokenizer, row)

        token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        prompt_tokens = len(token_ids)

        records.append(
            {
                "index": index,
                "id": row.get("_id", str(index)),
                "reported_length": row.get("length"),
                "prompt_tokens": prompt_tokens,
                "bucket": get_bucket(prompt_tokens),
                "answer_count": len(row.get("answers", [])),
            }
        )

    token_lengths = sorted(
        record["prompt_tokens"] for record in records
    )

    def percentile(percent):
        position = round(
            (len(token_lengths) - 1) * percent
        )
        return token_lengths[position]

    print("\nQasper-E token statistics")
    print("=" * 50)
    print(f"Examples: {len(records)}")
    print(f"Minimum:  {min(token_lengths):,}")
    print(f"Median:   {percentile(0.50):,}")
    print(f"P75:      {percentile(0.75):,}")
    print(f"P90:      {percentile(0.90):,}")
    print(f"Maximum:  {max(token_lengths):,}")

    bucket_counts = Counter(
        record["bucket"] for record in records
    )

    print("\nExamples by token-length bucket")
    print("=" * 50)

    bucket_order = [
        "0-2K",
        "2K-4K",
        "4K-8K",
        "8K-16K",
        "16K-32K",
        ">32K",
    ]

    for bucket in bucket_order:
        print(f"{bucket:>8}: {bucket_counts[bucket]:3d}")

    with LENGTHS_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index",
                "id",
                "reported_length",
                "prompt_tokens",
                "bucket",
                "answer_count",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    # Select reproducible examples close to these lengths.
    target_lengths = [
        2048,
        4096,
        8192,
        16384,
    ]

    selected = []
    used_indices = set()

    for target in target_lengths:
        candidates = [
            record
            for record in records
            if record["index"] not in used_indices
        ]

        closest = min(
            candidates,
            key=lambda record: abs(
                record["prompt_tokens"] - target
            ),
        )

        used_indices.add(closest["index"])

        selected.append(
            {
                "target_tokens": target,
                **closest,
            }
        )

    with SELECTED_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selected,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nSelected examples")
    print("=" * 50)

    for item in selected:
        print(
            f"Target {item['target_tokens']:>6,} tokens "
            f"-> index {item['index']:>3}, "
            f"actual {item['prompt_tokens']:>6,}, "
            f"bucket {item['bucket']}"
        )

    print("\nSaved files:")
    print(LENGTHS_CSV)
    print(SELECTED_JSON)


if __name__ == "__main__":
    main()
