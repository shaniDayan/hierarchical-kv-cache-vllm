#!/usr/bin/env python3

import csv
import json
import re
import string
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path.home() / "kv_cache_project"

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "original_vllm"
    / "longbench"
)

AUTO_CSV = RESULTS_DIR / "qasper_e_auto.csv"
INT8_CSV = RESULTS_DIR / "qasper_e_int8_per_token_head.csv"

AUTO_JSONL = RESULTS_DIR / "qasper_e_auto.jsonl"
INT8_JSONL = RESULTS_DIR / "qasper_e_int8_per_token_head.jsonl"

OUTPUT_CSV = RESULTS_DIR / "qasper_e_auto_vs_int8.csv"


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value):
        punctuation = set(string.punctuation)
        return "".join(
            character
            for character in value
            if character not in punctuation
        )

    def fix_whitespace(value):
        return " ".join(value.split())

    return fix_whitespace(
        remove_articles(
            remove_punctuation(text.lower())
        )
    )


def token_f1(prediction, reference):
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()

    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    shared_tokens = sum(common.values())

    if shared_tokens == 0:
        return 0.0

    precision = shared_tokens / len(prediction_tokens)
    recall = shared_tokens / len(reference_tokens)

    return (
        2 * precision * recall
        / (precision + recall)
    )


def qa_f1(prediction, references):
    if not references:
        return 0.0

    return max(
        token_f1(prediction, reference)
        for reference in references
    )


def load_csv(path):
    with path.open("r", encoding="utf-8") as file:
        return {
            int(row["dataset_index"]): row
            for row in csv.DictReader(file)
        }


def load_jsonl(path):
    records = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)
            records[int(row["dataset_index"])] = row

    return records


def main():
    auto_metrics = load_csv(AUTO_CSV)
    int8_metrics = load_csv(INT8_CSV)

    auto_outputs = load_jsonl(AUTO_JSONL)
    int8_outputs = load_jsonl(INT8_JSONL)

    indices = sorted(
        set(auto_metrics)
        & set(int8_metrics)
        & set(auto_outputs)
        & set(int8_outputs),
        key=lambda index: int(
            auto_metrics[index]["prompt_tokens"]
        ),
    )

    comparison = []

    for index in indices:
        auto_metric = auto_metrics[index]
        int8_metric = int8_metrics[index]

        auto_output = auto_outputs[index]
        int8_output = int8_outputs[index]

        references = auto_output["reference_answers"]

        auto_text = auto_output["generated_text"].strip()
        int8_text = int8_output["generated_text"].strip()

        auto_time = float(
            auto_metric["generation_time_seconds"]
        )
        int8_time = float(
            int8_metric["generation_time_seconds"]
        )

        auto_total_tps = float(
            auto_metric["total_tokens_per_second"]
        )
        int8_total_tps = float(
            int8_metric["total_tokens_per_second"]
        )

        record = {
            "dataset_index": index,
            "prompt_tokens": int(
                auto_metric["prompt_tokens"]
            ),
            "auto_output_tokens": int(
                auto_metric["output_tokens"]
            ),
            "int8_output_tokens": int(
                int8_metric["output_tokens"]
            ),
            "auto_time_seconds": auto_time,
            "int8_time_seconds": int8_time,
            "int8_time_change_percent": (
                (int8_time - auto_time)
                / auto_time
                * 100
            ),
            "auto_total_tokens_per_second": auto_total_tps,
            "int8_total_tokens_per_second": int8_total_tps,
            "int8_throughput_change_percent": (
                (int8_total_tps - auto_total_tps)
                / auto_total_tps
                * 100
            ),
            "auto_f1": qa_f1(
                auto_text,
                references,
            ) * 100,
            "int8_f1": qa_f1(
                int8_text,
                references,
            ) * 100,
            "f1_change": (
                qa_f1(int8_text, references)
                - qa_f1(auto_text, references)
            ) * 100,
            "outputs_identical": auto_text == int8_text,
            "question": auto_output["question"],
            "reference_answers": json.dumps(
                references,
                ensure_ascii=False,
            ),
            "auto_answer": auto_text,
            "int8_answer": int8_text,
        }

        comparison.append(record)

    fieldnames = list(comparison[0].keys())

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(comparison)

    print("=" * 76)
    print("Qasper-E: auto vs. INT8")
    print("=" * 76)

    for record in comparison:
        print(
            f"\nPrompt: {record['prompt_tokens']:,} tokens"
        )
        print(
            f"Time change: "
            f"{record['int8_time_change_percent']:+.2f}%"
        )
        print(
            f"Auto F1: {record['auto_f1']:.2f}"
        )
        print(
            f"INT8 F1: {record['int8_f1']:.2f}"
        )
        print(
            f"F1 change: {record['f1_change']:+.2f}"
        )
        print(
            f"Identical output: "
            f"{record['outputs_identical']}"
        )
        print(f"Auto: {record['auto_answer']}")
        print(f"INT8: {record['int8_answer']}")

    mean_auto_f1 = sum(
        record["auto_f1"]
        for record in comparison
    ) / len(comparison)

    mean_int8_f1 = sum(
        record["int8_f1"]
        for record in comparison
    ) / len(comparison)

    print("\n" + "=" * 76)
    print(f"Mean auto F1: {mean_auto_f1:.2f}")
    print(f"Mean INT8 F1: {mean_int8_f1:.2f}")
    print(
        f"Mean difference: "
        f"{mean_int8_f1 - mean_auto_f1:+.2f}"
    )
    print(f"Saved comparison: {OUTPUT_CSV}")
    print("=" * 76)


if __name__ == "__main__":
    main()
