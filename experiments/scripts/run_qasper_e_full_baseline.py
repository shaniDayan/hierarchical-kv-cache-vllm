#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


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
    / "qasper_e_full"
)

MODEL_NAME = os.environ.get(
    "QWEN_LOCAL_MODEL",
    "Qwen/Qwen3-0.6B",
)

MAX_MODEL_LEN = 32_768
MAX_OUTPUT_TOKENS = 128
GPU_MEMORY_UTILIZATION = 0.50
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Qasper-E dataset on the original vLLM."
        )
    )

    parser.add_argument(
        "--kv-cache-dtype",
        required=True,
        choices=[
            "auto",
            "int8_per_token_head",
        ],
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error

    return rows


def load_completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()

    completed: set[int] = set()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                row = json.loads(line)
                completed.add(int(row["dataset_index"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    return completed


def build_prompt(
    tokenizer: AutoTokenizer,
    row: dict[str, Any],
) -> str:
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


def length_bucket(token_count: int) -> str:
    if token_count <= 2_048:
        return "0-2K"
    if token_count <= 4_096:
        return "2K-4K"
    if token_count <= 8_192:
        return "4K-8K"
    if token_count <= 16_384:
        return "8K-16K"
    if token_count <= 32_768:
        return "16K-32K"
    return ">32K"


def save_environment(
    path: Path,
    kv_cache_dtype: str,
) -> None:
    information = {
        "model": MODEL_NAME,
        "dataset": "qasper_e",
        "dataset_path": str(DATASET_PATH),
        "kv_cache_dtype": kv_cache_dtype,
        "model_dtype": "float16",
        "max_model_len": MAX_MODEL_LEN,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "seed": SEED,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            information,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    args = parse_args()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this script inside a SLURM GPU job."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prefix = f"qasper_e_full_{args.kv_cache_dtype}"

    jsonl_path = RESULTS_DIR / f"{prefix}.jsonl"
    csv_path = RESULTS_DIR / f"{prefix}.csv"
    environment_path = RESULTS_DIR / f"{prefix}_environment.json"

    rows = load_jsonl(DATASET_PATH)
    completed_indices = load_completed_indices(jsonl_path)

    print("=" * 76)
    print("Complete Qasper-E baseline")
    print(f"Model:              {MODEL_NAME}")
    print(f"Dataset examples:   {len(rows)}")
    print(f"KV cache dtype:     {args.kv_cache_dtype}")
    print("Model dtype:        float16")
    print(f"Maximum model len:  {MAX_MODEL_LEN:,}")
    print(f"Maximum output:     {MAX_OUTPUT_TOKENS}")
    print(f"Already completed:  {len(completed_indices)}")
    print(f"Remaining:          {len(rows) - len(completed_indices)}")
    print("=" * 76)

    save_environment(
        environment_path,
        args.kv_cache_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        disable_log_stats=True,
        seed=SEED,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    print("\nRunning warm-up request...")

    llm.generate(
        ["Reply with the single word OK."],
        SamplingParams(
            temperature=0.0,
            max_tokens=4,
        ),
        use_tqdm=False,
    )

    print("Warm-up completed.\n")

    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0

    csv_fields = [
        "kv_cache_dtype",
        "dataset",
        "dataset_index",
        "example_id",
        "reported_length",
        "length_bucket",
        "prompt_tokens",
        "output_tokens",
        "generation_time_seconds",
        "output_tokens_per_second",
        "total_tokens_per_second",
        "finish_reason",
    ]

    with (
        jsonl_path.open("a", encoding="utf-8") as jsonl_file,
        csv_path.open("a", encoding="utf-8", newline="") as csv_file,
    ):
        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_fields,
        )

        if not csv_exists:
            writer.writeheader()
            csv_file.flush()

        for dataset_index, row in enumerate(rows):
            if dataset_index in completed_indices:
                continue

            prompt = build_prompt(tokenizer, row)

            prompt_tokens_from_tokenizer = len(
                tokenizer.encode(
                    prompt,
                    add_special_tokens=False,
                )
            )

            if (
                prompt_tokens_from_tokenizer
                + MAX_OUTPUT_TOKENS
                > MAX_MODEL_LEN
            ):
                raise RuntimeError(
                    f"Example {dataset_index} requires "
                    f"{prompt_tokens_from_tokenizer + MAX_OUTPUT_TOKENS} "
                    f"tokens, exceeding max_model_len={MAX_MODEL_LEN}"
                )

            print("-" * 76)
            print(
                f"Example {dataset_index + 1}/{len(rows)} "
                f"(dataset index {dataset_index})"
            )
            print(
                f"Prompt tokens: {prompt_tokens_from_tokenizer:,}"
            )
            print(
                f"Bucket:        "
                f"{length_bucket(prompt_tokens_from_tokenizer)}"
            )

            start_time = time.perf_counter()

            request_outputs = llm.generate(
                [prompt],
                sampling_params,
                use_tqdm=False,
            )

            elapsed_seconds = time.perf_counter() - start_time

            request_output = request_outputs[0]
            completion = request_output.outputs[0]

            generated_text = completion.text
            output_tokens = len(completion.token_ids)
            actual_prompt_tokens = len(
                request_output.prompt_token_ids
            )

            output_tokens_per_second = (
                output_tokens / elapsed_seconds
                if elapsed_seconds > 0
                else 0.0
            )

            total_tokens = (
                actual_prompt_tokens + output_tokens
            )

            total_tokens_per_second = (
                total_tokens / elapsed_seconds
                if elapsed_seconds > 0
                else 0.0
            )

            csv_record = {
                "kv_cache_dtype": args.kv_cache_dtype,
                "dataset": "qasper_e",
                "dataset_index": dataset_index,
                "example_id": row.get(
                    "_id",
                    str(dataset_index),
                ),
                "reported_length": row.get("length"),
                "length_bucket": length_bucket(
                    actual_prompt_tokens
                ),
                "prompt_tokens": actual_prompt_tokens,
                "output_tokens": output_tokens,
                "generation_time_seconds": round(
                    elapsed_seconds,
                    6,
                ),
                "output_tokens_per_second": round(
                    output_tokens_per_second,
                    4,
                ),
                "total_tokens_per_second": round(
                    total_tokens_per_second,
                    4,
                ),
                "finish_reason": completion.finish_reason,
            }

            jsonl_record = {
                **csv_record,
                "question": row["input"],
                "reference_answers": row.get(
                    "answers",
                    [],
                ),
                "generated_text": generated_text,
            }

            writer.writerow(csv_record)
            csv_file.flush()

            jsonl_file.write(
                json.dumps(
                    jsonl_record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            jsonl_file.flush()

            print(f"Output tokens: {output_tokens}")
            print(f"Time:          {elapsed_seconds:.4f} seconds")
            print(
                f"Answer:        "
                f"{generated_text[:180].strip()}"
            )

    final_completed = load_completed_indices(jsonl_path)

    print("\n" + "=" * 76)
    print("Qasper-E run finished")
    print(f"Completed examples: {len(final_completed)}/{len(rows)}")
    print(f"CSV:               {csv_path}")
    print(f"JSONL:             {jsonl_path}")
    print(f"Environment:       {environment_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
