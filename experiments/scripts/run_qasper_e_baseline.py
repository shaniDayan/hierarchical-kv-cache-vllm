#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
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

SELECTION_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "original_vllm"
    / "longbench"
    / "qasper_e_selected_examples.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "original_vllm"
    / "longbench"
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_MODEL_LEN = 16_384
MAX_OUTPUT_TOKENS = 128
GPU_MEMORY_UTILIZATION = 0.50
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the original vLLM Qasper-E baseline."
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


def write_environment_file(
    path: Path,
    kv_cache_dtype: str,
) -> None:
    environment = {
        "model": MODEL_NAME,
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
            environment,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    args = parse_args()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    if not SELECTION_PATH.exists():
        raise FileNotFoundError(SELECTION_PATH)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_prefix = f"qasper_e_{args.kv_cache_dtype}"

    csv_path = RESULTS_DIR / f"{output_prefix}.csv"
    jsonl_path = RESULTS_DIR / f"{output_prefix}.jsonl"
    environment_path = (
        RESULTS_DIR / f"{output_prefix}_environment.json"
    )

    rows = load_jsonl(DATASET_PATH)

    with SELECTION_PATH.open("r", encoding="utf-8") as file:
        selected_examples = json.load(file)

    selected_examples = sorted(
        selected_examples,
        key=lambda item: item["target_tokens"],
    )

    print("=" * 72)
    print("Original vLLM Qasper-E baseline")
    print(f"Model:              {MODEL_NAME}")
    print(f"KV cache dtype:     {args.kv_cache_dtype}")
    print("Model dtype:        float16")
    print(f"Maximum model len:  {MAX_MODEL_LEN}")
    print(f"Maximum output:     {MAX_OUTPUT_TOKENS}")
    print(f"GPU utilization:    {GPU_MEMORY_UTILIZATION}")
    print(f"Selected examples:  {len(selected_examples)}")
    print("=" * 72)

    write_environment_file(
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

    # Warm-up generation is not included in the measurements.
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

    csv_records: list[dict[str, Any]] = []
    jsonl_records: list[dict[str, Any]] = []

    for position, selected in enumerate(
        selected_examples,
        start=1,
    ):
        dataset_index = int(selected["index"])
        row = rows[dataset_index]

        prompt = build_prompt(tokenizer, row)

        tokenizer_prompt_tokens = len(
            tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )
        )

        expected_tokens = int(selected["prompt_tokens"])

        if tokenizer_prompt_tokens != expected_tokens:
            raise RuntimeError(
                "Prompt token count changed for dataset index "
                f"{dataset_index}: expected {expected_tokens}, "
                f"received {tokenizer_prompt_tokens}"
            )

        print("-" * 72)
        print(
            f"Running example {position}/{len(selected_examples)}"
        )
        print(f"Dataset index:       {dataset_index}")
        print(
            f"Target prompt length: "
            f"{selected['target_tokens']:,}"
        )
        print(
            f"Actual prompt length: "
            f"{tokenizer_prompt_tokens:,}"
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
            "example_id": row.get("_id", str(dataset_index)),
            "target_prompt_tokens": selected["target_tokens"],
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
        }

        jsonl_record = {
            **csv_record,
            "question": row["input"],
            "reference_answers": row.get("answers", []),
            "generated_text": generated_text,
        }

        csv_records.append(csv_record)
        jsonl_records.append(jsonl_record)

        print(f"Output tokens:       {output_tokens}")
        print(
            f"Generation time:     "
            f"{elapsed_seconds:.4f} seconds"
        )
        print(
            f"Output throughput:   "
            f"{output_tokens_per_second:.2f} tokens/second"
        )
        print(
            "Generated answer:     "
            f"{generated_text[:300].strip()}"
        )

    fieldnames = [
        "kv_cache_dtype",
        "dataset",
        "dataset_index",
        "example_id",
        "target_prompt_tokens",
        "prompt_tokens",
        "output_tokens",
        "generation_time_seconds",
        "output_tokens_per_second",
        "total_tokens_per_second",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(csv_records)

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in jsonl_records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\n" + "=" * 72)
    print("Baseline completed")
    print(f"CSV summary:  {csv_path}")
    print(f"Full outputs: {jsonl_path}")
    print(f"Environment:  {environment_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
