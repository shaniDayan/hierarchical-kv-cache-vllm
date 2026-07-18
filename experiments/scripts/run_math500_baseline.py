#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

PROMPT_TEMPLATE = r"""
Solve the following mathematics problem.

Give a concise solution and end with the final answer in LaTeX using exactly:

\boxed{{final answer}}

Do not write anything after the boxed answer.

Problem:

{problem}
""".strip()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MATH-500 baseline using FP16 or INT8 KV cache."
        )
    )

    parser.add_argument(
        "--kv-cache-dtype",
        required=True,
        choices=[
            "auto",
            "int8_per_token_head",
        ],
        help="KV-cache format used by vLLM.",
    )

    parser.add_argument(
        "--dataset-path",
        default="data/math500/test.jsonl",
        help="Path to the local MATH-500 JSONL file.",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "experiments/results/"
            "original_vllm/math500"
        ),
        help="Directory in which results are stored.",
    )

    parser.add_argument(
        "--tag",
        default="full",
        help=(
            "Name added to the result filename, "
            "for example smoke10 or full."
        ),
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First dataset index to include.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of examples to run.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of prompts submitted to vLLM together.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens per problem.",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum total model context length.",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory available to vLLM.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Sampling seed shared by both baseline runs.",
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error

            records.append(record)

    return records


def build_prompt(
    tokenizer: AutoTokenizer,
    problem: str,
) -> str:
    user_message = PROMPT_TEMPLATE.format(
        problem=problem
    )

    messages = [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_final_response(text: str) -> str:
    text = text.strip()

    if "</think>" in text:
        return text.rsplit(
            "</think>",
            maxsplit=1,
        )[-1].strip()

    return text


def verify_math_answer(
    gold_answer: str,
    generated_answer: str,
) -> tuple[bool, str | None]:
    try:
        # MATH-500 gold answers are LaTeX expressions.
        # Wrapping them in a LaTeX environment improves parsing.
        gold_parsed = parse(
            f"${gold_answer}$"
        )

        prediction_parsed = parse(
            generated_answer
        )

        is_correct = bool(
            verify(
                gold_parsed,
                prediction_parsed,
            )
        )

        return is_correct, None

    except Exception as error:
        error_message = (
            f"{type(error).__name__}: {error}"
        )

        return False, error_message


def read_existing_results(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    return load_jsonl(path)


def append_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        file.flush()


def save_csv(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    dataframe = pd.DataFrame(records)

    dataframe.to_csv(
        path,
        index=False,
    )


def print_summary(
    records: list[dict[str, Any]],
    elapsed_seconds: float,
) -> None:
    if not records:
        print("No result records were found.")
        return

    result_df = pd.DataFrame(records)

    correct_count = int(
        result_df["correct"].sum()
    )

    total_count = len(result_df)

    accuracy = (
        100
        * correct_count
        / total_count
    )

    average_output_tokens = (
        result_df["output_tokens"].mean()
    )

    truncated_count = int(
        result_df["reached_max_tokens"].sum()
    )

    verification_errors = int(
        result_df[
            "verification_error"
        ].notna().sum()
    )

    print()
    print("=" * 70)
    print("MATH-500 SUMMARY")
    print("=" * 70)

    print(
        f"KV cache dtype: "
        f"{result_df['kv_cache_dtype'].iloc[0]}"
    )

    print(
        f"Examples: {total_count}"
    )

    print(
        f"Correct: {correct_count}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print(
        f"Average output tokens: "
        f"{average_output_tokens:.2f}"
    )

    print(
        f"Reached max tokens: "
        f"{truncated_count}"
    )

    print(
        f"Verification errors: "
        f"{verification_errors}"
    )

    print(
        f"Current invocation time: "
        f"{elapsed_seconds:.2f} seconds"
    )

    print()
    print("Accuracy by level:")

    level_summary = (
        result_df
        .groupby("level")
        .agg(
            examples=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    level_summary["accuracy"] *= 100

    print(
        level_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )

    print()
    print("Accuracy by subject:")

    subject_summary = (
        result_df
        .groupby("subject")
        .agg(
            examples=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
        .sort_values("subject")
    )

    subject_summary["accuracy"] *= 100

    print(
        subject_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )

    print("=" * 70)


def main() -> None:
    args = parse_arguments()

    project_root = Path.cwd()

    dataset_path = Path(
        args.dataset_path
    ).expanduser()

    if not dataset_path.is_absolute():
        dataset_path = (
            project_root
            / dataset_path
        )

    output_dir = Path(
        args.output_dir
    ).expanduser()

    if not output_dir.is_absolute():
        output_dir = (
            project_root
            / output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"MATH-500 file was not found: "
            f"{dataset_path}"
        )

    model_name = os.environ.get(
        "QWEN_LOCAL_MODEL",
        DEFAULT_MODEL,
    )

    result_stem = (
        f"math500_{args.tag}_"
        f"{args.kv_cache_dtype}"
    )

    jsonl_path = (
        output_dir
        / f"{result_stem}.jsonl"
    )

    csv_path = (
        output_dir
        / f"{result_stem}.csv"
    )

    all_examples = load_jsonl(
        dataset_path
    )

    if len(all_examples) != 500:
        raise RuntimeError(
            "Expected exactly 500 MATH-500 examples, "
            f"but found {len(all_examples)}."
        )

    end_index = min(
        args.start_index + args.limit,
        len(all_examples),
    )

    selected_examples: list[
        dict[str, Any]
    ] = []

    for dataset_index in range(
        args.start_index,
        end_index,
    ):
        example = dict(
            all_examples[dataset_index]
        )

        example["dataset_index"] = (
            dataset_index
        )

        selected_examples.append(
            example
        )

    existing_results = (
        read_existing_results(
            jsonl_path
        )
    )

    completed_indices = {
        int(record["dataset_index"])
        for record in existing_results
    }

    pending_examples = [
        example
        for example in selected_examples
        if example["dataset_index"]
        not in completed_indices
    ]

    print(
        f"Model: {model_name}"
    )

    print(
        f"KV cache dtype: "
        f"{args.kv_cache_dtype}"
    )

    print(
        f"Selected examples: "
        f"{len(selected_examples)}"
    )

    print(
        f"Already completed: "
        f"{len(selected_examples) - len(pending_examples)}"
    )

    print(
        f"Pending examples: "
        f"{len(pending_examples)}"
    )

    print(
        f"Results: {jsonl_path}"
    )

    if not pending_examples:
        print(
            "All selected examples are already complete."
        )

        save_csv(
            existing_results,
            csv_path,
        )

        print_summary(
            existing_results,
            elapsed_seconds=0.0,
        )

        return

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    )

    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        dtype="half",
        kv_cache_dtype=(
            args.kv_cache_dtype
        ),
        tensor_parallel_size=1,
        max_model_len=(
            args.max_model_len
        ),
        gpu_memory_utilization=(
            args.gpu_memory_utilization
        ),
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    invocation_start = (
        time.perf_counter()
    )

    newly_completed = 0

    for batch_start in range(
        0,
        len(pending_examples),
        args.batch_size,
    ):
        batch_examples = (
            pending_examples[
                batch_start:
                batch_start
                + args.batch_size
            ]
        )

        batch_prompts = [
            build_prompt(
                tokenizer,
                example["problem"],
            )
            for example in batch_examples
        ]

        batch_start_time = (
            time.perf_counter()
        )

        request_outputs = llm.generate(
            batch_prompts,
            sampling_params,
            use_tqdm=False,
        )

        batch_elapsed = (
            time.perf_counter()
            - batch_start_time
        )

        batch_records: list[
            dict[str, Any]
        ] = []

        batch_correct = 0

        for example, request_output in zip(
            batch_examples,
            request_outputs,
        ):
            if not request_output.outputs:
                raise RuntimeError(
                    "vLLM returned no output "
                    f"for dataset index "
                    f"{example['dataset_index']}."
                )

            generated_output = (
                request_output.outputs[0]
            )

            raw_generated_text = (
                generated_output.text
            )

            final_response = (
                extract_final_response(
                    raw_generated_text
                )
            )

            correct, verification_error = (
                verify_math_answer(
                    str(example["answer"]),
                    final_response,
                )
            )

            output_tokens = len(
                generated_output.token_ids
            )

            prompt_token_ids = getattr(
                request_output,
                "prompt_token_ids",
                None,
            )

            if prompt_token_ids is None:
                prompt_tokens = len(
                    tokenizer.encode(
                        batch_prompts[
                            len(batch_records)
                        ]
                    )
                )
            else:
                prompt_tokens = len(
                    prompt_token_ids
                )

            finish_reason = getattr(
                generated_output,
                "finish_reason",
                None,
            )

            reached_max_tokens = (
                output_tokens
                >= args.max_tokens
            )

            record = {
                "dataset": "MATH-500",
                "dataset_index": int(
                    example["dataset_index"]
                ),
                "unique_id": example.get(
                    "unique_id"
                ),
                "subject": example.get(
                    "subject"
                ),
                "level": example.get(
                    "level"
                ),
                "problem": example.get(
                    "problem"
                ),
                "gold_solution": example.get(
                    "solution"
                ),
                "gold_answer": example.get(
                    "answer"
                ),
                "generated_text": (
                    raw_generated_text
                ),
                "final_response": (
                    final_response
                ),
                "correct": bool(correct),
                "verification_error": (
                    verification_error
                ),
                "prompt_tokens": int(
                    prompt_tokens
                ),
                "output_tokens": int(
                    output_tokens
                ),
                "finish_reason": (
                    finish_reason
                ),
                "reached_max_tokens": bool(
                    reached_max_tokens
                ),
                "batch_elapsed_seconds": (
                    batch_elapsed
                ),
                "batch_size": len(
                    batch_examples
                ),
                "model": model_name,
                "kv_cache_dtype": (
                    args.kv_cache_dtype
                ),
                "thinking_enabled": False,
                "temperature": 0.0,
                "seed": args.seed,
                "max_tokens": (
                    args.max_tokens
                ),
                "max_model_len": (
                    args.max_model_len
                ),
            }

            batch_records.append(
                record
            )

            if correct:
                batch_correct += 1

        append_jsonl(
            jsonl_path,
            batch_records,
        )

        newly_completed += len(
            batch_records
        )

        completed_now = (
            len(selected_examples)
            - len(pending_examples)
            + newly_completed
        )

        print(
            f"Completed "
            f"{completed_now}/"
            f"{len(selected_examples)} "
            f"| batch correct "
            f"{batch_correct}/"
            f"{len(batch_records)} "
            f"| batch time "
            f"{batch_elapsed:.2f}s"
        )

    invocation_elapsed = (
        time.perf_counter()
        - invocation_start
    )

    all_results = read_existing_results(
        jsonl_path
    )

    all_results = sorted(
        all_results,
        key=lambda record: int(
            record["dataset_index"]
        ),
    )

    save_csv(
        all_results,
        csv_path,
    )

    print()
    print(
        f"JSONL saved to: {jsonl_path}"
    )

    print(
        f"CSV saved to: {csv_path}"
    )

    print_summary(
        all_results,
        invocation_elapsed,
    )


if __name__ == "__main__":
    main()
