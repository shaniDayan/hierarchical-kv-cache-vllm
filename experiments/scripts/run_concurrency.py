import argparse
import csv
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPT_LENGTH = 1024
BATCH_SIZES = [40, 48, 56, 64]

def build_prompt(tokenizer, target_tokens: int) -> str:
    """Create a synthetic prompt with exactly target_tokens tokens."""
    base_text = (
        "Large language model inference uses a key value cache to avoid "
        "recomputing previous attention states. PagedAttention stores these "
        "states in fixed-size memory blocks and manages them dynamically. "
    )

    repeated_text = base_text * 300
    token_ids = tokenizer.encode(
        repeated_text,
        add_special_tokens=False,
    )

    if len(token_ids) < target_tokens:
        raise RuntimeError(
            f"Could not create a prompt with {target_tokens} tokens."
        )

    return tokenizer.decode(token_ids[:target_tokens])


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        choices=["auto", "int8_per_token_head"],
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help="CSV file in which the results will be saved.",
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    prompt = build_prompt(tokenizer, PROMPT_LENGTH)
    actual_prompt_length = len(
        tokenizer.encode(prompt, add_special_tokens=False)
    )

    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=2048,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
    )

    os.makedirs(
        os.path.dirname(args.output_file),
        exist_ok=True,
    )

    results = []

    for batch_size in BATCH_SIZES:
        prompts = [prompt] * batch_size

        print("\n" + "=" * 60)
        print(f"KV cache dtype: {args.kv_cache_dtype}")
        print(f"Concurrent requests: {batch_size}")
        print(f"Prompt tokens per request: {actual_prompt_length}")
        print("=" * 60)

        # Warmup for this batch size.
        print("Running warmup...")
        llm.generate(prompts, sampling_params)

        # Measured execution.
        print("Running measured experiment...")
        start_time = time.perf_counter()

        outputs = llm.generate(prompts, sampling_params)

        elapsed_time = time.perf_counter() - start_time

        total_output_tokens = sum(
            len(output.outputs[0].token_ids)
            for output in outputs
        )

        total_input_tokens = (
            actual_prompt_length * batch_size
        )

        output_throughput = (
            total_output_tokens / elapsed_time
        )

        request_throughput = batch_size / elapsed_time

        total_token_throughput = (
            total_input_tokens + total_output_tokens
        ) / elapsed_time

        result = {
            "kv_cache_dtype": args.kv_cache_dtype,
            "concurrent_requests": batch_size,
            "prompt_tokens_per_request": actual_prompt_length,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "batch_completion_time_seconds": round(
                elapsed_time,
                4,
            ),
            "requests_per_second": round(
                request_throughput,
                3,
            ),
            "output_tokens_per_second": round(
                output_throughput,
                2,
            ),
            "total_tokens_per_second": round(
                total_token_throughput,
                2,
            ),
        }

        results.append(result)

        print("Result:")
        print(result)

    with open(
        args.output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()