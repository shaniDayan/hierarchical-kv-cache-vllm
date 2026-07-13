import argparse
import csv
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPT_LENGTHS = [128, 512, 1024, 1536]


def build_prompt(tokenizer, target_tokens: int) -> str:
    """Build a synthetic prompt containing exactly target_tokens tokens."""
    base_text = (
        "Large language model inference uses a key value cache to avoid "
        "recomputing previous attention states. PagedAttention stores these "
        "states in fixed-size memory blocks and manages them dynamically. "
    )

    repeated_text = base_text * 200
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

    for target_length in PROMPT_LENGTHS:
        prompt = build_prompt(tokenizer, target_length)
        actual_length = len(
            tokenizer.encode(prompt, add_special_tokens=False)
        )

        print("\n" + "=" * 60)
        print(f"KV cache dtype: {args.kv_cache_dtype}")
        print(f"Target prompt length: {target_length}")
        print(f"Actual prompt length: {actual_length}")
        print("=" * 60)

        print("Running warmup...")
        llm.generate([prompt], sampling_params)

        print("Running measured experiment...")
        start_time = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        elapsed_time = time.perf_counter() - start_time

        generated_tokens = len(
            outputs[0].outputs[0].token_ids
        )
        throughput = generated_tokens / elapsed_time

        result = {
            "kv_cache_dtype": args.kv_cache_dtype,
            "prompt_tokens": actual_length,
            "generated_tokens": generated_tokens,
            "generation_time_seconds": round(elapsed_time, 4),
            "output_tokens_per_second": round(throughput, 2),
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