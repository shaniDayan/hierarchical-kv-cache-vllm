import argparse
import json
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help="KV cache dtype: auto or int8_per_token_head",
    )
    parser.add_argument(
        "--model",
        default=(
            "/home/shani.dayan/.cache/huggingface/hub/"
            "models--Qwen--Qwen3-0.6B/snapshots/"
            "c1899de289a04d12100db370d81485cdf75e47ca"
        ),
        help="Local model path or Hugging Face model ID",
    )
    parser.add_argument(
        "--result-json",
        default=None,
        help="Optional path for saving measured generation results as JSON",
    )
    args = parser.parse_args()

    model_name = args.model

    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"KV cache dtype: {args.kv_cache_dtype}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    llm = LLM(
        model=model_name,
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

    prompt = (
        "Explain why the KV cache is important during large language "
        "model inference and how PagedAttention manages it."
    )

    # First generation compiles and warms up the required Triton kernels.
    print("\nRunning warmup...")
    llm.generate([prompt], sampling_params)

    # The second generation is the measured run.
    print("\nRunning measured experiment...")
    start_time = time.perf_counter()

    outputs = llm.generate([prompt], sampling_params)

    elapsed_time = time.perf_counter() - start_time

    completion = outputs[0].outputs[0]
    generated_text = completion.text
    generated_token_ids = list(completion.token_ids)
    generated_tokens = len(generated_token_ids)
    tokens_per_second = generated_tokens / elapsed_time

    print("\nGenerated text:")
    print(generated_text)

    print("\nGenerated token IDs:")
    print(generated_token_ids)

    print("\nExperiment results:")
    print(f"KV cache dtype: {args.kv_cache_dtype}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Generation time: {elapsed_time:.4f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/second")

    if args.result_json is not None:
        result = {
            "model": model_name,
            "kv_cache_dtype": args.kv_cache_dtype,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 128,
            "generated_text": generated_text,
            "generated_token_ids": generated_token_ids,
            "generated_tokens": generated_tokens,
            "generation_time_seconds": elapsed_time,
            "throughput_tokens_per_second": tokens_per_second,
        }
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved result JSON: {result_path}")


if __name__ == "__main__":
    main()
