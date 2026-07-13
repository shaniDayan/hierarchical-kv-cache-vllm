import argparse
import time

import torch
from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help="KV cache dtype: auto or int8_per_token_head",
    )
    args = parser.parse_args()

    model_name = "Qwen/Qwen3-0.6B"

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

    generated_text = outputs[0].outputs[0].text
    generated_tokens = len(outputs[0].outputs[0].token_ids)
    tokens_per_second = generated_tokens / elapsed_time

    print("\nGenerated text:")
    print(generated_text)

    print("\nExperiment results:")
    print(f"KV cache dtype: {args.kv_cache_dtype}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Generation time: {elapsed_time:.4f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/second")


if __name__ == "__main__":
    main()