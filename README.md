# Hierarchical KV Cache Management for Efficient LLM Inference

This project investigates hierarchical KV cache management for efficient
large language model inference using vLLM.

The main idea is to classify KV-cache pages according to their importance:

- **HOT** pages: stored at high precision
- **WARM** pages: stored using INT8 quantization
- **COLD** pages: stored using lower precision or potentially offloaded

The project aims to reduce GPU memory usage while preserving model accuracy.

## Repository Structure

- `external/vllm/` — vLLM source code used as the implementation base
- `experiments/` — experiment scripts and evaluation code
- `notes/` — project notes and code exploration
- `01_inspect_qwen_kv_cache.ipynb` — initial KV-cache inspection notebook

## Reference Implementation

KVQuant is kept locally as a reference implementation and is not included
in this repository.

The project is based on the vLLM version recorded in:

- `VLLM_BASE_COMMIT.txt`
- `VLLM_BASE_BRANCH.txt`
