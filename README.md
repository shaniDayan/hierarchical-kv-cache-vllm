# Time-Aware Hierarchical KV Cache Management for Concurrent LLM Inference

This project investigates time-aware hierarchical KV cache management for efficient concurrent large language model inference using vLLM.

During multi-user LLM serving, some conversations are actively generating tokens while others may remain idle before receiving another user turn. Keeping the complete KV cache of every inactive conversation in high precision on the GPU can waste limited device memory.

The project classifies each conversation according to the time elapsed since its most recent user turn:

* **HOT** — an active or recently active conversation.
* **WARM** — a temporarily inactive conversation.
* **COLD** — a conversation that has remained inactive for a longer period.

The hierarchy policy operates at the conversation level, while KV cache storage and migration are managed using vLLM physical blocks.

## Target Storage Hierarchy

| State | Planned representation          | Location |
| ----- | ------------------------------- | -------- |
| HOT   | FP16/BF16                       | GPU      |
| WARM  | INT8 with per-token-head scales | GPU      |
| COLD  | Quantized representation        | CPU      |

When a WARM conversation becomes active again, its historical INT8 blocks remain WARM while newly generated KV blocks are stored as HOT FP16 blocks. The modified attention implementation can read historical WARM blocks and new HOT blocks during the same attention operation.

For a COLD conversation, the planned resume path restores its quantized history from CPU memory into the WARM GPU tier before generation continues.

## Current Implementation

The current prototype includes:

* HOT, WARM, and COLD hierarchy metadata;
* request-level activity tracking;
* configurable HOT and COLD idle-time thresholds;
* deterministic conversation-level state classification;
* propagation of idle-state transitions to all eligible KV blocks of a request;
* conservative handling of shared prefix-cache blocks by keeping them HOT;
* a physical INT8 WARM KV cache pool on the GPU;
* HOT-to-WARM physical block mappings;
* experimental quantization of complete HOT blocks into WARM storage;
* mixed HOT/WARM reads in the Triton attention kernel;
* deterministic All-HOT versus mixed HOT/WARM output comparison.

The current physical-tier implementation is still experimental. HOT-to-WARM quantization has been validated for a debug block, but general multi-block migration, WARM slot management, GPU memory reclamation, COLD CPU offloading, and restoration are under development.

## Time-Aware Policy

For a conversation with idle time (T_{\text{idle}}), the logical policy is:

```text
T_idle < T_hot                  → HOT
T_hot ≤ T_idle < T_cold         → WARM
T_idle ≥ T_cold                 → COLD
```

The thresholds can be configured through:

```text
--kv-cache-hot-idle-threshold-seconds
--kv-cache-cold-idle-threshold-seconds
```

Both thresholds must be supplied together and satisfy:

```text
0 ≤ T_hot < T_cold
```

## Research Questions

The project evaluates:

1. How much GPU KV cache memory can be saved by demoting inactive conversations?
2. How does the hierarchy affect the number of concurrent conversations that can be maintained?
3. What latency is introduced when a WARM or COLD conversation resumes?
4. How does INT8 historical KV storage affect generated tokens and task-level accuracy?
5. How do different idle-time thresholds and workload patterns affect the memory–latency trade-off?

## Evaluation Plan

The system will be compared against:

* an All-HOT FP16/BF16 baseline;
* uniform INT8 KV cache quantization;
* the time-aware HOT/WARM/COLD hierarchy.

Measurements include:

* GPU KV cache memory;
* HOT, WARM, and COLD block counts;
* concurrent session capacity;
* throughput;
* request latency;
* time to first token;
* inactive-session resume latency;
* deterministic token agreement;
* MATH-500 accuracy;
* long-context quality.

The workload plan combines:

* controlled synthetic multi-turn conversations for deterministic transition tests;
* BurstGPT traces for realistic arrivals, session IDs, idle intervals, and concurrency;
* ShareGPT conversations for realistic multi-turn content;
* MATH-500 and LongBench/Qasper for quality evaluation.

## Repository Structure

* `external/vllm/` — vendored vLLM source code and project modifications
* `experiments/scripts/` — experiment and evaluation scripts
* `experiments/notebooks/` — analysis notebooks
* `experiments/results/` — experiment outputs and figures
* `notes/` — environment and project notes
* `VLLM_BASE_COMMIT.txt` — original vLLM base commit
* `VLLM_BASE_BRANCH.txt` — original vLLM base branch

## Reference Work

The project builds on:

* **PagedAttention**, which manages KV cache memory using fixed-size physical blocks;
* **KVQuant**, which studies low-bit KV cache quantization and its effect on model quality;
* real-world LLM serving traces such as **BurstGPT**.

KVQuant is kept locally as a reference implementation and is not included in this repository.

## Status

The logical time-aware control path and the experimental mixed HOT/WARM attention path are implemented separately. The next implementation stages connect logical state transitions to general physical block migration, add WARM slot management, and implement the COLD CPU tier.
