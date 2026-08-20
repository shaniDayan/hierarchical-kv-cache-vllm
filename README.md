# Time-Aware Hierarchical KV Cache Management for Concurrent LLM Inference

This project investigates time-aware hierarchical KV cache management for efficient concurrent large language model inference using vLLM.

During multi-user LLM serving, some conversations are actively generating tokens while others remain idle between user turns. Keeping the entire KV cache of inactive conversations in high precision on the GPU can waste device memory.

The system classifies each conversation based on the elapsed time since its most recent user turn:

* **HOT** — an active or recently active conversation.
* **WARM** — a temporarily inactive conversation.
* **COLD** — a conversation that has remained inactive for an extended period.

The hierarchy policy operates at the conversation level, while KV cache storage and migrations are executed per physical KV block.

## Target Storage Hierarchy

| State | Representation                  | Location | Status |
| ----- | ------------------------------- | -------- | ------ |
| HOT   | FP16/BF16                       | GPU      | Implemented |
| WARM  | INT8 with per-token-head scales | GPU      | Implemented |
| COLD  | Quantized representation        | CPU      | Deferred / Planned |

When a WARM conversation becomes active again, its historical INT8 blocks remain WARM while newly generated KV blocks are allocated as HOT FP16 blocks. The modified attention kernel reads historical WARM blocks and new HOT blocks during the same attention operation.

> [!NOTE]
> Physical COLD CPU offloading, restoration, and prefetching are not implemented and are intentionally deferred. While `classify_request_kv_state()` logically returns COLD when $T_{\text{idle}} \ge T_{\text{cold}}$, the scheduler clamps a direct HOT $\rightarrow$ COLD classification to HOT $\rightarrow$ WARM, and a request already in WARM remains in WARM rather than scheduling a WARM $\rightarrow$ COLD transition.

## Implemented Architecture

The codebase currently implements:

* **Time-aware request classification**: Deterministic idle-time evaluation based on configurable HOT/COLD thresholds ($0 \le T_{\text{hot}} < T_{\text{cold}}$).
* **Physical multi-block HOT->WARM migration**: CUDA-based quantization and copy of eligible complete HOT blocks, with current-stream synchronization before a SUCCESS result is emitted.
* **INT8 WARM GPU storage**: Compact INT8 tensor representation with per-token per-head scaling factors.
* **Logical-block WARM slot ownership**: Dynamic tracking of logical block positions within WARM residency tables.
* **Transactional reservation and cleanup**: Two-phase reservation guaranteeing safe rollback upon capacity or validation errors.
* **HOT block reclamation and reuse**: Automatic release and reuse of freed HOT blocks upon successful WARM migration commit.
* **Conservative prefix handling**: Shared prefix blocks with `ref_cnt > 1` remain HOT to protect active sessions.
* **Mixed HOT/WARM Triton attention reads**: Custom attention kernels reading mixed-precision block tables in a single pass.
* **Ordered worker transition results**: Strict sequencing of state transition acknowledgments from worker to scheduler.
* **CUDA-fenced SUCCESS acknowledgment**: Stream synchronization fencing memory writes before commit.
* **Robust status handling**: Explicit support and rollback paths for `SUCCESS`, `RETRYABLE_CAPACITY`, and `STALE_VALIDATION`.
* **Deferred pending-transition lifecycle**: Safe handling of streaming updates, client finish, and client abort arriving while transition ACKs are in flight.
* **Strict runtime configuration validation**: Enforced runtime checks when thresholds are active:
  * `0 <= T_hot < T_cold`
  * `VLLM_USE_V2_MODEL_RUNNER=1`
  * `HKV_ENABLE_PHYSICAL_TIERS=1`
  * `HKV_ENABLE_MULTI_BLOCK_WARM_MIGRATION=1`
  * `HKV_DEBUG_MIXED_READ=1`
  * `HKV_WARM_POOL_BLOCKS` must be an integer greater than zero
* **Qwen-Bailian trace parser and replay runner**: Real-world multi-turn conversational replay tools for validation.

## Verification & Correctness Milestone

The implementation has reached a verified end-to-end correctness milestone:

* **Scheduler & Worker Test Suite**: 58 passed (`external/vllm/tests/v1/streaming_input/test_scheduler_streaming.py`, `external/vllm/tests/v1/worker/test_gpu_worker_kv_state.py`).
* **Trace & Replay Test Suite**: 16 passed (`experiments/tests/test_qwen_bailian_trace.py`, `experiments/tests/test_run_qwen_bailian_replay.py`).
* **Qwen-Bailian Correctness Replay**:
  * Successfully replayed 9 real-world conversational sessions across 25 turns.
  * Observed a peak of 73 WARM blocks across 6 demoted sessions.
  * Verified allocator consistency with zero leaks and clean post-run teardown.
  * Achieved exact output token agreement between hierarchical KV cache and an all-HOT baseline.

> [!IMPORTANT]
> The current replay serves as an end-to-end correctness milestone and not a performance claim. Memory capacity benefits, migration latency, equal-memory concurrency, TTFT, and kernel overhead remain to be evaluated at scale.

## Time-Aware Policy

For a conversation with idle time $T_{\text{idle}}$, the logical classification policy is:

```text
T_idle < T_hot                  → HOT
T_hot ≤ T_idle < T_cold         → WARM
T_idle ≥ T_cold                 → COLD (scheduler clamps/keeps in WARM)
```

The thresholds are configured via vLLM command-line options:

```text
--kv-cache-hot-idle-threshold-seconds <seconds>
--kv-cache-cold-idle-threshold-seconds <seconds>
```

Both thresholds must be supplied together and satisfy $0 \le T_{\text{hot}} < T_{\text{cold}}$.

## Evaluation Plan

The system is evaluated against:

* an All-HOT FP16/BF16 baseline;
* uniform INT8 KV cache quantization;
* the time-aware HOT/WARM hierarchy.

Workloads:
* **Qwen-Bailian multi-turn trace workflow**: Anonymized real-world usage traces preserving session structure, timestamps, input/output lengths, and block-hash prefix relationships (with token blocks deterministically derived from hashes, containing no original conversation text).
* **MATH-500**: Mathematical reasoning accuracy baseline.
* **LongBench / Qasper**: Long-context comprehension quality baseline.

Key metrics to evaluate:
* Effective GPU KV cache memory savings;
* Peak concurrent conversation capacity under identical GPU memory limits;
* Time to first token (TTFT) and inter-token latency on resumed sessions;
* HOT->WARM migration overhead and kernel execution profile.

## Current Limitations

* **COLD tier deferred**: Physical COLD CPU offload, CPU memory pooling, prefetching, and restoration paths are not implemented.
* **Shared prefix protection**: Shared prefix-cache blocks with `ref_cnt > 1` remain HOT and are not demoted.
* **Parallelism constraints**: Completion proof currently supports only single-worker execution (`TP=1`, `PP=1`, `DP=1`).
* **Cache layout constraints**: Physical migration currently requires a single KV-cache group and `blocks_per_kv_block == 1`.
* **Synchronous stream copy**: Migration uses the current CUDA stream and synchronizes before emitting a `SUCCESS` transition result (no asynchronous overlap with model compute).
* **Performance evaluation scope**: Quantified memory-capacity benefits, TTFT impact, migration jitter, and kernel execution overhead remain to be measured across broader workloads.

## Repository Structure

* `external/vllm/` — vendored vLLM source code and project modifications.
* `experiments/scripts/` — trace parser and replay runner scripts:
  * `qwen_bailian_trace.py` — parser and preprocessor for Qwen-Bailian conversational traces.
  * `run_qwen_bailian_replay.py` — multi-session time-aware replay runner with KV state validation.
* `experiments/notebooks/` — baseline analysis and evaluation notebooks:
  * `00_original_vllm_performance_baseline.ipynb` — baseline throughput and latency benchmarks.
  * `01_inspect_qwen_kv_cache.ipynb` — KV cache inspection and block allocation experiments.
  * `02_longbench_qasper_baseline.ipynb` — LongBench/Qasper long-context accuracy evaluation.
  * `03_math500_fp16_int8_analysis.ipynb` — MATH-500 FP16 vs INT8 quantization accuracy analysis.
* `experiments/tests/` — unit and regression tests for trace parsing and replay workflows.
* `experiments/results/` — selected compact result summaries and figures.
* `notes/` — environment configuration and design notes.
* `VLLM_BASE_COMMIT.txt` — original vLLM base commit hash.
* `VLLM_BASE_BRANCH.txt` — original vLLM base branch.

## Reference Work

The project builds on:

* **PagedAttention**, which manages KV cache memory using fixed-size physical blocks;
* **KVQuant**, which studies low-bit KV cache quantization and its effect on model quality;
* Multi-turn conversational serving traces from real-world LLM workloads.
