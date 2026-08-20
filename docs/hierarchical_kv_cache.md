# Hierarchical KV Cache Implementation Guide

This document describes the time-aware hierarchical KV-cache system added to vLLM
as part of this project. Its goal is to explain the overall architecture, document
the modified and newly added files, show how the components interact, and provide
a reference for runtime validation, test suites, and experimental replay tools.

The system extends the standard vLLM KV-cache architecture by allowing request
history to move between different storage tiers according to elapsed inactivity
between conversational turns. The primary objective is to reduce the high-precision
GPU KV-cache capacity occupied by idle conversational sessions while allowing them
to resume seamlessly without recomputing their full context.

## Target Storage Hierarchy

| Tier | Representation | Storage Location | Implementation Status |
| :--- | :--- | :--- | :--- |
| **HOT** | FP16 / BF16 high-precision | GPU memory (standard vLLM cache) | Fully implemented |
| **WARM** | INT8 with dynamic per-token-head scales | Dedicated GPU WARM pool | Fully implemented |
| **COLD** | Quantized representation | CPU memory (host offload) | Deferred / Planned |

### Logical Hierarchy vs. Physical Implementation

The classification model defines three logical states:

- **HOT** — active or recently active conversational sessions.
- **WARM** — temporarily inactive conversational sessions.
- **COLD** — sessions that have remained inactive for an extended period.

The current implementation completes the full **HOT → WARM** physical lifecycle:
- Inactivity-based session classification in the scheduler;
- Multi-block physical quantization (FP16/BF16 → INT8 with inline FP32 scales);
- Transactional WARM slot reservation and logical residency tracking;
- CUDA-fenced execution and ordered worker-to-scheduler result signaling;
- Reclaiming and recycling physical HOT blocks in the standard block pool;
- Single-pass mixed HOT/WARM attention reads during resumed inference;
- Non-blocking handling of streaming updates and client finish/abort during in-flight migrations;
- Robust lifecycle cleanup upon request completion, preemption, or abort.

> [!NOTE]
> **Physical COLD Tier Deferred**: CPU offloading, CPU memory pooling, prefetching,
> and host-to-device restoration paths are not implemented and are intentionally deferred.
> While `classify_request_kv_state()` logically returns `COLD` when $T_{\text{idle}} \ge T_{\text{cold}}$,
> the scheduler clamps a direct $\text{HOT} \rightarrow \text{COLD}$ classification to
> $\text{HOT} \rightarrow \text{WARM}$, and a request already in `WARM` remains in `WARM`
> rather than scheduling a $\text{WARM} \rightarrow \text{COLD}$ transition.

---

## Architecture Overview

A conversational request begins execution in the **HOT** state.

```text
       Active Session
             |
             v
   Standard HOT Allocation (FP16/BF16)
             |
             | Inactivity T_idle >= T_hot
             v
   HOT -> WARM Demotion (INT8 Quantization)
             |
             v
   Physical HOT Block Reclaimed & Recycled
             |
             | Session Resumes
             v
   Mixed Attention: WARM History (INT8) + New Generation (HOT FP16)
             |
             | Subsequent Inactivity
             v
   Additional Completed HOT Blocks -> WARM
```

### End-to-End Transition Lifecycle

1. **Inactivity Classification**:
   The scheduler monitors idle resumable sessions waiting for streaming input (`WAITING_FOR_STREAMING_REQ`). When elapsed idle time satisfies $T_{\text{idle}} \ge T_{\text{hot}}$, the scheduler identifies candidate requests for demotion.
2. **Planning**:
   The scheduler queries `KVCacheManager.plan_request_kv_state()`. The manager selects eligible completed prefix blocks (`num_computed_tokens // block_size`), skipping shared prefix blocks (`ref_cnt > 1`), incomplete tail blocks, and null placeholders.
3. **Dispatch**:
   The planned block transitions are packaged into a `KVCacheStateTransition` with a unique transition ID and attached to `SchedulerOutput.kv_cache_state_transitions`.
4. **Worker Execution & Quantization**:
   `Worker` forwards transitions to `GPUModelRunner.handle_kv_cache_state_transitions()`. The runner validates request presence and block tables, then delegates to `HKVWarmMigrationManager.migrate()`.
   - `HKVWarmSlotAllocator` transactionally reserves required WARM slots.
   - `quantize_hkv_blocks_to_warm()` launches the Triton per-token-head quantization kernel, converting FP16/BF16 KV blocks into INT8 format with FP32 scales stored in padded head dimensions.
   - If quantization succeeds, the reservation is committed, logical residency `(request_id, cache_group_index, logical_block_index) -> warm_slot_id` is recorded, and the residency revision counter increments.
   - If an error occurs (such as capacity exhaustion), newly allocated slots are rolled back.
5. **Ordered Result Signaling & CUDA Stream Fencing**:
   `GPUModelRunner` synchronizes the current CUDA stream (`torch.cuda.current_stream(device).synchronize()`) before emitting a `SUCCESS` result when new quantization work was enqueued. Results are assembled in index-preserved order (`indexed_results`) and returned to the scheduler via `ModelRunnerOutput.kv_cache_transition_results`.
6. **Validation & Scheduler Commit**:
   `Scheduler._validate_kv_cache_transition_results()` validates result signatures and ordering against pending transitions. On `SUCCESS`, `KVCacheManager.commit_request_kv_transition()` revalidates source blocks, substitutes `null_block` placeholders in the logical block table, and releases the original physical HOT blocks back to `BlockPool.free_blocks()` for immediate reuse.
7. **Mixed Resumption**:
   When a conversational session resumes, newly generated tokens allocate fresh HOT blocks. During attention, `GPUModelRunner.prepare_attn()` translates logical WARM residency into `hkv_warm_slot_table`. The Triton unified attention kernel performs disjoint memory loads, reading INT8 KV data + dynamic scales for WARM blocks and high-precision KV data for HOT blocks in a single pass.
8. **Lifecycle & Cleanup**:
   When requests finish, abort, or are preempted, `GPUModelRunner.finish_requests()` calls `hkv_warm_migration_manager.release_request(req_id)`, freeing WARM slots back to the allocator heap.

---

## Main Components

### Scheduler (`vllm/v1/core/sched/scheduler.py`)
- **Responsibilities**: Request lifecycle management, inactivity monitoring, transition dispatch, transition result validation, and deferred lifecycle handling.
- **HOT → WARM Role**: Scans `WAITING_FOR_STREAMING_REQ` sessions in `_classify_idle_kv_sessions()`. Generates `KVCacheStateTransition` descriptors. Validates incoming worker acknowledgments in `_validate_kv_cache_transition_results()`.
- **In-Flight Lifecycle Protection**:
  - If a streaming input arrives while a transition is pending, `_try_promote_blocked_waiting_request()` returns `False`, leaving the session in `WAITING_FOR_STREAMING_REQ` and buffering tokens in `streaming_queue` without assertion failures.
  - If a client finish or abort arrives while a transition is pending, the request is recorded in `_pending_finish_requests` and its removal is deferred until the worker ACK completes.

### KV Cache Manager (`vllm/v1/core/kv_cache_manager.py`)
- **Responsibilities**: Logical-to-physical block mapping, block lifecycle accounting, and physical block allocation/reclamation.
- **HOT → WARM Role**:
  - `plan_request_kv_state()`: Non-mutating pass selecting completed prefix blocks while keeping tail blocks and shared blocks (`ref_cnt > 1`) HOT.
  - `commit_request_kv_transition()`: Revalidates physical block IDs and private ownership (`ref_cnt == 1`), replaces logical HOT entries with `null_block`, and frees original physical HOT blocks back to the free pool.

### GPU Worker & Model Runner (`vllm/v1/worker/gpu_worker.py`, `vllm/v1/worker/gpu/model_runner.py`)
- **Responsibilities**: GPU execution loop, HOT/WARM tensor allocation, attention metadata preparation, and transition execution.
- **HOT → WARM Role**:
  - `gpu_worker.py`: Validates runtime configuration and dispatches transitions to `GPUModelRunner`.
  - `model_runner.py`: Owns `hkv_hot_kv_caches`, `hkv_warm_kv_caches`, `hkv_warm_slot_table`, and `hkv_warm_migration_manager`. Preserves transition result ordering via `indexed_results`. Synchronizes CUDA stream before emitting `SUCCESS`. Rebuilds `hkv_warm_slot_table` in `prepare_attn()`. Cleans up WARM slots in `finish_requests()`.

### WARM Migration Manager (`vllm/v1/worker/gpu/hkv_migration.py`)
- **Responsibilities**: Deterministic WARM slot management and transactional migration coordination.
- **HOT → WARM Role**:
  - `HKVWarmSlotAllocator`: Manages slot pool `[0, capacity - 1]` with a free-slot min-heap (smallest available slot allocation), bidirectional mappings, and two-phase reservation (`reserve_many`, `commit`, `rollback`).
  - `HKVWarmMigrationManager`: Manages logical residency `(request_id, cache_group_index, logical_block_index) -> warm_slot_id`. Coordinates validation, slot reservation, kernel invocation, and error rollback (`HKVWarmCapacityError`, `HKVWarmStaleValidationError`). Note: legacy physical `hot_to_warm_maps` has been completely removed from constructor and migration flow.

### Quantization Routines (`vllm/v1/worker/gpu/attn_utils.py`)
- **Responsibilities**: Physical WARM GPU cache allocation and INT8 multi-block quantization.
- **HOT → WARM Role**:
  - `initialize_hkv_warm_kv_caches()`: Allocates dedicated `torch.int8` GPU tensors with padded head dimensions for inline FP32 scale storage (`int8_per_token_head`).
  - `quantize_hkv_blocks_to_warm()`: Gathers source HOT blocks and launches `triton_reshape_and_cache_flash_per_token_head_quant`, quantizing high-precision tensors and writing INT8 data and FP32 scales into destination WARM slots.

### Mixed HOT/WARM Attention (`vllm/v1/attention/backends/triton_attn.py`, `vllm/v1/attention/ops/triton_unified_attention.py`)
- **Responsibilities**: Attention forward execution over heterogeneous storage tiers.
- **HOT → WARM Role**:
  - `triton_attn.py`: Binds `hkv_warm_kv_cache`, `hkv_warm_slot_table`, and scale views, passing them to `unified_attention()`. Gated by `HKV_DEBUG_MIXED_READ`.
  - `triton_unified_attention.py`: Checks `hkv_warm_slot_table[req, logical_block]`. Loads WARM INT8 data via `warm_slot` for WARM lanes; loads HOT high-precision data via `block_tables` for HOT lanes. Fuses K-scale into $Q \cdot K$ score calculation and V-scale into softmax probabilities $(P \cdot v\_scale) \cdot V$.

---

## File-by-File Reference

### `external/vllm/vllm/config/scheduler.py`

**Original role in vLLM**
`SchedulerConfig` is the configuration dataclass parameterizing the runtime scheduler (token budgets, chunking, scheduling intervals).

**Role in Hierarchical KV Cache**
Stores and validates inactivity thresholds and enforces required runtime environment settings.

**Changes introduced**
- Added configuration fields:
  - `kv_cache_hot_idle_threshold_seconds: float | None = None`
  - `kv_cache_cold_idle_threshold_seconds: float | None = None`
- Validation logic in `__post_init__()`:
  - Both thresholds must be set together or both omitted;
  - Both must be finite, non-negative numbers;
  - $0 \le T_{\text{hot}} < T_{\text{cold}}$ (COLD must be strictly greater than HOT);
  - **Strict Runtime Environment Checks** (enforced when thresholds are set):
    - `VLLM_USE_V2_MODEL_RUNNER == "1"`
    - `HKV_ENABLE_PHYSICAL_TIERS == "1"`
    - `HKV_ENABLE_MULTI_BLOCK_WARM_MIGRATION == "1"`
    - `HKV_DEBUG_MIXED_READ == "1"`
    - `HKV_WARM_POOL_BLOCKS` must be an integer greater than zero.

---

### `external/vllm/vllm/engine/arg_utils.py`

**Original role in vLLM**
`EngineArgs` parses command-line arguments and configuration options, converting them into internal config objects.

**Role in Hierarchical KV Cache**
Exposes CLI flags and engine arguments for inactivity thresholds and passes them to `SchedulerConfig`.

**Changes introduced**
- Added `kv_cache_hot_idle_threshold_seconds` and `kv_cache_cold_idle_threshold_seconds` to `EngineArgs`.
- Added CLI argument parser options in the scheduler argument group:
  - `--kv-cache-hot-idle-threshold-seconds`
  - `--kv-cache-cold-idle-threshold-seconds`
- Updated `create_scheduler_config()` to forward these options.

---

### `external/vllm/vllm/v1/request.py`

**Original role in vLLM**
`Request` represents a single request/session, tracking tokens, arrival time, execution status, and streaming metadata.

**Role in Hierarchical KV Cache**
Maintains per-request activity timestamps and logical KV hierarchy state.

**Changes introduced**
- Added `last_activity_time: float` (initialized to `arrival_time`).
- Added `kv_cache_state: KVBlockState` (initialized to `KVBlockState.HOT`).
- Added `mark_activity(activity_time: float | None = None)`.
- Added `get_idle_time(current_time: float | None = None) -> float`:
  $$\max(0.0, \text{current\_time} - \text{last\_activity\_time})$$

---

### `external/vllm/vllm/v1/kv_cache_state.py`

**Original role in vLLM**
New module added by this project defining hierarchy data structures and classification helpers.

**Changes introduced**
- `KVBlockState`: Enum (`HOT`, `WARM`, `COLD`).
- `KVCacheTransitionStatus`: Enum (`SUCCESS`, `RETRYABLE_CAPACITY`, `STALE_VALIDATION`, `FATAL`).
- `KVCacheBlockTransition`: Dataclass storing `logical_block_index` and `source_hot_block_id`.
- `KVCacheStateTransition`: Scheduler-to-worker command with unique `transition_id`, `request_id`, state bounds, and `changed_blocks`.
- `KVCacheTransitionResult`: Worker-to-scheduler acknowledgment with execution status and error string.
- `normalize_transition_blocks()`: Canonicalizes block representations into tuples of `(cache_group_index, logical_block_index, source_hot_block_id)`.
- `classify_request_kv_state(idle_seconds, hot_threshold, cold_threshold) -> KVBlockState`:
  - $T_{\text{idle}} < T_{\text{hot}} \implies \text{HOT}$
  - $T_{\text{hot}} \le T_{\text{idle}} < T_{\text{cold}} \implies \text{WARM}$
  - $T_{\text{idle}} \ge T_{\text{cold}} \implies \text{COLD}$

---

### `external/vllm/vllm/v1/core/sched/scheduler.py`

**Original role in vLLM**
Central control-plane scheduler managing queues, scheduling steps, request preemption, and worker coordination.

**Role in Hierarchical KV Cache**
Orchestrates idle session classification, transition creation, acknowledgment validation, and lifecycle coordination.

**Changes introduced**
- Reads HOT/COLD thresholds from `SchedulerConfig`.
- Tracks pending state with `_next_kv_transition_id`, `_pending_kv_transitions`, and `_pending_finish_requests`.
- `_classify_idle_kv_sessions()`:
  - Scans resumable sessions in `WAITING_FOR_STREAMING_REQ`.
  - Skips sessions with in-flight transitions.
  - Computes idle time and applies `classify_request_kv_state()`.
  - **Cold tier clamp**: Direct `HOT -> COLD` classifications are clamped to `HOT -> WARM`; `WARM -> COLD` transitions are not scheduled.
  - Queries `KVCacheManager.plan_request_kv_state()` and builds `KVCacheStateTransition` objects.
- `_validate_kv_cache_transition_results()`:
  - Validates returned results against pending transition signatures.
  - On `SUCCESS`: calls `KVCacheManager.commit_request_kv_transition()`, updates `request.kv_cache_state`, and clears pending records.
  - On `RETRYABLE_CAPACITY` / `STALE_VALIDATION`: clears pending state without committing, allowing future retry.
- **In-Flight Lifecycle Handling**:
  - `_try_promote_blocked_waiting_request()`: Returns `False` when `request_id in self._pending_kv_transitions`, preventing race conditions with streaming input arriving during migration.
  - `finish_requests()`: Defers client finish/abort arriving during pending transitions via `_pending_finish_requests`, executing cleanup only after worker ACK.

---

### `external/vllm/vllm/v1/core/kv_cache_manager.py`

**Original role in vLLM**
Manages block allocation, freeing, prefix caching, and logical-to-physical block tables.

**Role in Hierarchical KV Cache**
Plans eligible blocks for migration and commits physical HOT block reclamation upon worker success.

**Changes introduced**
- `plan_request_kv_state()`:
  - Non-mutating planning pass returning `KVCacheBlockTransition` entries.
  - Skips null blocks, shared blocks (`ref_cnt > 1`), and incomplete tail blocks (`num_computed_tokens // block_size`).
- `commit_request_kv_transition()`:
  - Revalidates cache group, logical index, matching `source_hot_block_id`, and private ownership (`ref_cnt == 1`).
  - Replaces logical HOT block table entry with `null_block`.
  - Returns physical HOT blocks to `BlockPool.free_blocks()` for immediate reuse.

---

### `external/vllm/vllm/v1/core/single_type_kv_cache_manager.py`

**Original role in vLLM**
Abstract base class for single-type KV-cache managers handling prefix hashing and block allocation.

**Changes introduced**
- Removed legacy window-based static demotion heuristics (`HKV_HOT_WINDOW_BLOCKS`, `HKV_WARM_WINDOW_BLOCKS`) in favor of the time-aware, scheduler-driven architecture.

---

### `external/vllm/vllm/v1/core/block_pool.py`

**Original role in vLLM**
Manages physical block allocations, reference counts, and the free-block pool.

**Role in Hierarchical KV Cache**
Ensures safe handling of `null_block` placeholders introduced during HOT block reclamation.

**Changes introduced**
- In `free_blocks()`: immediately skips `null_block` placeholders (`if block.is_null: continue`).
- Prevents invalid ref-count decrements or re-insertion of shared null blocks into free queues during request teardown.

---

### `external/vllm/vllm/v1/core/kv_cache_utils.py`

**Original role in vLLM**
Provides KV-cache utility structures and metadata.

**Changes introduced**
- Imports shared `KVBlockState` enum from `vllm.v1.kv_cache_state`.
- `KVCacheBlock.hierarchy_state` initialized to `KVBlockState.HOT`.
- `KVCacheBlock.reset_hierarchy_state()` resets reused physical blocks to `HOT` to prevent stale metadata carryover.

---

### `external/vllm/vllm/v1/core/sched/output.py` & `external/vllm/vllm/v1/outputs.py`

**Original role in vLLM**
Control-plane transport structures for scheduler outputs and worker outputs.

**Changes introduced**
- `SchedulerOutput`: Added `kv_cache_state_transitions: list[KVCacheStateTransition]`.
- `ModelRunnerOutput`: Added `kv_cache_transition_results: list[KVCacheTransitionResult]`.

---

### `external/vllm/vllm/v1/worker/gpu_worker.py`

**Original role in vLLM**
GPU worker wrapper receiving `SchedulerOutput` and driving the model runner.

**Changes introduced**
- `_handle_kv_cache_state_transitions()`:
  - Validates requested transitions.
  - Checks feature flag `is_hkv_multi_block_warm_migration_enabled`.
  - Forwards transitions to `GPUModelRunner.handle_kv_cache_state_transitions()`.

---

### `external/vllm/vllm/v1/worker/gpu/model_runner.py`

**Original role in vLLM**
Coordinates GPU forward execution, KV-cache allocation, and model runner output generation.

**Role in Hierarchical KV Cache**
Allocates HOT/WARM GPU cache tensors, coordinates migration, preserves result ordering, fences CUDA streams, and builds attention metadata.

**Changes introduced**
- Owns `hkv_hot_kv_caches`, `hkv_warm_kv_caches`, `hkv_warm_slot_table`, and `hkv_warm_migration_manager`.
- `handle_kv_cache_state_transitions()`:
  - Preserves exact transition result ordering via `indexed_results`.
  - Calls `HKVWarmMigrationManager.migrate()`.
  - Synchronizes current CUDA stream (`torch.cuda.current_stream(device).synchronize()`) before emitting `SUCCESS` when quantization work is enqueued.
- `prepare_attn()`: Translates logical WARM residency into `hkv_warm_slot_table` for mixed attention forward passes.
- `finish_requests()`: Calls `hkv_warm_migration_manager.release_request(req_id)` to release WARM slots upon request completion or preemption.

---

### `external/vllm/vllm/v1/worker/gpu/hkv_migration.py`

**Role in Hierarchical KV Cache**
Project-specific module managing logical WARM residency and deterministic slot allocation.

**Key Components**
- `HKVWarmLogicalKey`: `tuple[str, int, int]` representing `(request_id, cache_group_index, logical_block_index)`.
- `HKVWarmSlotAllocator`:
  - Free slots tracked in a `heapq` for deterministic, smallest-slot allocation.
  - Bidirectional lookup (`_key_to_slot`, `_slot_to_key`).
  - Two-phase transactional reservation (`reserve_many`, `commit`, `rollback`).
- `HKVWarmMigrationManager`:
  - Owns allocator, cache references, `warm_residency`, and `warm_residency_revision`.
  - Removed legacy `hot_to_warm_maps` from constructor and migration flow.
  - `migrate()`: Validates current block table, checks existing residency, reserves WARM slots, invokes `quantize_hkv_blocks_to_warm()`, commits reservation, and updates residency.
  - `release_request()`: Releases all WARM slots for a request and bumps revision.

---

### `external/vllm/vllm/v1/worker/gpu/attn_utils.py`

**Role in Hierarchical KV Cache**
Handles physical WARM cache tensor creation and Triton multi-block quantization kernel dispatch.

**Key Functions**
- `initialize_hkv_warm_kv_caches()`: Allocates `torch.int8` tensors with padded head dimensions for inline FP32 scale storage.
- `_get_hkv_per_token_head_scale_views()`: Creates strided FP32 scale tensor views over padded cache storage.
- `quantize_hkv_blocks_to_warm()`: Multi-block quantization function invoking `triton_reshape_and_cache_flash_per_token_head_quant`. Note: `hot_to_warm_maps` is optional (`None` by default) for legacy/debug callers.

---

### `external/vllm/vllm/v1/attention/backends/triton_attn.py` & `external/vllm/vllm/v1/attention/ops/triton_unified_attention.py`

**Role in Hierarchical KV Cache**
Performs single-pass mixed-precision attention reading from both HOT FP16 and WARM INT8 tiers.

**Key Mechanics**
- `triton_attn.py`: Passes `hkv_warm_k`, `hkv_warm_v`, `hkv_warm_slot_table`, and scale views into `unified_attention()`. Gated by `HKV_DEBUG_MIXED_READ`.
- `triton_unified_attention.py`:
  - Derives logical block index: `seq_offset // BLOCK_SIZE`.
  - Reads `warm_slot = hkv_warm_slot_table[req, logical_block]`.
  - Disjoint loads: WARM lanes load from INT8 WARM tensors; HOT lanes load from high-precision HOT cache via standard block tables.
  - Fused dequantization: K-scale applied directly in $Q \cdot K$ score scaling; V-scale applied directly to softmax probabilities $(P \cdot v\_scale) \cdot V$.
  - Compile-time assertion preventing mixed reads with tensor descriptors (`USE_TD`).

---

## Experiments, Trace Parsing & Replay Tools

The repository contains tools for validating time-aware hierarchical KV caching on real-world conversational workloads.

### `experiments/scripts/qwen_bailian_trace.py`
- **Purpose**: Parser and preprocessor for anonymized Qwen-Bailian multi-turn conversational traces.
- **Features**:
  - Preserves session structure, timestamps, turn numbers, input/output token lengths, and prefix block hashes.
  - Deterministically reconstructs token IDs from block hashes without exposing or requiring original conversation text.
  - Zero external vLLM/GPU dependencies, allowing rapid unit testing on CPU.
- **Key Data Structures**:
  - `BailianRecord`: Represents a single raw trace record.
  - `ReplayTurn`: Structured turn descriptor containing session ID, root chat ID, turn index, scheduled send timestamp, and delta token IDs.

### `experiments/scripts/run_qwen_bailian_replay.py`
- **Purpose**: Multi-session conversational replay runner with time-aware HKV simulation and validation.
- **Features**:
  - Evaluates All-HOT FP16 baseline vs. Time-Aware Hierarchical HOT/WARM KV Cache.
  - Validates exact output token agreement between hierarchical KV cache and all-HOT baseline.
  - Tracks peak WARM blocks, demoted sessions, and verifies zero allocator leaks upon completion.
  - `HKVReplayWorkerExtension`: Inspects worker-side WARM residency and allocator consistency.

### Evaluation Results & Baseline Notebooks
- `experiments/results/qwen_bailian_replay/`:
  - `qwen_traceA_v0_all_hot.json`: Baseline execution record with all sessions remaining in HOT tier.
  - `qwen_traceA_v0_mixed.json`: Mixed HOT/WARM execution record demonstrating successful demotions and exact output match.
- `experiments/notebooks/`:
  - `00_original_vllm_performance_baseline.ipynb`: Baseline throughput and latency benchmarks.
  - `01_inspect_qwen_kv_cache.ipynb`: KV cache inspection and block allocation experiments.
  - `02_longbench_qasper_baseline.ipynb`: LongBench/Qasper long-context accuracy evaluation.
  - `03_math500_fp16_int8_analysis.ipynb`: MATH-500 FP16 vs INT8 quantization accuracy analysis.

---

## Verification, Testing & Correctness Milestone

The implementation has achieved a verified end-to-end correctness milestone across unit, integration, and trace replay test suites:

### Test Suites
- **Scheduler & Streaming Test Suite** (`external/vllm/tests/v1/streaming_input/test_scheduler_streaming.py`):
  - Validates idle session classification, streaming chunk scheduling, queue promotion, and lifecycle guards.
- **Worker KV State Test Suite** (`external/vllm/tests/v1/worker/test_gpu_worker_kv_state.py`):
  - Validates worker-side transition handling, ordered results, CUDA stream synchronization, in-flight finish/abort deferrals, and mixed attention execution.
- **Trace & Replay Test Suite** (`experiments/tests/test_qwen_bailian_trace.py`, `experiments/tests/test_run_qwen_bailian_replay.py`):
  - Validates trace parsing, session grouping, replay plan generation, and replay worker extensions.
- **Core Hierarchy & Configuration Test Suites** (`external/vllm/tests/v1/core/test_kv_cache_hierarchy.py`, `external/vllm/tests/v1/worker/test_hkv_migration.py`, `external/vllm/tests/v1/test_kv_cache_state.py`, `external/vllm/tests/test_config.py`):
  - Validates threshold configuration constraints, required environment variables, WARM slot allocator invariants, transactional rollback, and block pool null placeholder handling.

### Correctness Milestone Summary
- **Conversational Replay**:
  - Replayed 9 real-world sessions across 25 conversational turns.
  - Observed a peak of 73 WARM blocks across 6 demoted sessions.
  - Verified allocator consistency with zero leaks and clean post-run teardown.
  - Achieved exact output token agreement between hierarchical KV cache and the all-HOT baseline.

> [!IMPORTANT]
> **Correctness Milestone vs. Performance Claims**: The replay run proves architectural and numerical correctness. Memory capacity savings under concurrency limits, TTFT impact, migration latency overhead, and kernel execution efficiency remain to be evaluated at scale.

---

## Known Limitations & Deferred Capabilities

1. **Physical COLD Tier Deferred**:
   CPU offloading, host memory pooling, asynchronous prefetching, and CPU-to-GPU restoration paths are intentionally deferred. Logical classification recognizes COLD, but the scheduler clamps direct $\text{HOT} \rightarrow \text{COLD}$ to $\text{HOT} \rightarrow \text{WARM}$ and maintains WARM sessions in WARM.
2. **Conservative Shared Prefix Handling**:
   Shared prefix-cache blocks with `ref_cnt > 1` remain HOT and are excluded from demotion to prevent invalidating concurrent sessions.
3. **Tail Block Retention**:
   Incomplete tail blocks (`num_computed_tokens % block_size != 0`) remain HOT and are not demoted until filled.
4. **Parallelism Constraints**:
   Current implementation supports single-worker execution (`TP=1`, `PP=1`, `DP=1`).
5. **Cache Layout Constraints**:
   Physical migration currently requires a single KV-cache group and `blocks_per_kv_block == 1`.
6. **Synchronous Stream Execution**:
   Migration executes on the current CUDA stream and synchronizes before emitting a `SUCCESS` result (no asynchronous compute/migration overlap).
7. **Triton Attention Tensor Descriptors (`USE_TD`)**:
   Mixed HKV reads are incompatible with the experimental Triton tensor-descriptor KV load path in the current kernel implementation.
