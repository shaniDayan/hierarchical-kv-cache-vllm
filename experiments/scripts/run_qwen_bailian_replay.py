"""Replay Qwen-Bailian chat sessions through vLLM streaming input."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.scripts.qwen_bailian_trace import (
    BailianRecord,
    ReplayTurn,
    build_replay_plan,
    group_linear_sessions,
    load_bailian_records,
)

RESULT_SCHEMA_VERSION = "3.0"
ATTENTION_BACKEND = "TRITON_ATTN"
MODEL_DTYPE = "float16"
BLOCK_SIZE = 16
DEFAULT_MAX_NUM_SEQS = 128
KV_MEMORY_FORMULA_VERSION = "qwen3-0.6b-hkv-persistent-v1"
SUPPORTED_BUDGET_MODEL = "Qwen/Qwen3-0.6B"
QWEN_ATTENTION_LAYERS = 28
QWEN_KV_HEADS = 8
QWEN_HEAD_DIM = 128
HOT_DTYPE_BYTES = 2
WARM_DTYPE_BYTES = 1
WARM_INLINE_SCALE_BYTES = 4
INT32_BYTES = 4
BLOCK_TABLE_ALIGNMENT_TOKENS = 128
TOTAL_KV_BUDGET_ENV = "HKV_TOTAL_KV_BUDGET_BYTES"
TOPOLOGY_ASSUMPTIONS = {
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "data_parallel_size": 1,
    "kv_cache_groups": 1,
    "blocks_per_kv_block": 1,
}


def hot_bytes_per_block() -> int:
    return (
        QWEN_ATTENTION_LAYERS
        * 2
        * BLOCK_SIZE
        * QWEN_KV_HEADS
        * QWEN_HEAD_DIM
        * HOT_DTYPE_BYTES
    )


def warm_bytes_per_slot() -> int:
    return (
        QWEN_ATTENTION_LAYERS
        * 2
        * BLOCK_SIZE
        * QWEN_KV_HEADS
        * (QWEN_HEAD_DIM + WARM_INLINE_SCALE_BYTES)
        * WARM_DTYPE_BYTES
    )


def warm_pool_storage_bytes(warm_pool_blocks: int) -> int:
    return warm_pool_blocks * warm_bytes_per_slot()


def warm_slot_table_storage_bytes(
    max_model_len: int,
    max_num_seqs: int,
) -> int:
    logical_blocks = (max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_alignment = BLOCK_TABLE_ALIGNMENT_TOKENS // BLOCK_SIZE
    aligned_blocks = (
        (logical_blocks + block_alignment - 1) // block_alignment
    ) * block_alignment
    return max_num_seqs * aligned_blocks * INT32_BYTES


def _validate_explicit_budget_layout(args: argparse.Namespace) -> None:
    if args.model != SUPPORTED_BUDGET_MODEL:
        raise ValueError(
            "explicit persistent KV budgeting supports only "
            f"{SUPPORTED_BUDGET_MODEL}; got {args.model}"
        )
    if (
        MODEL_DTYPE != "float16"
        or BLOCK_SIZE != 16
        or ATTENTION_BACKEND != "TRITON_ATTN"
        or TOPOLOGY_ASSUMPTIONS
        != {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "kv_cache_groups": 1,
            "blocks_per_kv_block": 1,
        }
    ):
        raise ValueError(
            "explicit persistent KV budgeting requires float16 HOT, block "
            "size 16, Triton attention, TP=PP=DP=1, one KV-cache group, "
            "and blocks_per_kv_block=1"
        )


def derive_persistent_kv_budget(args: argparse.Namespace) -> dict[str, Any]:
    total_budget = args.total_kv_budget_bytes
    layout = {
        "formula_version": KV_MEMORY_FORMULA_VERSION,
        "model": SUPPORTED_BUDGET_MODEL,
        "attention_layers": QWEN_ATTENTION_LAYERS,
        "kv_heads": QWEN_KV_HEADS,
        "head_dimension": QWEN_HEAD_DIM,
        "hot_dtype": MODEL_DTYPE,
        "hot_dtype_bytes": HOT_DTYPE_BYTES,
        "warm_dtype": "int8",
        "warm_dtype_bytes": WARM_DTYPE_BYTES,
        "warm_inline_scale_bytes_per_token_head": WARM_INLINE_SCALE_BYTES,
        "block_size": BLOCK_SIZE,
        "block_table_alignment_tokens": BLOCK_TABLE_ALIGNMENT_TOKENS,
        "attention_backend": ATTENTION_BACKEND,
        "enforce_eager": True,
        "topology": TOPOLOGY_ASSUMPTIONS,
    }
    hot_block_bytes = hot_bytes_per_block()
    warm_slot_bytes = warm_bytes_per_slot()
    common = {
        "total_kv_budget_bytes": total_budget,
        "max_num_seqs": args.max_num_seqs,
        "hot_bytes_per_block": hot_block_bytes,
        "warm_bytes_per_slot": warm_slot_bytes,
        "layout": layout,
    }
    if total_budget is None:
        return {
            **common,
            "derived_num_gpu_blocks": None,
            "derived_hot_kv_budget_bytes": None,
            "derived_warm_kv_storage_bytes": None,
            "derived_hot_to_warm_map_storage_bytes": None,
            "derived_warm_slot_table_storage_bytes": None,
            "derived_actual_persistent_kv_bytes": None,
            "derived_budget_slack_bytes": None,
            "block_rounding_tolerance_bytes": None,
        }

    _validate_explicit_budget_layout(args)
    if total_budget <= 0:
        raise ValueError("--total-kv-budget-bytes must be positive")
    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")
    if args.max_num_seqs <= 0:
        raise ValueError("--max-num-seqs must be positive")

    mixed = args.kv_mode == "mixed"
    if mixed and (
        args.warm_pool_blocks is None or args.warm_pool_blocks <= 0
    ):
        raise ValueError(
            "mixed explicit budgeting requires positive --warm-pool-blocks"
        )
    warm_bytes = (
        warm_pool_storage_bytes(args.warm_pool_blocks) if mixed else 0
    )
    slot_table_bytes = (
        warm_slot_table_storage_bytes(args.max_model_len, args.max_num_seqs)
        if mixed
        else 0
    )
    fixed_mixed_bytes = warm_bytes + slot_table_bytes
    if mixed and fixed_mixed_bytes >= total_budget:
        raise ValueError(
            "mixed WARM pool and slot table require "
            f"{fixed_mixed_bytes} bytes, which must be less than the "
            f"configured total KV budget {total_budget}"
        )

    map_bytes_per_hot_block = (
        QWEN_ATTENTION_LAYERS * INT32_BYTES if mixed else 0
    )
    bytes_per_budgeted_block = hot_block_bytes + map_bytes_per_hot_block
    num_gpu_blocks = (
        total_budget - fixed_mixed_bytes
    ) // bytes_per_budgeted_block
    if num_gpu_blocks <= 0:
        raise ValueError(
            "configured total KV budget derives no usable HOT blocks: "
            f"total={total_budget}, fixed_mixed={fixed_mixed_bytes}, "
            f"bytes_per_budgeted_block={bytes_per_budgeted_block}"
        )

    hot_budget_bytes = num_gpu_blocks * hot_block_bytes
    map_bytes = num_gpu_blocks * map_bytes_per_hot_block
    actual_total = hot_budget_bytes + map_bytes + fixed_mixed_bytes
    return {
        **common,
        "derived_num_gpu_blocks": num_gpu_blocks,
        "derived_hot_kv_budget_bytes": hot_budget_bytes,
        "derived_warm_kv_storage_bytes": warm_bytes,
        "derived_hot_to_warm_map_storage_bytes": map_bytes,
        "derived_warm_slot_table_storage_bytes": slot_table_bytes,
        "derived_actual_persistent_kv_bytes": actual_total,
        "derived_budget_slack_bytes": total_budget - actual_total,
        "block_rounding_tolerance_bytes": bytes_per_budgeted_block - 1,
    }


def unique_storage_bytes(value: Any) -> int:
    """Count unique underlying tensor storage, including nested containers."""
    import torch

    seen: set[tuple[str, int, int]] = set()

    def visit(node: Any) -> int:
        if node is None:
            return 0
        if isinstance(node, torch.Tensor):
            storage = node.untyped_storage()
            key = (
                str(node.device),
                storage.data_ptr(),
                storage.nbytes(),
            )
            if key in seen:
                return 0
            seen.add(key)
            return int(storage.nbytes())
        if isinstance(node, dict):
            return sum(visit(item) for item in node.values())
        if isinstance(node, (list, tuple, set)):
            return sum(visit(item) for item in node)
        return 0

    return visit(value)


def hot_kv_storage_source(model_runner: Any) -> Any:
    """Prefer HKV HOT tensors; otherwise use ordinary runner KV caches."""
    hot_caches = getattr(model_runner, "hkv_hot_kv_caches", None)
    if unique_storage_bytes(hot_caches) > 0:
        return hot_caches
    return getattr(model_runner, "kv_caches", None)


class HKVReplayWorkerExtension:
    """Expose aggregate WARM state without modifying the worker."""

    def inspect_hkv_replay(self) -> dict[str, Any]:
        import torch

        manager = getattr(
            self.model_runner, "hkv_warm_migration_manager", None
        )
        residency = manager.warm_residency if manager is not None else {}
        allocator = manager.allocator if manager is not None else None
        hot_bytes = unique_storage_bytes(hot_kv_storage_source(self.model_runner))
        warm_bytes = unique_storage_bytes(
            getattr(self.model_runner, "hkv_warm_kv_caches", None)
        )
        map_bytes = unique_storage_bytes(
            getattr(self.model_runner, "hkv_hot_to_warm_maps", None)
        )
        slot_table_bytes = unique_storage_bytes(
            getattr(self.model_runner, "hkv_warm_slot_table", None)
        )
        kv_cache_config = getattr(self.model_runner, "kv_cache_config", None)
        num_gpu_blocks = getattr(kv_cache_config, "num_blocks", 0)
        cache_config = getattr(self.model_runner, "cache_config", None)
        derived_hot_budget = getattr(
            cache_config, "kv_cache_memory_bytes", None
        )
        configured_total_str = os.getenv(TOTAL_KV_BUDGET_ENV)
        configured_total = (
            int(configured_total_str) if configured_total_str else None
        )
        actual_persistent = (
            hot_bytes + warm_bytes + map_bytes + slot_table_bytes
        )
        return {
            "warm_blocks": len(residency),
            "warm_requests": len({key[0] for key in residency}),
            "owned_warm_slots": (
                allocator.num_owned_slots if allocator is not None else 0
            ),
            "allocator_consistent": (
                allocator is None or allocator.num_owned_slots == len(residency)
            ),
            "max_gpu_allocated_bytes": (
                torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else 0
            ),
            "max_gpu_reserved_bytes": (
                torch.cuda.max_memory_reserved()
                if torch.cuda.is_available()
                else 0
            ),
            "num_gpu_blocks": num_gpu_blocks,
            "hot_kv_storage_bytes": hot_bytes,
            "warm_kv_storage_bytes": warm_bytes,
            "hot_to_warm_map_storage_bytes": map_bytes,
            "warm_slot_table_storage_bytes": slot_table_bytes,
            "actual_persistent_kv_bytes": actual_persistent,
            "configured_total_kv_budget_bytes": configured_total,
            "derived_hot_kv_budget_bytes": derived_hot_budget,
            "budget_slack_bytes": (
                configured_total - actual_persistent
                if configured_total is not None
                else None
            ),
        }


@dataclass(slots=True)
class TurnResult:
    chat_id: int
    turn: int
    scheduled_send_seconds: float
    actual_send_seconds: float
    send_lateness_seconds: float
    first_output_seconds: float
    finished_seconds: float
    ttft_seconds: float
    latency_seconds: float
    generated_tokens: int
    generated_token_ids: list[int]
    configured_max_tokens_per_turn: int
    effective_max_tokens: int
    is_resume: bool
    finish_reason: str | None = None


@dataclass(slots=True)
class SessionResult:
    root_chat_id: int
    session_id: str
    turns: int
    final_input_tokens: int
    trace_output_tokens: int
    scheduled_first_seconds: float
    scheduled_last_seconds: float
    generated_tokens: int = 0
    completed_turns: int = 0
    first_output_seconds: float | None = None
    finished_seconds: float | None = None
    max_send_lateness_seconds: float = 0.0
    generated_token_sha256: str = ""
    generated_token_sample: list[int] = field(default_factory=list)
    turn_results: list[TurnResult] = field(default_factory=list)


def select_sessions(
    records: list[BailianRecord],
    *,
    request_type: str | None,
    min_turns: int,
    max_input_length: int | None,
    max_sessions: int | None,
) -> list[BailianRecord]:
    """Filter whole sessions so no parent chain is accidentally broken."""
    eligible: list[tuple[int, list[BailianRecord]]] = []
    for root_id, session in group_linear_sessions(records).items():
        if len(session) < min_turns:
            continue
        if request_type and any(
            record.request_type != request_type for record in session
        ):
            continue
        if max_input_length and any(
            record.input_length > max_input_length for record in session
        ):
            continue
        eligible.append((root_id, session))

    eligible.sort(key=lambda item: (item[1][0].timestamp, item[0]))
    if max_sessions is not None:
        eligible = eligible[:max_sessions]
    if not eligible:
        raise ValueError("no Bailian sessions match the requested filters")
    return [record for _, session in eligible for record in session]


def tokenizer_vocab_size(tokenizer: Any) -> int:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        vocab_size = len(tokenizer)
    if vocab_size <= 0:
        raise ValueError("tokenizer vocabulary is empty")
    return vocab_size


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trace_identifier(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as trace_file:
        for chunk in iter(lambda: trace_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "sha256": digest.hexdigest()}


def get_git_metadata() -> dict[str, str | bool | None]:
    project_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def generation_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "seed": args.seed,
        "ignore_eos": True,
        "output_kind": "delta",
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "generation_policy": {
            "name": "final_turn_only_multi_token",
            "non_final_turn_max_tokens": 1,
            "final_turn_max_tokens": args.max_tokens_per_turn,
            "description": (
                "Non-final synthetic trace turns generate one discardable token; "
                "only the final turn uses the configured maximum."
            ),
        },
    }


def selection_metadata(
    selected: list[BailianRecord],
) -> dict[str, Any]:
    session_ids = sorted({record.chat_id for record in selected if record.turn == 1})
    turn_ids = [
        {
            "root_chat_id": record.chat_id if record.turn == 1 else None,
            "chat_id": record.chat_id,
            "parent_chat_id": record.parent_chat_id,
            "turn": record.turn,
        }
        for record in selected
    ]
    return {
        "selected_session_ids": session_ids,
        "selected_session_count": len(session_ids),
        "selected_request_count": len(selected),
        "selection_sha256": sha256_json(turn_ids),
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int((len(ordered) - 1) * fraction)
    return ordered[index]


UNAVAILABLE_RUNTIME_REASON = (
    "not exposed by the existing HKV worker inspector"
)


def unavailable_runtime_field(
    reason: str = UNAVAILABLE_RUNTIME_REASON,
) -> dict[str, Any]:
    return {"available": False, "reason": reason}


def create_engine(args: argparse.Namespace) -> Any:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    return AsyncLLM.from_engine_args(
        AsyncEngineArgs(**build_engine_args_kwargs(args))
    )


def collect_session_progress(
    sessions: list[SessionResult],
    *,
    selected_session_count: int,
    selected_request_count: int,
) -> dict[str, int]:
    completed_session_count = sum(
        1 for session in sessions if session.completed_turns == session.turns
    )
    completed_request_count = sum(
        session.completed_turns for session in sessions
    )
    return {
        "selected_session_count": selected_session_count,
        "completed_session_count": completed_session_count,
        "incomplete_session_count": (
            selected_session_count - completed_session_count
        ),
        "selected_request_count": selected_request_count,
        "completed_request_count": completed_request_count,
        "incomplete_request_count": (
            selected_request_count - completed_request_count
        ),
        "total_generated_tokens": sum(
            session.generated_tokens for session in sessions
        ),
    }


def build_termination(
    *,
    timed_out: bool,
    configured_timeout_seconds: float,
    service_window_duration_seconds: float | None,
    last_output_received_seconds: float | None,
    progress: dict[str, int],
    runtime_at_timeout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seconds_since_last_completed_turn = None
    if (
        last_output_received_seconds is not None
        and service_window_duration_seconds is not None
    ):
        seconds_since_last_completed_turn = max(
            0.0,
            service_window_duration_seconds - last_output_received_seconds,
        )
    termination: dict[str, Any] = {
        "status": "timeout" if timed_out else "completed",
        "configured_timeout_seconds": configured_timeout_seconds,
        "timed_out": timed_out,
        "service_window_duration_seconds": service_window_duration_seconds,
        "seconds_since_last_completed_turn": seconds_since_last_completed_turn,
        **progress,
    }
    if timed_out:
        termination["runtime_at_timeout"] = runtime_at_timeout
    return termination


def build_timeout_runtime_snapshot(
    observation: dict[str, Any],
    inspect_state: dict[str, Any] | None,
    *,
    warm_pool_blocks: int,
) -> dict[str, Any]:
    state = inspect_state if inspect_state is not None else observation
    unavailable = [
        "hot_utilization",
        "hot_allocated_blocks",
        "hot_free_blocks",
        "pending_transitions",
        "successful_transitions",
        "retryable_capacity_transitions",
        "stale_transitions",
        "preemption_count",
        "num_running_reqs",
        "num_waiting_reqs",
    ]
    snapshot: dict[str, Any] = {
        field_name: unavailable_runtime_field() for field_name in unavailable
    }
    snapshot.update(
        {
            "num_gpu_blocks": state.get("num_gpu_blocks"),
            "hot_kv_storage_bytes": state.get("hot_kv_storage_bytes"),
            "warm_occupied_slots": state.get("warm_blocks"),
            "warm_occupied_requests": state.get("warm_requests"),
            "owned_warm_slots": state.get("owned_warm_slots"),
            "warm_capacity_blocks": warm_pool_blocks,
            "allocator_consistent": state.get("allocator_consistent"),
            "unavailable_fields": unavailable,
        }
    )
    return snapshot


def print_replay_progress(
    *,
    elapsed_seconds: float,
    progress: dict[str, int],
    hot_utilization: float | None,
) -> None:
    extra = ""
    if hot_utilization is not None:
        extra = f" hot_util={hot_utilization:.3f}"
    print(
        f"[replay] elapsed={elapsed_seconds:.1f}s "
        f"turns={progress['completed_request_count']}/"
        f"{progress['selected_request_count']} "
        f"sessions={progress['completed_session_count']}/"
        f"{progress['selected_session_count']} "
        f"tokens={progress['total_generated_tokens']}"
        f"{extra}",
        flush=True,
    )


def session_results_from_outcomes(
    outcomes: list[Any],
) -> tuple[list[SessionResult], list[float]]:
    sessions: list[SessionResult] = []
    lateness: list[float] = []
    for outcome in outcomes:
        if isinstance(outcome, tuple) and len(outcome) == 2:
            session, values = outcome
            if isinstance(session, SessionResult):
                sessions.append(session)
                lateness.extend(values)
    return sessions, lateness


def record_available_turn_results(
    result: SessionResult,
    turns: list[ReplayTurn],
    *,
    actual_send_times: list[float | None],
    turn_first_output_times: list[float | None],
    turn_finished_times: list[float | None],
    turn_generated_tokens: list[int],
    turn_generated_token_ids: list[list[int]],
    turn_finish_reasons: list[str | None],
    max_tokens_per_turn: int,
    effective_max_tokens: list[int],
) -> None:
    result.turn_results.clear()
    for i, turn in enumerate(turns):
        actual_send = actual_send_times[i]
        first_out = turn_first_output_times[i]
        finished = turn_finished_times[i]
        if actual_send is None or first_out is None or finished is None:
            continue
        send_lateness = max(0.0, actual_send - turn.send_at_seconds)
        result.turn_results.append(
            TurnResult(
                chat_id=turn.chat_id,
                turn=turn.turn,
                scheduled_send_seconds=turn.send_at_seconds,
                actual_send_seconds=actual_send,
                send_lateness_seconds=send_lateness,
                first_output_seconds=first_out,
                finished_seconds=finished,
                ttft_seconds=first_out - actual_send,
                latency_seconds=finished - actual_send,
                generated_tokens=turn_generated_tokens[i],
                generated_token_ids=turn_generated_token_ids[i],
                configured_max_tokens_per_turn=max_tokens_per_turn,
                effective_max_tokens=effective_max_tokens[i],
                is_resume=turn.turn > 1,
                finish_reason=turn_finish_reasons[i],
            )
        )


async def inspect_worker(engine: Any) -> dict[str, Any]:
    states = await engine.engine_core.collective_rpc_async("inspect_hkv_replay")
    if not states:
        raise RuntimeError("HKV worker inspection returned no states")
    result = {
        "warm_blocks": sum(state["warm_blocks"] for state in states),
        "warm_requests": sum(state["warm_requests"] for state in states),
        "owned_warm_slots": sum(
            state["owned_warm_slots"] for state in states
        ),
        "allocator_consistent": all(
            state["allocator_consistent"] for state in states
        ),
        "max_gpu_allocated_bytes": max(
            state["max_gpu_allocated_bytes"] for state in states
        ),
        "max_gpu_reserved_bytes": max(
            state["max_gpu_reserved_bytes"] for state in states
        ),
    }
    for field_name in (
        "num_gpu_blocks",
        "hot_kv_storage_bytes",
        "warm_kv_storage_bytes",
        "hot_to_warm_map_storage_bytes",
        "warm_slot_table_storage_bytes",
        "actual_persistent_kv_bytes",
    ):
        result[field_name] = sum(state[field_name] for state in states)
    for field_name in (
        "configured_total_kv_budget_bytes",
        "derived_hot_kv_budget_bytes",
        "budget_slack_bytes",
    ):
        values = {state[field_name] for state in states}
        if len(values) != 1:
            raise RuntimeError(
                f"HKV workers disagree on {field_name}: "
                f"{sorted(values, key=repr)}"
            )
        result[field_name] = values.pop()
    return result


def update_observation(summary: dict[str, Any], state: dict[str, Any]) -> None:
    summary["samples"] += 1
    summary["warm_observed"] |= state["warm_blocks"] > 0
    for source, target in (
        ("warm_blocks", "peak_warm_blocks"),
        ("warm_requests", "peak_warm_requests"),
        ("owned_warm_slots", "peak_owned_warm_slots"),
        ("max_gpu_allocated_bytes", "max_gpu_allocated_bytes"),
        ("max_gpu_reserved_bytes", "max_gpu_reserved_bytes"),
    ):
        summary[target] = max(summary[target], state[source])
    summary["allocator_consistent"] &= state["allocator_consistent"]
    for field_name in (
        "num_gpu_blocks",
        "hot_kv_storage_bytes",
        "warm_kv_storage_bytes",
        "hot_to_warm_map_storage_bytes",
        "warm_slot_table_storage_bytes",
        "actual_persistent_kv_bytes",
        "configured_total_kv_budget_bytes",
        "derived_hot_kv_budget_bytes",
        "budget_slack_bytes",
    ):
        summary[field_name] = state[field_name]


def build_runtime_memory_accounting(
    budget: dict[str, Any],
    runtime_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "num_gpu_blocks": runtime_state["num_gpu_blocks"],
        "hot_kv_storage_bytes": runtime_state["hot_kv_storage_bytes"],
        "warm_kv_storage_bytes": runtime_state["warm_kv_storage_bytes"],
        "hot_to_warm_map_storage_bytes": runtime_state[
            "hot_to_warm_map_storage_bytes"
        ],
        "warm_slot_table_storage_bytes": runtime_state[
            "warm_slot_table_storage_bytes"
        ],
        "actual_persistent_kv_bytes": runtime_state[
            "actual_persistent_kv_bytes"
        ],
        "configured_total_kv_budget_bytes": runtime_state[
            "configured_total_kv_budget_bytes"
        ],
        "derived_hot_kv_budget_bytes": runtime_state[
            "derived_hot_kv_budget_bytes"
        ],
        "budget_slack_bytes": runtime_state["budget_slack_bytes"],
    }


def validate_runtime_memory_accounting(
    budget: dict[str, Any],
    runtime: dict[str, Any],
) -> list[str]:
    if budget["total_kv_budget_bytes"] is None:
        return []
    expected = {
        "num_gpu_blocks": budget["derived_num_gpu_blocks"],
        "hot_kv_storage_bytes": budget["derived_hot_kv_budget_bytes"],
        "warm_kv_storage_bytes": budget["derived_warm_kv_storage_bytes"],
        "hot_to_warm_map_storage_bytes": budget[
            "derived_hot_to_warm_map_storage_bytes"
        ],
        "warm_slot_table_storage_bytes": budget[
            "derived_warm_slot_table_storage_bytes"
        ],
        "actual_persistent_kv_bytes": budget[
            "derived_actual_persistent_kv_bytes"
        ],
        "configured_total_kv_budget_bytes": budget[
            "total_kv_budget_bytes"
        ],
        "derived_hot_kv_budget_bytes": budget[
            "derived_hot_kv_budget_bytes"
        ],
        "budget_slack_bytes": budget["derived_budget_slack_bytes"],
    }
    differences = {
        field_name: {
            "expected": expected_value,
            "actual": runtime[field_name],
        }
        for field_name, expected_value in expected.items()
        if runtime[field_name] != expected_value
    }
    total_budget = budget["total_kv_budget_bytes"]
    if runtime["actual_persistent_kv_bytes"] > total_budget:
        differences["configured_total_kv_budget_bytes"] = {
            "expected_maximum": total_budget,
            "actual": runtime["actual_persistent_kv_bytes"],
        }
    if differences:
        return [
            "persistent KV runtime accounting differs from the explicit "
            f"budget derivation: {differences}"
        ]
    return []


async def observe_hkv(
    engine: Any,
    stop: asyncio.Event,
    interval: float,
    summary: dict[str, Any],
    on_tick: Any | None = None,
) -> None:
    while not stop.is_set():
        try:
            update_observation(summary, await inspect_worker(engine))
        except Exception:
            if stop.is_set():
                break
        if on_tick is not None:
            on_tick()
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass


def load_vllm_sampling_types() -> tuple[Any, Any, Any]:
    from vllm import SamplingParams
    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind

    return SamplingParams, StreamingInput, RequestOutputKind


async def run_session(
    engine: Any,
    turns: list[ReplayTurn],
    replay_started: float,
    seed: int,
    max_tokens_per_turn: int = 1,
    run_timing: dict[str, Any] | None = None,
    live_sessions: list[SessionResult] | None = None,
) -> tuple[SessionResult, list[float]]:
    SamplingParams, StreamingInput, RequestOutputKind = (
        load_vllm_sampling_types()
    )

    result = SessionResult(
        root_chat_id=turns[0].root_chat_id,
        session_id=turns[0].session_id,
        turns=len(turns),
        final_input_tokens=turns[-1].input_length,
        trace_output_tokens=sum(turn.trace_output_length for turn in turns),
        scheduled_first_seconds=turns[0].send_at_seconds,
        scheduled_last_seconds=turns[-1].send_at_seconds,
    )
    if live_sessions is not None:
        live_sessions.append(result)
    common = {
        "temperature": 0.0,
        "seed": seed,
        "ignore_eos": True,
        "output_kind": RequestOutputKind.DELTA,
    }
    effective_max_tokens = [
        1 if index < len(turns) - 1 else max_tokens_per_turn
        for index in range(len(turns))
    ]
    # Per-turn StreamingInput sampling params keep non-final turns at 1 token.
    # The base request must not impose a smaller lifetime cap than the final
    # turn: effective_max_tokens[0] is 1 on every multi-turn session.
    base_params = SamplingParams(max_tokens=max(effective_max_tokens), **common)
    turn_finished = [asyncio.Event() for _ in turns]
    lateness_values: list[float] = []

    actual_send_times: list[float | None] = [None] * len(turns)
    turn_first_output_times: list[float | None] = [None] * len(turns)
    turn_finished_times: list[float | None] = [None] * len(turns)
    turn_generated_tokens: list[int] = [0] * len(turns)
    turn_generated_token_ids: list[list[int]] = [[] for _ in turns]
    turn_finish_reasons: list[str | None] = [None] * len(turns)

    async def inputs():
        for i, turn in enumerate(turns):
            target = replay_started + turn.send_at_seconds
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            now = time.perf_counter() - replay_started
            actual_send = max(0.0, now)
            actual_send_times[i] = actual_send
            if run_timing is not None and run_timing["first_request_sent"] is None:
                run_timing["first_request_sent"] = utc_now()
                run_timing["first_request_sent_seconds"] = actual_send
            lateness = max(0.0, actual_send - turn.send_at_seconds)
            lateness_values.append(lateness)
            result.max_send_lateness_seconds = max(
                result.max_send_lateness_seconds, lateness
            )
            yield StreamingInput(
                {"prompt_token_ids": list(turn.delta_token_ids)},
                SamplingParams(max_tokens=effective_max_tokens[i], **common),
            )
        # Do not send vLLM's stream-finished sentinel before the last output.
        await turn_finished[-1].wait()

    digest = hashlib.sha256()
    finish_index = 0
    cancelled = False
    try:
        async for output in engine.generate(
            inputs(), base_params, result.session_id
        ):
            if not output.outputs:
                continue
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            now = time.perf_counter() - replay_started
            if run_timing is not None:
                run_timing["last_output_received"] = utc_now()
                run_timing["last_output_received_seconds"] = now
            if token_ids and result.first_output_seconds is None:
                result.first_output_seconds = now
            if token_ids and finish_index < len(turns):
                if turn_first_output_times[finish_index] is None:
                    turn_first_output_times[finish_index] = now
                turn_generated_tokens[finish_index] += len(token_ids)
                turn_generated_token_ids[finish_index].extend(token_ids)
            for token_id in token_ids:
                digest.update(int(token_id).to_bytes(8, "little"))
            room = 16 - len(result.generated_token_sample)
            if room > 0:
                result.generated_token_sample.extend(token_ids[:room])
            result.generated_tokens += len(token_ids)
            if completion.finish_reason and finish_index < len(turn_finished):
                finished_now = time.perf_counter() - replay_started
                turn_finished_times[finish_index] = finished_now
                turn_finish_reasons[finish_index] = completion.finish_reason
                if turn_first_output_times[finish_index] is None:
                    turn_first_output_times[finish_index] = finished_now
                turn_finished[finish_index].set()
                finish_index += 1
                result.completed_turns = finish_index
    except asyncio.CancelledError:
        cancelled = True

    result.finished_seconds = time.perf_counter() - replay_started
    result.generated_token_sha256 = digest.hexdigest()
    record_available_turn_results(
        result,
        turns,
        actual_send_times=actual_send_times,
        turn_first_output_times=turn_first_output_times,
        turn_finished_times=turn_finished_times,
        turn_generated_tokens=turn_generated_tokens,
        turn_generated_token_ids=turn_generated_token_ids,
        turn_finish_reasons=turn_finish_reasons,
        max_tokens_per_turn=max_tokens_per_turn,
        effective_max_tokens=effective_max_tokens,
    )
    if cancelled:
        return result, lateness_values
    if result.completed_turns != result.turns:
        raise RuntimeError(
            f"session {result.root_chat_id} completed "
            f"{result.completed_turns}/{result.turns} turns"
        )
    if len(result.turn_results) != result.turns:
        raise RuntimeError(
            f"session {result.root_chat_id} turn {turns[0].turn} "
            f"(chat_id={turns[0].chat_id}) has incomplete timing"
        )

    return result, lateness_values


def metric_distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values, default=None),
    }


def compute_turn_metrics_summary(
    sessions: list[SessionResult],
) -> dict[str, dict[str, float | None]]:
    all_turns = [
        turn for session in sessions for turn in session.turn_results
    ]
    resumed_turns = [turn for turn in all_turns if turn.is_resume]

    return {
        "all_turn_ttft_seconds": metric_distribution(
            [t.ttft_seconds for t in all_turns if t.ttft_seconds is not None]
        ),
        "resumed_turn_ttft_seconds": metric_distribution(
            [t.ttft_seconds for t in resumed_turns if t.ttft_seconds is not None]
        ),
        "all_turn_latency_seconds": metric_distribution(
            [t.latency_seconds for t in all_turns if t.latency_seconds is not None]
        ),
        "resumed_turn_latency_seconds": metric_distribution(
            [t.latency_seconds for t in resumed_turns if t.latency_seconds is not None]
        ),
    }


def compute_run_metrics(
    *,
    requests: int,
    generated_tokens: int,
    workload_start: float,
    last_output_received: float,
    cleanup_start: float,
    shutdown_start: float,
    shutdown_complete: float,
) -> dict[str, float]:
    service_window = last_output_received - workload_start
    if service_window <= 0:
        raise ValueError("service window duration must be positive")
    return {
        "service_window_duration_seconds": service_window,
        "cleanup_duration_seconds": shutdown_start - cleanup_start,
        "shutdown_duration_seconds": shutdown_complete - shutdown_start,
        "requests_per_second": requests / service_window,
        "output_tokens_per_second_service_window": (
            generated_tokens / service_window
        ),
    }


def validate_timing_and_turns(sessions: list[SessionResult]) -> list[str]:
    errors: list[str] = []
    for session in sessions:
        if session.completed_turns != session.turns:
            errors.append(
                f"session {session.root_chat_id} completed "
                f"{session.completed_turns}/{session.turns} turns"
            )
        if len(session.turn_results) != session.turns:
            errors.append(
                f"session {session.root_chat_id} recorded "
                f"{len(session.turn_results)}/{session.turns} turn results"
            )
        if session.turns > 1 and not session.turn_results:
            errors.append(
                f"resumed session {session.root_chat_id} is missing per-turn metrics"
            )

        for field_name in (
            "scheduled_first_seconds",
            "scheduled_last_seconds",
            "max_send_lateness_seconds",
            "first_output_seconds",
            "finished_seconds",
        ):
            val = getattr(session, field_name)
            if val is not None and val < 0:
                errors.append(
                    f"session {session.root_chat_id} has negative {field_name}: {val}"
                )

        for turn in session.turn_results:
            if turn.turn > 1 and not turn.is_resume:
                errors.append(
                    f"session {session.root_chat_id} turn {turn.turn} "
                    "has is_resume=False; expected True"
                )
            if turn.turn == 1 and turn.is_resume:
                errors.append(
                    f"session {session.root_chat_id} turn {turn.turn} "
                    "has is_resume=True; expected False"
                )
            for tf in (
                "scheduled_send_seconds",
                "actual_send_seconds",
                "send_lateness_seconds",
                "first_output_seconds",
                "finished_seconds",
                "ttft_seconds",
                "latency_seconds",
            ):
                val = getattr(turn, tf)
                if val is None or val < 0:
                    errors.append(
                        f"session {session.root_chat_id} turn {turn.turn} "
                        f"has invalid/negative {tf}: {val}"
                    )
    return errors


LEGITIMATE_EARLY_STOP_REASONS = frozenset({"abort"})
SILENT_ONE_TOKEN_FINISH_REASONS = frozenset({None, "length"})


def validate_final_turn_generation_limits(
    sessions: list[SessionResult],
) -> list[str]:
    """Reject runs where every multi-token final turn is silently capped at 1.

    ignore_eos=True is used, so a normal final turn should reach its configured
    maximum unless the runtime reports a legitimate early termination reason.
    """
    multi_token_finals: list[TurnResult] = []
    silently_limited: list[TurnResult] = []
    for session in sessions:
        if not session.turn_results:
            continue
        final = session.turn_results[-1]
        if final.effective_max_tokens <= 1:
            continue
        multi_token_finals.append(final)
        if final.generated_tokens >= final.effective_max_tokens:
            continue
        reason = final.finish_reason
        if reason in LEGITIMATE_EARLY_STOP_REASONS:
            continue
        if (
            final.generated_tokens == 1
            and reason in SILENT_ONE_TOKEN_FINISH_REASONS
        ):
            silently_limited.append(final)
    if multi_token_finals and len(silently_limited) == len(multi_token_finals):
        return [
            "final turns were silently limited to one token despite "
            "effective_max_tokens > 1; ignore_eos=True so a normal final "
            "turn should reach its configured maximum unless the runtime "
            "reports a legitimate early termination reason"
        ]
    return []


def build_experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    mixed = args.kv_mode == "mixed"
    memory_budget = derive_persistent_kv_budget(args)
    return {
        "experiment_mode": args.experiment_mode,
        "kv_mode": args.kv_mode,
        "model": args.model,
        "seed": args.seed,
        "time_scale": args.time_scale,
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "generation_parameters": generation_parameters(args),
        "generation_policy": generation_parameters(args)["generation_policy"],
        "thresholds": {
            "demotion_start_utilization": (
                args.demotion_start_utilization if mixed else None
            ),
            "demotion_stop_utilization": (
                args.demotion_stop_utilization if mixed else None
            ),
            "hot_idle_threshold_seconds": None,
            "cold_idle_threshold_seconds": None,
        },
        "warm_pool_blocks": args.warm_pool_blocks if mixed else 0,
        "hkv_settings": {
            "physical_tiers": mixed,
            "multi_block_warm_migration": mixed,
            "mixed_attention_read": mixed,
        },
        "max_sessions": args.max_sessions,
        "min_turns": args.min_turns,
        "max_input_length": args.max_input_length,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "persistent_kv_memory_budget": memory_budget,
        "attention_backend": ATTENTION_BACKEND,
        "dtype": MODEL_DTYPE,
        "topology_assumptions": TOPOLOGY_ASSUMPTIONS,
    }


def build_experiment_fingerprint(
    args: argparse.Namespace,
    trace: dict[str, str],
    selection: dict[str, Any],
) -> dict[str, Any]:
    memory_budget = derive_persistent_kv_budget(args)
    fields = {
        "experiment_mode": args.experiment_mode,
        "model": args.model,
        "trace": trace,
        "seed": args.seed,
        "selection_sha256": selection["selection_sha256"],
        "selected_session_count": selection["selected_session_count"],
        "selected_request_count": selection["selected_request_count"],
        "time_scale": args.time_scale,
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "generation_parameters": generation_parameters(args),
        "generation_policy": generation_parameters(args)["generation_policy"],
        "gpu_memory_configuration": {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "block_size": BLOCK_SIZE,
            "total_kv_budget_bytes": memory_budget[
                "total_kv_budget_bytes"
            ],
            "max_num_seqs": args.max_num_seqs,
            "kv_memory_formula_version": KV_MEMORY_FORMULA_VERSION,
        },
        "attention_backend": ATTENTION_BACKEND,
        "enable_prefix_caching": False,
        "dtype": MODEL_DTYPE,
        "topology_assumptions": TOPOLOGY_ASSUMPTIONS,
    }
    return {"fields": fields, "sha256": sha256_json(fields)}


def build_engine_args_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    mixed = args.kv_mode == "mixed"
    memory_budget = derive_persistent_kv_budget(args)
    return {
        "model": args.model,
        "dtype": MODEL_DTYPE,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_memory_bytes": memory_budget[
            "derived_hot_kv_budget_bytes"
        ],
        "block_size": BLOCK_SIZE,
        "attention_backend": ATTENTION_BACKEND,
        "worker_extension_cls": (
            "experiments.scripts.run_qwen_bailian_replay."
            "HKVReplayWorkerExtension"
        ),
        "kv_cache_hot_idle_threshold_seconds": None,
        "kv_cache_cold_idle_threshold_seconds": None,
        "kv_cache_demotion_start_utilization": (
            args.demotion_start_utilization if mixed else None
        ),
        "kv_cache_demotion_stop_utilization": (
            args.demotion_stop_utilization if mixed else None
        ),
    }


def build_reproducibility_metadata(
    args: argparse.Namespace,
    trace: dict[str, str],
    selection: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    experiment_config = build_experiment_config(args)
    memory_budget = experiment_config["persistent_kv_memory_budget"]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_mode": args.experiment_mode,
        "kv_mode": args.kv_mode,
        "model": args.model,
        "seed": args.seed,
        "trace": trace,
        "selection": selection,
        "time_scale": args.time_scale,
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "generation_parameters": generation_parameters(args),
        "generation_policy": generation_parameters(args)["generation_policy"],
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "persistent_kv_memory_budget": memory_budget,
        "configured_hkv": experiment_config["hkv_settings"],
        "warm_pool_blocks": experiment_config["warm_pool_blocks"],
        "thresholds": experiment_config["thresholds"],
        "attention_backend": ATTENTION_BACKEND,
        "dtype": MODEL_DTYPE,
        "topology_assumptions": TOPOLOGY_ASSUMPTIONS,
        "git": get_git_metadata(),
        "timing": timing,
    }


def _fingerprint_differences(
    current: Any,
    baseline: Any,
    prefix: str = "",
) -> list[str]:
    if isinstance(current, dict) and isinstance(baseline, dict):
        differences: list[str] = []
        for key in sorted(current.keys() | baseline.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in current or key not in baseline:
                differences.append(path)
            else:
                differences.extend(
                    _fingerprint_differences(current[key], baseline[key], path)
                )
        return differences
    return [] if current == baseline else [prefix]


def validate_baseline_schema(
    experiment_mode: str,
    baseline: dict[str, Any],
) -> None:
    if "experiment_fingerprint" not in baseline:
        raise ValueError(
            "legacy baseline unsupported: baseline is missing the experiment "
            "fingerprint and persistent KV-budget metadata"
        )
    if experiment_mode == "performance":
        baseline_budget = (
            baseline.get("experiment_config", {})
            .get("persistent_kv_memory_budget", {})
            .get("total_kv_budget_bytes")
        )
        if (
            baseline.get("schema_version") != RESULT_SCHEMA_VERSION
            or baseline_budget is None
        ):
            raise ValueError(
                "legacy baseline unsupported for performance comparison: "
                f"schema {RESULT_SCHEMA_VERSION} with an explicit persistent "
                "KV-memory budget is required"
            )


def compare_baseline(
    result: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    validate_baseline_schema(result["experiment_mode"], baseline)
    current_fingerprint = result["experiment_fingerprint"]["fields"]
    baseline_fingerprint = baseline["experiment_fingerprint"]["fields"]
    fingerprint_differences = _fingerprint_differences(
        current_fingerprint,
        baseline_fingerprint,
    )
    if fingerprint_differences:
        raise ValueError(
            "baseline experiment fingerprint differs in required fields: "
            + ", ".join(fingerprint_differences)
        )

    def ordered_turn_tokens(
        value: dict[str, Any],
    ) -> dict[tuple[int, int, int], list[int]]:
        return {
            (
                int(session["root_chat_id"]),
                int(turn["chat_id"]),
                int(turn["turn"]),
            ): list(turn["generated_token_ids"])
            for session in value["sessions"]
            for turn in session["turn_results"]
        }

    expected = ordered_turn_tokens(baseline)
    actual = ordered_turn_tokens(result)
    mismatched = sorted(
        key
        for key in expected.keys() | actual.keys()
        if expected.get(key) != actual.get(key)
    )
    return {
        "fingerprint_match": True,
        "ordered_turn_token_match": not mismatched,
        "token_comparison_role": (
            "correctness_gate"
            if result["experiment_mode"] == "correctness"
            else "diagnostic_only"
        ),
        "mismatched_turns": [
            {
                "root_chat_id": key[0],
                "chat_id": key[1],
                "turn": key[2],
                "baseline_token_ids": expected.get(key),
                "current_token_ids": actual.get(key),
            }
            for key in mismatched
        ],
        "policy_configuration": {
            "baseline": baseline["experiment_config"],
            "current": result["experiment_config"],
        },
        "interpretation": (
            "deterministic correctness check"
            if result["experiment_mode"] == "correctness"
            else "diagnostic reproducibility check, not a quality metric"
        ),
    }


def token_comparison_validation_errors(
    experiment_mode: str,
    comparison: dict[str, Any],
) -> list[str]:
    if (
        experiment_mode == "correctness"
        and not comparison["ordered_turn_token_match"]
    ):
        return [
            "ordered per-turn output differs: "
            f"{comparison['mismatched_turns']}"
        ]
    return []


async def drain_warm_residency(
    engine: Any,
    observation: dict[str, Any],
    cleanup_timeout: float,
) -> dict[str, Any]:
    deadline = time.perf_counter() + cleanup_timeout
    final_state = await inspect_worker(engine)
    while final_state["warm_blocks"] and time.perf_counter() < deadline:
        await asyncio.sleep(0.05)
        final_state = await inspect_worker(engine)
    update_observation(observation, final_state)
    observation["final_warm_blocks"] = final_state["warm_blocks"]
    observation["cleanup_complete"] = final_state["warm_blocks"] == 0
    return final_state


def cli_exit_status(result: dict[str, Any]) -> int:
    if result.get("termination", {}).get("timed_out"):
        return 1
    return 0


async def run(args: argparse.Namespace) -> dict[str, Any]:
    run_started_at = utc_now()
    baseline = None
    if args.baseline_json:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        validate_baseline_schema(args.experiment_mode, baseline)
    selected = select_sessions(
        load_bailian_records(args.trace),
        request_type=None if args.request_type == "all" else args.request_type,
        min_turns=args.min_turns,
        max_input_length=args.max_input_length or None,
        max_sessions=args.max_sessions or None,
    )
    trace_metadata = trace_identifier(args.trace)
    selection = selection_metadata(selected)
    mixed = args.kv_mode == "mixed"
    memory_budget = derive_persistent_kv_budget(args)
    if memory_budget["total_kv_budget_bytes"] is None:
        os.environ.pop(TOTAL_KV_BUDGET_ENV, None)
    else:
        os.environ[TOTAL_KV_BUDGET_ENV] = str(
            memory_budget["total_kv_budget_bytes"]
        )
    engine = create_engine(args)
    observation = {
        "samples": 0,
        "warm_observed": False,
        "peak_warm_blocks": 0,
        "peak_warm_requests": 0,
        "peak_owned_warm_slots": 0,
        "allocator_consistent": True,
        "max_gpu_allocated_bytes": 0,
        "max_gpu_reserved_bytes": 0,
        "num_gpu_blocks": 0,
        "hot_kv_storage_bytes": 0,
        "warm_kv_storage_bytes": 0,
        "hot_to_warm_map_storage_bytes": 0,
        "warm_slot_table_storage_bytes": 0,
        "actual_persistent_kv_bytes": 0,
        "configured_total_kv_budget_bytes": None,
        "derived_hot_kv_budget_bytes": None,
        "budget_slack_bytes": None,
        "final_warm_blocks": None,
        "cleanup_complete": None,
    }
    stop_observer = asyncio.Event()
    observer: asyncio.Task | None = None
    tasks: list[asyncio.Task] = []
    live_sessions: list[SessionResult] = []
    run_timing: dict[str, Any] = {
        "replay_plan_ready": None,
        "workload_start": None,
        "first_request_sent": None,
        "first_request_sent_seconds": None,
        "last_output_received": None,
        "last_output_received_seconds": None,
        "cleanup_start": None,
        "shutdown_complete": None,
    }
    cleanup_started_perf: float | None = None
    shutdown_started_perf: float | None = None
    shutdown_complete_perf: float | None = None
    timed_out = False
    timeout_snapshot: dict[str, Any] | None = None
    session_results: list[SessionResult] = []
    lateness: list[float] = []
    final_state: dict[str, Any] | None = None
    scheduled_duration = 0.0
    workload_started = 0.0

    def progress_tick() -> None:
        if run_timing["workload_start"] is None:
            return
        elapsed = time.perf_counter() - workload_started
        progress = collect_session_progress(
            live_sessions,
            selected_session_count=selection["selected_session_count"],
            selected_request_count=selection["selected_request_count"],
        )
        print_replay_progress(
            elapsed_seconds=elapsed,
            progress=progress,
            hot_utilization=None,
        )

    try:
        plan = build_replay_plan(
            selected,
            vocab_size=tokenizer_vocab_size(engine.get_tokenizer()),
            time_scale=args.time_scale,
            seed=args.seed,
        )
        run_timing["replay_plan_ready"] = utc_now()
        scheduled_duration = max(
            turn.send_at_seconds for turns in plan.values() for turn in turns
        )
        workload_started = time.perf_counter()
        run_timing["workload_start"] = utc_now()
        observer = asyncio.create_task(
            observe_hkv(
                engine,
                stop_observer,
                args.metrics_interval,
                observation,
                on_tick=progress_tick,
            )
        )
        tasks = [
            asyncio.create_task(
                run_session(
                    engine,
                    turns,
                    workload_started,
                    args.seed,
                    args.max_tokens_per_turn,
                    run_timing,
                    live_sessions,
                )
            )
            for _, turns in sorted(plan.items())
        ]
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=args.timeout)
        except TimeoutError:
            timed_out = True
            cleanup_started_perf = time.perf_counter()
            run_timing["cleanup_start"] = utc_now()
            try:
                timeout_snapshot = await inspect_worker(engine)
                update_observation(observation, timeout_snapshot)
            except Exception:
                timeout_snapshot = None
        else:
            cleanup_started_perf = time.perf_counter()
            run_timing["cleanup_start"] = utc_now()
            final_state = await drain_warm_residency(
                engine, observation, args.cleanup_timeout
            )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        gathered: list[Any] = []
        if tasks:
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
        session_results, lateness = session_results_from_outcomes(gathered)
        if timed_out:
            try:
                final_state = await drain_warm_residency(
                    engine, observation, args.cleanup_timeout
                )
            except Exception:
                observation["cleanup_complete"] = False
        stop_observer.set()
        if observer is not None:
            await observer
        shutdown_started_perf = time.perf_counter()
        engine.shutdown()
        shutdown_complete_perf = time.perf_counter()
        run_timing["shutdown_complete"] = utc_now()

    progress = collect_session_progress(
        session_results,
        selected_session_count=selection["selected_session_count"],
        selected_request_count=selection["selected_request_count"],
    )
    generated = progress["total_generated_tokens"]
    last_output_received_seconds = run_timing["last_output_received_seconds"]
    if (
        shutdown_started_perf is None
        or shutdown_complete_perf is None
        or cleanup_started_perf is None
    ):
        raise RuntimeError("cleanup and shutdown timing was not completed")
    if timed_out:
        service_window = cleanup_started_perf - workload_started
        run_metrics = {
            "service_window_duration_seconds": service_window,
            "cleanup_duration_seconds": shutdown_started_perf - cleanup_started_perf,
            "shutdown_duration_seconds": (
                shutdown_complete_perf - shutdown_started_perf
            ),
            "requests_per_second": None,
            "output_tokens_per_second_service_window": None,
        }
    else:
        if last_output_received_seconds is None:
            raise RuntimeError("no model output was received during the workload")
        last_output_received_perf = (
            workload_started + last_output_received_seconds
        )
        run_metrics = compute_run_metrics(
            requests=progress["selected_request_count"],
            generated_tokens=generated,
            workload_start=workload_started,
            last_output_received=last_output_received_perf,
            cleanup_start=cleanup_started_perf,
            shutdown_start=shutdown_started_perf,
            shutdown_complete=shutdown_complete_perf,
        )
        service_window = run_metrics["service_window_duration_seconds"]
    turn_metrics_summary = compute_turn_metrics_summary(session_results)
    experiment_config = build_experiment_config(args)
    fingerprint = build_experiment_fingerprint(
        args,
        trace_metadata,
        selection,
    )
    timing_metadata = {
        **run_timing,
        "start": run_started_at,
        "end": run_timing["shutdown_complete"],
        "service_window_duration_seconds": run_metrics[
            "service_window_duration_seconds"
        ],
        "cleanup_duration_seconds": run_metrics[
            "cleanup_duration_seconds"
        ],
        "shutdown_duration_seconds": run_metrics[
            "shutdown_duration_seconds"
        ],
    }
    reproducibility = build_reproducibility_metadata(
        args,
        trace_metadata,
        selection,
        timing_metadata,
    )
    runtime_source = final_state if final_state is not None else observation
    runtime_memory = build_runtime_memory_accounting(
        memory_budget,
        runtime_source,
    )
    termination = build_termination(
        timed_out=timed_out,
        configured_timeout_seconds=args.timeout,
        service_window_duration_seconds=service_window,
        last_output_received_seconds=last_output_received_seconds,
        progress=progress,
        runtime_at_timeout=(
            build_timeout_runtime_snapshot(
                observation,
                timeout_snapshot,
                warm_pool_blocks=args.warm_pool_blocks,
            )
            if timed_out
            else None
        ),
    )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_mode": args.experiment_mode,
        "kv_mode": args.kv_mode,
        "trace": trace_metadata,
        "model": args.model,
        "seed": args.seed,
        "time_scale": args.time_scale,
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "max_num_seqs": args.max_num_seqs,
        "generation_parameters": generation_parameters(args),
        "generation_policy": generation_parameters(args)["generation_policy"],
        "selected_sessions": selection["selected_session_count"],
        "selected_requests": selection["selected_request_count"],
        "selection": selection,
        "scheduled_duration_seconds": scheduled_duration,
        "timing": timing_metadata,
        "termination": termination,
        "total_final_input_tokens": sum(
            session.final_input_tokens for session in session_results
        ),
        "total_trace_output_tokens": sum(
            session.trace_output_tokens for session in session_results
        ),
        "total_generated_tokens": generated,
        "requests_per_second": run_metrics["requests_per_second"],
        "output_tokens_per_second_service_window": run_metrics[
            "output_tokens_per_second_service_window"
        ],
        "decode_only_tokens_per_second": None,
        "throughput_definition": (
            "Output tokens and completed turns divided by workload_start to "
            "last_output_received. Decode-only throughput is unavailable from "
            "the replay callbacks. Multi-token decoding is measured only on "
            "each session's final turn; non-final synthetic trace turns use "
            "one token to avoid contaminating later trace history."
        ),
        "send_lateness_seconds": {
            "p50": percentile(lateness, 0.50),
            "p95": percentile(lateness, 0.95),
            "max": max(lateness, default=None),
        },
        "all_turn_ttft_seconds": turn_metrics_summary["all_turn_ttft_seconds"],
        "resumed_turn_ttft_seconds": turn_metrics_summary[
            "resumed_turn_ttft_seconds"
        ],
        "all_turn_latency_seconds": turn_metrics_summary[
            "all_turn_latency_seconds"
        ],
        "resumed_turn_latency_seconds": turn_metrics_summary[
            "resumed_turn_latency_seconds"
        ],
        "experiment_config": experiment_config,
        "experiment_fingerprint": fingerprint,
        "persistent_kv_memory_budget": memory_budget,
        "runtime_persistent_kv_memory": runtime_memory,
        "git": reproducibility["git"],
        "reproducibility": reproducibility,
        "hkv_observation": observation,
        "sessions": [asdict(session) for session in session_results],
    }
    validation_errors: list[str] = []
    if timed_out:
        result["validation"] = {
            "passed": False,
            "errors": [
                "workload timed out before all sessions completed"
            ],
            "full_validation_performed": False,
            "full_validation_unavailable_reason": "timed_out",
        }
        if not observation["allocator_consistent"]:
            result["validation"]["errors"].append(
                "WARM allocator ownership became inconsistent"
            )
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    if mixed and not observation["warm_observed"]:
        validation_errors.append("mixed mode never observed WARM residency")
    if not mixed and observation["warm_observed"]:
        validation_errors.append("all-hot mode unexpectedly observed WARM residency")
    if not mixed:
        unexpected_mixed_storage = {
            field_name: runtime_memory[field_name]
            for field_name in (
                "warm_kv_storage_bytes",
                "hot_to_warm_map_storage_bytes",
                "warm_slot_table_storage_bytes",
            )
            if runtime_memory[field_name] != 0
        }
        if unexpected_mixed_storage:
            validation_errors.append(
                "all-hot mode allocated mixed-only persistent storage: "
                f"{unexpected_mixed_storage}"
            )
    if not observation["allocator_consistent"]:
        validation_errors.append("WARM allocator ownership became inconsistent")
    if not observation["cleanup_complete"]:
        validation_errors.append("WARM residency was not released after replay")
    validation_errors.extend(
        validate_runtime_memory_accounting(memory_budget, runtime_memory)
    )

    validation_errors.extend(validate_timing_and_turns(session_results))
    validation_errors.extend(
        validate_final_turn_generation_limits(session_results)
    )

    if baseline is not None:
        result["baseline_comparison"] = compare_baseline(result, baseline)
        validation_errors.extend(
            token_comparison_validation_errors(
                args.experiment_mode,
                result["baseline_comparison"],
            )
        )

    result["validation"] = {
        "passed": not validation_errors,
        "errors": validation_errors,
        "full_validation_performed": True,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if validation_errors:
        raise AssertionError("; ".join(validation_errors))
    return result

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-mode",
        choices=("correctness", "performance"),
        required=True,
    )
    parser.add_argument(
        "--kv-mode",
        choices=("all-hot", "mixed"),
        required=True,
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path(
            "datasets/qwen_bailian/subsets/qwen_traceA_multiturn_v0.jsonl"
        ),
    )
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--time-scale", type=float, default=0.01)
    parser.add_argument("--request-type", default="text")
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--total-kv-budget-bytes", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens-per-turn", type=int)
    parser.add_argument("--demotion-start-utilization", type=float)
    parser.add_argument("--demotion-stop-utilization", type=float)
    parser.add_argument("--warm-pool-blocks", type=int)
    parser.add_argument("--metrics-interval", type=float, default=0.05)
    parser.add_argument("--cleanup-timeout", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.experiment_mode == "correctness":
        if args.max_tokens_per_turn is None:
            args.max_tokens_per_turn = 1
        elif args.max_tokens_per_turn <= 0:
            parser.error("--max-tokens-per-turn must be positive")
    elif (
        args.max_tokens_per_turn is None
        or args.max_tokens_per_turn <= 1
    ):
        parser.error(
            "performance mode requires --max-tokens-per-turn greater than 1"
        )
    if (
        args.experiment_mode == "performance"
        and args.total_kv_budget_bytes is None
    ):
        parser.error(
            "performance mode requires positive --total-kv-budget-bytes"
        )

    mixed = args.kv_mode == "mixed"
    threshold_pair = (
        args.demotion_start_utilization,
        args.demotion_stop_utilization,
    )
    if mixed and args.baseline_json is None:
        parser.error("--baseline-json is required in mixed mode")
    if mixed and any(value is None for value in threshold_pair):
        parser.error(
            "mixed mode requires both --demotion-start-utilization and "
            "--demotion-stop-utilization"
        )
    if not mixed and any(value is not None for value in threshold_pair):
        parser.error("pressure thresholds are valid only in mixed mode")
    if mixed:
        start, stop = threshold_pair
        assert start is not None and stop is not None
        if (
            not math.isfinite(start)
            or not math.isfinite(stop)
            or not 0.0 <= stop < start <= 1.0
        ):
            parser.error(
                "pressure thresholds must satisfy 0 <= stop < start <= 1"
            )
        if args.warm_pool_blocks is None or args.warm_pool_blocks <= 0:
            parser.error("mixed mode requires positive --warm-pool-blocks")
    elif args.warm_pool_blocks is not None:
        parser.error("--warm-pool-blocks is valid only in mixed mode")
    else:
        args.warm_pool_blocks = 0

    if args.time_scale <= 0 or args.min_turns <= 0:
        parser.error("time scale and minimum turns must be positive")
    if args.max_model_len <= 0:
        parser.error("--max-model-len must be positive")
    if args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be positive")
    if args.max_input_length < 0 or args.max_sessions < 0:
        parser.error("session limits must be non-negative")
    if args.total_kv_budget_bytes is not None:
        try:
            derive_persistent_kv_budget(args)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def apply_hkv_environment(args: argparse.Namespace) -> None:
    """Apply the replay process environment before the engine starts."""
    mixed = args.kv_mode == "mixed"
    os.environ.update({
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "HKV_ENABLE_PHYSICAL_TIERS": "1" if mixed else "0",
        "HKV_WARM_POOL_BLOCKS": str(args.warm_pool_blocks) if mixed else "0",
        "HKV_DEBUG_DEMOTE_ONE_BLOCK": "0",
        # Drop a stale shell setting. The resume-stall tracer has been removed.
        "HKV_DEBUG_RESUME_PROGRESS": "0",
        "HKV_ENABLE_MULTI_BLOCK_WARM_MIGRATION": "1" if mixed else "0",
        "HKV_DEBUG_MIXED_READ": "1" if mixed else "0",
    })


def main() -> None:
    args = parse_args()
    apply_hkv_environment(args)
    result = asyncio.run(run(args))
    printable = {
        key: result[key]
        for key in (
            "experiment_mode",
            "kv_mode",
            "selected_sessions",
            "selected_requests",
            "timing",
            "total_generated_tokens",
            "requests_per_second",
            "output_tokens_per_second_service_window",
            "generation_policy",
            "send_lateness_seconds",
            "all_turn_ttft_seconds",
            "resumed_turn_ttft_seconds",
            "all_turn_latency_seconds",
            "resumed_turn_latency_seconds",
            "experiment_config",
            "persistent_kv_memory_budget",
            "runtime_persistent_kv_memory",
            "hkv_observation",
            "validation",
        )
        if key in result
    }
    if "baseline_comparison" in result:
        printable["baseline_comparison"] = result["baseline_comparison"]
    if "termination" in result:
        printable["termination"] = result["termination"]
    print(json.dumps(printable, indent=2))
    raise SystemExit(cli_exit_status(result))


if __name__ == "__main__":
    main()
