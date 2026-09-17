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

RESULT_SCHEMA_VERSION = "2.0"
ATTENTION_BACKEND = "TRITON_ATTN"
MODEL_DTYPE = "float16"
BLOCK_SIZE = 16
TOPOLOGY_ASSUMPTIONS = {
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "data_parallel_size": 1,
    "kv_cache_groups": 1,
    "blocks_per_kv_block": 1,
}


class HKVReplayWorkerExtension:
    """Expose aggregate WARM state without modifying the worker."""

    def inspect_hkv_replay(self) -> dict[str, Any]:
        import torch

        manager = getattr(
            self.model_runner, "hkv_warm_migration_manager", None
        )
        residency = manager.warm_residency if manager is not None else {}
        allocator = manager.allocator if manager is not None else None
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


async def inspect_worker(engine: Any) -> dict[str, Any]:
    states = await engine.engine_core.collective_rpc_async("inspect_hkv_replay")
    if not states:
        raise RuntimeError("HKV worker inspection returned no states")
    return {
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


async def observe_hkv(
    engine: Any,
    stop: asyncio.Event,
    interval: float,
    summary: dict[str, Any],
) -> None:
    while not stop.is_set():
        update_observation(summary, await inspect_worker(engine))
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass


async def run_session(
    engine: Any,
    turns: list[ReplayTurn],
    replay_started: float,
    seed: int,
    max_tokens_per_turn: int = 1,
    run_timing: dict[str, Any] | None = None,
) -> tuple[SessionResult, list[float]]:
    from vllm import SamplingParams
    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind

    result = SessionResult(
        root_chat_id=turns[0].root_chat_id,
        session_id=turns[0].session_id,
        turns=len(turns),
        final_input_tokens=turns[-1].input_length,
        trace_output_tokens=sum(turn.trace_output_length for turn in turns),
        scheduled_first_seconds=turns[0].send_at_seconds,
        scheduled_last_seconds=turns[-1].send_at_seconds,
    )
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
    base_params = SamplingParams(max_tokens=effective_max_tokens[0], **common)
    turn_finished = [asyncio.Event() for _ in turns]
    lateness_values: list[float] = []

    actual_send_times: list[float | None] = [None] * len(turns)
    turn_first_output_times: list[float | None] = [None] * len(turns)
    turn_finished_times: list[float | None] = [None] * len(turns)
    turn_generated_tokens: list[int] = [0] * len(turns)
    turn_generated_token_ids: list[list[int]] = [[] for _ in turns]

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
    async for output in engine.generate(inputs(), base_params, result.session_id):
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
            if turn_first_output_times[finish_index] is None:
                turn_first_output_times[finish_index] = finished_now
            turn_finished[finish_index].set()
            finish_index += 1
            result.completed_turns = finish_index

    result.finished_seconds = time.perf_counter() - replay_started
    result.generated_token_sha256 = digest.hexdigest()
    if result.completed_turns != result.turns:
        raise RuntimeError(
            f"session {result.root_chat_id} completed "
            f"{result.completed_turns}/{result.turns} turns"
        )

    for i, turn in enumerate(turns):
        actual_send = actual_send_times[i]
        first_out = turn_first_output_times[i]
        finished = turn_finished_times[i]
        if actual_send is None or first_out is None or finished is None:
            raise RuntimeError(
                f"session {result.root_chat_id} turn {turn.turn} "
                f"(chat_id={turn.chat_id}) has incomplete timing"
            )
        send_lateness = max(0.0, actual_send - turn.send_at_seconds)
        ttft = first_out - actual_send
        latency = finished - actual_send
        result.turn_results.append(
            TurnResult(
                chat_id=turn.chat_id,
                turn=turn.turn,
                scheduled_send_seconds=turn.send_at_seconds,
                actual_send_seconds=actual_send,
                send_lateness_seconds=send_lateness,
                first_output_seconds=first_out,
                finished_seconds=finished,
                ttft_seconds=ttft,
                latency_seconds=latency,
                generated_tokens=turn_generated_tokens[i],
                generated_token_ids=turn_generated_token_ids[i],
                configured_max_tokens_per_turn=max_tokens_per_turn,
                effective_max_tokens=effective_max_tokens[i],
                is_resume=turn.turn > 1,
            )
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


def build_experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    mixed = args.kv_mode == "mixed"
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
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "attention_backend": ATTENTION_BACKEND,
        "dtype": MODEL_DTYPE,
        "topology_assumptions": TOPOLOGY_ASSUMPTIONS,
    }


def build_experiment_fingerprint(
    args: argparse.Namespace,
    trace: dict[str, str],
    selection: dict[str, Any],
) -> dict[str, Any]:
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
        },
        "attention_backend": ATTENTION_BACKEND,
        "dtype": MODEL_DTYPE,
        "topology_assumptions": TOPOLOGY_ASSUMPTIONS,
    }
    return {"fields": fields, "sha256": sha256_json(fields)}


def build_engine_args_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    mixed = args.kv_mode == "mixed"
    return {
        "model": args.model,
        "dtype": MODEL_DTYPE,
        "enforce_eager": True,
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
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


def compare_baseline(
    result: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
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


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    run_started_at = utc_now()
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
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(**build_engine_args_kwargs(args))
    )
    observation = {
        "samples": 0,
        "warm_observed": False,
        "peak_warm_blocks": 0,
        "peak_warm_requests": 0,
        "peak_owned_warm_slots": 0,
        "allocator_consistent": True,
        "max_gpu_allocated_bytes": 0,
        "max_gpu_reserved_bytes": 0,
        "final_warm_blocks": None,
        "cleanup_complete": None,
    }
    stop_observer = asyncio.Event()
    observer: asyncio.Task | None = None
    tasks: list[asyncio.Task] = []
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
                engine, stop_observer, args.metrics_interval, observation
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
                )
            )
            for _, turns in sorted(plan.items())
        ]
        completed = await asyncio.wait_for(
            asyncio.gather(*tasks), timeout=args.timeout
        )
        session_results = [item[0] for item in completed]
        lateness = [value for item in completed for value in item[1]]
        cleanup_started_perf = time.perf_counter()
        run_timing["cleanup_start"] = utc_now()

        deadline = time.perf_counter() + args.cleanup_timeout
        final_state = await inspect_worker(engine)
        while final_state["warm_blocks"] and time.perf_counter() < deadline:
            await asyncio.sleep(0.05)
            final_state = await inspect_worker(engine)
        update_observation(observation, final_state)
        observation["final_warm_blocks"] = final_state["warm_blocks"]
        observation["cleanup_complete"] = final_state["warm_blocks"] == 0
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        stop_observer.set()
        if observer is not None:
            await observer
        shutdown_started_perf = time.perf_counter()
        engine.shutdown()
        shutdown_complete_perf = time.perf_counter()
        run_timing["shutdown_complete"] = utc_now()

    requests = sum(session.turns for session in session_results)
    generated = sum(session.generated_tokens for session in session_results)
    last_output_received_seconds = run_timing["last_output_received_seconds"]
    if last_output_received_seconds is None:
        raise RuntimeError("no model output was received during the workload")
    last_output_received_perf = (
        workload_started + last_output_received_seconds
    )
    if (
        cleanup_started_perf is None
        or shutdown_started_perf is None
        or shutdown_complete_perf is None
    ):
        raise RuntimeError("cleanup and shutdown timing was not completed")
    run_metrics = compute_run_metrics(
        requests=requests,
        generated_tokens=generated,
        workload_start=workload_started,
        last_output_received=last_output_received_perf,
        cleanup_start=cleanup_started_perf,
        shutdown_start=shutdown_started_perf,
        shutdown_complete=shutdown_complete_perf,
    )
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
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_mode": args.experiment_mode,
        "kv_mode": args.kv_mode,
        "trace": trace_metadata,
        "model": args.model,
        "seed": args.seed,
        "time_scale": args.time_scale,
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "generation_parameters": generation_parameters(args),
        "generation_policy": generation_parameters(args)["generation_policy"],
        "selected_sessions": len(session_results),
        "selected_requests": requests,
        "selection": selection,
        "scheduled_duration_seconds": scheduled_duration,
        "timing": timing_metadata,
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
        "git": reproducibility["git"],
        "reproducibility": reproducibility,
        "hkv_observation": observation,
        "sessions": [asdict(session) for session in session_results],
    }
    validation_errors: list[str] = []
    if mixed and not observation["warm_observed"]:
        validation_errors.append("mixed mode never observed WARM residency")
    if not mixed and observation["warm_observed"]:
        validation_errors.append("all-hot mode unexpectedly observed WARM residency")
    if not observation["allocator_consistent"]:
        validation_errors.append("WARM allocator ownership became inconsistent")
    if not observation["cleanup_complete"]:
        validation_errors.append("WARM residency was not released after replay")

    validation_errors.extend(validate_timing_and_turns(session_results))

    if args.baseline_json:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
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
    if args.max_input_length < 0 or args.max_sessions < 0:
        parser.error("session limits must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    mixed = args.kv_mode == "mixed"
    os.environ.update({
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "HKV_ENABLE_PHYSICAL_TIERS": "1" if mixed else "0",
        "HKV_WARM_POOL_BLOCKS": str(args.warm_pool_blocks) if mixed else "0",
        "HKV_DEBUG_DEMOTE_ONE_BLOCK": "0",
        "HKV_ENABLE_MULTI_BLOCK_WARM_MIGRATION": "1" if mixed else "0",
        "HKV_DEBUG_MIXED_READ": "1" if mixed else "0",
    })
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
            "hkv_observation",
            "validation",
        )
        if key in result
    }
    if "baseline_comparison" in result:
        printable["baseline_comparison"] = result["baseline_comparison"]
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()