"""Replay Qwen-Bailian chat sessions through vLLM streaming input."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from experiments.scripts.qwen_bailian_trace import (
    BailianRecord,
    ReplayTurn,
    build_replay_plan,
    group_linear_sessions,
    load_bailian_records,
)


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
    base_params = SamplingParams(max_tokens=1, **common)
    turn_finished = [asyncio.Event() for _ in turns]
    lateness_values: list[float] = []

    async def inputs():
        for turn in turns:
            target = replay_started + turn.send_at_seconds
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            lateness = max(0.0, time.perf_counter() - target)
            lateness_values.append(lateness)
            result.max_send_lateness_seconds = max(
                result.max_send_lateness_seconds, lateness
            )
            # One sampled token is deliberate: vLLM discards this final,
            # uncomputed token when the next synthetic trace delta arrives.
            yield StreamingInput(
                {"prompt_token_ids": list(turn.delta_token_ids)},
                SamplingParams(max_tokens=1, **common),
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
        if token_ids and result.first_output_seconds is None:
            result.first_output_seconds = time.perf_counter() - replay_started
        for token_id in token_ids:
            digest.update(int(token_id).to_bytes(8, "little"))
        room = 16 - len(result.generated_token_sample)
        if room > 0:
            result.generated_token_sample.extend(token_ids[:room])
        result.generated_tokens += len(token_ids)
        if completion.finish_reason and finish_index < len(turn_finished):
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
    return result, lateness_values


def compare_baseline(
    sessions: list[SessionResult], baseline: dict[str, Any]
) -> dict[str, Any]:
    keys = ("generated_tokens", "generated_token_sha256")
    expected = {
        int(item["root_chat_id"]): tuple(item[key] for key in keys)
        for item in baseline["sessions"]
    }
    actual = {
        item.root_chat_id: (item.generated_tokens, item.generated_token_sha256)
        for item in sessions
    }
    mismatched = sorted(
        root_id
        for root_id in expected.keys() | actual.keys()
        if expected.get(root_id) != actual.get(root_id)
    )
    return {
        "exact_session_output_match": not mismatched,
        "mismatched_root_chat_ids": mismatched,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    selected = select_sessions(
        load_bailian_records(args.trace),
        request_type=None if args.request_type == "all" else args.request_type,
        min_turns=args.min_turns,
        max_input_length=args.max_input_length or None,
        max_sessions=args.max_sessions or None,
    )
    mixed = args.mode == "mixed"
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=args.model,
            dtype="float16",
            enforce_eager=True,
            seed=args.seed,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            block_size=16,
            attention_backend="TRITON_ATTN",
            worker_extension_cls=(
                "experiments.scripts.run_qwen_bailian_replay."
                "HKVReplayWorkerExtension"
            ),
            kv_cache_hot_idle_threshold_seconds=(
                args.hot_idle_threshold if mixed else None
            ),
            kv_cache_cold_idle_threshold_seconds=(
                args.cold_idle_threshold if mixed else None
            ),
        )
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
    started = time.perf_counter()
    try:
        plan = build_replay_plan(
            selected,
            vocab_size=tokenizer_vocab_size(engine.get_tokenizer()),
            time_scale=args.time_scale,
            seed=args.seed,
        )
        scheduled_duration = max(
            turn.send_at_seconds for turns in plan.values() for turn in turns
        )
        observer = asyncio.create_task(
            observe_hkv(
                engine, stop_observer, args.metrics_interval, observation
            )
        )
        tasks = [
            asyncio.create_task(run_session(engine, turns, started, args.seed))
            for _, turns in sorted(plan.items())
        ]
        completed = await asyncio.wait_for(
            asyncio.gather(*tasks), timeout=args.timeout
        )
        session_results = [item[0] for item in completed]
        lateness = [value for item in completed for value in item[1]]

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
        engine.shutdown()

    elapsed = time.perf_counter() - started
    requests = sum(session.turns for session in session_results)
    generated = sum(session.generated_tokens for session in session_results)
    result: dict[str, Any] = {
        "mode": args.mode,
        "trace": str(args.trace),
        "model": args.model,
        "time_scale": args.time_scale,
        "output_tokens_mode": "one_per_turn_for_prefix_correctness",
        "selected_sessions": len(session_results),
        "selected_requests": requests,
        "scheduled_duration_seconds": scheduled_duration,
        "replay_elapsed_seconds": elapsed,
        "total_final_input_tokens": sum(
            session.final_input_tokens for session in session_results
        ),
        "total_trace_output_tokens": sum(
            session.trace_output_tokens for session in session_results
        ),
        "total_generated_tokens": generated,
        "requests_per_second": requests / elapsed,
        "generated_tokens_per_second": generated / elapsed,
        "send_lateness_seconds": {
            "p50": percentile(lateness, 0.50),
            "p95": percentile(lateness, 0.95),
            "max": max(lateness, default=None),
        },
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

    if args.baseline_json:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        result["baseline_comparison"] = compare_baseline(
            session_results, baseline
        )
        if not result["baseline_comparison"]["exact_session_output_match"]:
            validation_errors.append(
                "mixed output differs for sessions "
                f"{result['baseline_comparison']['mismatched_root_chat_ids']}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all-hot", "mixed"), required=True)
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
    parser.add_argument("--hot-idle-threshold", type=float, default=1.0)
    parser.add_argument("--cold-idle-threshold", type=float, default=3600.0)
    parser.add_argument("--warm-pool-blocks", type=int, default=128)
    parser.add_argument("--metrics-interval", type=float, default=0.05)
    parser.add_argument("--cleanup-timeout", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.mode == "mixed" and args.baseline_json is None:
        parser.error("--baseline-json is required in mixed mode")
    if args.time_scale <= 0 or args.min_turns <= 0:
        parser.error("time scale and minimum turns must be positive")
    if args.max_input_length < 0 or args.max_sessions < 0:
        parser.error("session limits must be non-negative")
    if args.warm_pool_blocks <= 0:
        parser.error("--warm-pool-blocks must be positive")
    return args


def main() -> None:
    args = parse_args()
    mixed = args.mode == "mixed"
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
            "mode",
            "selected_sessions",
            "selected_requests",
            "replay_elapsed_seconds",
            "total_generated_tokens",
            "requests_per_second",
            "send_lateness_seconds",
            "hkv_observation",
            "validation",
        )
    }
    if "baseline_comparison" in result:
        printable["baseline_comparison"] = result["baseline_comparison"]
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()