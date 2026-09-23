"""Resume-quality GSM8K experiment for hierarchical KV cache."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from experiments.scripts.hkv_gsm8k import (
    BLOCK_SIZE,
    DEFAULT_BALLAST_GENERATION_TOKENS,
    DEFAULT_BALLAST_SAFETY_MARGIN_BLOCKS,
    RESULT_SCHEMA_VERSION,
    TURN2_TEXT,
    Gsm8kSubset,
    PromptPlan,
    SessionQuality,
    build_capacity_report,
    build_prompt_plans,
    complete_blocks_are_warm,
    exact_length_token_ids,
    fingerprint_differences,
    load_gsm8k_subset,
    mixed_exclusion_reason,
    pair_quality,
    prompt_fingerprint_fields,
    quality_summary,
    score_generated_text,
    sha256_json,
)
from experiments.scripts.run_qwen_bailian_replay import (
    ATTENTION_BACKEND,
    MODEL_DTYPE,
    TOTAL_KV_BUDGET_ENV,
    TOPOLOGY_ASSUMPTIONS,
    HKVReplayWorkerExtension,
    build_engine_args_kwargs,
    build_runtime_memory_accounting,
    derive_persistent_kv_budget,
    drain_warm_residency,
    get_git_metadata,
    inspect_worker,
    update_observation,
    utc_now,
    validate_runtime_memory_accounting,
)

SUPPORTED_BUDGET_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_DATASET_DIR = Path("experiments/tests/fixtures/gsm8k_tiny")
MIXED_READ_STATS_ENV = "HKV_DEBUG_MIXED_READ_STATS"
RESUME_PROGRESS_DEBUG_ENV = "HKV_DEBUG_RESUME_PROGRESS"
# Keep vLLM async scheduling so Mixed ballast prefill can complete. Quality
# comparison serializes turn-2 resume instead of disabling async scheduling.
GSM8K_ASYNC_SCHEDULING = True
GSM8K_SEQUENTIAL_RESUME = True


class MixedDemotionBarrierTimeout(TimeoutError):
    """Raised when Mixed WARM confirmation does not complete in time."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class HKVGsm8kWorkerExtension(HKVReplayWorkerExtension):
    """Aggregate replay inspect plus per-request mixed-read evidence."""

    def inspect_hkv_request(self, request_id: str) -> dict[str, Any]:
        import torch

        runner = self.model_runner
        manager = getattr(runner, "hkv_warm_migration_manager", None)
        allocator = manager.allocator if manager is not None else None
        residency_items = (
            () if manager is None else sorted(manager.warm_residency.items())
        )
        index = runner.req_states.req_id_to_index.get(request_id)
        blocks, computed = [], None
        if index is not None:
            count = int(runner.block_tables.num_blocks.np[0, index])
            table = runner.block_tables.block_tables[0].gpu
            blocks = table[index, :count].cpu().tolist()
            computed = int(runner.req_states.num_computed_tokens_np[index])
        residency = [
            {
                "key": list(key),
                "warm_slot_id": entry.warm_slot_id,
                "temporary_shadow_hot_block_id": (
                    entry.temporary_shadow_hot_block_id
                ),
            }
            for key, entry in residency_items
            if key[0] == request_id
        ]
        allocator_matches = allocator is None or all(
            allocator.lookup(key) == entry.warm_slot_id
            for key, entry in residency_items
        )
        ownership_count_matches = allocator is None or (
            allocator.num_owned_slots == len(residency_items)
        )
        stats = getattr(runner, "hkv_mixed_read_stats", None) or {}
        return {
            "request_id": request_id,
            "num_computed_tokens": computed,
            "block_ids": blocks,
            "warm_residency": residency,
            "allocator_matches_residency": allocator_matches,
            "ownership_count_matches": ownership_count_matches,
            "mixed_read_stats": dict(stats.get(request_id, {})),
            "max_gpu_allocated_bytes": (
                torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else 0
            ),
        }

    def inspect_hkv_mixed_read_stats(self) -> dict[str, Any]:
        stats = getattr(self.model_runner, "hkv_mixed_read_stats", None) or {}
        return {
            "requests": {
                request_id: dict(payload)
                for request_id, payload in stats.items()
            }
        }


@dataclass(slots=True)
class SequentialResumeTracker:
    """Record turn-2 start/finish so overlapping resumes are detectable."""

    active: set[str] = field(default_factory=set)
    order: list[str] = field(default_factory=list)
    overlap_events: list[tuple[str, ...]] = field(default_factory=list)
    max_active: int = 0

    def start_turn2(self, session_id: str) -> None:
        if self.active:
            self.overlap_events.append(tuple(sorted(self.active | {session_id})))
        self.active.add(session_id)
        self.max_active = max(self.max_active, len(self.active))
        self.order.append(session_id)

    def finish_turn2(self, session_id: str) -> None:
        self.active.discard(session_id)


@dataclass(slots=True)
class EvaluatedSessionState:
    plan: PromptPlan
    turn1_done: asyncio.Event = field(default_factory=asyncio.Event)
    resume_event: asyncio.Event = field(default_factory=asyncio.Event)
    turn2_start_event: asyncio.Event = field(default_factory=asyncio.Event)
    turn2_done: asyncio.Event = field(default_factory=asyncio.Event)
    turn2_text: str = ""
    turn2_token_ids: list[int] = field(default_factory=list)
    completed_turns: int = 0
    pre_resume_inspect: dict[str, Any] | None = None
    error: str | None = None


def apply_hkv_environment(args: argparse.Namespace) -> None:
    mixed = args.kv_mode == "mixed"
    os.environ.update(
        {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
            "HKV_ENABLE_PHYSICAL_TIERS": "1" if mixed else "0",
            "HKV_WARM_POOL_BLOCKS": (
                str(args.warm_pool_blocks) if mixed else "0"
            ),
            "HKV_DEBUG_DEMOTE_ONE_BLOCK": "0",
            RESUME_PROGRESS_DEBUG_ENV: "0",
            "HKV_ENABLE_MULTI_BLOCK_WARM_MIGRATION": "1" if mixed else "0",
            "HKV_DEBUG_MIXED_READ": "1" if mixed else "0",
            MIXED_READ_STATS_ENV: "1",
        }
    )
    if args.total_kv_budget_bytes is None:
        os.environ.pop(TOTAL_KV_BUDGET_ENV, None)
    else:
        os.environ[TOTAL_KV_BUDGET_ENV] = str(args.total_kv_budget_bytes)


def gsm8k_engine_args_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = build_engine_args_kwargs(args)
    kwargs["worker_extension_cls"] = (
        "experiments.scripts.run_hkv_gsm8k_resume_quality."
        "HKVGsm8kWorkerExtension"
    )
    kwargs["enable_prefix_caching"] = False
    kwargs["enforce_eager"] = True
    kwargs["async_scheduling"] = GSM8K_ASYNC_SCHEDULING
    kwargs["attention_backend"] = ATTENTION_BACKEND
    kwargs["dtype"] = MODEL_DTYPE
    return kwargs


def generation_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "seed": args.seed,
        "output_kind": "delta",
        "turn1_max_tokens": 1,
        "turn1_ignore_eos": True,
        "turn2_max_tokens": args.max_tokens,
        "turn2_ignore_eos": False,
        "enable_thinking": False,
        "pad_history_to_block_size": False,
        "generation_policy": {
            "name": "gsm8k_resume_turn2_scored",
            "description": (
                "Turn 1 generates one discardable token after the natural "
                "GSM8K prompt. Ballast, if any, stays allocated until the "
                "pre-resume barrier succeeds and is cancelled before turn 2. "
                "Only turn 2 is scored."
            ),
        },
    }


def build_experiment_fingerprint(
    args: argparse.Namespace,
    subset: Gsm8kSubset,
    plans: list[PromptPlan],
) -> dict[str, Any]:
    memory_budget = derive_persistent_kv_budget(args)
    fields = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "model": args.model,
        "seed": args.seed,
        "dataset": {
            "train_sha256": subset.train_sha256,
            "test_sha256": subset.test_sha256,
            "start_index": args.start_index,
            "num_questions": args.num_questions,
            "num_shots": args.num_shots,
            "shot_start_index": args.shot_start_index,
            "shot_indices": [shot.index for shot in subset.shots],
        },
        "prompts": prompt_fingerprint_fields(plans),
        "generation_parameters": generation_parameters(args),
        "gpu_memory_configuration": {
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "block_size": BLOCK_SIZE,
            "total_kv_budget_bytes": memory_budget["total_kv_budget_bytes"],
            "max_num_seqs": args.max_num_seqs,
        },
        "num_ballast_sessions": args.num_ballast_sessions,
        "attention_backend": ATTENTION_BACKEND,
        "dtype": MODEL_DTYPE,
        "topology_assumptions": TOPOLOGY_ASSUMPTIONS,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "async_scheduling": GSM8K_ASYNC_SCHEDULING,
        "sequential_resume": GSM8K_SEQUENTIAL_RESUME,
    }
    return {"fields": fields, "sha256": sha256_json(fields)}


def load_tokenizer(model: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


async def inspect_request(engine: Any, request_id: str) -> dict[str, Any]:
    states = await engine.engine_core.collective_rpc_async(
        "inspect_hkv_request",
        args=(request_id,),
    )
    if not states:
        raise RuntimeError(f"no inspect_hkv_request state for {request_id}")
    return states[0]


async def inspect_mixed_read_stats(engine: Any) -> dict[str, Any]:
    states = await engine.engine_core.collective_rpc_async(
        "inspect_hkv_mixed_read_stats"
    )
    merged: dict[str, Any] = {}
    for state in states:
        merged.update(state.get("requests", {}))
    return merged


async def sample_hkv_observation(
    engine: Any,
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Record one worker snapshot without resetting peak occupancy."""
    state = await inspect_worker(engine)
    update_observation(observation, state)
    return state


def _eos_token_ids_from_tokenizer(tokenizer: Any) -> frozenset[int]:
    ids: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.add(int(eos))
    for attr in ("eod_id", "im_end_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            ids.add(int(value))
    return frozenset(ids)


def _turn2_output_is_complete(
    output: Any,
    completion: Any | None,
    eos_token_ids: frozenset[int],
) -> bool:
    """Turn 2 can finish without a follow-up empty finish_reason chunk."""
    if completion is not None and completion.finish_reason:
        return True
    token_ids = list(getattr(completion, "token_ids", None) or [])
    if eos_token_ids and any(int(token_id) in eos_token_ids for token_id in token_ids):
        return True
    return bool(getattr(output, "finished", False))


def _load_sampling_types() -> tuple[Any, Any, Any]:
    from vllm import SamplingParams
    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind

    return SamplingParams, StreamingInput, RequestOutputKind


def record_pre_resume_inspect(
    state: EvaluatedSessionState,
    snapshot: dict[str, Any],
    *,
    kv_mode: str,
    validation_errors: list[str],
) -> dict[str, Any]:
    """Store the inspect snapshot taken immediately before this session resumes."""
    snapshot = dict(snapshot)
    confirmed = _session_warm_confirmed(
        snapshot,
        state.plan.complete_historical_blocks,
        state.plan.session_id,
    )
    mixed = kv_mode == "mixed"
    snapshot["pre_resume_warm_confirmed"] = bool(mixed and confirmed)
    state.pre_resume_inspect = snapshot
    if mixed and not confirmed:
        validation_errors.append(
            f"{state.plan.session_id} did not confirm WARM residency "
            "on every complete historical block before resume"
        )
    if not mixed and (
        snapshot.get("warm_residency")
        or int(
            (snapshot.get("mixed_read_stats") or {}).get("mixed_read_steps")
            or 0
        )
    ):
        validation_errors.append(
            f"{state.plan.session_id} unexpectedly used WARM in all-hot"
        )
    return snapshot


def _task_failure(task: asyncio.Task | None) -> BaseException | None:
    if task is None or not task.done() or task.cancelled():
        return None
    return task.exception()


async def _wait_for_session_event(
    event: asyncio.Event,
    *,
    task: asyncio.Task | None,
    timeout: float,
    session_id: str,
    what: str,
) -> None:
    """Wait for a per-session event, failing immediately if the task died."""
    failure = _task_failure(task)
    if failure is not None:
        raise RuntimeError(
            f"{session_id} task ended before {what}: {type(failure).__name__}: "
            f"{failure}"
        ) from failure
    if task is None:
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                f"sequential turn-2 resume timed out for {session_id} "
                f"waiting for {what}"
            ) from exc
        return

    event_task = asyncio.create_task(event.wait())
    done, pending = await asyncio.wait(
        {event_task, task},
        timeout=timeout,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if event_task in pending:
        event_task.cancel()
        with suppress(asyncio.CancelledError):
            await event_task
    if event.is_set():
        failure = _task_failure(task)
        if failure is not None:
            raise RuntimeError(
                f"{session_id} task failed after {what}: "
                f"{type(failure).__name__}: {failure}"
            ) from failure
        return
    failure = _task_failure(task)
    if failure is not None:
        raise RuntimeError(
            f"{session_id} task ended before {what}: {type(failure).__name__}: "
            f"{failure}"
        ) from failure
    if task.done():
        raise RuntimeError(
            f"{session_id} task ended before {what} without completing turn 2"
        )
    raise TimeoutError(
        f"sequential turn-2 resume timed out for {session_id} waiting for {what}"
        f" (task_done={task.done()}, cancelled={task.cancelled()})"
    )


async def resume_turn2_sequentially(
    states: list[EvaluatedSessionState],
    *,
    inspect_before_resume: Any,
    kv_mode: str,
    timeout: float,
    validation_errors: list[str],
    tasks: Sequence[asyncio.Task] | None = None,
    order: list[str] | None = None,
) -> list[str]:
    """Resume turn 2 one evaluated session at a time in plan order."""
    if order is None:
        order = []
    task_by_id = {}
    if tasks is not None:
        for state, task in zip(states, tasks):
            task_by_id[state.plan.session_id] = task
    for state in states:
        session_id = state.plan.session_id
        task = task_by_id.get(session_id)
        snapshot = await inspect_before_resume(state)
        record_pre_resume_inspect(
            state,
            snapshot,
            kv_mode=kv_mode,
            validation_errors=validation_errors,
        )
        order.append(session_id)
        failure = _task_failure(task)
        if failure is not None:
            raise RuntimeError(
                f"{session_id} died before turn-2 resume: "
                f"{type(failure).__name__}: {failure}"
            ) from failure
        if task is not None and task.done():
            raise RuntimeError(
                f"{session_id} session coroutine is not alive when turn-2 "
                "resume was requested"
            )
        state.resume_event.set()
        await _wait_for_session_event(
            state.turn2_start_event,
            task=task,
            timeout=timeout,
            session_id=session_id,
            what="turn2_start_event",
        )
        await _wait_for_session_event(
            state.turn2_done,
            task=task,
            timeout=timeout,
            session_id=session_id,
            what="turn2_done",
        )
        if state.completed_turns < 2:
            raise RuntimeError(
                f"{session_id} released turn2_done without generating turn 2 "
                f"(completed_turns={state.completed_turns}, error={state.error})"
            )
    return order


async def run_evaluated_session(
    engine: Any,
    state: EvaluatedSessionState,
    *,
    seed: int,
    max_tokens: int,
    retain_event: asyncio.Event,
    tracker: SequentialResumeTracker | None = None,
    eos_token_ids: frozenset[int] | None = None,
) -> None:
    SamplingParams, StreamingInput, RequestOutputKind = _load_sampling_types()
    plan = state.plan
    eos_token_ids = eos_token_ids or frozenset()
    common = {
        "temperature": 0.0,
        "seed": seed,
        "output_kind": RequestOutputKind.DELTA,
    }
    turn1_params = SamplingParams(max_tokens=1, ignore_eos=True, **common)
    turn2_params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=False,
        **common,
    )
    base_params = SamplingParams(
        max_tokens=max(max_tokens, 1),
        ignore_eos=False,
        **common,
    )

    async def inputs():
        yield StreamingInput(
            {"prompt_token_ids": list(plan.turn1_token_ids)},
            turn1_params,
        )
        await state.resume_event.wait()
        if tracker is not None:
            tracker.start_turn2(plan.session_id)
        state.turn2_start_event.set()
        yield StreamingInput(
            {"prompt_token_ids": list(plan.turn2_token_ids)},
            turn2_params,
        )
        # Keep the input stream open until every evaluated session has finished
        # turn 2. Do not wait on turn2_done here: handle_inputs() pulls the
        # next chunk immediately, and that wait would sit on the input-stream
        # task rather than in the generate() consumer.
        await retain_event.wait()

    try:
        async for output in engine.generate(
            inputs(),
            base_params,
            plan.session_id,
        ):
            completion = output.outputs[0] if output.outputs else None
            if not state.turn2_start_event.is_set():
                if completion is not None and completion.finish_reason:
                    state.completed_turns = 1
                    state.turn1_done.set()
                continue
            if completion is not None:
                state.turn2_text += completion.text or ""
                state.turn2_token_ids.extend(list(completion.token_ids or []))
            if _turn2_output_is_complete(output, completion, eos_token_ids):
                state.completed_turns = 2
                if tracker is not None:
                    tracker.finish_turn2(plan.session_id)
                state.turn2_done.set()
    except asyncio.CancelledError:
        state.error = "cancelled"
        raise
    except Exception as exc:
        state.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state.turn1_done.set()
        state.turn2_start_event.set()
        if tracker is not None and plan.session_id in tracker.active:
            tracker.finish_turn2(plan.session_id)
        state.turn2_done.set()


async def run_ballast_session(
    engine: Any,
    request_id: str,
    prompt_token_ids: Sequence[int],
    seed: int,
    max_tokens: int,
    hold_event: asyncio.Event,
) -> None:
    SamplingParams, StreamingInput, RequestOutputKind = _load_sampling_types()
    params = SamplingParams(
        temperature=0.0,
        seed=seed,
        max_tokens=max(max_tokens, 1),
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )

    async def inputs():
        yield StreamingInput(
            {"prompt_token_ids": list(prompt_token_ids)},
            params,
        )
        await hold_event.wait()

    try:
        async for _ in engine.generate(inputs(), params, request_id):
            pass
    except asyncio.CancelledError:
        raise


async def wait_for_turn1(states: list[EvaluatedSessionState], timeout: float) -> None:
    await asyncio.wait_for(
        asyncio.gather(*(state.turn1_done.wait() for state in states)),
        timeout=timeout,
    )


def _session_warm_confirmed(
    inspect_state: dict[str, Any],
    complete_historical_blocks: int,
    request_id: str | None = None,
) -> bool:
    return complete_blocks_are_warm(
        complete_historical_blocks=complete_historical_blocks,
        residency=inspect_state.get("warm_residency") or [],
        block_ids=inspect_state.get("block_ids"),
        request_id=request_id or inspect_state.get("request_id"),
        kv_group=0,
    )


def _task_is_running(task: asyncio.Task | None) -> bool:
    return task is not None and not task.done()


def mixed_demotion_barrier_diagnostics(
    *,
    ballast_plan: dict[str, Any],
    states: list[EvaluatedSessionState],
    latest: dict[str, dict[str, Any]],
    worker_state: dict[str, Any] | None,
    ballast_ids: list[str],
    ballast_tasks: list[asyncio.Task],
    ballast_snapshots: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tasks_by_id = {
        request_id: task
        for request_id, task in zip(ballast_ids, ballast_tasks)
    }
    evaluated = []
    for state in states:
        snapshot = latest.get(state.plan.session_id) or {}
        evaluated.append(
            {
                "session_id": state.plan.session_id,
                "complete_historical_blocks": (
                    state.plan.complete_historical_blocks
                ),
                "warm_confirmed": _session_warm_confirmed(
                    snapshot,
                    state.plan.complete_historical_blocks,
                    state.plan.session_id,
                ),
                "block_ids": snapshot.get("block_ids"),
                "warm_residency": snapshot.get("warm_residency"),
                "allocator_matches_residency": snapshot.get(
                    "allocator_matches_residency"
                ),
                "ownership_count_matches": snapshot.get(
                    "ownership_count_matches"
                ),
                "num_computed_tokens": snapshot.get("num_computed_tokens"),
            }
        )
    ballast_status = []
    for request_id in ballast_ids:
        snapshot = (ballast_snapshots or {}).get(request_id) or {}
        task = tasks_by_id.get(request_id)
        ballast_status.append(
            {
                "request_id": request_id,
                "running": _task_is_running(task),
                "task_done": task.done() if task is not None else None,
                "task_exception": (
                    repr(task.exception())
                    if task is not None and task.done() and not task.cancelled()
                    else None
                ),
                "block_ids": snapshot.get("block_ids"),
                "warm_residency": snapshot.get("warm_residency"),
                "num_computed_tokens": snapshot.get("num_computed_tokens"),
            }
        )
    observed_hot_blocks = sum(
        len(item.get("block_ids") or [])
        for item in list(latest.values()) + list((ballast_snapshots or {}).values())
    )
    return {
        "required_ballast_blocks": ballast_plan.get("required_ballast_blocks"),
        "planned_blocks_per_request": ballast_plan.get(
            "planned_blocks_per_request"
        ),
        "prompt_tokens_per_request": ballast_plan.get(
            "prompt_tokens_per_request"
        ),
        "generation_tokens_per_request": ballast_plan.get(
            "generation_tokens_per_request"
        ),
        "total_planned_ballast_blocks": ballast_plan.get(
            "total_planned_ballast_blocks"
        ),
        "safety_margin_blocks": ballast_plan.get("safety_margin_blocks"),
        "estimated_peak_hot_blocks": ballast_plan.get(
            "estimated_peak_hot_blocks"
        ),
        "observed_hot_block_ids": observed_hot_blocks,
        "observed_warm_occupancy": {
            "warm_blocks": (worker_state or {}).get("warm_blocks"),
            "warm_requests": (worker_state or {}).get("warm_requests"),
            "owned_warm_slots": (worker_state or {}).get("owned_warm_slots"),
            "num_gpu_blocks": (worker_state or {}).get("num_gpu_blocks"),
        },
        "allocator_consistent": (worker_state or {}).get(
            "allocator_consistent"
        ),
        "evaluated_sessions": evaluated,
        "ballast_tasks": ballast_status,
    }


async def _inspect_optional(engine: Any, request_id: str) -> dict[str, Any]:
    try:
        return await inspect_request(engine, request_id)
    except Exception as exc:
        return {"request_id": request_id, "inspect_error": repr(exc)}


async def wait_mixed_demotion_barrier(
    engine: Any,
    states: list[EvaluatedSessionState],
    timeout: float,
    poll_interval: float,
    *,
    ballast_plan: dict[str, Any],
    ballast_ids: list[str],
    ballast_tasks: list[asyncio.Task],
) -> dict[str, dict[str, Any]]:
    deadline = time.perf_counter() + timeout
    latest: dict[str, dict[str, Any]] = {}
    failure = "timed out"
    while time.perf_counter() < deadline:
        ready = True
        for state in states:
            snapshot = await inspect_request(engine, state.plan.session_id)
            latest[state.plan.session_id] = snapshot
            if not snapshot.get("allocator_matches_residency"):
                ready = False
                continue
            if not snapshot.get("ownership_count_matches"):
                ready = False
                continue
            if not _session_warm_confirmed(
                snapshot,
                state.plan.complete_historical_blocks,
                state.plan.session_id,
            ):
                ready = False
        if ready:
            return latest
        dead_ballast = [
            request_id
            for request_id, task in zip(ballast_ids, ballast_tasks)
            if task.done()
        ]
        if dead_ballast:
            failure = (
                "ballast tasks finished before WARM confirmation: "
                + ", ".join(dead_ballast)
            )
            break
        await asyncio.sleep(poll_interval)

    worker_state: dict[str, Any] | None
    try:
        worker_state = await inspect_worker(engine)
    except Exception as exc:
        worker_state = {"inspect_error": repr(exc)}
    ballast_snapshots = {
        request_id: await _inspect_optional(engine, request_id)
        for request_id in ballast_ids
    }
    diagnostics = mixed_demotion_barrier_diagnostics(
        ballast_plan=ballast_plan,
        states=states,
        latest=latest,
        worker_state=worker_state,
        ballast_ids=ballast_ids,
        ballast_tasks=ballast_tasks,
        ballast_snapshots=ballast_snapshots,
    )
    raise MixedDemotionBarrierTimeout(
        "Mixed demotion barrier failed "
        f"({failure}) before every complete historical block was confirmed "
        "WARM: " + json.dumps(diagnostics, default=str),
        diagnostics,
    )


async def abort_ballast(
    engine: Any,
    request_ids: list[str],
    tasks: list[asyncio.Task],
) -> None:
    if request_ids:
        await engine.abort(request_ids)
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def wait_ballast_released(
    engine: Any,
    ballast_ids: set[str],
    timeout: float,
) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout
    latest = await inspect_worker(engine)
    while time.perf_counter() < deadline:
        latest = await inspect_worker(engine)
        if latest.get("allocator_consistent") is False:
            await asyncio.sleep(0.05)
            continue
        occupancy_ids: set[str] = set()
        for request_id in ballast_ids:
            snapshot = await inspect_request(engine, request_id)
            if snapshot.get("warm_residency") or snapshot.get("block_ids"):
                occupancy_ids.add(request_id)
        if not occupancy_ids:
            return latest
        await asyncio.sleep(0.05)
    return latest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GSM8K two-turn resume-quality comparison for All-HOT vs Mixed HKV."
        )
    )
    parser.add_argument(
        "--kv-mode",
        choices=("all-hot", "mixed"),
        required=True,
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--model", default=SUPPORTED_BUDGET_MODEL)
    parser.add_argument("--num-questions", type=int, default=4)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--shot-start-index", type=int, default=0)
    parser.add_argument("--num-ballast-sessions", type=int, default=8)
    parser.add_argument(
        "--ballast-safety-margin-blocks",
        type=int,
        default=DEFAULT_BALLAST_SAFETY_MARGIN_BLOCKS,
        help=(
            "Extra HOT blocks above ceil(demotion_start * usable_hot). "
            f"Default {DEFAULT_BALLAST_SAFETY_MARGIN_BLOCKS}."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--total-kv-budget-bytes", type=int)
    parser.add_argument("--demotion-start-utilization", type=float)
    parser.add_argument("--demotion-stop-utilization", type=float)
    parser.add_argument("--warm-pool-blocks", type=int)
    parser.add_argument("--metrics-interval", type=float, default=0.05)
    parser.add_argument("--demotion-timeout", type=float, default=30.0)
    parser.add_argument("--cleanup-timeout", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Tokenize prompts, report capacity, and exit without starting vLLM.",
    )
    args = parser.parse_args(argv)

    if args.num_questions <= 0 or args.num_shots < 0:
        parser.error("question and shot counts are invalid")
    if args.max_tokens <= 0 or args.seed < 0:
        parser.error("max tokens must be positive and seed non-negative")
    if args.max_model_len <= 0 or args.max_num_seqs <= 0:
        parser.error("model length and max_num_seqs must be positive")
    if args.num_ballast_sessions < 0:
        parser.error("ballast count must be non-negative")
    if args.ballast_safety_margin_blocks < 0:
        parser.error("ballast safety margin must be non-negative")
    if not args.preflight and args.result_json is None:
        parser.error("--result-json is required unless --preflight is set")

    mixed = args.kv_mode == "mixed"
    thresholds = (
        args.demotion_start_utilization,
        args.demotion_stop_utilization,
    )
    if mixed and any(value is None for value in thresholds):
        parser.error(
            "mixed mode requires --demotion-start-utilization and "
            "--demotion-stop-utilization"
        )
    if not mixed and any(value is not None for value in thresholds):
        parser.error("pressure thresholds are valid only in mixed mode")
    if mixed:
        start, stop = thresholds
        assert start is not None and stop is not None
        if not 0.0 <= stop < start <= 1.0:
            parser.error("pressure thresholds must satisfy 0 <= stop < start <= 1")
        if args.warm_pool_blocks is None or args.warm_pool_blocks <= 0:
            parser.error("mixed mode requires positive --warm-pool-blocks")
        if args.num_ballast_sessions <= 0:
            parser.error("mixed mode requires a positive --num-ballast-sessions")
        if args.baseline_json is None and not args.preflight:
            parser.error("--baseline-json is required in mixed mode")
    elif args.warm_pool_blocks is not None:
        parser.error("--warm-pool-blocks is valid only in mixed mode")
    else:
        args.warm_pool_blocks = 0

    if args.total_kv_budget_bytes is not None:
        try:
            derive_persistent_kv_budget(args)
        except ValueError as exc:
            parser.error(str(exc))
    elif not args.preflight:
        parser.error("--total-kv-budget-bytes is required for an engine run")
    return args


def prepare_workload(
    args: argparse.Namespace,
    tokenizer: Any,
) -> tuple[Gsm8kSubset, list[PromptPlan], dict[str, Any], dict[str, Any]]:
    subset = load_gsm8k_subset(
        args.dataset_dir,
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        start_index=args.start_index,
        shot_start_index=args.shot_start_index,
    )
    plans = build_prompt_plans(subset, tokenizer)
    memory_budget = derive_persistent_kv_budget(args)
    capacity = build_capacity_report(
        plans,
        kv_mode=args.kv_mode,
        memory_budget=memory_budget,
        warm_pool_blocks=args.warm_pool_blocks or 0,
        demotion_start_utilization=args.demotion_start_utilization,
        num_ballast_sessions=args.num_ballast_sessions,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        safety_margin_blocks=args.ballast_safety_margin_blocks,
        ballast_generation_tokens=DEFAULT_BALLAST_GENERATION_TOKENS,
    )
    return subset, plans, memory_budget, capacity


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _quality_from_state(
    state: EvaluatedSessionState,
    *,
    kv_mode: str,
    mixed_stats: dict[str, Any],
) -> SessionQuality:
    extracted, parse_failure, exact_match = score_generated_text(
        state.turn2_text,
        state.plan.gold_value,
    )
    inspect_state = state.pre_resume_inspect or {}
    stats = mixed_stats.get(state.plan.session_id) or inspect_state.get(
        "mixed_read_stats"
    ) or {}
    pre_stats = inspect_state.get("mixed_read_stats") or {}
    pre_steps = int(pre_stats.get("mixed_read_steps") or 0)
    post_steps = int(stats.get("mixed_read_steps") or 0)
    had_warm = post_steps > pre_steps
    pre_resume = bool(inspect_state.get("pre_resume_warm_confirmed"))
    if kv_mode == "mixed":
        exclusion = mixed_exclusion_reason(
            complete_historical_blocks=state.plan.complete_historical_blocks,
            pre_resume_warm_confirmed=pre_resume,
            attention_had_warm_slots=had_warm,
        )
    else:
        exclusion = None
    return SessionQuality(
        session_id=state.plan.session_id,
        question_index=state.plan.question_index,
        gold_value=state.plan.gold_value,
        final_text=state.turn2_text,
        extracted_value=extracted,
        parse_failure=parse_failure,
        exact_match=exact_match,
        complete_historical_blocks=state.plan.complete_historical_blocks,
        pre_resume_warm_confirmed=pre_resume,
        attention_had_warm_slots=had_warm,
        mixed_read_steps=int(stats.get("mixed_read_steps") or 0),
        warm_logical_blocks_observed=int(
            stats.get("warm_logical_blocks_observed") or 0
        ),
        exclusion_reason=exclusion,
        generated_token_ids=list(state.turn2_token_ids),
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = load_tokenizer(args.model)
    subset, plans, memory_budget, capacity = prepare_workload(args, tokenizer)
    fingerprint = build_experiment_fingerprint(args, subset, plans)
    preflight_payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kv_mode": args.kv_mode,
        "preflight": True,
        "dataset": {
            "train_sha256": subset.train_sha256,
            "test_sha256": subset.test_sha256,
            "num_questions": args.num_questions,
            "num_shots": args.num_shots,
        },
        "capacity": capacity,
        "experiment_fingerprint": fingerprint,
    }
    if args.preflight:
        write_json(args.result_json, preflight_payload)
        if not capacity["feasible"]:
            raise ValueError(
                "preflight found an infeasible configuration: "
                + "; ".join(capacity["refusal_reasons"])
            )
        return preflight_payload
    if not capacity["feasible"]:
        write_json(args.result_json, preflight_payload)
        raise ValueError(
            "refusing to start the engine: "
            + "; ".join(capacity["refusal_reasons"])
        )

    apply_hkv_environment(args)
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(**gsm8k_engine_args_kwargs(args))
    )
    mixed = args.kv_mode == "mixed"
    retain_event = asyncio.Event()
    ballast_hold_event = asyncio.Event()
    sequential_tracker = SequentialResumeTracker()
    sequential_resume_order: list[str] = []
    states = [EvaluatedSessionState(plan=plan) for plan in plans]
    eos_token_ids = _eos_token_ids_from_tokenizer(tokenizer)
    ballast_plan = capacity["ballast_plan"]
    ballast_specs = [
        item
        for item in ballast_plan["requests"]
        if item["planned_blocks"] > 0
    ]
    ballast_ids = [item["request_id"] for item in ballast_specs]
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
    validation_errors: list[str] = []
    evaluated_tasks: list[asyncio.Task] = []
    ballast_tasks: list[asyncio.Task] = []
    baseline = None
    if args.baseline_json:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    try:
        evaluated_tasks = [
            asyncio.create_task(
                run_evaluated_session(
                    engine,
                    state,
                    seed=args.seed,
                    max_tokens=args.max_tokens,
                    retain_event=retain_event,
                    tracker=sequential_tracker,
                    eos_token_ids=eos_token_ids,
                )
            )
            for state in states
        ]
        await wait_for_turn1(states, args.timeout)
        ballast_tasks = [
            asyncio.create_task(
                run_ballast_session(
                    engine,
                    item["request_id"],
                    exact_length_token_ids(
                        item["prompt_tokens"],
                        fill_token_id=ballast_plan["fill_token_id"],
                    ),
                    args.seed,
                    item["generation_tokens"],
                    ballast_hold_event,
                )
            )
            for item in ballast_specs
        ]
        if mixed:
            await wait_mixed_demotion_barrier(
                engine,
                states,
                timeout=args.demotion_timeout,
                poll_interval=args.metrics_interval,
                ballast_plan=ballast_plan,
                ballast_ids=ballast_ids,
                ballast_tasks=ballast_tasks,
            )
        else:
            await asyncio.sleep(min(args.metrics_interval, 0.1))
        await sample_hkv_observation(engine, observation)
        await abort_ballast(engine, ballast_ids, ballast_tasks)
        ballast_tasks = []
        released = await wait_ballast_released(
            engine,
            set(ballast_ids),
            args.cleanup_timeout,
        )
        update_observation(observation, released)
        if mixed and not released.get("allocator_consistent", True):
            validation_errors.append(
                "allocator became inconsistent after ballast release"
            )

        async def inspect_before_resume(state: EvaluatedSessionState) -> dict[str, Any]:
            return await inspect_request(engine, state.plan.session_id)

        sequential_resume_order = await resume_turn2_sequentially(
            states,
            inspect_before_resume=inspect_before_resume,
            kv_mode=args.kv_mode,
            timeout=args.timeout,
            validation_errors=validation_errors,
            tasks=evaluated_tasks,
            order=sequential_resume_order,
        )
        if sequential_tracker.overlap_events or sequential_tracker.max_active > 1:
            validation_errors.append(
                "turn-2 resumes overlapped: "
                + repr(list(sequential_tracker.overlap_events))
            )
        retain_event.set()
        await asyncio.wait_for(
            asyncio.gather(*evaluated_tasks),
            timeout=args.timeout,
        )
        mixed_stats = await inspect_mixed_read_stats(engine)
        await sample_hkv_observation(engine, observation)
        final_state = await drain_warm_residency(
            engine,
            observation,
            args.cleanup_timeout,
        )
    except Exception as exc:
        validation_errors.append(f"{type(exc).__name__}: {exc}")
        mixed_stats = {}
        final_state = None
        for state in states:
            state.resume_event.set()
            state.turn2_start_event.set()
            state.turn2_done.set()
        retain_event.set()
        for task in evaluated_tasks + ballast_tasks:
            if not task.done():
                task.cancel()
        if evaluated_tasks or ballast_tasks:
            await asyncio.gather(
                *evaluated_tasks,
                *ballast_tasks,
                return_exceptions=True,
            )
        try:
            mixed_stats = await inspect_mixed_read_stats(engine)
        except Exception:
            mixed_stats = {}
        try:
            final_state = await inspect_worker(engine)
            observation["allocator_consistent"] &= bool(
                final_state.get("allocator_consistent", True)
            )
        except Exception:
            final_state = None
    finally:
        engine.shutdown()

    qualities = [
        _quality_from_state(state, kv_mode=args.kv_mode, mixed_stats=mixed_stats)
        for state in states
    ]
    summary = quality_summary(qualities, kv_mode=args.kv_mode)
    runtime_source = final_state if final_state is not None else observation
    runtime_memory = None
    if runtime_source and "actual_persistent_kv_bytes" in runtime_source:
        runtime_memory = build_runtime_memory_accounting(
            memory_budget,
            runtime_source,
        )
        validation_errors.extend(
            validate_runtime_memory_accounting(memory_budget, runtime_memory)
        )
    if mixed and summary["n_eligible"] == 0:
        validation_errors.append(
            "no Mixed session was eligible: complete historical WARM before "
            "resume and a resumed mixed-read step are both required"
        )
    if observation.get("allocator_consistent") is False:
        validation_errors.append("WARM allocator ownership became inconsistent")
    if observation.get("cleanup_complete") is False:
        validation_errors.append("WARM residency was not released after the run")

    sessions_json = []
    for state, quality in zip(states, qualities):
        payload = asdict(quality)
        payload["turn1_token_count"] = state.plan.turn1_token_count
        payload["hkv_before_resume"] = state.pre_resume_inspect
        payload["completed_turns"] = state.completed_turns
        payload["error"] = state.error
        sessions_json.append(payload)

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kv_mode": args.kv_mode,
        "model": args.model,
        "seed": args.seed,
        "async_scheduling": GSM8K_ASYNC_SCHEDULING,
        "sequential_resume": GSM8K_SEQUENTIAL_RESUME,
        "sequential_resume_order": sequential_resume_order,
        "dataset": {
            "name": "gsm8k",
            "source": "local-jsonl",
            "dataset_dir": str(args.dataset_dir),
            "train_sha256": subset.train_sha256,
            "test_sha256": subset.test_sha256,
            "start_index": args.start_index,
            "num_questions": args.num_questions,
            "num_shots": args.num_shots,
            "shot_start_index": args.shot_start_index,
            "shot_indices": [shot.index for shot in subset.shots],
        },
        "generation_parameters": generation_parameters(args),
        "engine_configuration": {
            "async_scheduling": GSM8K_ASYNC_SCHEDULING,
            "sequential_resume": GSM8K_SEQUENTIAL_RESUME,
            "enable_prefix_caching": False,
            "enforce_eager": True,
        },
        "experiment_fingerprint": fingerprint,
        "capacity": capacity,
        "persistent_kv_memory_budget": memory_budget,
        "runtime_persistent_kv_memory": runtime_memory,
        "hkv_observation": observation,
        "quality": summary,
        "git": get_git_metadata(),
        "timing": {"end": utc_now()},
        "sessions": sessions_json,
        "validation": {
            "passed": not validation_errors,
            "errors": validation_errors,
            "full_validation_performed": True,
        },
    }
    if baseline is not None:
        baseline_fields = baseline.get("experiment_fingerprint", {}).get(
            "fields"
        )
        diffs = fingerprint_differences(
            fingerprint["fields"],
            baseline_fields or {},
        )
        if diffs:
            validation_errors.append(
                "baseline experiment fingerprint differs: " + ", ".join(diffs)
            )
            result["validation"]["errors"] = validation_errors
            result["validation"]["passed"] = False
        else:
            baseline_sessions = [
                SessionQuality(
                    session_id=item["session_id"],
                    question_index=item["question_index"],
                    gold_value=item["gold_value"],
                    final_text=item.get("final_text", ""),
                    extracted_value=item["extracted_value"],
                    parse_failure=item["parse_failure"],
                    exact_match=item["exact_match"],
                    complete_historical_blocks=item[
                        "complete_historical_blocks"
                    ],
                    pre_resume_warm_confirmed=item[
                        "pre_resume_warm_confirmed"
                    ],
                    attention_had_warm_slots=item["attention_had_warm_slots"],
                    mixed_read_steps=item.get("mixed_read_steps", 0),
                    warm_logical_blocks_observed=item.get(
                        "warm_logical_blocks_observed", 0
                    ),
                    exclusion_reason=item.get("exclusion_reason"),
                    generated_token_ids=item.get("generated_token_ids") or [],
                )
                for item in baseline.get("sessions", [])
            ]
            result["paired_comparison"] = pair_quality(
                baseline_sessions,
                qualities,
            )
    write_json(args.result_json, result)
    if validation_errors:
        raise AssertionError("; ".join(validation_errors))
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = asyncio.run(run(args))
    printable = {
        key: result[key]
        for key in (
            "schema_version",
            "kv_mode",
            "capacity",
            "quality",
            "paired_comparison",
            "validation",
            "hkv_observation",
        )
        if key in result
    }
    print(json.dumps(printable, indent=2))
    if args.preflight:
        raise SystemExit(0 if result["capacity"]["feasible"] else 2)
    raise SystemExit(0 if result["validation"]["passed"] else 1)


if __name__ == "__main__":
    main()
