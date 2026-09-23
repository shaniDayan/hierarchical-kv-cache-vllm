"""Deterministic CPU-only characterization of the Qwen-Bailian workload.

This script is the source of truth for Phase 1 of the final evaluation. It
answers what the trace contains, how retained sessions overlap, and which
replay sizes should pressure a 4 GiB persistent KV budget under the Mixed
HOT/WARM demotion policy. It has no GPU dependency and does not run vLLM.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from experiments.scripts.qwen_bailian_trace import BLOCK_SIZE, BailianRecord
from experiments.scripts.run_qwen_bailian_replay import (
    DEFAULT_MAX_NUM_SEQS,
    KV_MEMORY_FORMULA_VERSION,
    SUPPORTED_BUDGET_MODEL,
    derive_persistent_kv_budget,
    hot_bytes_per_block,
    percentile,
    selection_metadata,
    sha256_json,
    trace_identifier,
    warm_bytes_per_slot,
)

WORKLOAD_STATISTICS_SCHEMA_VERSION = "1.1"
ALL_HOT_CAPACITY_BLOCKS = 2340
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_TOTAL_KV_BUDGET_BYTES = 4 * 1024**3
DEFAULT_SESSION_COUNTS = [25, 50, 100, 250, 542, 750, 1000, 1500]
REQUIRED_PREFIX_SESSION_COUNTS = (250, 542, 750, 1000, 1500)
DEFAULT_DEMOTION_POLICIES = (
    (0.90, 0.75),
    (0.80, 0.65),
    (0.70, 0.55),
)
PRIMARY_START_UTILIZATION = 0.80
PRIMARY_STOP_UTILIZATION = 0.65
PRIMARY_WARM_POOL_BLOCKS = 1024
PREFERRED_HIGH_LOAD_PEAK_MIN_BLOCKS = 2600
PREFERRED_HIGH_LOAD_PEAK_MAX_BLOCKS = 2750
BIAS_RELATIVE_THRESHOLD = 0.15
UTILIZATION_EPS = 1e-12
REQUIRED_PREFIX_ELIGIBLE_THRESHOLD = 250
RETURN_AFTER_THRESHOLDS_SECONDS = (
    ("10_seconds", 10.0),
    ("30_seconds", 30.0),
    ("60_seconds", 60.0),
    ("5_minutes", 5.0 * 60.0),
    ("10_minutes", 10.0 * 60.0),
    ("30_minutes", 30.0 * 60.0),
)
REQUIRED_TRACE_KEYS = (
    "chat_id",
    "parent_chat_id",
    "timestamp",
    "input_length",
    "output_length",
    "type",
    "turn",
    "hash_ids",
)
FIGURE_STYLE = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
COLOR_PRIMARY = "#1f4e79"
COLOR_SECONDARY = "#c45911"
COLOR_CAPACITY = "#b22222"
COLOR_START_THRESHOLD = "#e65100"
COLOR_STOP_TARGET = "#2e7d32"
COLOR_MIXED_TOTAL = "#6a1b9a"
SERIES_COLORS = (
    "#1f4e79",
    "#c45911",
    "#2e7d32",
    "#6a1b9a",
    "#00838f",
    "#ef6c00",
    "#5d4037",
    "#c62828",
    "#455a64",
)


@dataclass(frozen=True, slots=True)
class ParsedTrace:
    records: list[BailianRecord]
    rejections: list[dict[str, Any]]
    total_non_empty_rows: int
    blank_lines: int


@dataclass(frozen=True, slots=True)
class ClassifiedSessions:
    complete: dict[int, list[BailianRecord]]
    incomplete: list[dict[str, Any]]
    duplicate_chat_ids: list[int]


@dataclass(frozen=True, slots=True)
class BlockDemand:
    tokens: int
    complete_blocks: int
    incomplete_tail_tokens: int
    incomplete_tail_blocks: int
    hot_blocks: int


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    return value


def format_float_key(value: float) -> str:
    return format(float(value), ".15g")


def tokens_to_block_demand(
    token_count: int,
    *,
    block_size: int = BLOCK_SIZE,
) -> BlockDemand:
    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    complete_blocks = token_count // block_size
    tail_tokens = token_count % block_size
    tail_blocks = 1 if tail_tokens else 0
    return BlockDemand(
        tokens=token_count,
        complete_blocks=complete_blocks,
        incomplete_tail_tokens=tail_tokens,
        incomplete_tail_blocks=tail_blocks,
        hot_blocks=complete_blocks + tail_blocks,
    )


def cumulative_session_context_tokens(
    session: list[BailianRecord],
) -> dict[str, int | bool]:
    """Measure context from cumulative input_length without summing turns."""
    if not session:
        raise ValueError("session is empty")
    first = session[0].input_length
    final = session[-1].input_length
    maximum = max(record.input_length for record in session)
    reconstructed = first
    for previous, current in zip(session, session[1:]):
        reconstructed += current.input_length - previous.input_length
    naive_double_count = sum(record.input_length for record in session)
    return {
        "first_input_length": first,
        "final_input_length": final,
        "max_input_length": maximum,
        "reconstructed_from_deltas": reconstructed,
        "naive_sum_of_turn_input_lengths": naive_double_count,
        "uses_cumulative_final_context": True,
        "reconstructed_matches_final": reconstructed == final,
        "final_equals_maximum": final == maximum,
    }


def numeric_distribution(values: Iterable[float | int]) -> dict[str, float | int | None]:
    materialised = [float(value) for value in values]
    if not materialised:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(materialised),
        "min": min(materialised),
        "mean": statistics.fmean(materialised),
        "median": percentile(materialised, 0.50),
        "p75": percentile(materialised, 0.75),
        "p90": percentile(materialised, 0.90),
        "p95": percentile(materialised, 0.95),
        "p99": percentile(materialised, 0.99),
        "max": max(materialised),
    }


def load_trace_records(path: str | Path) -> ParsedTrace:
    records: list[BailianRecord] = []
    rejections: list[dict[str, Any]] = []
    total_non_empty_rows = 0
    blank_lines = 0
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                blank_lines += 1
                continue
            total_non_empty_rows += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                rejections.append(
                    {
                        "line_number": line_number,
                        "reason": "json_decode_error",
                        "detail": str(exc),
                    }
                )
                continue
            if not isinstance(value, dict):
                rejections.append(
                    {
                        "line_number": line_number,
                        "reason": "row_is_not_an_object",
                        "detail": type(value).__name__,
                    }
                )
                continue
            missing = [
                key for key in REQUIRED_TRACE_KEYS if key not in value
            ]
            if missing:
                rejections.append(
                    {
                        "line_number": line_number,
                        "reason": "missing_keys",
                        "detail": ",".join(missing),
                    }
                )
                continue
            try:
                records.append(BailianRecord.from_mapping(value))
            except (KeyError, TypeError, ValueError) as exc:
                rejections.append(
                    {
                        "line_number": line_number,
                        "reason": "invalid_record",
                        "detail": str(exc),
                    }
                )
    return ParsedTrace(
        records=records,
        rejections=rejections,
        total_non_empty_rows=total_non_empty_rows,
        blank_lines=blank_lines,
    )


def _resolve_root(
    chat_id: int,
    by_id: dict[int, BailianRecord],
) -> tuple[int | None, str | None]:
    current = chat_id
    seen: set[int] = set()
    while True:
        if current in seen:
            return None, "cycle"
        seen.add(current)
        record = by_id.get(current)
        if record is None:
            return None, "missing_parent"
        if record.parent_chat_id == -1:
            return record.chat_id, None
        current = record.parent_chat_id


def _linear_session_issue(session: list[BailianRecord]) -> str | None:
    if not session:
        return "empty_session"
    if session[0].parent_chat_id != -1:
        return "missing_root"
    if session[0].turn != 1:
        return "root_turn_is_not_1"
    for previous, current in zip(session, session[1:]):
        if current.parent_chat_id != previous.chat_id:
            return "branches_or_skips_parent"
        if current.turn != previous.turn + 1:
            return "non_consecutive_turns"
        if current.timestamp < previous.timestamp:
            return "timestamps_move_backwards"
        if current.input_length < previous.input_length:
            return "input_length_shrank"
        complete_parent_blocks = previous.input_length // BLOCK_SIZE
        if (
            current.hash_ids[:complete_parent_blocks]
            != previous.hash_ids[:complete_parent_blocks]
        ):
            return "complete_block_prefix_changed"
    return None


def classify_linear_sessions(
    records: Iterable[BailianRecord],
) -> ClassifiedSessions:
    by_id: dict[int, BailianRecord] = {}
    duplicate_chat_ids: list[int] = []
    for record in records:
        if record.chat_id in by_id:
            duplicate_chat_ids.append(record.chat_id)
            continue
        by_id[record.chat_id] = record

    grouped: dict[int | str, list[BailianRecord]] = defaultdict(list)
    incomplete: list[dict[str, Any]] = []
    for record in by_id.values():
        root_id, walk_issue = _resolve_root(record.chat_id, by_id)
        if walk_issue is not None or root_id is None:
            grouped[f"fragment-{record.chat_id}"].append(record)
            continue
        grouped[root_id].append(record)

    complete: dict[int, list[BailianRecord]] = {}
    for key, session in grouped.items():
        session.sort(key=lambda item: (item.turn, item.timestamp, item.chat_id))
        if isinstance(key, str):
            incomplete.append(
                {
                    "root_chat_id": session[0].chat_id,
                    "request_count": len(session),
                    "reason": "missing_parent_or_cycle",
                    "chat_ids": [record.chat_id for record in session],
                }
            )
            continue
        issue = _linear_session_issue(session)
        if issue is not None:
            incomplete.append(
                {
                    "root_chat_id": key,
                    "request_count": len(session),
                    "reason": issue,
                    "chat_ids": [record.chat_id for record in session],
                }
            )
            continue
        complete[key] = session
    return ClassifiedSessions(
        complete=complete,
        incomplete=incomplete,
        duplicate_chat_ids=duplicate_chat_ids,
    )


def inter_turn_gaps_seconds(session: list[BailianRecord]) -> list[float]:
    return [
        current.timestamp - previous.timestamp
        for previous, current in zip(session, session[1:])
    ]


def session_is_text_only(session: list[BailianRecord]) -> bool:
    return all(record.request_type == "text" for record in session)


def session_max_input_length(session: list[BailianRecord]) -> int:
    return max(record.input_length for record in session)


def filter_sessions(
    sessions: dict[int, list[BailianRecord]],
    *,
    request_type: str | None,
    min_turns: int,
    max_input_length: int | None,
) -> dict[int, list[BailianRecord]]:
    eligible: dict[int, list[BailianRecord]] = {}
    for root_id, session in sessions.items():
        if len(session) < min_turns:
            continue
        if request_type is not None and any(
            record.request_type != request_type for record in session
        ):
            continue
        if max_input_length is not None and any(
            record.input_length > max_input_length for record in session
        ):
            continue
        eligible[root_id] = session
    return eligible


def sorted_session_items(
    sessions: dict[int, list[BailianRecord]],
) -> list[tuple[int, list[BailianRecord]]]:
    return sorted(
        sessions.items(),
        key=lambda item: (item[1][0].timestamp, item[0]),
    )


def select_first_n_sessions(
    sessions: dict[int, list[BailianRecord]],
    max_sessions: int,
) -> dict[int, list[BailianRecord]]:
    if max_sessions < 0:
        raise ValueError("max_sessions must be non-negative")
    selected = sorted_session_items(sessions)[:max_sessions]
    return {root_id: session for root_id, session in selected}


def flatten_sessions(
    sessions: dict[int, list[BailianRecord]],
) -> list[BailianRecord]:
    return [
        record
        for _, session in sorted_session_items(sessions)
        for record in session
    ]


def selection_hash(sessions: dict[int, list[BailianRecord]]) -> dict[str, Any]:
    if not sessions:
        return {
            "selected_session_ids": [],
            "selected_session_count": 0,
            "selected_request_count": 0,
            "selection_sha256": sha256_json([]),
        }
    return selection_metadata(flatten_sessions(sessions))


def return_after_counts(
    sessions: dict[int, list[BailianRecord]],
) -> dict[str, dict[str, float | int]]:
    totals = len(sessions)
    result: dict[str, dict[str, float | int]] = {}
    max_gaps = [
        max(inter_turn_gaps_seconds(session), default=0.0)
        for session in sessions.values()
    ]
    for name, threshold in RETURN_AFTER_THRESHOLDS_SECONDS:
        count = sum(1 for gap in max_gaps if gap >= threshold)
        result[name] = {
            "threshold_seconds": threshold,
            "session_count": count,
            "session_percentage": (
                100.0 * count / totals if totals else 0.0
            ),
        }
    return result


def session_statistics(
    sessions: dict[int, list[BailianRecord]],
) -> dict[str, Any]:
    turns = [len(session) for session in sessions.values()]
    first_input = [session[0].input_length for session in sessions.values()]
    final_input = [session[-1].input_length for session in sessions.values()]
    max_input = [session_max_input_length(session) for session in sessions.values()]
    output_per_turn = [
        record.output_length
        for session in sessions.values()
        for record in session
    ]
    output_per_session = [
        sum(record.output_length for record in session)
        for session in sessions.values()
    ]
    requests = list(turns)
    durations = [
        session[-1].timestamp - session[0].timestamp
        for session in sessions.values()
    ]
    gaps = [
        gap
        for session in sessions.values()
        for gap in inter_turn_gaps_seconds(session)
    ]
    context_checks = [
        cumulative_session_context_tokens(session)
        for session in sessions.values()
    ]
    return {
        "session_count": len(sessions),
        "request_count": sum(requests),
        "turns_per_session": numeric_distribution(turns),
        "first_input_length": numeric_distribution(first_input),
        "final_input_length": numeric_distribution(final_input),
        "max_input_length": numeric_distribution(max_input),
        "output_length_per_turn": numeric_distribution(output_per_turn),
        "total_output_length_per_session": numeric_distribution(
            output_per_session
        ),
        "total_requests_per_session": numeric_distribution(requests),
        "trace_duration_seconds_per_session": numeric_distribution(durations),
        "inter_turn_gap_seconds": numeric_distribution(gaps),
        "sessions_returning_after": return_after_counts(sessions),
        "cumulative_context_checks": {
            "sessions_checked": len(context_checks),
            "reconstructed_matches_final": sum(
                1
                for item in context_checks
                if item["reconstructed_matches_final"]
            ),
            "final_equals_maximum": sum(
                1 for item in context_checks if item["final_equals_maximum"]
            ),
        },
    }


def sweep_retained_occupancy(
    sessions: dict[int, list[BailianRecord]],
) -> dict[str, Any]:
    """Idle retained occupancy from first turn until the final turn.

    A session is counted as retained on [first_turn, final_turn). Arrival
    timestamps cannot identify simultaneously active GPU inference.
    Occupancy uses the cumulative input_length of the last arrived turn,
    converted with ceil(tokens / 16). Turn input lengths are not summed.
    """
    events: list[tuple[float, int, int]] = []
    for session in sessions.values():
        if len(session) < 2:
            continue
        for index, record in enumerate(session[:-1]):
            blocks = tokens_to_block_demand(record.input_length).hot_blocks
            next_timestamp = session[index + 1].timestamp
            session_open = 1 if index == 0 else 0
            session_close = 1 if index == len(session) - 2 else 0
            events.append((record.timestamp, session_open, blocks))
            events.append((next_timestamp, -session_close, -blocks))
    events.sort(key=lambda item: (item[0], item[1], item[2]))

    retained = 0
    hot_blocks = 0
    peak_retained = 0
    peak_hot_blocks = 0
    samples: list[dict[str, float | int]] = []
    index = 0
    while index < len(events):
        timestamp = events[index][0]
        while index < len(events) and events[index][0] == timestamp:
            retained += events[index][1]
            hot_blocks += events[index][2]
            index += 1
        peak_retained = max(peak_retained, retained)
        peak_hot_blocks = max(peak_hot_blocks, hot_blocks)
        samples.append(
            {
                "timestamp": timestamp,
                "retained_sessions": retained,
                "estimated_hot_blocks": hot_blocks,
            }
        )
    return {
        "samples": samples,
        "peak_retained_sessions": peak_retained,
        "peak_estimated_hot_blocks": peak_hot_blocks,
        "interpretation": (
            "Retained sessions are conversations holding KV between their "
            "first and final turn. This is not GPU inference concurrency "
            "and is not measured runtime concurrency."
        ),
    }


def _span(records: Iterable[BailianRecord]) -> tuple[float, float]:
    materialised = list(records)
    if not materialised:
        return 0.0, 0.0
    timestamps = [record.timestamp for record in materialised]
    return min(timestamps), max(timestamps)


def one_second_bucket_arrivals(
    records: Iterable[BailianRecord],
    *,
    time_scale: float,
) -> dict[str, Any]:
    materialised = list(records)
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")
    if not materialised:
        return {
            "time_scale": time_scale,
            "simulated_duration_seconds": 0.0,
            "request_count": 0,
            "mean_arrivals_per_simulated_second": 0.0,
            "peak_arrivals_in_one_second_buckets": 0,
            "occupied_one_second_buckets": 0,
            "bucket_counts": [],
        }
    start, end = _span(materialised)
    scaled_times = [(record.timestamp - start) * time_scale for record in materialised]
    simulated_duration = max((end - start) * time_scale, 0.0)
    bucket_count = max(1, math.floor(max(scaled_times)) + 1)
    if simulated_duration > 0:
        bucket_count = max(bucket_count, math.ceil(simulated_duration))
    buckets = [0] * bucket_count
    for scaled in scaled_times:
        index = min(int(scaled), bucket_count - 1)
        if index < 0:
            index = 0
        buckets[index] += 1
    occupied = sum(1 for count in buckets if count)
    return {
        "time_scale": time_scale,
        "simulated_duration_seconds": simulated_duration,
        "request_count": len(materialised),
        "mean_arrivals_per_simulated_second": (
            len(materialised) / simulated_duration if simulated_duration else None
        ),
        "peak_arrivals_in_one_second_buckets": max(buckets, default=0),
        "occupied_one_second_buckets": occupied,
        "bucket_counts": buckets,
    }


def resample_step_maxima(
    samples: list[dict[str, float | int]],
    *,
    start: float,
    end: float,
    time_scale: float,
) -> dict[str, Any]:
    duration = max((end - start) * time_scale, 0.0)
    if duration <= 0 or not samples:
        zeros = {
            "bucket_size_simulated_seconds": 1.0,
            "retained_sessions": [],
            "estimated_hot_blocks": [],
            "median_retained_sessions": 0 if not samples else None,
            "p95_retained_sessions": 0 if not samples else None,
            "median_estimated_hot_blocks": 0 if not samples else None,
            "p95_estimated_hot_blocks": 0 if not samples else None,
        }
        if not samples:
            zeros["median_retained_sessions"] = 0
            zeros["p95_retained_sessions"] = 0
            zeros["median_estimated_hot_blocks"] = 0
            zeros["p95_estimated_hot_blocks"] = 0
        return zeros

    scaled_samples = [
        (
            (float(sample["timestamp"]) - start) * time_scale,
            int(sample["retained_sessions"]),
            int(sample["estimated_hot_blocks"]),
        )
        for sample in samples
    ]
    bucket_count = max(1, math.ceil(duration))
    retained_buckets = [0] * bucket_count
    block_buckets = [0] * bucket_count
    current_retained = 0
    current_blocks = 0
    sample_index = 0
    for bucket in range(bucket_count):
        bucket_end = float(bucket + 1)
        while (
            sample_index < len(scaled_samples)
            and scaled_samples[sample_index][0] <= bucket_end
        ):
            current_retained = scaled_samples[sample_index][1]
            current_blocks = scaled_samples[sample_index][2]
            retained_buckets[bucket] = max(
                retained_buckets[bucket], current_retained
            )
            block_buckets[bucket] = max(block_buckets[bucket], current_blocks)
            sample_index += 1
        retained_buckets[bucket] = max(retained_buckets[bucket], current_retained)
        block_buckets[bucket] = max(block_buckets[bucket], current_blocks)
    return {
        "bucket_size_simulated_seconds": 1.0,
        "retained_sessions": retained_buckets,
        "estimated_hot_blocks": block_buckets,
        "median_retained_sessions": percentile(retained_buckets, 0.50),
        "p95_retained_sessions": percentile(retained_buckets, 0.95),
        "median_estimated_hot_blocks": percentile(block_buckets, 0.50),
        "p95_estimated_hot_blocks": percentile(block_buckets, 0.95),
    }


def arrival_and_retention(
    sessions: dict[int, list[BailianRecord]],
    time_scales: Iterable[float],
) -> dict[str, Any]:
    records = flatten_sessions(sessions)
    start, end = _span(records)
    occupancy = sweep_retained_occupancy(sessions)
    by_scale: dict[str, Any] = {}
    for time_scale in time_scales:
        arrivals = one_second_bucket_arrivals(records, time_scale=time_scale)
        resampled = resample_step_maxima(
            occupancy["samples"],
            start=start,
            end=end,
            time_scale=time_scale,
        )
        by_scale[format_float_key(time_scale)] = {
            "time_scale": time_scale,
            "requests_arriving_per_simulated_second": arrivals[
                "mean_arrivals_per_simulated_second"
            ],
            "peak_arrivals_in_one_second_buckets": arrivals[
                "peak_arrivals_in_one_second_buckets"
            ],
            "simulated_duration_seconds": arrivals["simulated_duration_seconds"],
            "peak_retained_sessions": occupancy["peak_retained_sessions"],
            "median_retained_sessions": resampled["median_retained_sessions"],
            "p95_retained_sessions": resampled["p95_retained_sessions"],
            "peak_estimated_hot_blocks": occupancy["peak_estimated_hot_blocks"],
            "median_estimated_hot_blocks": resampled[
                "median_estimated_hot_blocks"
            ],
            "p95_estimated_hot_blocks": resampled["p95_estimated_hot_blocks"],
            "retained_sessions_are_not_gpu_inference_concurrency": True,
            "measured_runtime_concurrency_requires_gpu_replay": True,
            "overlap_invariant_to_uniform_time_scaling": True,
        }
    return {
        "trace_start_timestamp": start,
        "trace_end_timestamp": end,
        "trace_span_seconds": end - start,
        "peak_retained_sessions": occupancy["peak_retained_sessions"],
        "peak_estimated_hot_blocks": occupancy["peak_estimated_hot_blocks"],
        "interpretation": occupancy["interpretation"],
        "by_time_scale": by_scale,
        "event_samples": occupancy["samples"],
    }


def selected_workload_demand(
    sessions: dict[int, list[BailianRecord]],
    *,
    requested_session_count: int,
    time_scales: Iterable[float],
    all_hot_blocks: int,
) -> dict[str, Any]:
    selected = select_first_n_sessions(sessions, requested_session_count)
    metadata = selection_hash(selected)
    context = [
        cumulative_session_context_tokens(session)
        for session in selected.values()
    ]
    total_context_tokens = sum(int(item["final_input_length"]) for item in context)
    demands = [
        tokens_to_block_demand(int(item["final_input_length"]))
        for item in context
    ]
    complete_blocks = sum(item.complete_blocks for item in demands)
    incomplete_tail_blocks = sum(item.incomplete_tail_blocks for item in demands)
    incomplete_tail_tokens = sum(item.incomplete_tail_tokens for item in demands)
    final_hot_blocks_if_all_resident = sum(item.hot_blocks for item in demands)
    retention = arrival_and_retention(selected, time_scales)
    peak_hot = int(retention["peak_estimated_hot_blocks"])
    excess_blocks = max(0, peak_hot - all_hot_blocks)
    return {
        "requested_session_count": requested_session_count,
        "selected_session_count": metadata["selected_session_count"],
        "selected_request_count": metadata["selected_request_count"],
        "selected_session_ids": metadata["selected_session_ids"],
        "selection_sha256": metadata["selection_sha256"],
        "selection_order": "first_n_by_first_turn_timestamp_then_root_chat_id",
        "input_length_semantics": "cumulative_context_tokens",
        "total_context_token_demand": total_context_tokens,
        "complete_block_demand": complete_blocks,
        "incomplete_tail_token_demand": incomplete_tail_tokens,
        "incomplete_tail_block_demand": incomplete_tail_blocks,
        "final_hot_blocks_if_all_sessions_resident": (
            final_hot_blocks_if_all_resident
        ),
        "naive_double_counted_turn_input_tokens": sum(
            int(item["naive_sum_of_turn_input_lengths"]) for item in context
        ),
        "peak_estimated_hot_blocks": peak_hot,
        "peak_retained_sessions": retention["peak_retained_sessions"],
        "all_hot_capacity_blocks": all_hot_blocks,
        "peak_exceeds_all_hot_capacity": peak_hot > all_hot_blocks,
        "excess_hot_blocks": excess_blocks,
        "excess_hot_percent": (
            100.0 * excess_blocks / all_hot_blocks if all_hot_blocks else None
        ),
        "arrival_and_retention": retention,
        "caveat": (
            "Peak estimated HOT blocks count retained cumulative prompts "
            "between turns. They are not measured GPU inference concurrency."
        ),
    }


def _budget_args(
    *,
    kv_mode: str,
    warm_pool_blocks: int,
    total_kv_budget_bytes: int,
    max_model_len: int,
    max_num_seqs: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=SUPPORTED_BUDGET_MODEL,
        kv_mode=kv_mode,
        warm_pool_blocks=warm_pool_blocks if kv_mode == "mixed" else 0,
        total_kv_budget_bytes=total_kv_budget_bytes,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )


def equal_memory_capacity_table(
    *,
    total_kv_budget_bytes: int,
    warm_pool_sizes: Iterable[int],
    max_model_len: int,
    max_num_seqs: int,
) -> dict[str, Any]:
    all_hot = derive_persistent_kv_budget(
        _budget_args(
            kv_mode="all-hot",
            warm_pool_blocks=0,
            total_kv_budget_bytes=total_kv_budget_bytes,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )
    )
    all_hot_blocks = int(all_hot["derived_num_gpu_blocks"])
    all_hot_tokens = all_hot_blocks * BLOCK_SIZE
    configurations = {
        "all_hot": {
            "kv_mode": "all-hot",
            "warm_pool_blocks": 0,
            "hot_blocks": all_hot_blocks,
            "total_logical_blocks": all_hot_blocks,
            "logical_token_capacity": all_hot_tokens,
            "capacity_gain_over_all_hot_percent": 0.0,
            "warm_storage_bytes": all_hot["derived_warm_kv_storage_bytes"],
            "map_bytes": all_hot["derived_hot_to_warm_map_storage_bytes"],
            "slot_table_bytes": all_hot["derived_warm_slot_table_storage_bytes"],
            "actual_persistent_kv_bytes": all_hot[
                "derived_actual_persistent_kv_bytes"
            ],
            "budget_slack_bytes": all_hot["derived_budget_slack_bytes"],
            "within_total_budget": (
                all_hot["derived_actual_persistent_kv_bytes"]
                <= total_kv_budget_bytes
            ),
            "budget": all_hot,
        }
    }
    for warm_pool_blocks in warm_pool_sizes:
        mixed = derive_persistent_kv_budget(
            _budget_args(
                kv_mode="mixed",
                warm_pool_blocks=warm_pool_blocks,
                total_kv_budget_bytes=total_kv_budget_bytes,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
            )
        )
        hot_blocks = int(mixed["derived_num_gpu_blocks"])
        logical_blocks = hot_blocks + warm_pool_blocks
        tokens = logical_blocks * BLOCK_SIZE
        gain = 100.0 * (tokens - all_hot_tokens) / all_hot_tokens
        configurations[f"mixed_warm_{warm_pool_blocks}"] = {
            "kv_mode": "mixed",
            "warm_pool_blocks": warm_pool_blocks,
            "hot_blocks": hot_blocks,
            "total_logical_blocks": logical_blocks,
            "logical_token_capacity": tokens,
            "capacity_gain_over_all_hot_percent": gain,
            "warm_storage_bytes": mixed["derived_warm_kv_storage_bytes"],
            "map_bytes": mixed["derived_hot_to_warm_map_storage_bytes"],
            "slot_table_bytes": mixed["derived_warm_slot_table_storage_bytes"],
            "actual_persistent_kv_bytes": mixed[
                "derived_actual_persistent_kv_bytes"
            ],
            "budget_slack_bytes": mixed["derived_budget_slack_bytes"],
            "within_total_budget": (
                mixed["derived_actual_persistent_kv_bytes"]
                <= total_kv_budget_bytes
            ),
            "budget": mixed,
        }
    return {
        "formula_version": KV_MEMORY_FORMULA_VERSION,
        "total_kv_budget_bytes": total_kv_budget_bytes,
        "block_size_tokens": BLOCK_SIZE,
        "hot_bytes_per_block": hot_bytes_per_block(),
        "warm_bytes_per_slot": warm_bytes_per_slot(),
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "all_configurations_within_budget": all(
            item["within_total_budget"] for item in configurations.values()
        ),
        "configurations": configurations,
    }


def histogram_data(
    values: list[float],
    *,
    bins: int | list[float],
) -> dict[str, Any]:
    if not values:
        return {"bins": [], "counts": [], "values": []}
    try:
        import numpy as np
    except ImportError:
        return {"bins": None, "counts": None, "values": list(values)}
    counts, edges = np.histogram(values, bins=bins)
    return {
        "bins": [float(edge) for edge in edges],
        "counts": [int(count) for count in counts],
        "values": list(values),
    }


def mixed_hot_start_threshold_blocks(
    hot_blocks: int,
    start_utilization: float,
) -> int:
    """Smallest HOT occupancy with occupancy / hot_blocks >= start_utilization.

    Mixed WARM-1024 has 1,812 HOT blocks. 1,449 / 1,812 < 0.80, while
    1,450 / 1,812 >= 0.80, so demotion begins at 1,450 HOT blocks.
    """
    if hot_blocks <= 0:
        return 0
    if start_utilization <= 0:
        return 0
    if start_utilization >= 1:
        return hot_blocks
    return min(
        hot_blocks,
        math.ceil(hot_blocks * start_utilization - UTILIZATION_EPS),
    )


def mixed_hot_stop_target_blocks(
    hot_blocks: int,
    stop_utilization: float,
) -> int:
    """Largest HOT occupancy with occupancy / hot_blocks <= stop_utilization."""
    if hot_blocks <= 0:
        return 0
    if stop_utilization <= 0:
        return 0
    if stop_utilization >= 1:
        return hot_blocks
    return min(
        hot_blocks,
        math.floor(hot_blocks * stop_utilization + UTILIZATION_EPS),
    )


def policy_key(start_utilization: float, stop_utilization: float) -> str:
    return f"start_{start_utilization:g}_stop_{stop_utilization:g}"


def policy_aware_pressure(
    *,
    peak_estimated_hot_blocks: int,
    mixed_hot_blocks: int,
    warm_pool_blocks: int,
    start_utilization: float,
    stop_utilization: float,
    all_hot_blocks: int,
) -> dict[str, Any]:
    start_blocks = mixed_hot_start_threshold_blocks(
        mixed_hot_blocks, start_utilization
    )
    stop_blocks = mixed_hot_stop_target_blocks(
        mixed_hot_blocks, stop_utilization
    )
    peak = int(peak_estimated_hot_blocks)
    mixed_total = int(mixed_hot_blocks) + int(warm_pool_blocks)
    projected_to_stop = max(0, peak - stop_blocks)
    max_demotion = min(projected_to_stop, int(warm_pool_blocks))
    hot_after = peak - max_demotion
    hot_util_after = (
        hot_after / mixed_hot_blocks if mixed_hot_blocks else None
    )
    return {
        "start_utilization": start_utilization,
        "stop_utilization": stop_utilization,
        "mixed_hot_blocks": mixed_hot_blocks,
        "warm_pool_blocks": warm_pool_blocks,
        "mixed_hot_start_threshold_blocks": start_blocks,
        "mixed_hot_stop_target_blocks": stop_blocks,
        "peak_estimated_retained_blocks": peak,
        "projected_blocks_to_demote_to_stop": projected_to_stop,
        "warm_pool_can_hold_demotion": projected_to_stop <= warm_pool_blocks,
        "maximum_demotable_blocks": max_demotion,
        "projected_hot_blocks_after_max_demotion": hot_after,
        "projected_hot_utilization_after_max_demotion": hot_util_after,
        "policy_would_start_demotion": peak >= start_blocks and start_blocks > 0,
        "all_hot_capacity_exceeded": peak > all_hot_blocks,
        "mixed_total_logical_capacity_exceeded": peak > mixed_total,
        "mixed_total_logical_blocks": mixed_total,
    }


def policy_pressure_for_peak(
    *,
    peak_estimated_hot_blocks: int,
    capacity: dict[str, Any],
    policies: Iterable[tuple[float, float]] = DEFAULT_DEMOTION_POLICIES,
) -> dict[str, Any]:
    all_hot_blocks = int(capacity["configurations"]["all_hot"]["hot_blocks"])
    result: dict[str, Any] = {}
    for name, config in capacity["configurations"].items():
        if config["kv_mode"] != "mixed":
            continue
        result[name] = {
            policy_key(start, stop): policy_aware_pressure(
                peak_estimated_hot_blocks=peak_estimated_hot_blocks,
                mixed_hot_blocks=int(config["hot_blocks"]),
                warm_pool_blocks=int(config["warm_pool_blocks"]),
                start_utilization=start,
                stop_utilization=stop,
                all_hot_blocks=all_hot_blocks,
            )
            for start, stop in policies
        }
    return result


def attach_policy_pressure(
    selection: dict[str, Any],
    *,
    capacity: dict[str, Any],
    policies: Iterable[tuple[float, float]] = DEFAULT_DEMOTION_POLICIES,
) -> dict[str, Any]:
    peak = int(selection["peak_estimated_hot_blocks"])
    all_hot_blocks = int(capacity["configurations"]["all_hot"]["hot_blocks"])
    selection["policy_pressure"] = policy_pressure_for_peak(
        peak_estimated_hot_blocks=peak,
        capacity=capacity,
        policies=policies,
    )
    mixed_flags: dict[str, Any] = {}
    for name, config in capacity["configurations"].items():
        if config["kv_mode"] != "mixed":
            continue
        total = int(config["total_logical_blocks"])
        mixed_flags[name] = {
            "hot_blocks": int(config["hot_blocks"]),
            "warm_pool_blocks": int(config["warm_pool_blocks"]),
            "total_logical_blocks": total,
            "peak_exceeds_mixed_total_logical": peak > total,
        }
    selection["mixed_logical_capacity"] = mixed_flags
    selection["all_hot_capacity_blocks"] = all_hot_blocks
    selection["peak_exceeds_all_hot_capacity"] = peak > all_hot_blocks
    return selection


def flatten_policy_pressure_rows(
    selections: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count, selection in selections.items():
        for mixed_name, policies in selection.get("policy_pressure", {}).items():
            for policy_name, item in policies.items():
                rows.append(
                    {
                        "requested_sessions": int(count),
                        "selected_sessions": selection["selected_session_count"],
                        "mixed_configuration": mixed_name,
                        "policy": policy_name,
                        **item,
                    }
                )
    rows.sort(
        key=lambda row: (
            row["requested_sessions"],
            row["mixed_configuration"],
            row["start_utilization"],
        )
    )
    return rows


class PrefixPeakIndex:
    """Cached peak retained demand of timestamp-ordered prefixes."""

    def __init__(self, sessions: dict[int, list[BailianRecord]]) -> None:
        self.ordered = sorted_session_items(sessions)
        self._cache: dict[int, dict[str, int]] = {}

    def __len__(self) -> int:
        return len(self.ordered)

    def peak(self, session_count: int) -> dict[str, int]:
        count = max(0, min(int(session_count), len(self.ordered)))
        if count == 0:
            return {
                "session_count": 0,
                "peak_estimated_hot_blocks": 0,
                "peak_retained_sessions": 0,
            }
        cached = self._cache.get(count)
        if cached is not None:
            return cached
        occupancy = sweep_retained_occupancy(dict(self.ordered[:count]))
        result = {
            "session_count": count,
            "peak_estimated_hot_blocks": int(
                occupancy["peak_estimated_hot_blocks"]
            ),
            "peak_retained_sessions": int(occupancy["peak_retained_sessions"]),
        }
        self._cache[count] = result
        return result

    def smallest_n(self, predicate) -> int | None:
        if not self.ordered:
            return None
        low = 1
        high = len(self.ordered)
        found: int | None = None
        while low <= high:
            mid = (low + high) // 2
            if predicate(self.peak(mid)["peak_estimated_hot_blocks"]):
                found = mid
                high = mid - 1
            else:
                low = mid + 1
        return found

    def largest_n(self, predicate) -> int | None:
        if not self.ordered:
            return None
        low = 1
        high = len(self.ordered)
        found: int | None = None
        while low <= high:
            mid = (low + high) // 2
            if predicate(self.peak(mid)["peak_estimated_hot_blocks"]):
                found = mid
                low = mid + 1
            else:
                high = mid - 1
        return found

    def smallest_reaching(
        self,
        target_blocks: int,
        *,
        comparison: str = "ge",
    ) -> dict[str, int | None]:
        if comparison == "gt":
            predicate = lambda peak, target=target_blocks: peak > target
        elif comparison == "ge":
            predicate = lambda peak, target=target_blocks: peak >= target
        else:
            raise ValueError(f"unsupported comparison: {comparison}")
        smallest = self.smallest_n(predicate)
        full = self.peak(len(self.ordered))
        at_smallest = self.peak(smallest) if smallest is not None else full
        return {
            "smallest_session_count": smallest,
            "peak_estimated_hot_blocks": at_smallest["peak_estimated_hot_blocks"],
            "peak_retained_sessions": at_smallest["peak_retained_sessions"],
            "full_eligible_peak_estimated_hot_blocks": full[
                "peak_estimated_hot_blocks"
            ],
            "full_eligible_peak_retained_sessions": full["peak_retained_sessions"],
            "searched_session_count": len(self.ordered),
            "target_blocks": target_blocks,
            "comparison": comparison,
        }


def smallest_prefix_count_exceeding_capacity(
    sessions: dict[int, list[BailianRecord]],
    *,
    all_hot_blocks: int,
) -> dict[str, int | None]:
    return PrefixPeakIndex(sessions).smallest_reaching(
        all_hot_blocks, comparison="gt"
    )


def smallest_prefix_reaching_peak(
    sessions: dict[int, list[BailianRecord]],
    *,
    target_blocks: int,
    comparison: str = "ge",
) -> dict[str, int | None]:
    return PrefixPeakIndex(sessions).smallest_reaching(
        target_blocks, comparison=comparison
    )


def prefix_peak_target_specs(capacity: dict[str, Any]) -> list[dict[str, Any]]:
    all_hot = int(capacity["configurations"]["all_hot"]["hot_blocks"])
    mixed = capacity["configurations"].get(
        f"mixed_warm_{PRIMARY_WARM_POOL_BLOCKS}"
    )
    mixed_hot = int(mixed["hot_blocks"]) if mixed else 0
    mixed_total = int(mixed["total_logical_blocks"]) if mixed else 0
    start_80 = mixed_hot_start_threshold_blocks(mixed_hot, 0.80)
    return [
        {
            "name": "mixed_hot_80_percent",
            "description": (
                "80% of Mixed-1024 HOT capacity (demotion-start threshold)"
            ),
            "target_blocks": start_80,
        },
        {
            "name": "all_hot_100_percent",
            "description": "100% of All-HOT capacity",
            "target_blocks": all_hot,
        },
        {
            "name": "all_hot_110_percent",
            "description": "110% of All-HOT capacity",
            "target_blocks": math.ceil(all_hot * 1.10 - UTILIZATION_EPS),
        },
        {
            "name": "all_hot_120_percent",
            "description": "120% of All-HOT capacity",
            "target_blocks": math.ceil(all_hot * 1.20 - UTILIZATION_EPS),
        },
        {
            "name": "mixed_total_90_percent",
            "description": "90% of Mixed-1024 total logical capacity",
            "target_blocks": math.ceil(mixed_total * 0.90 - UTILIZATION_EPS),
        },
        {
            "name": "mixed_total_95_percent",
            "description": "95% of Mixed-1024 total logical capacity",
            "target_blocks": math.ceil(mixed_total * 0.95 - UTILIZATION_EPS),
        },
        {
            "name": "mixed_total_100_percent",
            "description": "100% of Mixed-1024 total logical capacity",
            "target_blocks": mixed_total,
        },
    ]


def search_prefix_peak_targets(
    index: PrefixPeakIndex,
    capacity: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in prefix_peak_target_specs(capacity):
        found = index.smallest_reaching(int(spec["target_blocks"]), comparison="ge")
        rows.append({**spec, **found})
    return rows


def _closest_prefix_to_peak(
    index: PrefixPeakIndex,
    *,
    low: int,
    high: int,
    target_peak: float,
) -> dict[str, int]:
    reaching = None
    lo, hi = low, high
    while lo <= hi:
        mid = (lo + hi) // 2
        if index.peak(mid)["peak_estimated_hot_blocks"] >= target_peak:
            reaching = mid
            hi = mid - 1
        else:
            lo = mid + 1
    candidates = [high if reaching is None else reaching]
    if reaching is not None and reaching - 1 >= low:
        candidates.append(reaching - 1)
    return min(
        (index.peak(count) for count in candidates),
        key=lambda item: (
            abs(item["peak_estimated_hot_blocks"] - target_peak),
            item["session_count"],
        ),
    )


def find_high_load_prefix(
    index: PrefixPeakIndex,
    *,
    preferred_min_blocks: int = PREFERRED_HIGH_LOAD_PEAK_MIN_BLOCKS,
    preferred_max_blocks: int = PREFERRED_HIGH_LOAD_PEAK_MAX_BLOCKS,
    all_hot_blocks: int,
    mixed_total_blocks: int,
) -> dict[str, Any]:
    empty = {
        "preferred_peak_min_blocks": preferred_min_blocks,
        "preferred_peak_max_blocks": preferred_max_blocks,
        "all_hot_capacity_blocks": all_hot_blocks,
        "mixed_total_logical_blocks": mixed_total_blocks,
        "range_satisfied": False,
        "session_count": None,
        "peak_estimated_hot_blocks": 0,
        "peak_retained_sessions": 0,
        "clearly_above_all_hot_capacity": False,
        "below_mixed_total_logical_capacity": True,
        "closest_below_session_count": None,
        "closest_below_peak_estimated_hot_blocks": None,
        "closest_above_session_count": None,
        "closest_above_peak_estimated_hot_blocks": None,
        "note": "No eligible sessions were available for high-load selection.",
    }
    if not index.ordered:
        return empty

    n_ge_min = index.smallest_n(lambda peak: peak >= preferred_min_blocks)
    n_le_max = index.largest_n(lambda peak: peak <= preferred_max_blocks)
    n_below = index.largest_n(lambda peak: peak < preferred_min_blocks)
    n_above = index.smallest_n(lambda peak: peak > preferred_max_blocks)
    below = index.peak(n_below) if n_below is not None else None
    above = index.peak(n_above) if n_above is not None else None

    range_satisfied = (
        n_ge_min is not None
        and n_le_max is not None
        and n_ge_min <= n_le_max
    )
    if range_satisfied:
        chosen = _closest_prefix_to_peak(
            index,
            low=n_ge_min,
            high=n_le_max,
            target_peak=(preferred_min_blocks + preferred_max_blocks) / 2,
        )
    else:
        candidates: list[tuple[float, dict[str, int], str]] = []
        if below is not None:
            candidates.append(
                (
                    float(preferred_min_blocks - below["peak_estimated_hot_blocks"]),
                    below,
                    "below_preferred_range",
                )
            )
        if above is not None:
            candidates.append(
                (
                    float(above["peak_estimated_hot_blocks"] - preferred_max_blocks),
                    above,
                    "above_preferred_range",
                )
            )
        if not candidates:
            chosen = index.peak(len(index))
        else:
            chosen = min(
                candidates,
                key=lambda item: (item[0], item[1]["session_count"]),
            )[1]

    peak = int(chosen["peak_estimated_hot_blocks"])
    if range_satisfied:
        note = (
            f"Timestamp-ordered prefix of {chosen['session_count']} sessions "
            f"has peak retained demand {peak} blocks, inside the preferred "
            f"{preferred_min_blocks}–{preferred_max_blocks} band, above the "
            f"{all_hot_blocks}-block All-HOT capacity and below the "
            f"{mixed_total_blocks}-block Mixed-1024 total logical capacity."
        )
    else:
        note = (
            "No timestamp-ordered prefix has peak retained demand in the "
            f"preferred {preferred_min_blocks}–{preferred_max_blocks}-block "
            f"band while remaining below Mixed-1024 total logical capacity. "
            f"Closest prefix is {chosen['session_count']} sessions "
            f"(peak {peak} blocks)."
        )
    return {
        "preferred_peak_min_blocks": preferred_min_blocks,
        "preferred_peak_max_blocks": preferred_max_blocks,
        "all_hot_capacity_blocks": all_hot_blocks,
        "mixed_total_logical_blocks": mixed_total_blocks,
        "range_satisfied": range_satisfied,
        "session_count": chosen["session_count"],
        "peak_estimated_hot_blocks": peak,
        "peak_retained_sessions": chosen["peak_retained_sessions"],
        "clearly_above_all_hot_capacity": peak > all_hot_blocks,
        "below_mixed_total_logical_capacity": peak < mixed_total_blocks,
        "closest_below_session_count": (
            below["session_count"] if below is not None else None
        ),
        "closest_below_peak_estimated_hot_blocks": (
            below["peak_estimated_hot_blocks"] if below is not None else None
        ),
        "closest_above_session_count": (
            above["session_count"] if above is not None else None
        ),
        "closest_above_peak_estimated_hot_blocks": (
            above["peak_estimated_hot_blocks"] if above is not None else None
        ),
        "note": note,
    }


def resolved_session_counts(
    requested: Iterable[int],
    eligible_count: int,
    extra_counts: Iterable[int] = (),
) -> list[int]:
    counts = {int(count) for count in requested if int(count) > 0}
    counts.update(int(count) for count in extra_counts if int(count) > 0)
    if eligible_count >= REQUIRED_PREFIX_ELIGIBLE_THRESHOLD:
        counts.update(REQUIRED_PREFIX_SESSION_COUNTS)
    if eligible_count > 0:
        counts.add(eligible_count)
    return sorted(counts)


def _relative_difference(prefix_value: float | None, pool_value: float | None) -> float | None:
    if prefix_value is None or pool_value is None:
        return None
    if pool_value == 0:
        return None if prefix_value == 0 else None
    return (prefix_value - pool_value) / abs(pool_value)


def compare_selection_bias_metrics(
    prefix_sessions: dict[int, list[BailianRecord]],
    pool_sessions: dict[int, list[BailianRecord]],
) -> dict[str, Any]:
    prefix_stats = session_statistics(prefix_sessions)
    pool_stats = session_statistics(pool_sessions)
    metrics: dict[str, Any] = {}
    material_metrics: list[str] = []
    for name in (
        "turns_per_session",
        "final_input_length",
        "inter_turn_gap_seconds",
    ):
        prefix_dist = prefix_stats[name]
        pool_dist = pool_stats[name]
        median_rel = _relative_difference(prefix_dist["median"], pool_dist["median"])
        mean_rel = _relative_difference(prefix_dist["mean"], pool_dist["mean"])
        p95_rel = _relative_difference(prefix_dist["p95"], pool_dist["p95"])
        median_abs = None
        if prefix_dist["median"] is not None and pool_dist["median"] is not None:
            median_abs = prefix_dist["median"] - pool_dist["median"]
        material = any(
            diff is not None and abs(diff) >= BIAS_RELATIVE_THRESHOLD
            for diff in (median_rel, mean_rel)
        )
        if material:
            material_metrics.append(name)
        metrics[name] = {
            "prefix": prefix_dist,
            "eligible_pool": pool_dist,
            "median_abs_diff": median_abs,
            "median_relative_diff": median_rel,
            "mean_relative_diff": mean_rel,
            "p95_relative_diff": p95_rel,
            "material": material,
        }
    return {
        "prefix_session_count": len(prefix_sessions),
        "eligible_pool_session_count": len(pool_sessions),
        "metrics": metrics,
        "material_metrics": material_metrics,
        "material_bias": bool(material_metrics),
    }


def selection_bias_report(
    *,
    eligible: dict[int, list[BailianRecord]],
    prefix_counts: Iterable[int],
    high_load_session_count: int | None = None,
) -> dict[str, Any]:
    prefixes: dict[str, Any] = {}
    material_prefixes: list[str] = []
    for count in prefix_counts:
        if count <= 0:
            continue
        selected = select_first_n_sessions(eligible, count)
        comparison = compare_selection_bias_metrics(selected, eligible)
        prefixes[str(count)] = comparison
        if comparison["material_bias"] and count < len(eligible):
            material_prefixes.append(str(count))
    material = bool(material_prefixes)
    high_load_key = (
        str(high_load_session_count) if high_load_session_count else None
    )
    high_load_material = (
        high_load_key in material_prefixes if high_load_key is not None else False
    )
    if not eligible:
        interpretation = "No eligible sessions were available for bias comparison."
        follow_up = "Replay selection is not redesigned in this analysis."
    elif material:
        details = []
        for name in material_prefixes:
            metrics = prefixes[name]["material_metrics"]
            details.append(f"{name} sessions on {', '.join(metrics)}")
        interpretation = (
            "Timestamp-ordered prefixes differ materially from the complete "
            "eligible pool at the "
            f"{BIAS_RELATIVE_THRESHOLD:.0%} relative-difference threshold: "
            + "; ".join(details)
            + "."
        )
        if high_load_material:
            follow_up = (
                "A peak-window or deterministic sampled selection should be "
                "considered before GPU replay. This analysis does not redesign "
                "the timestamp-ordered prefix rule."
            )
        else:
            follow_up = (
                "The recommended high-load prefix is not materially biased "
                "versus the eligible pool. A peak-window or deterministic "
                "sampled selection can be considered for the earliest "
                "small/medium prefixes if gap-tail matching matters; this "
                "analysis does not redesign the timestamp-ordered prefix rule."
            )
    else:
        interpretation = (
            "Earliest timestamp-ordered prefixes do not differ materially "
            "from the complete eligible pool on turns per session, final "
            "context length, or inter-turn gap at the "
            f"{BIAS_RELATIVE_THRESHOLD:.0%} relative-difference threshold."
        )
        follow_up = (
            "Timestamp-ordered prefixes remain acceptable for Phase 1. A "
            "peak-window or deterministic sampled selection is not required "
            "by these bias metrics."
        )
    return {
        "method": "timestamp_ordered_prefix_versus_eligible_pool",
        "metrics_compared": [
            "turns_per_session",
            "final_input_length",
            "inter_turn_gap_seconds",
        ],
        "material_relative_threshold": BIAS_RELATIVE_THRESHOLD,
        "prefixes": prefixes,
        "material_prefixes": material_prefixes,
        "material_bias": material,
        "high_load_material_bias": high_load_material,
        "interpretation": interpretation,
        "follow_up": follow_up,
    }


def _selection_or_none(
    selections: dict[str, Any],
    count: int | None,
) -> dict[str, Any] | None:
    if count is None:
        return None
    return selections.get(str(count))


def _primary_policy_row(
    selection: dict[str, Any] | None,
    *,
    mixed_name: str = f"mixed_warm_{PRIMARY_WARM_POOL_BLOCKS}",
    start: float = PRIMARY_START_UTILIZATION,
    stop: float = PRIMARY_STOP_UTILIZATION,
) -> dict[str, Any] | None:
    if selection is None:
        return None
    return (
        selection.get("policy_pressure", {})
        .get(mixed_name, {})
        .get(policy_key(start, stop))
    )


def build_recommendation(
    *,
    session_counts: list[int],
    selections: dict[str, Any],
    capacity: dict[str, Any],
    time_scales: list[float],
    primary_max_input_length: int,
    eligible_count: int,
    session_stats: dict[str, Any],
    prefix_pressure: dict[str, int | None],
    high_load: dict[str, Any],
    selection_bias: dict[str, Any],
) -> dict[str, Any]:
    all_hot_blocks = int(capacity["configurations"]["all_hot"]["hot_blocks"])
    mixed = capacity["configurations"].get(
        f"mixed_warm_{PRIMARY_WARM_POOL_BLOCKS}"
    )
    mixed_hot = int(mixed["hot_blocks"]) if mixed else 0
    mixed_total = int(mixed["total_logical_blocks"]) if mixed else 0
    start_blocks = mixed_hot_start_threshold_blocks(
        mixed_hot, PRIMARY_START_UTILIZATION
    )
    stop_blocks = mixed_hot_stop_target_blocks(
        mixed_hot, PRIMARY_STOP_UTILIZATION
    )
    ordered = sorted(session_counts)
    smoke = 25 if 25 in ordered else (ordered[0] if ordered else None)
    medium = 250 if 250 in ordered else (
        ordered[len(ordered) // 2] if ordered else None
    )
    smallest = prefix_pressure.get("smallest_session_count")
    high = high_load.get("session_count")
    requested_exceeding = [
        count
        for count in ordered
        if selections.get(str(count), {}).get("peak_exceeds_all_hot_capacity")
    ]

    medium_stats = _selection_or_none(selections, medium)
    high_stats = _selection_or_none(selections, high)
    medium_policy = _primary_policy_row(medium_stats)
    high_policy = _primary_policy_row(high_stats)
    if high_stats is not None:
        high_peak = int(high_stats["peak_estimated_hot_blocks"])
        high_retained = high_stats["peak_retained_sessions"]
        span = high_stats["arrival_and_retention"]["trace_span_seconds"]
    else:
        high_peak = int(high_load.get("peak_estimated_hot_blocks") or 0)
        high_retained = high_load.get("peak_retained_sessions")
        fallback_selection = selections.get(str(ordered[-1])) if ordered else None
        span = (
            fallback_selection["arrival_and_retention"]["trace_span_seconds"]
            if fallback_selection
            else 0.0
        )

    medium_peak = (
        int(medium_stats["peak_estimated_hot_blocks"]) if medium_stats else 0
    )
    if medium_policy and medium_policy["policy_would_start_demotion"]:
        medium_note = (
            f"{medium} timestamp-ordered eligible sessions peak at "
            f"{medium_peak} retained blocks. Mixed-1024 demotion starts at "
            f"{start_blocks} HOT blocks "
            f"(ceil({mixed_hot} x {PRIMARY_START_UTILIZATION:g})), so this "
            "medium load is expected to begin demotion even though it does "
            f"not exhaust the {mixed_hot}-block HOT pool or the "
            f"{all_hot_blocks}-block All-HOT capacity. Reaching the "
            f"{stop_blocks}-block stop target would demote "
            f"{medium_policy['projected_blocks_to_demote_to_stop']} blocks; "
            f"WARM-{PRIMARY_WARM_POOL_BLOCKS} "
            f"{'can' if medium_policy['warm_pool_can_hold_demotion'] else 'cannot'} "
            "hold that amount."
        )
    elif medium is not None:
        medium_note = (
            f"{medium} timestamp-ordered eligible sessions peak at "
            f"{medium_peak} retained blocks, which is below the Mixed-1024 "
            f"demotion-start threshold of {start_blocks} HOT blocks."
        )
    else:
        medium_note = "No medium-load prefix was selected."

    if smallest is not None:
        smallest_peak = prefix_pressure.get("peak_estimated_hot_blocks")
        all_hot_note = (
            f"{smallest} timestamp-ordered eligible sessions are the smallest "
            "prefix whose peak estimated retained demand exceeds the "
            f"{all_hot_blocks}-block All-HOT capacity "
            f"(peak {smallest_peak} blocks, only "
            f"{max(0, int(smallest_peak or 0) - all_hot_blocks)} blocks above "
            "capacity). That margin is not a robust high-pressure comparison."
        )
    else:
        all_hot_note = (
            "No timestamp-ordered prefix of the eligible pool has an "
            f"estimated retained-HOT peak above the {all_hot_blocks}-block "
            "All-HOT capacity."
        )

    high_note = high_load.get("note", "")
    if high_policy is not None:
        high_policy_note = (
            f"At the primary Mixed-1024 policy start={PRIMARY_START_UTILIZATION:g} "
            f"/ stop={PRIMARY_STOP_UTILIZATION:g}, the high-load peak of "
            f"{high_peak} blocks would demote "
            f"{high_policy['projected_blocks_to_demote_to_stop']} blocks to "
            f"reach the {stop_blocks}-block stop target. WARM-"
            f"{PRIMARY_WARM_POOL_BLOCKS} "
            f"{'can' if high_policy['warm_pool_can_hold_demotion'] else 'cannot'} "
            "hold that demotion; projected HOT occupancy after the maximum "
            f"possible demotion is "
            f"{high_policy['projected_hot_blocks_after_max_demotion']} blocks "
            f"({high_policy['projected_hot_utilization_after_max_demotion']:.3f} "
            "of Mixed HOT) if the WARM pool saturates."
        )
    else:
        high_policy_note = (
            "High-load policy pressure could not be computed because the "
            "selected prefix was not materialized."
        )

    warm_choice = None
    warm_reason = "No mixed WARM pool sizes were requested."
    mixed_configs = {
        name: item
        for name, item in capacity["configurations"].items()
        if item["kv_mode"] == "mixed"
    }
    if mixed_configs:
        largest = max(
            mixed_configs.values(),
            key=lambda item: item["warm_pool_blocks"],
        )
        gaps = session_stats.get("inter_turn_gap_seconds", {})
        median_gap = gaps.get("median") or 0.0
        p95_gap = gaps.get("p95") or 0.0
        preferred = largest
        if PRIMARY_WARM_POOL_BLOCKS in {
            int(item["warm_pool_blocks"]) for item in mixed_configs.values()
        }:
            if median_gap >= 30.0 or p95_gap >= 300.0 or high_peak > mixed_hot:
                preferred = next(
                    item
                    for item in mixed_configs.values()
                    if int(item["warm_pool_blocks"]) == PRIMARY_WARM_POOL_BLOCKS
                )
        warm_choice = int(preferred["warm_pool_blocks"])
        demote_to_stop = (
            high_policy["projected_blocks_to_demote_to_stop"]
            if high_policy
            else max(0, high_peak - stop_blocks)
        )
        warm_reason = (
            f"High-load peak estimated retained demand is {high_peak} blocks "
            f"across {high_retained} retained sessions. Excess over All-HOT "
            f"is {max(0, high_peak - all_hot_blocks)} blocks; Mixed-1024 total "
            f"logical capacity is {mixed_total} blocks. Reaching the primary "
            f"stop target requires demoting {demote_to_stop} blocks. "
            f"WARM-{warm_choice} provides {preferred['warm_pool_blocks']} "
            "logical overflow slots. Inter-turn gaps are long enough that "
            "idle sessions, not only in-flight decode, dominate retained KV."
        )

    practical_scales = [scale for scale in time_scales if scale < 1.0]
    primary_scale = (
        0.01
        if 0.01 in time_scales
        else (practical_scales[0] if practical_scales else time_scales[0])
    )
    fallback_scale = (
        0.005
        if 0.005 in time_scales and 0.005 != primary_scale
        else (
            min(practical_scales)
            if practical_scales and min(practical_scales) != primary_scale
            else primary_scale
        )
    )
    bias_note = selection_bias.get("interpretation", "")
    follow_up = selection_bias.get("follow_up", "")
    return {
        "primary_max_input_length": primary_max_input_length,
        "eligible_session_pool": eligible_count,
        "smallest_pressure_session_count": smallest,
        "smallest_requested_count_exceeding_all_hot": (
            min(requested_exceeding) if requested_exceeding else None
        ),
        "timestamp_ordered_prefix_pressure": prefix_pressure,
        "high_load_prefix_search": high_load,
        "smoke_load_session_count": smoke,
        "medium_load_session_count": medium,
        "high_load_session_count": high,
        "primary_time_scale": primary_scale,
        "fallback_time_scale": fallback_scale,
        "primary_warm_pool_blocks": warm_choice,
        "primary_demotion_policy": {
            "start_utilization": PRIMARY_START_UTILIZATION,
            "stop_utilization": PRIMARY_STOP_UTILIZATION,
            "mixed_hot_blocks": mixed_hot,
            "mixed_hot_start_threshold_blocks": start_blocks,
            "mixed_hot_stop_target_blocks": stop_blocks,
            "mixed_total_logical_blocks": mixed_total,
        },
        "all_hot_capacity_blocks": all_hot_blocks,
        "selection_bias_material": selection_bias.get("material_bias"),
        "evidence": [
            (
                f"Mixed-1024 has {mixed_hot} HOT blocks. Demotion under the "
                f"primary policy begins at {start_blocks} HOT blocks "
                f"({PRIMARY_START_UTILIZATION:g} utilization) and targets "
                f"{stop_blocks} HOT blocks ({PRIMARY_STOP_UTILIZATION:g}). "
                f"{max(start_blocks - 1, 0)}/{mixed_hot} < "
                f"{PRIMARY_START_UTILIZATION:g} while {start_blocks}/"
                f"{mixed_hot} >= {PRIMARY_START_UTILIZATION:g}."
            ),
            medium_note,
            all_hot_note,
            high_note,
            high_policy_note,
            (
                "Retained-session overlap is invariant to uniform time "
                "scaling. Time scale only changes simulated arrival density "
                "and wall-clock replay duration."
            ),
            (
                f"High-load scheduled duration is approximately "
                f"{span * primary_scale:.1f}s at time_scale={primary_scale} "
                f"and {span * fallback_scale:.1f}s at time_scale="
                f"{fallback_scale}, versus {span:.1f}s at time_scale=1.0."
            ),
            warm_reason,
            bias_note,
            follow_up,
            (
                "Selection is the same timestamp-ordered rule as "
                "run_qwen_bailian_replay.select_sessions; --seed does not "
                "shuffle sessions."
            ),
        ],
        "replay_cli_hints": {
            "max_input_length": primary_max_input_length,
            "min_turns": 2,
            "request_type": "text",
            "max_sessions_smoke": smoke,
            "max_sessions_smallest_pressure": smallest,
            "max_sessions_medium": medium,
            "max_sessions_high": high,
            "time_scale": primary_scale,
            "fallback_time_scale": fallback_scale,
            "warm_pool_blocks": warm_choice,
            "demotion_start_utilization": PRIMARY_START_UTILIZATION,
            "demotion_stop_utilization": PRIMARY_STOP_UTILIZATION,
            "total_kv_budget_bytes": capacity["total_kv_budget_bytes"],
        },
    }


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(FIGURE_STYLE)
    return plt


def _save_figure(fig: Any, plt: Any, directory: Path, stem: str) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    png_path = directory / f"{stem}.png"
    pdf_path = directory / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _figure_session_counts(
    session_counts: list[int],
    selections: dict[str, Any],
) -> list[int]:
    preferred = [25, 250, 542, 750, 1000, 1500]
    available = [count for count in preferred if str(count) in selections]
    analyzed = sorted(int(key) for key in selections)
    if analyzed:
        if analyzed[0] not in available:
            available.insert(0, analyzed[0])
        if analyzed[-1] not in available:
            available.append(analyzed[-1])
    in_band: list[tuple[int, int]] = []
    midpoint = (
        PREFERRED_HIGH_LOAD_PEAK_MIN_BLOCKS + PREFERRED_HIGH_LOAD_PEAK_MAX_BLOCKS
    ) / 2
    for key, selection in selections.items():
        peak = int(selection.get("peak_estimated_hot_blocks") or 0)
        count = int(key)
        if (
            PREFERRED_HIGH_LOAD_PEAK_MIN_BLOCKS
            <= peak
            <= PREFERRED_HIGH_LOAD_PEAK_MAX_BLOCKS
        ):
            in_band.append((abs(peak - midpoint), count))
    if in_band:
        best = min(in_band)[1]
        if best not in available:
            available.append(best)
    if not available:
        available = list(session_counts)
    return sorted(dict.fromkeys(available))


def _series_colors(count: int) -> list[Any]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    return [cmap(index % 10) for index in range(count)]


def generate_figures(
    *,
    eligible: dict[int, list[BailianRecord]],
    selections: dict[str, Any],
    capacity: dict[str, Any],
    figures_dir: Path,
    session_counts: list[int],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    plt = _configure_matplotlib()
    turns = [len(session) for session in eligible.values()]
    first_input = [session[0].input_length for session in eligible.values()]
    final_input = [session[-1].input_length for session in eligible.values()]
    gaps = [
        gap
        for session in eligible.values()
        for gap in inter_turn_gaps_seconds(session)
        if gap > 0
    ]
    zero_gaps = sum(
        1
        for session in eligible.values()
        for gap in inter_turn_gaps_seconds(session)
        if gap == 0
    )
    figure_data: dict[str, Any] = {}
    figure_paths: dict[str, dict[str, str]] = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    max_turns = max(turns) if turns else 1
    bins = range(1, max_turns + 2)
    ax.hist(turns, bins=bins, color=COLOR_PRIMARY, align="left", rwidth=0.9)
    ax.set_title("Turns per Eligible Text-Only Multi-Turn Session")
    ax.set_xlabel("Turns per session")
    ax.set_ylabel("Number of sessions")
    ax.set_ylim(bottom=0)
    figure_data["turns_per_session"] = histogram_data(turns, bins=list(bins))
    figure_paths["turns_per_session"] = _save_figure(
        fig, plt, figures_dir, "turns_per_session"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        first_input,
        bins=30,
        color=COLOR_SECONDARY,
        alpha=0.65,
        label="First-turn cumulative input length",
    )
    ax.hist(
        final_input,
        bins=30,
        color=COLOR_PRIMARY,
        alpha=0.55,
        label="Final/max cumulative context length",
    )
    ax.set_title("Cumulative Context Length of Eligible Sessions")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Number of sessions")
    ax.set_ylim(bottom=0)
    ax.legend()
    figure_data["context_length_distribution"] = {
        "first_input_length": histogram_data(first_input, bins=30),
        "final_input_length": histogram_data(final_input, bins=30),
        "note": (
            "input_length is cumulative context, not an incremental turn "
            "delta. Final and maximum lengths match for monotonic sessions."
        ),
    }
    figure_paths["context_length_distribution"] = _save_figure(
        fig, plt, figures_dir, "context_length_distribution"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    if gaps:
        import numpy as np

        positive = [gap for gap in gaps if gap > 0]
        min_gap = min(positive)
        max_gap = max(positive)
        bins = np.logspace(math.log10(min_gap), math.log10(max_gap), 30)
        ax.hist(positive, bins=bins, color=COLOR_PRIMARY)
        ax.set_xscale("log")
        figure_data["inter_turn_gap_distribution"] = histogram_data(
            positive, bins=[float(edge) for edge in bins]
        )
    else:
        ax.text(0.5, 0.5, "No positive inter-turn gaps", ha="center")
        figure_data["inter_turn_gap_distribution"] = histogram_data([], bins=10)
    ax.set_title("Inter-Turn Idle Gaps of Eligible Sessions")
    ax.set_xlabel("Gap between consecutive turns (seconds, log scale)")
    ax.set_ylabel("Number of turn-to-turn gaps")
    ax.set_ylim(bottom=0)
    figure_data["inter_turn_gap_distribution"]["zero_gap_count"] = zero_gaps
    figure_paths["inter_turn_gap_distribution"] = _save_figure(
        fig, plt, figures_dir, "inter_turn_gap_distribution"
    )

    plot_counts = _figure_session_counts(session_counts, selections)
    fig, ax = plt.subplots(figsize=(9, 5))
    retained_series: dict[str, Any] = {
        "time_seconds": None,
        "series": {},
        "plotted_session_counts": plot_counts,
    }
    for color, count in zip(_series_colors(len(plot_counts)), plot_counts):
        selection = selections[str(count)]
        retention = selection["arrival_and_retention"]
        start = retention["trace_start_timestamp"]
        end = retention["trace_end_timestamp"]
        resampled = resample_step_maxima(
            retention["event_samples"],
            start=start,
            end=end,
            time_scale=1.0,
        )
        times = list(range(len(resampled["retained_sessions"])))
        if retained_series["time_seconds"] is None:
            retained_series["time_seconds"] = times
        ax.plot(
            times,
            resampled["retained_sessions"],
            color=color,
            linewidth=1.6,
            label=f"{count} sessions",
        )
        retained_series["series"][str(count)] = resampled["retained_sessions"]
    ax.set_title("Retained Idle Sessions Over Trace Time")
    ax.set_xlabel("Trace time (seconds)")
    ax.set_ylabel("Retained sessions (not GPU inference concurrency)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2)
    retained_series["note"] = (
        "A session is retained between its first and final turn. Values are "
        "1-second bucket maxima in unscaled trace time. Overlap is invariant "
        "to uniform time scaling."
    )
    figure_data["retained_sessions_over_time"] = retained_series
    figure_paths["retained_sessions_over_time"] = _save_figure(
        fig, plt, figures_dir, "retained_sessions_over_time"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    hot_series: dict[str, Any] = {
        "time_seconds": None,
        "series": {},
        "plotted_session_counts": plot_counts,
    }
    all_hot_blocks = int(capacity["configurations"]["all_hot"]["hot_blocks"])
    mixed_1024 = capacity["configurations"].get("mixed_warm_1024")
    start_blocks = None
    stop_blocks = None
    mixed_total = None
    if mixed_1024 is not None:
        mixed_hot = int(mixed_1024["hot_blocks"])
        start_blocks = mixed_hot_start_threshold_blocks(
            mixed_hot, PRIMARY_START_UTILIZATION
        )
        stop_blocks = mixed_hot_stop_target_blocks(
            mixed_hot, PRIMARY_STOP_UTILIZATION
        )
        mixed_total = int(mixed_1024["total_logical_blocks"])
    for color, count in zip(_series_colors(len(plot_counts)), plot_counts):
        selection = selections[str(count)]
        retention = selection["arrival_and_retention"]
        start = retention["trace_start_timestamp"]
        end = retention["trace_end_timestamp"]
        resampled = resample_step_maxima(
            retention["event_samples"],
            start=start,
            end=end,
            time_scale=1.0,
        )
        times = list(range(len(resampled["estimated_hot_blocks"])))
        if hot_series["time_seconds"] is None:
            hot_series["time_seconds"] = times
        ax.plot(
            times,
            resampled["estimated_hot_blocks"],
            color=color,
            linewidth=1.6,
            label=f"{count} sessions",
        )
        hot_series["series"][str(count)] = resampled["estimated_hot_blocks"]
    if start_blocks is not None:
        ax.axhline(
            start_blocks,
            color=COLOR_START_THRESHOLD,
            linestyle="-.",
            linewidth=1.6,
            label=(
                f"Mixed-1024 {PRIMARY_START_UTILIZATION:g} start "
                f"({start_blocks} blocks)"
            ),
        )
    if stop_blocks is not None:
        ax.axhline(
            stop_blocks,
            color=COLOR_STOP_TARGET,
            linestyle=":",
            linewidth=1.8,
            label=(
                f"Mixed-1024 {PRIMARY_STOP_UTILIZATION:g} stop "
                f"({stop_blocks} blocks)"
            ),
        )
    ax.axhline(
        all_hot_blocks,
        color=COLOR_CAPACITY,
        linestyle="--",
        linewidth=1.8,
        label=f"All-HOT capacity ({all_hot_blocks} blocks)",
    )
    if mixed_total is not None:
        ax.axhline(
            mixed_total,
            color=COLOR_MIXED_TOTAL,
            linestyle="--",
            linewidth=1.5,
            label=f"Mixed-1024 total logical ({mixed_total} blocks)",
        )
    ax.set_title(
        "Estimated Retained HOT-Block Demand vs Mixed Pressure Thresholds"
    )
    ax.set_xlabel("Trace time (seconds)")
    ax.set_ylabel("Estimated retained HOT blocks (16 tokens/block)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2)
    hot_series["all_hot_capacity_blocks"] = all_hot_blocks
    hot_series["mixed_1024_start_threshold_blocks"] = start_blocks
    hot_series["mixed_1024_stop_target_blocks"] = stop_blocks
    hot_series["mixed_1024_total_logical_blocks"] = mixed_total
    hot_series["note"] = (
        "Demand uses ceil(cumulative input_length / 16) of the last arrived "
        "turn for each retained session. Intermediate turn lengths are not "
        "summed. Horizontal lines mark Mixed-1024 demotion start/stop, "
        "All-HOT capacity, and Mixed-1024 total logical capacity."
    )
    figure_data["retained_hot_blocks_vs_capacity"] = hot_series
    figure_paths["retained_hot_blocks_vs_capacity"] = _save_figure(
        fig, plt, figures_dir, "retained_hot_blocks_vs_capacity"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    tokens = []
    colors = [COLOR_PRIMARY, "#2e7d32", "#6a1b9a"]
    for key, label in (
        ("all_hot", "All-HOT"),
        ("mixed_warm_512", "Mixed WARM-512"),
        ("mixed_warm_1024", "Mixed WARM-1024"),
    ):
        config = capacity["configurations"].get(key)
        if config is None:
            continue
        labels.append(label)
        tokens.append(config["logical_token_capacity"])
    bars = ax.bar(labels, tokens, color=colors[: len(labels)])
    ax.set_title("Equal-Memory Logical Token Capacity (4 GiB Persistent KV)")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Logical token capacity (tokens)")
    ax.set_ylim(bottom=0, top=max(tokens) * 1.15 if tokens else 1)
    for bar, value in zip(bars, tokens):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
        )
    figure_data["equal_memory_logical_capacity"] = {
        "labels": labels,
        "logical_token_capacity": tokens,
    }
    figure_paths["equal_memory_logical_capacity"] = _save_figure(
        fig, plt, figures_dir, "equal_memory_logical_capacity"
    )
    return figure_data, figure_paths


def trace_schema_interpretation() -> dict[str, Any]:
    return {
        "session_identifier": (
            "root chat_id (parent_chat_id == -1). The trace has no verified "
            "user identifier, so this analysis reports sessions, not users."
        ),
        "fields": list(REQUIRED_TRACE_KEYS),
        "input_length_semantics": "cumulative_context_tokens",
        "input_length_is_not": "incremental_turn_delta",
        "evidence": [
            (
                "qwen_bailian_trace._extend_session_tokens treats "
                "input_length as the reconstructed session token length and "
                "emits only the suffix since the previous turn."
            ),
            (
                "BailianRecord.hash_ids length is ceil(input_length / 16), "
                "matching a full-context block hash list rather than a delta."
            ),
            (
                "Replay rejects a child whose input_length shrank or whose "
                "complete-block hash prefix differs from its parent."
            ),
            (
                "In the raw trace every parent-child pair has "
                "child.input_length > parent.input_length, and "
                "child.input_length - parent.input_length >= "
                "parent.output_length."
            ),
        ],
        "kv_occupancy_rule": (
            "Context-token demand of a session is its final cumulative "
            "input_length. Occupancy over time uses the last arrived "
            "cumulative input_length. Summing per-turn input_length values "
            "would double-count history."
        ),
        "concurrency_definitions": {
            "arriving_requests": (
                "Trace rows after time scaling, counted in one-second buckets."
            ),
            "retained_open_sessions": (
                "Sessions between first and final turn. Not GPU concurrency."
            ),
            "active_inference_requests": (
                "Cannot be known from arrival timestamps alone."
            ),
            "measured_runtime_concurrency": (
                "Requires a GPU replay; not estimated here."
            ),
        },
    }


def analyze_workload(
    parsed: ParsedTrace,
    *,
    seed: int,
    max_input_lengths: list[int],
    time_scales: list[float],
    session_counts: list[int],
    total_kv_budget_bytes: int,
    warm_pool_blocks: list[int],
    max_model_len: int,
    max_num_seqs: int,
    request_type: str | None,
    min_turns: int,
    figures_dir: Path | None = None,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    classified = classify_linear_sessions(parsed.records)
    complete = classified.complete
    complete_records = flatten_sessions(complete)
    type_counts = Counter(record.request_type for record in parsed.records)
    session_type_profiles: Counter[str] = Counter()
    for session in complete.values():
        types = tuple(sorted({record.request_type for record in session}))
        if len(types) == 1:
            session_type_profiles[types[0]] += 1
        else:
            session_type_profiles["mixed:" + ",".join(types)] += 1

    single_turn = {
        root_id: session
        for root_id, session in complete.items()
        if len(session) == 1
    }
    multi_turn = {
        root_id: session
        for root_id, session in complete.items()
        if len(session) >= 2
    }
    text_multi = filter_sessions(
        multi_turn,
        request_type=request_type,
        min_turns=min_turns,
        max_input_length=None,
    )
    filtered = {
        str(limit): filter_sessions(
            text_multi,
            request_type=request_type,
            min_turns=min_turns,
            max_input_length=limit,
        )
        for limit in max_input_lengths
    }
    primary_max_input = max_input_lengths[0]
    eligible = filtered[str(primary_max_input)]

    rejection_reasons = Counter(item["reason"] for item in parsed.rejections)
    if classified.duplicate_chat_ids:
        rejection_reasons["duplicate_chat_id"] += len(
            classified.duplicate_chat_ids
        )
    incomplete_reasons = Counter(
        item["reason"] for item in classified.incomplete
    )

    capacity = equal_memory_capacity_table(
        total_kv_budget_bytes=total_kv_budget_bytes,
        warm_pool_sizes=warm_pool_blocks,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    all_hot_blocks = int(capacity["configurations"]["all_hot"]["hot_blocks"])
    mixed_1024 = capacity["configurations"].get(
        f"mixed_warm_{PRIMARY_WARM_POOL_BLOCKS}"
    )
    mixed_total = (
        int(mixed_1024["total_logical_blocks"]) if mixed_1024 is not None else 0
    )
    prefix_index = PrefixPeakIndex(eligible)
    prefix_pressure = prefix_index.smallest_reaching(
        all_hot_blocks, comparison="gt"
    )
    prefix_targets = search_prefix_peak_targets(prefix_index, capacity)
    high_load = find_high_load_prefix(
        prefix_index,
        all_hot_blocks=all_hot_blocks,
        mixed_total_blocks=mixed_total,
    )
    extra_counts = [
        int(row["smallest_session_count"])
        for row in prefix_targets
        if row.get("smallest_session_count") is not None
    ]
    if high_load.get("session_count"):
        extra_counts.append(int(high_load["session_count"]))
    analyzed_counts = resolved_session_counts(
        session_counts,
        len(eligible),
        extra_counts=extra_counts,
    )
    selections = {
        str(count): attach_policy_pressure(
            selected_workload_demand(
                eligible,
                requested_session_count=count,
                time_scales=time_scales,
                all_hot_blocks=all_hot_blocks,
            ),
            capacity=capacity,
        )
        for count in analyzed_counts
    }
    eligible_retention = arrival_and_retention(eligible, time_scales)
    session_stats = {
        "complete_sessions": session_statistics(complete),
        "multi_turn_sessions": session_statistics(multi_turn),
        "text_only_multi_turn_sessions": session_statistics(text_multi),
        **{
            f"text_only_multi_turn_max_input_{limit}": session_statistics(
                filtered[str(limit)]
            )
            for limit in max_input_lengths
        },
    }
    bias_counts = [
        count
        for count in (
            *session_counts,
            25,
            *REQUIRED_PREFIX_SESSION_COUNTS,
            high_load.get("session_count"),
            len(eligible),
        )
        if isinstance(count, int) and 0 < count <= max(len(eligible), 0)
    ]
    selection_bias = selection_bias_report(
        eligible=eligible,
        prefix_counts=sorted(set(bias_counts)),
        high_load_session_count=high_load.get("session_count"),
    )
    recommendation = build_recommendation(
        session_counts=analyzed_counts,
        selections=selections,
        capacity=capacity,
        time_scales=time_scales,
        primary_max_input_length=primary_max_input,
        eligible_count=len(eligible),
        session_stats=session_stats[
            f"text_only_multi_turn_max_input_{primary_max_input}"
        ],
        prefix_pressure=prefix_pressure,
        high_load=high_load,
        selection_bias=selection_bias,
    )

    figure_data: dict[str, Any] = {}
    figure_paths: dict[str, dict[str, str]] = {}
    if figures_dir is not None:
        figure_data, figure_paths = generate_figures(
            eligible=eligible,
            selections=selections,
            capacity=capacity,
            figures_dir=figures_dir,
            session_counts=analyzed_counts,
        )
    eligible_retention.pop("event_samples", None)
    for selection in selections.values():
        selection["arrival_and_retention"].pop("event_samples", None)

    trace_meta = (
        trace_identifier(trace_path)
        if trace_path is not None and trace_path.exists()
        else None
    )
    return jsonable(
        {
            "schema_version": WORKLOAD_STATISTICS_SCHEMA_VERSION,
            "analysis_constants": {
                "seed": seed,
                "seed_affects_session_selection": False,
                "block_size_tokens": BLOCK_SIZE,
                "hot_bytes_per_block": hot_bytes_per_block(),
                "warm_bytes_per_slot": warm_bytes_per_slot(),
                "total_kv_budget_bytes": total_kv_budget_bytes,
                "max_input_lengths": max_input_lengths,
                "primary_max_input_length": primary_max_input,
                "time_scales": time_scales,
                "session_counts": session_counts,
                "analyzed_session_counts": analyzed_counts,
                "warm_pool_blocks": warm_pool_blocks,
                "demotion_policies": [
                    {
                        "start_utilization": start,
                        "stop_utilization": stop,
                    }
                    for start, stop in DEFAULT_DEMOTION_POLICIES
                ],
                "primary_demotion_policy": {
                    "start_utilization": PRIMARY_START_UTILIZATION,
                    "stop_utilization": PRIMARY_STOP_UTILIZATION,
                },
                "max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs,
                "request_type": request_type,
                "min_turns": min_turns,
                "kv_memory_formula_version": KV_MEMORY_FORMULA_VERSION,
            },
            "trace": trace_meta,
            "trace_schema_interpretation": trace_schema_interpretation(),
            "trace_counts": {
                "total_non_empty_rows": parsed.total_non_empty_rows,
                "blank_lines": parsed.blank_lines,
                "valid_rows": len(parsed.records),
                "complete_sessions": len(complete),
                "incomplete_sessions": len(classified.incomplete),
                "single_turn_sessions": len(single_turn),
                "multi_turn_sessions": len(multi_turn),
                "text_only_multi_turn_sessions": len(text_multi),
                "filtered_text_only_multi_turn_sessions": {
                    str(limit): len(filtered[str(limit)])
                    for limit in max_input_lengths
                },
                "request_type_counts": dict(type_counts),
                "complete_session_type_profiles": dict(session_type_profiles),
                "malformed_or_rejected_rows": len(parsed.rejections)
                + len(classified.duplicate_chat_ids),
                "rejection_reasons": dict(rejection_reasons),
                "incomplete_session_reasons": dict(incomplete_reasons),
                "rejections": parsed.rejections,
                "duplicate_chat_ids": classified.duplicate_chat_ids,
                "incomplete_sessions_detail": classified.incomplete,
            },
            "session_statistics": session_stats,
            "arrival_and_retention": {
                "population": (
                    f"text-only multi-turn sessions with max input_length "
                    f"<= {primary_max_input}"
                ),
                "session_count": len(eligible),
                **eligible_retention,
            },
            "workload_selections": selections,
            "policy_pressure_table": flatten_policy_pressure_rows(selections),
            "prefix_peak_targets": prefix_targets,
            "high_load_prefix_search": high_load,
            "selection_bias": selection_bias,
            "equal_memory_capacity": capacity,
            "recommendation": recommendation,
            "figure_data": figure_data,
            "figure_paths": figure_paths,
            "caveats": [
                "input_length is cumulative context and must not be summed across turns.",
                "Retained sessions are not GPU inference concurrency.",
                "Active in-flight decode cannot be recovered from arrival timestamps.",
                "Measured runtime concurrency requires a GPU replay.",
                (
                    "Retained HOT occupancy uses the last arrived cumulative "
                    "prompt. Trace output tokens are already included in the "
                    "next turn's input_length and are not added again."
                ),
                (
                    "--seed is recorded for replay compatibility but does not "
                    "change timestamp-ordered session selection."
                ),
                (
                    "Mixed demotion is utilization-based. Occupancy can start "
                    "demotion before the HOT pool is exhausted; the primary "
                    f"Mixed-1024 start threshold is "
                    f"{PRIMARY_START_UTILIZATION:g} of HOT blocks."
                ),
            ],
            "valid_complete_request_count": len(complete_records),
        }
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only Qwen-Bailian workload characterization"
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("datasets/qwen_bailian/raw/qwen_traceA_blksz_16.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-input-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048],
    )
    parser.add_argument(
        "--time-scales",
        type=float,
        nargs="+",
        default=[1.0, 0.01, 0.005],
    )
    parser.add_argument(
        "--session-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_SESSION_COUNTS),
    )
    parser.add_argument(
        "--total-kv-budget-bytes",
        type=int,
        default=DEFAULT_TOTAL_KV_BUDGET_BYTES,
    )
    parser.add_argument(
        "--warm-pool-blocks",
        type=int,
        nargs="+",
        default=[512, 1024],
    )
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument("--request-type", default="text")
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "experiments/results/qwen_bailian_analysis/workload_statistics.json"
        ),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("experiments/figures/qwen_bailian_workload"),
    )
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if any(value <= 0 for value in args.max_input_lengths):
        parser.error("--max-input-lengths must be positive")
    if any(value <= 0 for value in args.time_scales):
        parser.error("--time-scales must be positive")
    if any(value < 0 for value in args.session_counts):
        parser.error("--session-counts must be non-negative")
    if args.total_kv_budget_bytes <= 0:
        parser.error("--total-kv-budget-bytes must be positive")
    if any(value <= 0 for value in args.warm_pool_blocks):
        parser.error("--warm-pool-blocks must be positive")
    if args.max_model_len <= 0 or args.max_num_seqs <= 0 or args.min_turns <= 0:
        parser.error("model length, max-num-seqs and min-turns must be positive")
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    parsed = load_trace_records(args.trace)
    result = analyze_workload(
        parsed,
        seed=args.seed,
        max_input_lengths=list(args.max_input_lengths),
        time_scales=list(args.time_scales),
        session_counts=list(args.session_counts),
        total_kv_budget_bytes=args.total_kv_budget_bytes,
        warm_pool_blocks=list(args.warm_pool_blocks),
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        request_type=None if args.request_type == "all" else args.request_type,
        min_turns=args.min_turns,
        figures_dir=None if args.skip_figures else args.figures_dir,
        trace_path=args.trace,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    printable = {
        "output_json": str(args.output_json),
        "trace_counts": {
            key: result["trace_counts"][key]
            for key in (
                "total_non_empty_rows",
                "complete_sessions",
                "incomplete_sessions",
                "single_turn_sessions",
                "multi_turn_sessions",
                "text_only_multi_turn_sessions",
                "filtered_text_only_multi_turn_sessions",
                "request_type_counts",
                "malformed_or_rejected_rows",
                "rejection_reasons",
            )
        },
        "recommendation": result["recommendation"],
        "equal_memory_capacity": {
            name: {
                "hot_blocks": item["hot_blocks"],
                "total_logical_blocks": item["total_logical_blocks"],
                "logical_token_capacity": item["logical_token_capacity"],
                "capacity_gain_over_all_hot_percent": item[
                    "capacity_gain_over_all_hot_percent"
                ],
                "within_total_budget": item["within_total_budget"],
            }
            for name, item in result["equal_memory_capacity"][
                "configurations"
            ].items()
        },
        "figure_paths": result["figure_paths"],
    }
    print(json.dumps(printable, indent=2))
    return result


if __name__ == "__main__":
    main()
