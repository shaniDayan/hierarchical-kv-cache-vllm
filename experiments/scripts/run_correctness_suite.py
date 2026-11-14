#!/usr/bin/env python3
"""Run the hierarchical KV correctness suite.

Each selected mode and session count launches the Bailian capacity replay
through ``experiments/scripts/run_qwen_bailian_replay.py``. After that run
finishes, the suite measures GSM8K exact-match accuracy with the ballast
method in ``experiments/scripts/hkv_gsm8k.py`` (via
``run_hkv_gsm8k_resume_quality.py``). The default dataset is the full
GSM8K split under ``datasets/gsm8k/openai_grade_school_math/``, and every
question in its test split is scored. ``--gsm8k-trace`` overrides that
directory. A timed GSM8K trace passed with ``--gsm8k-trace`` is preferred
when a runner for it exists; otherwise the suite falls back to ballast.
Replay and accuracy JSON files are written
under ``experiments/results/correctness_suite/``.

Run from the repository root:

    PYTHONPATH=$PWD python experiments/scripts/run_correctness_suite.py --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPLAY_SCRIPT = REPO_ROOT / "experiments" / "scripts" / "run_qwen_bailian_replay.py"
GSM8K_SCRIPT = (
    REPO_ROOT / "experiments" / "scripts" / "run_hkv_gsm8k_resume_quality.py"
)
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "correctness_suite"
# Full raw trace. The multiturn subset has only 9 sessions, so the published
# 125/150/200 session counts require this file. Override with --trace.
DEFAULT_TRACE = (
    REPO_ROOT / "datasets" / "qwen_bailian" / "raw" / "qwen_traceA_blksz_16.jsonl"
)

# Modes the suite knows how to ask for. Support is decided by the replay
# runner: a mode it rejects as an invalid --kv-mode choice is reported and
# skipped. uniform-int16 is listed so that gap stays visible.
SUITE_MODES = ("all-hot", "mixed", "uniform-int8", "uniform-int16")
# uniform-int8 is stable and is loaded from cache unless explicitly requested
# via --modes. Previous result JSONs in the results directory are not deleted.
DEFAULT_MODES = ("all-hot", "mixed")
CACHED_BASELINE_MODE = "uniform-int8"
# Full GSM8K split (train.jsonl and test.jsonl). --gsm8k-trace replaces this.
DEFAULT_GSM8K_DATASET = (
    REPO_ROOT / "datasets" / "gsm8k" / "openai_grade_school_math"
)
ACCURACY_METHOD_BALLAST = "ballast_gsm8k"
ACCURACY_METHOD_TIMED = "timed_gsm8k_trace"
DEFAULT_SESSION_COUNTS = (125, 150, 200)
# Published capacity configuration (README / 3.5 GiB Qwen3-0.6B runs).
DEFAULT_TOTAL_KV_BUDGET_BYTES = 3758096384
DEFAULT_TIME_SCALE = 0.05
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_TOKENS_PER_TURN = 32
DEFAULT_MIN_TURNS = 1
DEFAULT_MAX_INPUT_LENGTH = 8176
DEFAULT_MAX_MODEL_LEN = 9216
DEFAULT_MAX_NUM_SEQS = 2048
DEFAULT_GPU_MEMORY_UTILIZATION = 0.6
DEFAULT_WARM_POOL_BLOCKS = 1024
DEFAULT_DEMOTION_START = 0.80
DEFAULT_DEMOTION_STOP = 0.65

# Drop-in for the timed GSM8K workload. Leave this None until that trace
# exists. A dataset directory (train.jsonl and test.jsonl) enables the
# current ballast resume-quality script. A trace file is recognized below
# but not launched yet.
TIMED_GSM8K_TRACE: Path | None = None

SUMMARY_COLUMNS = (
    "mode",
    "sessions",
    "completed",
    "timed_out",
    "max_concurrent",
    "median_latency",
    "accuracy_exact_match",
)


def _from_cwd(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path.cwd() / path


def parse_modes(value: str) -> list[str]:
    modes: list[str] = []
    for part in value.split(","):
        mode = part.strip()
        if not mode:
            continue
        if mode not in SUITE_MODES:
            raise argparse.ArgumentTypeError(
                f"unknown mode {mode!r}; choose from {', '.join(SUITE_MODES)}"
            )
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise argparse.ArgumentTypeError("at least one mode is required")
    return [mode for mode in SUITE_MODES if mode in modes]


def parse_session_counts(value: str) -> list[int]:
    counts: list[int] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            count = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"session count {text!r} is not an integer"
            ) from exc
        if count <= 0:
            raise argparse.ArgumentTypeError("session counts must be positive")
        counts.append(count)
    if not counts:
        raise argparse.ArgumentTypeError("at least one session count is required")
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capacity replay plus GSM8K exact-match accuracy for each "
            "hierarchical KV mode and session count. Accuracy scores every "
            "question in the chosen dataset's test split."
        )
    )
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=parse_modes(",".join(DEFAULT_MODES)),
        help=(
            "Comma-separated modes: "
            + ", ".join(SUITE_MODES)
            + ". Default: all-hot,mixed. uniform-int8 is stable and is "
            "loaded from previous results unless requested here. all-hot "
            "runs before mixed so mixed can use its result as --baseline-json."
        ),
    )
    parser.add_argument(
        "--session-counts",
        type=parse_session_counts,
        default=parse_session_counts(
            ",".join(str(count) for count in DEFAULT_SESSION_COUNTS)
        ),
        help="Comma-separated session counts. Default: 125,150,200.",
    )
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--total-kv-budget-bytes",
        type=int,
        default=DEFAULT_TOTAL_KV_BUDGET_BYTES,
    )
    parser.add_argument("--time-scale", type=float, default=DEFAULT_TIME_SCALE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--max-tokens-per-turn",
        type=int,
        default=DEFAULT_MAX_TOKENS_PER_TURN,
    )
    parser.add_argument("--min-turns", type=int, default=DEFAULT_MIN_TURNS)
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=DEFAULT_MAX_INPUT_LENGTH,
    )
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--warm-pool-blocks",
        type=int,
        default=DEFAULT_WARM_POOL_BLOCKS,
    )
    parser.add_argument(
        "--demotion-start-utilization",
        type=float,
        default=DEFAULT_DEMOTION_START,
    )
    parser.add_argument(
        "--demotion-stop-utilization",
        type=float,
        default=DEFAULT_DEMOTION_STOP,
    )
    parser.add_argument(
        "--gsm8k-trace",
        type=Path,
        default=TIMED_GSM8K_TRACE,
        help=(
            "Timed GSM8K trace file, or a dataset directory with train.jsonl "
            "and test.jsonl for the ballast check. The default ballast "
            "directory is datasets/gsm8k/openai_grade_school_math/. A trace "
            "file is used when a timed runner is available; otherwise "
            "accuracy falls back to the ballast method."
        ),
    )
    parser.add_argument(
        "--accuracy-questions",
        type=int,
        default=None,
        help=(
            "Questions to score from the chosen dataset's test split. "
            "The default scores every question in test.jsonl."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run and do not launch them.",
    )
    args = parser.parse_args(argv)
    if args.time_scale <= 0 or args.timeout <= 0:
        parser.error("--time-scale and --timeout must be positive")
    if args.total_kv_budget_bytes <= 0:
        parser.error("--total-kv-budget-bytes must be positive")
    if args.max_tokens_per_turn <= 1:
        parser.error(
            "capacity replay requires --max-tokens-per-turn greater than 1"
        )
    if args.min_turns <= 0 or args.max_input_length < 0:
        parser.error("turn and input-length limits are invalid")
    if args.max_model_len <= 0 or args.max_num_seqs <= 0:
        parser.error("--max-model-len and --max-num-seqs must be positive")
    if args.warm_pool_blocks <= 0:
        parser.error("--warm-pool-blocks must be positive")
    if args.accuracy_questions is not None and args.accuracy_questions <= 0:
        parser.error("--accuracy-questions must be positive")
    args.trace = _from_cwd(args.trace)
    args.results_dir = _from_cwd(args.results_dir)
    if args.gsm8k_trace is not None:
        args.gsm8k_trace = _from_cwd(args.gsm8k_trace)
    start = args.demotion_start_utilization
    stop = args.demotion_stop_utilization
    if (
        not math.isfinite(start)
        or not math.isfinite(stop)
        or not 0.0 <= stop < start <= 1.0
    ):
        parser.error("pressure thresholds must satisfy 0 <= stop < start <= 1")
    return args


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(REPO_ROOT)
    current = env.get("PYTHONPATH", "")
    parts = [root]
    if current:
        parts.append(current)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def replay_argv(
    args: argparse.Namespace,
    *,
    mode: str,
    sessions: int,
    result_json: Path,
    baseline_json: Path | None,
) -> list[str]:
    argv = [
        "--experiment-mode",
        "performance",
        "--kv-mode",
        mode,
        "--trace",
        str(args.trace),
        "--result-json",
        str(result_json),
        "--total-kv-budget-bytes",
        str(args.total_kv_budget_bytes),
        "--max-sessions",
        str(sessions),
        "--min-turns",
        str(args.min_turns),
        "--max-input-length",
        str(args.max_input_length),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--time-scale",
        str(args.time_scale),
        "--max-tokens-per-turn",
        str(args.max_tokens_per_turn),
        "--timeout",
        str(args.timeout),
        "--seed",
        str(args.seed),
    ]
    if mode == "mixed":
        if baseline_json is None:
            raise ValueError("mixed replay requires an all-hot baseline JSON")
        argv.extend(
            [
                "--baseline-json",
                str(baseline_json),
                "--warm-pool-blocks",
                str(args.warm_pool_blocks),
                "--demotion-start-utilization",
                str(args.demotion_start_utilization),
                "--demotion-stop-utilization",
                str(args.demotion_stop_utilization),
            ]
        )
    return argv


def replay_command(argv: list[str]) -> list[str]:
    return [sys.executable, str(REPLAY_SCRIPT), *argv]


def _runner_error_line(message: str) -> str:
    marker = "error:"
    for line in message.splitlines():
        index = line.find(marker)
        if index != -1:
            return line[index + len(marker):].strip()
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    return lines[-1] if lines else "rejected by the replay runner"


def runner_rejection(argv: list[str]) -> str | None:
    """Return the replay parser error, or None when the command is accepted."""
    from experiments.scripts.run_qwen_bailian_replay import parse_args as parse_replay

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            parse_replay(argv)
        except SystemExit:
            return stderr.getvalue().strip() or "rejected by the replay runner"
    return None


def unsupported_mode_reason(args: argparse.Namespace, mode: str) -> str | None:
    """Skip modes the replay engine does not accept yet.

    Other parser failures are configuration errors and abort the suite before
    any GPU work starts.
    """
    argv = replay_argv(
        args,
        mode=mode,
        sessions=args.session_counts[0],
        result_json=Path("probe.json"),
        baseline_json=Path("probe-baseline.json"),
    )
    message = runner_rejection(argv)
    if message is None:
        return None
    detail = _runner_error_line(message)
    if "invalid choice" in message and "--kv-mode" in message:
        return (
            f"{mode} is not supported by the replay engine ({detail}); skipping"
        )
    raise SystemExit(f"replay engine rejected {mode}: {detail}")


def result_name(mode: str, sessions: int, *, accuracy: bool = False) -> str:
    suffix = "_accuracy" if accuracy else ""
    return f"{mode}_sessions{sessions}{suffix}.json"


def gsm8k_dataset_dir(candidate: Path | None) -> Path | None:
    if candidate is None or not candidate.is_dir():
        return None
    train = candidate / "train.jsonl"
    test = candidate / "test.jsonl"
    if train.is_file() and test.is_file():
        return candidate
    return None


def ballast_dataset(args: argparse.Namespace) -> Path:
    """Dataset directory for the ballast accuracy check."""
    candidate = args.gsm8k_trace
    if candidate is not None and candidate.is_dir():
        dataset = gsm8k_dataset_dir(candidate)
        if dataset is None:
            raise SystemExit(
                f"{candidate} is not a GSM8K dataset directory "
                "(expected train.jsonl and test.jsonl)"
            )
        return dataset
    return DEFAULT_GSM8K_DATASET


def dataset_test_question_count(dataset: Path) -> int:
    """Questions in test.jsonl, using the same skip rules as the GSM8K loader."""
    path = dataset / "test.jsonl"
    count = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip() and not line.lstrip().startswith("#"):
                count += 1
    if count <= 0:
        raise SystemExit(f"{path} contains no GSM8K test questions")
    return count


def accuracy_question_count(args: argparse.Namespace, dataset: Path) -> int:
    """How many test questions this accuracy run should score.

    An explicit --accuracy-questions value is a cap. Otherwise the run
    scores every question in the chosen dataset.
    """
    if args.accuracy_questions is not None:
        return args.accuracy_questions
    return dataset_test_question_count(dataset)


def build_timed_gsm8k_command(
    args: argparse.Namespace,
    *,
    mode: str,
    result_json: Path,
    baseline_json: Path | None,
) -> list[str] | None:
    """Command for a timed GSM8K trace, or None when that runner is not ready.

    ``--gsm8k-trace`` pointing at a trace file is the drop-in. Return the
    runner command here once the trace format exists; the suite prefers this
    over ballast. ``mode``, ``result_json``, and ``baseline_json`` are the
    same cell the ballast command receives.
    """
    candidate = args.gsm8k_trace
    if candidate is None or not candidate.is_file():
        return None
    # The timed runner is not implemented yet. Build its argv from mode,
    # result_json, and baseline_json when the trace format is ready.
    return None


def build_ballast_command(
    args: argparse.Namespace,
    *,
    mode: str,
    result_json: Path,
    baseline_json: Path | None,
) -> list[str]:
    """Invoke the ballast exact-match run in run_hkv_gsm8k_resume_quality.py.

    That script plans ballast with ``hkv_gsm8k.plan_ballast_allocation`` and
    scores answers with ``hkv_gsm8k.score_generated_text``.
    """
    dataset = ballast_dataset(args)
    command = [
        sys.executable,
        str(GSM8K_SCRIPT),
        "--kv-mode",
        mode,
        "--dataset-dir",
        str(dataset),
        "--result-json",
        str(result_json),
        "--num-questions",
        str(accuracy_question_count(args, dataset)),
        "--total-kv-budget-bytes",
        str(args.total_kv_budget_bytes),
        "--seed",
        str(args.seed),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if mode == "mixed":
        if baseline_json is None:
            raise ValueError("mixed accuracy check requires an all-hot baseline")
        command.extend(
            [
                "--baseline-json",
                str(baseline_json),
                "--warm-pool-blocks",
                str(args.warm_pool_blocks),
                "--demotion-start-utilization",
                str(args.demotion_start_utilization),
                "--demotion-stop-utilization",
                str(args.demotion_stop_utilization),
            ]
        )
    return command


def gsm8k_rejection(argv: list[str]) -> str | None:
    """Return the GSM8K parser error, or None when the command is accepted."""
    from experiments.scripts.run_hkv_gsm8k_resume_quality import (
        parse_args as parse_gsm8k,
    )

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            parse_gsm8k(argv)
        except SystemExit:
            return stderr.getvalue().strip() or "rejected by the GSM8K script"
    return None


def accuracy_skip_reason(
    args: argparse.Namespace,
    mode: str,
) -> str | None:
    """Skip accuracy when the ballast script does not accept this kv mode.

    Other parser failures are configuration errors and abort before any
    accuracy run starts.
    """
    dataset = ballast_dataset(args)
    if not (dataset / "train.jsonl").is_file() or not (dataset / "test.jsonl").is_file():
        return (
            f"GSM8K dataset {dataset} is missing train.jsonl or test.jsonl; "
            "skipping accuracy"
        )
    command = build_ballast_command(
        args,
        mode=mode,
        result_json=Path("probe-accuracy.json"),
        baseline_json=Path("probe-accuracy-baseline.json"),
    )
    message = gsm8k_rejection(command[2:])
    if message is None:
        return None
    detail = _runner_error_line(message)
    if "invalid choice" in message and "--kv-mode" in message:
        return (
            f"{mode} is not accepted by the GSM8K accuracy script "
            f"({GSM8K_SCRIPT.name}: {detail}); skipping accuracy"
        )
    raise SystemExit(f"GSM8K accuracy script rejected {mode}: {detail}")


def timed_trace_requested(args: argparse.Namespace) -> bool:
    candidate = args.gsm8k_trace
    return candidate is not None and candidate.is_file()


def run_command(command: list[str], *, dry_run: bool) -> int | None:
    print(shlex.join(command), flush=True)
    if dry_run:
        return None
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=child_env(),
        check=False,
    )
    return completed.returncode


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"could not read {path}: {exc}", flush=True)
        return None
    if not isinstance(payload, dict):
        print(f"could not read {path}: expected a JSON object", flush=True)
        return None
    return payload


def session_window(session: dict[str, Any]) -> tuple[float, float] | None:
    """In-flight interval from the first sent turn until the session finishes."""
    turns = session.get("turn_results") or []
    if not turns:
        return None
    start = turns[0].get("actual_send_seconds")
    if start is None:
        return None
    end = session.get("finished_seconds")
    if end is None:
        end = math.inf
    return float(start), float(end)


def max_concurrent_sessions(result: dict[str, Any]) -> int | None:
    events: list[tuple[float, int]] = []
    for session in result.get("sessions") or []:
        if not isinstance(session, dict):
            continue
        window = session_window(session)
        if window is None:
            continue
        start, end = window
        events.append((start, 1))
        events.append((end, -1))
    if not events:
        return None
    # Close a session before opening another that shares its endpoint.
    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    peak = 0
    for _timestamp, delta in events:
        current += delta
        if current > peak:
            peak = current
    return peak


def metrics_from_replay(result: dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {
            "completed": None,
            "timed_out": None,
            "max_concurrent": None,
            "median_latency": None,
        }
    termination = result.get("termination") or {}
    completed = termination.get("completed_session_count")
    latency = result.get("all_turn_latency_seconds") or {}
    median = latency.get("p50") if isinstance(latency, dict) else None
    return {
        "completed": completed,
        "timed_out": termination.get("timed_out"),
        "max_concurrent": max_concurrent_sessions(result),
        "median_latency": median,
    }


def accuracy_record(
    result: dict[str, Any] | None,
    *,
    method: str,
) -> dict[str, Any]:
    """Map a GSM8K result JSON onto the suite's accuracy object."""
    from experiments.scripts.hkv_gsm8k import EXCLUSION_NOT_WARM_BEFORE_RESUME

    quality = {}
    if result is not None and isinstance(result.get("quality"), dict):
        quality = result["quality"]
    if not quality:
        return skipped_accuracy(method, "GSM8K result has no quality summary")
    excluded = quality.get("excluded_sessions") or []
    excluded_not_warm = sum(
        1
        for item in excluded
        if isinstance(item, dict)
        and item.get("reason") == EXCLUSION_NOT_WARM_BEFORE_RESUME
    )
    record: dict[str, Any] = {
        "exact_match": quality.get("accuracy_exact_match"),
        "excluded_not_warm": excluded_not_warm,
        "method": method,
    }
    for key, value in quality.items():
        if key not in record:
            record[key] = value
    return record


def skipped_accuracy(method: str, reason: str) -> dict[str, Any]:
    return {
        "exact_match": None,
        "excluded_not_warm": None,
        "method": method,
        "skipped": reason,
    }


def empty_row(mode: str, sessions: int, *, status: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "sessions": sessions,
        "completed": None,
        "timed_out": None,
        "max_concurrent": None,
        "median_latency": None,
        "accuracy_exact_match": None,
        "accuracy": None,
        "status": status,
        "replay_result_json": None,
        "accuracy_result_json": None,
        "replay_exit_code": None,
    }


def display_cell(column: str, row: dict[str, Any]) -> str:
    if column == "mode":
        label = str(row["mode"])
        if row.get("source") == "cached":
            return f"{label} (cached)"
        return label
    if column == "sessions":
        return str(row["sessions"])
    if column == "completed":
        if row["status"] == "skipped":
            return "skipped"
        if row["status"] == "failed":
            return "failed"
        if row["completed"] is None:
            return "n/a"
        return str(row["completed"])
    if column == "timed_out":
        value = row["timed_out"]
        if value is None:
            return "n/a"
        return "yes" if value else "no"
    if column == "max_concurrent":
        value = row["max_concurrent"]
        return "n/a" if value is None else str(value)
    if column == "median_latency":
        value = row["median_latency"]
        return "n/a" if value is None else f"{float(value):.4f}"
    if column == "accuracy_exact_match":
        value = row["accuracy_exact_match"]
        return "n/a" if value is None else f"{float(value):.3f}"
    return "n/a"


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    rendered = [
        [display_cell(column, row) for column in SUMMARY_COLUMNS] for row in rows
    ]
    widths = [
        max(len(column), *(len(line[index]) for line in rendered))
        if rendered
        else len(column)
        for index, column in enumerate(SUMMARY_COLUMNS)
    ]

    def format_line(cells: list[str]) -> str:
        return "  ".join(
            cell.ljust(width) for cell, width in zip(cells, widths)
        ).rstrip()

    print(format_line(list(SUMMARY_COLUMNS)))
    for line in rendered:
        print(format_line(line))


def launch_accuracy(
    args: argparse.Namespace,
    *,
    mode: str,
    sessions: int,
    result_json: Path,
    baseline_json: Path | None,
    skip_reason: str | None,
) -> dict[str, Any] | None:
    """Run the accuracy check after a finished capacity replay.

    Returns the accuracy object for summary.json. Dry-run prints the command
    and returns None.
    """
    timed = build_timed_gsm8k_command(
        args,
        mode=mode,
        result_json=result_json,
        baseline_json=baseline_json,
    )
    if timed is not None:
        run_command(timed, dry_run=args.dry_run)
        if args.dry_run:
            return None
        return _accuracy_from_disk(
            result_json,
            method=ACCURACY_METHOD_TIMED,
        )

    if skip_reason:
        print(skip_reason, flush=True)
        if args.dry_run:
            return None
        return skipped_accuracy(ACCURACY_METHOD_BALLAST, skip_reason)

    if (
        mode == "mixed"
        and not args.dry_run
        and (baseline_json is None or not baseline_json.is_file())
    ):
        reason = (
            f"mixed sessions={sessions}: accuracy baseline "
            f"{baseline_json} is missing; skipping accuracy"
        )
        print(reason, flush=True)
        return skipped_accuracy(ACCURACY_METHOD_BALLAST, reason)

    command = build_ballast_command(
        args,
        mode=mode,
        result_json=result_json,
        baseline_json=baseline_json,
    )
    run_command(command, dry_run=args.dry_run)
    if args.dry_run:
        return None
    return _accuracy_from_disk(result_json, method=ACCURACY_METHOD_BALLAST)


def _accuracy_from_disk(path: Path, *, method: str) -> dict[str, Any]:
    result = load_json(path)
    if result is None:
        reason = f"accuracy result was not written: {path}"
        print(reason, flush=True)
        return skipped_accuracy(method, reason)
    return accuracy_record(result, method=method)


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    baseline_uniform_int8: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": rows,
        "baseline_uniform_int8": baseline_uniform_int8,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cached_session_count(path: Path) -> int | None:
    prefix = f"{CACHED_BASELINE_MODE}_sessions"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(".json"):
        return None
    if name.endswith("_accuracy.json"):
        return None
    middle = name[len(prefix) : -len(".json")]
    if not middle.isdigit():
        return None
    return int(middle)


def load_cached_uniform_int8_rows(
    results_dir: Path,
    session_counts: list[int],
) -> list[dict[str, Any]]:
    """Read previous uniform-int8 results without launching a new run.

    uniform-int8 is stable and is loaded from cache unless explicitly
    requested. The JSON files themselves are left in place.
    """
    found: dict[int, Path] = {}
    if results_dir.is_dir():
        for path in results_dir.glob(f"{CACHED_BASELINE_MODE}_sessions*.json"):
            sessions = _cached_session_count(path)
            if sessions is not None:
                found[sessions] = path
    ordered = [count for count in session_counts if count in found]
    ordered.extend(sorted(count for count in found if count not in session_counts))
    rows: list[dict[str, Any]] = []
    for sessions in ordered:
        result_json = found[sessions]
        replay = load_json(result_json)
        if replay is None:
            continue
        accuracy_json = results_dir / result_name(
            CACHED_BASELINE_MODE, sessions, accuracy=True
        )
        accuracy_payload = None
        if accuracy_json.is_file():
            accuracy_payload = accuracy_record(
                load_json(accuracy_json),
                method=ACCURACY_METHOD_BALLAST,
            )
        row = empty_row(CACHED_BASELINE_MODE, sessions, status="cached")
        row.update(metrics_from_replay(replay))
        row["source"] = "cached"
        row["replay_result_json"] = str(result_json)
        row["accuracy"] = accuracy_payload
        if accuracy_json.is_file():
            row["accuracy_result_json"] = str(accuracy_json)
        if accuracy_payload is not None:
            row["accuracy_exact_match"] = accuracy_payload.get("exact_match")
        rows.append(row)
    return rows


def rows_with_cached_baseline(
    rows: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Place each cached uniform-int8 row beside the fresh rows for that count."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in baseline:
        grouped.setdefault(int(item["sessions"]), []).append(item)
    combined: list[dict[str, Any]] = []
    seen: set[int] = set()
    index = 0
    while index < len(rows):
        sessions = int(rows[index]["sessions"])
        while index < len(rows) and int(rows[index]["sessions"]) == sessions:
            combined.append(rows[index])
            index += 1
        seen.add(sessions)
        combined.extend(grouped.get(sessions, []))
    for sessions in sorted(set(grouped) - seen):
        combined.extend(grouped[sessions])
    return combined


def run_suite(args: argparse.Namespace) -> int:
    if not REPLAY_SCRIPT.is_file():
        raise SystemExit(f"replay runner not found: {REPLAY_SCRIPT}")

    skip_reasons: dict[str, str] = {}
    for mode in args.modes:
        reason = unsupported_mode_reason(args, mode)
        if reason:
            skip_reasons[mode] = reason
            print(reason, flush=True)

    if timed_trace_requested(args) and build_timed_gsm8k_command(
        args,
        mode=args.modes[0],
        result_json=Path("probe-accuracy.json"),
        baseline_json=Path("probe-accuracy-baseline.json"),
    ) is None:
        print(
            f"timed GSM8K trace {args.gsm8k_trace} is not ready; "
            "accuracy will use the ballast method",
            flush=True,
        )

    accuracy_skips: dict[str, str] = {}
    for mode in args.modes:
        if mode in skip_reasons:
            continue
        reason = accuracy_skip_reason(args, mode)
        if reason:
            accuracy_skips[mode] = reason
            print(reason, flush=True)

    rows: list[dict[str, Any]] = []
    failures = 0
    if not args.dry_run:
        args.results_dir.mkdir(parents=True, exist_ok=True)

    for sessions in args.session_counts:
        all_hot_json = args.results_dir / result_name("all-hot", sessions)
        all_hot_accuracy = args.results_dir / result_name(
            "all-hot", sessions, accuracy=True
        )
        for mode in args.modes:
            if mode in skip_reasons:
                row = empty_row(mode, sessions, status="skipped")
                row["skip_reason"] = skip_reasons[mode]
                rows.append(row)
                continue

            result_json = args.results_dir / result_name(mode, sessions)
            baseline_json = all_hot_json if mode == "mixed" else None
            if (
                mode == "mixed"
                and not args.dry_run
                and not all_hot_json.is_file()
            ):
                print(
                    f"mixed sessions={sessions}: all-hot baseline "
                    f"{all_hot_json} is missing; skipping",
                    flush=True,
                )
                row = empty_row(mode, sessions, status="failed")
                row["replay_result_json"] = str(result_json)
                rows.append(row)
                failures += 1
                continue

            argv = replay_argv(
                args,
                mode=mode,
                sessions=sessions,
                result_json=result_json,
                baseline_json=baseline_json,
            )
            exit_code = run_command(replay_command(argv), dry_run=args.dry_run)
            if not args.dry_run:
                print(
                    f"[suite] {mode} sessions={sessions} replay_exit={exit_code}",
                    flush=True,
                )

            replay_result = None if args.dry_run else load_json(result_json)
            capacity_finished = args.dry_run or replay_result is not None
            accuracy_json = args.results_dir / result_name(
                mode, sessions, accuracy=True
            )
            accuracy_baseline = all_hot_accuracy if mode == "mixed" else None
            accuracy_payload: dict[str, Any] | None = None
            if not capacity_finished:
                print(
                    f"{mode} sessions={sessions}: capacity run did not finish; "
                    "skipping accuracy",
                    flush=True,
                )
            else:
                accuracy_payload = launch_accuracy(
                    args,
                    mode=mode,
                    sessions=sessions,
                    result_json=accuracy_json,
                    baseline_json=accuracy_baseline,
                    skip_reason=accuracy_skips.get(mode),
                )

            if args.dry_run:
                continue

            row = empty_row(
                mode,
                sessions,
                status="ran" if replay_result is not None else "failed",
            )
            row.update(metrics_from_replay(replay_result))
            row["replay_result_json"] = str(result_json)
            row["replay_exit_code"] = exit_code
            row["accuracy"] = accuracy_payload
            if accuracy_json.is_file():
                row["accuracy_result_json"] = str(accuracy_json)
            if accuracy_payload is not None:
                row["accuracy_exact_match"] = accuracy_payload.get("exact_match")
            if row["status"] == "failed":
                failures += 1
            rows.append(row)

    if args.dry_run:
        return 0

    # uniform-int8 is stable and is loaded from cache unless explicitly
    # requested. Existing result files are not deleted or rewritten.
    baseline_uniform_int8: list[dict[str, Any]] = []
    if CACHED_BASELINE_MODE not in args.modes:
        baseline_uniform_int8 = load_cached_uniform_int8_rows(
            args.results_dir,
            list(args.session_counts),
        )

    print(flush=True)
    print_summary_table(rows_with_cached_baseline(rows, baseline_uniform_int8))
    write_summary(
        args.results_dir / "summary.json",
        rows,
        baseline_uniform_int8,
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raise SystemExit(run_suite(args))


if __name__ == "__main__":
    main()
