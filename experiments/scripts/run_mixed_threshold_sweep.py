#!/usr/bin/env python3
"""Sweep mixed-mode demotion thresholds on the Bailian capacity trace.

Each pair launches ``experiments/scripts/run_qwen_bailian_replay.py`` in
mixed mode for 200 sessions. The default grid is:

    (0.70, 0.55)
    (0.80, 0.65)  # current default
    (0.90, 0.75)
    (0.95, 0.80)

Result JSON files land in ``experiments/results/mixed_sweep/``. After the
runs finish, the script prints thresholds, completed sessions, timed_out,
median latency, and peak WARM.

Mixed performance replay also needs the published capacity settings and an
all-hot ``--baseline-json``. Those flags are included so the workload
matches the 200-session capacity run. This script does not launch the
correctness suite.

Run from the repository root:

    PYTHONPATH=$PWD python experiments/scripts/run_mixed_threshold_sweep.py --dry-run
"""

# parse_args: threshold grid, capacity trace, and baseline path.
# replay_argv: mixed capacity command for one demotion pair.
# metrics_from_replay: completed sessions, timeout, median latency, peak WARM.
# run_sweep: launch each pair, then print the summary table.

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
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "mixed_sweep"
DEFAULT_TRACE = (
    REPO_ROOT / "datasets" / "qwen_bailian" / "raw" / "qwen_traceA_blksz_16.jsonl"
)
DEFAULT_BASELINE_DIR = REPO_ROOT / "experiments" / "results" / "correctness_suite"

# Published Bailian capacity configuration. These match the 200-session
# all-hot result the mixed replay compares against.
DEFAULT_MAX_SESSIONS = 200
DEFAULT_TOTAL_KV_BUDGET_BYTES = 3758096384
DEFAULT_WARM_POOL_BLOCKS = 1024
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_TIME_SCALE = 0.05
DEFAULT_MAX_TOKENS_PER_TURN = 32
DEFAULT_MIN_TURNS = 1
DEFAULT_MAX_INPUT_LENGTH = 8176
DEFAULT_MAX_MODEL_LEN = 9216
DEFAULT_MAX_NUM_SEQS = 2048
DEFAULT_GPU_MEMORY_UTILIZATION = 0.6
DEFAULT_SEED = 0

# (start, stop). (0.80, 0.65) is the current mixed default.
DEFAULT_THRESHOLD_PAIRS = (
    (0.70, 0.55),
    (0.80, 0.65),
    (0.90, 0.75),
    (0.95, 0.80),
)

SUMMARY_COLUMNS = (
    "thresholds",
    "completed_sessions",
    "timed_out",
    "median_latency",
    "peak_warm",
)


def resolve_user_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def format_cli_number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def format_threshold(value: float) -> str:
    return f"{value:.2f}"


def default_baseline(sessions: int) -> Path:
    return DEFAULT_BASELINE_DIR / f"all-hot_sessions{sessions}.json"


def parse_threshold_pair(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.replace("/", ",").split(",")]
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"threshold pair {value!r} must be START,STOP"
        )
    try:
        start = float(parts[0])
        stop = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"threshold pair {value!r} is not two numbers"
        ) from exc
    if (
        not math.isfinite(start)
        or not math.isfinite(stop)
        or not 0.0 <= stop < start <= 1.0
    ):
        raise argparse.ArgumentTypeError(
            "pressure thresholds must satisfy 0 <= stop < start <= 1"
        )
    return start, stop


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run mixed mode on the Bailian capacity trace for several "
            "demotion threshold pairs, then print a summary table."
        )
    )
    parser.add_argument(
        "--threshold",
        action="append",
        dest="thresholds",
        type=parse_threshold_pair,
        metavar="START,STOP",
        help=(
            "Demotion pair as START,STOP. Repeat to replace the default "
            "grid: 0.70,0.55  0.80,0.65  0.90,0.75  0.95,0.80. "
            "0.80,0.65 is the current default."
        ),
    )
    parser.add_argument("--max-sessions", type=int, default=DEFAULT_MAX_SESSIONS)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help=(
            "All-hot result JSON for this session count. Default: "
            "experiments/results/correctness_suite/"
            "all-hot_sessions<sessions>.json."
        ),
    )
    parser.add_argument(
        "--total-kv-budget-bytes",
        type=int,
        default=DEFAULT_TOTAL_KV_BUDGET_BYTES,
    )
    parser.add_argument(
        "--warm-pool-blocks",
        type=int,
        default=DEFAULT_WARM_POOL_BLOCKS,
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run and do not launch them.",
    )
    args = parser.parse_args(argv)
    if args.thresholds is None:
        args.thresholds = list(DEFAULT_THRESHOLD_PAIRS)
    if args.max_sessions <= 0:
        parser.error("--max-sessions must be positive")
    if args.total_kv_budget_bytes <= 0:
        parser.error("--total-kv-budget-bytes must be positive")
    if args.warm_pool_blocks <= 0:
        parser.error("--warm-pool-blocks must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    labels = [pair_label(*pair) for pair in args.thresholds]
    if len(labels) != len(set(labels)):
        parser.error("threshold pairs must stay distinct at two decimal places")
    args.trace = resolve_user_path(args.trace)
    args.results_dir = resolve_user_path(args.results_dir)
    if args.baseline_json is None:
        args.baseline_json = default_baseline(args.max_sessions)
    else:
        args.baseline_json = resolve_user_path(args.baseline_json)
    return args


def pair_label(start: float, stop: float) -> str:
    return f"{format_threshold(start)}_{format_threshold(stop)}"


def result_path(args: argparse.Namespace, start: float, stop: float) -> Path:
    name = f"mixed_{args.max_sessions}_{pair_label(start, stop)}.json"
    return args.results_dir / name


def replay_argv(
    args: argparse.Namespace,
    start: float,
    stop: float,
    result_json: Path,
) -> list[str]:
    """Argv for one mixed capacity replay.

    The first flags are the sweep knobs. The rest are the published capacity
    settings required for a mixed performance run whose fingerprint matches
    the all-hot baseline.
    """
    return [
        "--experiment-mode",
        "performance",
        "--kv-mode",
        "mixed",
        "--max-sessions",
        str(args.max_sessions),
        "--demotion-start-utilization",
        format_threshold(start),
        "--demotion-stop-utilization",
        format_threshold(stop),
        "--warm-pool-blocks",
        str(args.warm_pool_blocks),
        "--total-kv-budget-bytes",
        str(args.total_kv_budget_bytes),
        "--trace",
        repo_relative(args.trace),
        "--result-json",
        repo_relative(result_json),
        "--timeout",
        format_cli_number(args.timeout),
        "--baseline-json",
        repo_relative(args.baseline_json),
        "--min-turns",
        str(DEFAULT_MIN_TURNS),
        "--max-input-length",
        str(DEFAULT_MAX_INPUT_LENGTH),
        "--max-model-len",
        str(DEFAULT_MAX_MODEL_LEN),
        "--max-num-seqs",
        str(DEFAULT_MAX_NUM_SEQS),
        "--time-scale",
        str(DEFAULT_TIME_SCALE),
        "--max-tokens-per-turn",
        str(DEFAULT_MAX_TOKENS_PER_TURN),
        "--gpu-memory-utilization",
        str(DEFAULT_GPU_MEMORY_UTILIZATION),
        "--seed",
        str(DEFAULT_SEED),
    ]


def replay_command(argv: list[str]) -> list[str]:
    return [sys.executable, str(REPLAY_SCRIPT), *argv]


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(REPO_ROOT)
    current = env.get("PYTHONPATH", "")
    parts = [root]
    if current:
        parts.append(current)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _runner_error_line(message: str) -> str:
    marker = "error:"
    for line in message.splitlines():
        index = line.find(marker)
        if index != -1:
            return line[index + len(marker):].strip()
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    return lines[-1] if lines else "rejected by the replay runner"


def replay_rejection(argv: list[str]) -> str | None:
    """Return the replay parser error, or None when the command is accepted."""
    from experiments.scripts.run_qwen_bailian_replay import parse_args as parse_replay

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            parse_replay(argv)
        except SystemExit:
            return stderr.getvalue().strip() or "rejected by the replay runner"
    return None


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


def file_mtime_ns(path: Path) -> int | None:
    if not path.is_file():
        return None
    return path.stat().st_mtime_ns


def metrics_from_replay(result: dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {
            "completed_sessions": None,
            "timed_out": None,
            "median_latency": None,
            "peak_warm": None,
        }
    termination = result.get("termination") or {}
    latency = result.get("all_turn_latency_seconds") or {}
    median = latency.get("p50") if isinstance(latency, dict) else None
    observation = result.get("hkv_observation") or {}
    peak_warm = (
        observation.get("peak_warm_blocks") if isinstance(observation, dict) else None
    )
    return {
        "completed_sessions": termination.get("completed_session_count"),
        "timed_out": termination.get("timed_out"),
        "median_latency": median,
        "peak_warm": peak_warm,
    }


def baseline_mismatch(args: argparse.Namespace, baseline: dict[str, Any]) -> str | None:
    """Return a message when the all-hot JSON cannot pair with this sweep."""
    fingerprint = baseline.get("experiment_fingerprint")
    fields = fingerprint.get("fields") if isinstance(fingerprint, dict) else None
    if not isinstance(fields, dict):
        return "baseline JSON has no experiment fingerprint"
    gpu = fields.get("gpu_memory_configuration")
    if not isinstance(gpu, dict):
        gpu = {}
    trace = fields.get("trace")
    trace_name = trace.get("name") if isinstance(trace, dict) else None
    expected = [
        ("experiment_mode", fields.get("experiment_mode"), "performance"),
        (
            "selected_session_count",
            fields.get("selected_session_count"),
            args.max_sessions,
        ),
        ("time_scale", fields.get("time_scale"), DEFAULT_TIME_SCALE),
        (
            "max_tokens_per_turn",
            fields.get("max_tokens_per_turn"),
            DEFAULT_MAX_TOKENS_PER_TURN,
        ),
        ("trace", trace_name, args.trace.name),
        (
            "total_kv_budget_bytes",
            gpu.get("total_kv_budget_bytes"),
            args.total_kv_budget_bytes,
        ),
        ("max_model_len", gpu.get("max_model_len"), DEFAULT_MAX_MODEL_LEN),
        ("max_num_seqs", gpu.get("max_num_seqs"), DEFAULT_MAX_NUM_SEQS),
        (
            "gpu_memory_utilization",
            gpu.get("gpu_memory_utilization"),
            DEFAULT_GPU_MEMORY_UTILIZATION,
        ),
    ]
    mismatched = [name for name, actual, wanted in expected if actual != wanted]
    if not mismatched:
        return None
    return (
        "all-hot baseline does not match this sweep in "
        + ", ".join(mismatched)
    )


def display_cell(column: str, row: dict[str, Any]) -> str:
    if column == "thresholds":
        return f"{format_threshold(row['start'])}/{format_threshold(row['stop'])}"
    if column == "completed_sessions":
        if row.get("missing"):
            return "failed"
        value = row["completed_sessions"]
        return "n/a" if value is None else str(value)
    if column == "timed_out":
        value = row["timed_out"]
        if value is None:
            return "n/a"
        return "yes" if value else "no"
    if column == "median_latency":
        value = row["median_latency"]
        return "n/a" if value is None else f"{float(value):.4f}"
    if column == "peak_warm":
        value = row["peak_warm"]
        return "n/a" if value is None else str(value)
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

    print(format_line(list(SUMMARY_COLUMNS)), flush=True)
    for line in rendered:
        print(format_line(line), flush=True)


def planned_runs(
    args: argparse.Namespace,
) -> list[tuple[float, float, Path, list[str]]]:
    runs: list[tuple[float, float, Path, list[str]]] = []
    for start, stop in args.thresholds:
        path = result_path(args, start, stop)
        runs.append((start, stop, path, replay_argv(args, start, stop, path)))
    return runs


def run_sweep(args: argparse.Namespace) -> int:
    if not REPLAY_SCRIPT.is_file():
        raise SystemExit(f"replay runner not found: {REPLAY_SCRIPT}")

    runs = planned_runs(args)
    for _start, _stop, _path, argv in runs:
        message = replay_rejection(argv)
        if message is not None:
            raise SystemExit(
                f"replay engine rejected the sweep command: "
                f"{_runner_error_line(message)}"
            )

    if args.dry_run:
        for _start, _stop, _path, argv in runs:
            run_command(replay_command(argv), dry_run=True)
        return 0

    baseline = load_json(args.baseline_json)
    if baseline is None:
        raise SystemExit(
            f"all-hot baseline JSON not found: {args.baseline_json}"
        )
    mismatch = baseline_mismatch(args, baseline)
    if mismatch is not None:
        raise SystemExit(f"{mismatch}: {args.baseline_json}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures = 0
    for start, stop, path, argv in runs:
        label = f"{format_threshold(start)}/{format_threshold(stop)}"
        before = file_mtime_ns(path)
        exit_code = run_command(replay_command(argv), dry_run=False)
        after = file_mtime_ns(path)
        # Ignore a result left by an earlier run when this launch did not
        # rewrite the file.
        fresh = after is not None and after != before
        result = load_json(path) if fresh else None
        row = {
            "start": start,
            "stop": stop,
            "missing": result is None,
            **metrics_from_replay(result),
        }
        rows.append(row)
        print(
            f"[sweep] {label} replay_exit={exit_code} "
            f"result={repo_relative(path)}",
            flush=True,
        )
        timed_out = row["timed_out"] is True
        if result is None or (exit_code not in (0, None) and not timed_out):
            failures += 1

    print(flush=True)
    print_summary_table(rows)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raise SystemExit(run_sweep(args))


if __name__ == "__main__":
    main()
