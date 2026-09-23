from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.scripts.analyze_qwen_bailian_workload import (
    WORKLOAD_STATISTICS_SCHEMA_VERSION,
    analyze_workload,
    arrival_and_retention,
    classify_linear_sessions,
    cumulative_session_context_tokens,
    equal_memory_capacity_table,
    filter_sessions,
    find_high_load_prefix,
    inter_turn_gaps_seconds,
    load_trace_records,
    mixed_hot_start_threshold_blocks,
    mixed_hot_stop_target_blocks,
    parse_args,
    policy_aware_pressure,
    PrefixPeakIndex,
    select_first_n_sessions,
    selected_workload_demand,
    selection_hash,
    sweep_retained_occupancy,
    smallest_prefix_count_exceeding_capacity,
    smallest_prefix_reaching_peak,
    tokens_to_block_demand,
)
from experiments.scripts.qwen_bailian_trace import BLOCK_SIZE, BailianRecord
from experiments.scripts.run_qwen_bailian_replay import (
    derive_persistent_kv_budget,
    hot_bytes_per_block,
    selection_metadata,
    warm_bytes_per_slot,
)


def make_record(
    *,
    chat_id: int,
    parent_chat_id: int,
    timestamp: float,
    input_length: int,
    turn: int,
    hash_ids: tuple[int, ...] | None = None,
    request_type: str = "text",
    output_length: int = 4,
) -> BailianRecord:
    expected = math.ceil(input_length / BLOCK_SIZE)
    if hash_ids is None:
        hash_ids = tuple(range(chat_id * 100, chat_id * 100 + expected))
    return BailianRecord(
        chat_id=chat_id,
        parent_chat_id=parent_chat_id,
        timestamp=timestamp,
        input_length=input_length,
        output_length=output_length,
        request_type=request_type,
        turn=turn,
        hash_ids=hash_ids,
    )


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def linear_session(
    root_id: int,
    lengths: list[int],
    *,
    start: float = 0.0,
    gap: float = 10.0,
    request_type: str = "text",
) -> list[BailianRecord]:
    records: list[BailianRecord] = []
    parent = -1
    for index, length in enumerate(lengths, start=1):
        chat_id = root_id if index == 1 else root_id * 100 + index
        previous_hashes = records[-1].hash_ids if records else ()
        expected = math.ceil(length / BLOCK_SIZE)
        hashes = tuple(
            previous_hashes[i] if i < len(previous_hashes) else chat_id * 1000 + i
            for i in range(expected)
        )
        records.append(
            make_record(
                chat_id=chat_id,
                parent_chat_id=parent,
                timestamp=start + (index - 1) * gap,
                input_length=length,
                turn=index,
                hash_ids=hashes,
                request_type=request_type,
            )
        )
        parent = chat_id
    return records


def test_tokens_to_block_demand_rounding():
    assert tokens_to_block_demand(0).hot_blocks == 0
    one = tokens_to_block_demand(1)
    assert one.complete_blocks == 0
    assert one.incomplete_tail_tokens == 1
    assert one.incomplete_tail_blocks == 1
    assert one.hot_blocks == 1
    exact = tokens_to_block_demand(16)
    assert exact.complete_blocks == 1
    assert exact.incomplete_tail_blocks == 0
    assert exact.hot_blocks == 1
    overflow = tokens_to_block_demand(17)
    assert overflow.complete_blocks == 1
    assert overflow.incomplete_tail_tokens == 1
    assert overflow.incomplete_tail_blocks == 1
    assert overflow.hot_blocks == 2


def test_cumulative_context_does_not_double_count_turns():
    session = linear_session(1, [10, 25, 40])
    context = cumulative_session_context_tokens(session)

    assert context["first_input_length"] == 10
    assert context["final_input_length"] == 40
    assert context["max_input_length"] == 40
    assert context["reconstructed_from_deltas"] == 40
    assert context["naive_sum_of_turn_input_lengths"] == 75
    assert context["reconstructed_matches_final"]
    assert context["final_equals_maximum"]

    demand = selected_workload_demand(
        {1: session},
        requested_session_count=1,
        time_scales=[1.0],
        all_hot_blocks=2340,
    )
    assert demand["total_context_token_demand"] == 40
    assert demand["naive_double_counted_turn_input_tokens"] == 75
    assert demand["complete_block_demand"] == 2
    assert demand["incomplete_tail_block_demand"] == 1
    assert demand["final_hot_blocks_if_all_sessions_resident"] == 3


def test_inter_turn_gaps_and_return_after_thresholds():
    session = linear_session(7, [8, 20, 36], start=0.0, gap=0.0)
    session[1] = make_record(
        chat_id=session[1].chat_id,
        parent_chat_id=session[0].chat_id,
        timestamp=15.0,
        input_length=20,
        turn=2,
        hash_ids=session[1].hash_ids,
    )
    session[2] = make_record(
        chat_id=session[2].chat_id,
        parent_chat_id=session[1].chat_id,
        timestamp=100.0,
        input_length=36,
        turn=3,
        hash_ids=session[2].hash_ids,
    )
    gaps = inter_turn_gaps_seconds(session)
    assert gaps == [15.0, 85.0]

    stats = analyze_workload(
        SimpleNamespace(
            records=session,
            rejections=[],
            total_non_empty_rows=3,
            blank_lines=0,
        ),
        seed=0,
        max_input_lengths=[1024],
        time_scales=[1.0],
        session_counts=[1],
        total_kv_budget_bytes=4 * 1024**3,
        warm_pool_blocks=[512, 1024],
        max_model_len=2048,
        max_num_seqs=128,
        request_type="text",
        min_turns=2,
    )["session_statistics"]["text_only_multi_turn_max_input_1024"][
        "sessions_returning_after"
    ]
    assert stats["10_seconds"]["session_count"] == 1
    assert stats["30_seconds"]["session_count"] == 1
    assert stats["60_seconds"]["session_count"] == 1
    assert stats["5_minutes"]["session_count"] == 0


def test_retained_session_timeline_and_hot_blocks():
    first = linear_session(1, [16, 32], start=0.0, gap=10.0)
    second = linear_session(2, [16, 16], start=5.0, gap=7.0)
    occupancy = sweep_retained_occupancy({1: first, 2: second})

    by_time = {
        sample["timestamp"]: sample for sample in occupancy["samples"]
    }
    assert occupancy["peak_retained_sessions"] == 2
    assert by_time[0.0]["retained_sessions"] == 1
    assert by_time[0.0]["estimated_hot_blocks"] == 1
    assert by_time[5.0]["retained_sessions"] == 2
    assert by_time[5.0]["estimated_hot_blocks"] == 2
    assert by_time[10.0]["retained_sessions"] == 1
    assert by_time[12.0]["retained_sessions"] == 0
    assert occupancy["peak_estimated_hot_blocks"] == 2


def test_time_scale_does_not_change_retained_overlap():
    sessions = {
        1: linear_session(1, [16, 48], start=0.0, gap=100.0),
        2: linear_session(2, [16, 32], start=50.0, gap=100.0),
    }
    first = arrival_and_retention(sessions, [1.0])
    second = arrival_and_retention(sessions, [0.01])
    assert first["peak_retained_sessions"] == second["peak_retained_sessions"]
    assert (
        first["peak_estimated_hot_blocks"]
        == second["peak_estimated_hot_blocks"]
    )
    assert first["by_time_scale"]["1"]["simulated_duration_seconds"] == 150.0
    assert second["by_time_scale"]["0.01"]["simulated_duration_seconds"] == 1.5


def test_smallest_prefix_count_exceeding_capacity():
    first = linear_session(1, [32, 32], start=0.0, gap=10.0)
    second = linear_session(2, [32, 32], start=1.0, gap=10.0)
    third = linear_session(3, [32, 32], start=2.0, gap=10.0)
    sessions = {1: first, 2: second, 3: third}
    result = smallest_prefix_count_exceeding_capacity(
        sessions,
        all_hot_blocks=3,
    )
    assert result["smallest_session_count"] == 2
    assert result["peak_estimated_hot_blocks"] > 3
    assert result["peak_retained_sessions"] >= 2
    assert result["full_eligible_peak_estimated_hot_blocks"] == 6
    none = smallest_prefix_count_exceeding_capacity(
        sessions,
        all_hot_blocks=100,
    )
    assert none["smallest_session_count"] is None


def test_deterministic_selection_and_stable_hash():
    late = linear_session(20, [8, 24], start=10.0)
    early = linear_session(10, [8, 24], start=1.0)
    pool = {20: late, 10: early}

    first = select_first_n_sessions(pool, 1)
    second = select_first_n_sessions(pool, 1)
    assert list(first) == [10]
    assert selection_hash(first) == selection_hash(second)
    assert selection_hash(first)["selection_sha256"] == selection_metadata(
        flatten := [record for session in first.values() for record in session]
    )["selection_sha256"]
    assert len(flatten) == 2

    third = select_first_n_sessions(pool, 1)
    assert selection_hash(first)["selection_sha256"] == selection_hash(third)[
        "selection_sha256"
    ]


def test_empty_and_malformed_inputs(tmp_path):
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")
    empty = load_trace_records(empty_path)
    assert empty.records == []
    assert empty.total_non_empty_rows == 0

    classified = classify_linear_sessions([])
    assert classified.complete == {}
    assert classified.incomplete == []

    malformed_path = tmp_path / "bad.jsonl"
    malformed_path.write_text(
        "\n{not-json}\n"
        + json.dumps({"chat_id": 1})
        + "\n"
        + json.dumps(
            {
                "chat_id": 2,
                "parent_chat_id": -1,
                "timestamp": 0.0,
                "input_length": 8,
                "output_length": 1,
                "type": "text",
                "turn": 1,
                "hash_ids": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = load_trace_records(malformed_path)
    reasons = {item["reason"] for item in parsed.rejections}
    assert "json_decode_error" in reasons
    assert "missing_keys" in reasons
    assert "invalid_record" in reasons
    assert parsed.records == []

    result = analyze_workload(
        parsed,
        seed=0,
        max_input_lengths=[1024],
        time_scales=[1.0],
        session_counts=[25],
        total_kv_budget_bytes=4 * 1024**3,
        warm_pool_blocks=[512],
        max_model_len=2048,
        max_num_seqs=128,
        request_type="text",
        min_turns=2,
    )
    assert result["trace_counts"]["complete_sessions"] == 0
    assert result["workload_selections"]["25"]["selected_session_count"] == 0


def test_incomplete_missing_parent_and_branch():
    root = make_record(
        chat_id=1,
        parent_chat_id=-1,
        timestamp=0.0,
        input_length=8,
        turn=1,
        hash_ids=(1,),
    )
    orphan = make_record(
        chat_id=9,
        parent_chat_id=8,
        timestamp=1.0,
        input_length=16,
        turn=2,
        hash_ids=(2,),
    )
    child_a = make_record(
        chat_id=2,
        parent_chat_id=1,
        timestamp=1.0,
        input_length=16,
        turn=2,
        hash_ids=(1,),
    )
    child_b = make_record(
        chat_id=3,
        parent_chat_id=1,
        timestamp=2.0,
        input_length=16,
        turn=2,
        hash_ids=(1,),
    )
    classified = classify_linear_sessions([root, orphan, child_a, child_b])
    reasons = {item["reason"] for item in classified.incomplete}
    assert 1 not in classified.complete
    assert "missing_parent_or_cycle" in reasons
    assert "branches_or_skips_parent" in reasons


def test_capacity_formulas_all_hot_and_mixed():
    total = 4 * 1024**3
    table = equal_memory_capacity_table(
        total_kv_budget_bytes=total,
        warm_pool_sizes=[512, 1024],
        max_model_len=2048,
        max_num_seqs=128,
    )
    replay_all_hot = derive_persistent_kv_budget(
        SimpleNamespace(
            model="Qwen/Qwen3-0.6B",
            kv_mode="all-hot",
            warm_pool_blocks=0,
            total_kv_budget_bytes=total,
            max_model_len=2048,
            max_num_seqs=128,
        )
    )
    all_hot = table["configurations"]["all_hot"]
    mixed_512 = table["configurations"]["mixed_warm_512"]
    mixed_1024 = table["configurations"]["mixed_warm_1024"]

    assert hot_bytes_per_block() == 1_835_008
    assert warm_bytes_per_slot() == 946_176
    assert all_hot["hot_blocks"] == total // 1_835_008
    assert all_hot["hot_blocks"] == 2340
    assert all_hot["logical_token_capacity"] == 37_440
    assert all_hot["hot_blocks"] == replay_all_hot["derived_num_gpu_blocks"]

    assert mixed_1024["hot_blocks"] == 1812
    assert mixed_1024["total_logical_blocks"] == 2836
    assert mixed_1024["logical_token_capacity"] == 45_376
    assert mixed_1024["capacity_gain_over_all_hot_percent"] == pytest.approx(
        21.196581196581197
    )

    slot_table = 128 * 128 * 4
    warm_512 = 512 * 946_176
    expected_512_hot = (total - warm_512 - slot_table) // (1_835_008 + 28 * 4)
    assert mixed_512["hot_blocks"] == expected_512_hot
    assert mixed_512["total_logical_blocks"] == expected_512_hot + 512
    assert mixed_512["logical_token_capacity"] == (
        expected_512_hot + 512
    ) * BLOCK_SIZE

    for config in table["configurations"].values():
        assert config["within_total_budget"]
        assert config["actual_persistent_kv_bytes"] <= total
        assert config["budget_slack_bytes"] >= 0
    assert table["all_configurations_within_budget"]


def test_json_output_schema_and_stable_selection_hash(tmp_path):
    rows = []
    for root_id, start in ((1, 0.0), (2, 1.0), (3, 2.0)):
        for record in linear_session(root_id, [8, 24, 40], start=start, gap=5.0):
            rows.append(
                {
                    "chat_id": record.chat_id,
                    "parent_chat_id": record.parent_chat_id,
                    "timestamp": record.timestamp,
                    "input_length": record.input_length,
                    "output_length": record.output_length,
                    "type": record.request_type,
                    "turn": record.turn,
                    "hash_ids": list(record.hash_ids),
                }
            )
    path = write_jsonl(tmp_path / "trace.jsonl", rows)
    parsed = load_trace_records(path)
    first = analyze_workload(
        parsed,
        seed=0,
        max_input_lengths=[1024, 2048],
        time_scales=[1.0, 0.01],
        session_counts=[1, 2],
        total_kv_budget_bytes=4 * 1024**3,
        warm_pool_blocks=[512, 1024],
        max_model_len=2048,
        max_num_seqs=128,
        request_type="text",
        min_turns=2,
        trace_path=path,
    )
    second = analyze_workload(
        parsed,
        seed=99,
        max_input_lengths=[1024, 2048],
        time_scales=[1.0, 0.01],
        session_counts=[1, 2],
        total_kv_budget_bytes=4 * 1024**3,
        warm_pool_blocks=[512, 1024],
        max_model_len=2048,
        max_num_seqs=128,
        request_type="text",
        min_turns=2,
        trace_path=path,
    )
    required = {
        "schema_version",
        "analysis_constants",
        "trace_schema_interpretation",
        "trace_counts",
        "session_statistics",
        "arrival_and_retention",
        "workload_selections",
        "policy_pressure_table",
        "prefix_peak_targets",
        "high_load_prefix_search",
        "selection_bias",
        "equal_memory_capacity",
        "recommendation",
        "figure_data",
        "figure_paths",
        "caveats",
    }
    assert required <= first.keys()
    assert first["schema_version"] == WORKLOAD_STATISTICS_SCHEMA_VERSION
    assert first["schema_version"] == "1.1"
    policy = first["workload_selections"]["1"]["policy_pressure"]["mixed_warm_1024"]
    assert "start_0.8_stop_0.65" in policy
    assert "start_0.9_stop_0.75" in policy
    assert "start_0.7_stop_0.55" in policy
    assert first["selection_bias"]["metrics_compared"] == [
        "turns_per_session",
        "final_input_length",
        "inter_turn_gap_seconds",
    ]
    assert first["recommendation"]["primary_demotion_policy"][
        "mixed_hot_start_threshold_blocks"
    ] == 1450
    assert first["trace_schema_interpretation"]["input_length_semantics"] == (
        "cumulative_context_tokens"
    )
    assert first["trace_counts"]["complete_sessions"] == 3
    assert first["trace_counts"]["text_only_multi_turn_sessions"] == 3
    assert first["workload_selections"]["1"]["selection_sha256"] == second[
        "workload_selections"
    ]["1"]["selection_sha256"]
    assert first["analysis_constants"]["seed_affects_session_selection"] is False
    encoded = json.dumps(first)
    json.loads(encoded)
    assert "event_samples" not in first["arrival_and_retention"]


def test_text_filter_and_max_input_length():
    text = linear_session(1, [8, 20], start=0.0)
    image = linear_session(2, [8, 20], start=1.0, request_type="image")
    long_text = linear_session(3, [8, 2000], start=2.0)
    complete = classify_linear_sessions(text + image + long_text).complete
    filtered = filter_sessions(
        complete,
        request_type="text",
        min_turns=2,
        max_input_length=1024,
    )
    assert list(filtered) == [1]


def test_parse_args_defaults():
    args = parse_args([])
    assert args.max_input_lengths == [1024, 2048]
    assert args.time_scales == [1.0, 0.01, 0.005]
    assert args.session_counts == [25, 50, 100, 250, 542, 750, 1000, 1500]
    assert args.total_kv_budget_bytes == 4 * 1024**3
    assert args.warm_pool_blocks == [512, 1024]
    assert args.seed == 0


def test_mixed_utilization_thresholds_map_to_integer_blocks():
    assert mixed_hot_start_threshold_blocks(1812, 0.80) == 1450
    assert mixed_hot_stop_target_blocks(1812, 0.65) == 1177
    assert 1449 / 1812 < 0.80
    assert 1450 / 1812 >= 0.80
    assert mixed_hot_start_threshold_blocks(1812, 0.90) == 1631
    assert mixed_hot_stop_target_blocks(1812, 0.75) == 1359
    assert mixed_hot_start_threshold_blocks(1812, 0.70) == 1269
    assert mixed_hot_stop_target_blocks(1812, 0.55) == 996
    assert mixed_hot_start_threshold_blocks(0, 0.80) == 0


def test_policy_aware_pressure_crosses_start_without_exhausting_hot():
    result = policy_aware_pressure(
        peak_estimated_hot_blocks=1465,
        mixed_hot_blocks=1812,
        warm_pool_blocks=1024,
        start_utilization=0.80,
        stop_utilization=0.65,
        all_hot_blocks=2340,
    )
    assert result["mixed_hot_start_threshold_blocks"] == 1450
    assert result["mixed_hot_stop_target_blocks"] == 1177
    assert result["peak_estimated_retained_blocks"] == 1465
    assert result["projected_blocks_to_demote_to_stop"] == 1465 - 1177
    assert result["warm_pool_can_hold_demotion"] is True
    assert result["projected_hot_blocks_after_max_demotion"] == 1177
    assert result["projected_hot_utilization_after_max_demotion"] == pytest.approx(
        1177 / 1812
    )
    assert result["policy_would_start_demotion"] is True
    assert result["all_hot_capacity_exceeded"] is False
    assert result["mixed_total_logical_capacity_exceeded"] is False


def test_policy_aware_pressure_warm_cannot_reach_stop_and_mixed_overflow():
    cannot_stop = policy_aware_pressure(
        peak_estimated_hot_blocks=2675,
        mixed_hot_blocks=1812,
        warm_pool_blocks=1024,
        start_utilization=0.80,
        stop_utilization=0.65,
        all_hot_blocks=2340,
    )
    assert cannot_stop["projected_blocks_to_demote_to_stop"] == 2675 - 1177
    assert cannot_stop["warm_pool_can_hold_demotion"] is False
    assert cannot_stop["projected_hot_blocks_after_max_demotion"] == 2675 - 1024
    assert cannot_stop["all_hot_capacity_exceeded"] is True
    assert cannot_stop["mixed_total_logical_capacity_exceeded"] is False

    overflow = policy_aware_pressure(
        peak_estimated_hot_blocks=3000,
        mixed_hot_blocks=1812,
        warm_pool_blocks=1024,
        start_utilization=0.80,
        stop_utilization=0.65,
        all_hot_blocks=2340,
    )
    assert overflow["mixed_total_logical_capacity_exceeded"] is True
    assert overflow["projected_hot_blocks_after_max_demotion"] == 3000 - 1024


def test_smallest_prefix_reaching_peak_and_high_load_band():
    sessions = {
        index: linear_session(index, [16, 16], start=float(index), gap=20.0)
        for index in range(1, 11)
    }
    reaching = smallest_prefix_reaching_peak(sessions, target_blocks=5)
    assert reaching["smallest_session_count"] == 5
    assert reaching["peak_estimated_hot_blocks"] == 5
    none = smallest_prefix_reaching_peak(sessions, target_blocks=20)
    assert none["smallest_session_count"] is None

    found = find_high_load_prefix(
        PrefixPeakIndex(sessions),
        preferred_min_blocks=5,
        preferred_max_blocks=7,
        all_hot_blocks=3,
        mixed_total_blocks=8,
    )
    assert found["range_satisfied"] is True
    assert found["session_count"] == 6
    assert found["peak_estimated_hot_blocks"] == 6
    assert found["clearly_above_all_hot_capacity"] is True
    assert found["below_mixed_total_logical_capacity"] is True

    missing = find_high_load_prefix(
        PrefixPeakIndex(sessions),
        preferred_min_blocks=20,
        preferred_max_blocks=25,
        all_hot_blocks=3,
        mixed_total_blocks=8,
    )
    assert missing["range_satisfied"] is False
    assert missing["session_count"] == 10
    assert missing["closest_below_session_count"] == 10
    assert "No timestamp-ordered prefix" in missing["note"]


def test_selection_bias_detects_early_prefix_difference():
    records = []
    for root_id in (1, 2):
        records.extend(
            linear_session(
                root_id,
                [16, 32],
                start=float(root_id),
                gap=5.0,
            )
        )
    for root_id in (3, 4):
        records.extend(
            linear_session(
                root_id,
                [64, 80, 96, 112],
                start=100.0 + root_id,
                gap=200.0,
            )
        )
    result = analyze_workload(
        SimpleNamespace(
            records=records,
            rejections=[],
            total_non_empty_rows=len(records),
            blank_lines=0,
        ),
        seed=0,
        max_input_lengths=[1024],
        time_scales=[1.0],
        session_counts=[2],
        total_kv_budget_bytes=4 * 1024**3,
        warm_pool_blocks=[512, 1024],
        max_model_len=2048,
        max_num_seqs=128,
        request_type="text",
        min_turns=2,
    )
    bias = result["selection_bias"]
    prefix = bias["prefixes"]["2"]
    assert prefix["material_bias"] is True
    assert "final_input_length" in prefix["material_metrics"]
    assert bias["material_bias"] is True
    assert "peak-window" in bias["follow_up"]
