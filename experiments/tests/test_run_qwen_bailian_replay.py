import asyncio
import json
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from experiments.scripts.qwen_bailian_trace import BailianRecord, ReplayTurn
from experiments.scripts.run_qwen_bailian_replay import (
    HKVReplayWorkerExtension,
    KV_MEMORY_FORMULA_VERSION,
    RESULT_SCHEMA_VERSION,
    SessionResult,
    TurnResult,
    build_engine_args_kwargs,
    build_experiment_config,
    build_experiment_fingerprint,
    build_reproducibility_metadata,
    build_runtime_memory_accounting,
    compare_baseline,
    compute_run_metrics,
    compute_turn_metrics_summary,
    derive_persistent_kv_budget,
    hot_bytes_per_block,
    parse_args,
    percentile,
    run_session,
    selection_metadata,
    select_sessions,
    token_comparison_validation_errors,
    tokenizer_vocab_size,
    validate_baseline_schema,
    validate_final_turn_generation_limits,
    validate_runtime_memory_accounting,
    validate_timing_and_turns,
    warm_bytes_per_slot,
    warm_pool_storage_bytes,
    warm_slot_table_storage_bytes,
)


def make_record(
    *,
    chat_id: int,
    parent_chat_id: int,
    timestamp: float,
    input_length: int,
    turn: int,
    hash_ids: tuple[int, ...],
    request_type: str = "text",
) -> BailianRecord:
    return BailianRecord(
        chat_id=chat_id,
        parent_chat_id=parent_chat_id,
        timestamp=timestamp,
        input_length=input_length,
        output_length=10,
        request_type=request_type,
        turn=turn,
        hash_ids=hash_ids,
    )


def make_turn_result(
    *,
    chat_id: int = 1,
    turn: int = 1,
    scheduled_send_seconds: float = 0.0,
    actual_send_seconds: float = 0.0,
    send_lateness_seconds: float = 0.0,
    first_output_seconds: float = 0.1,
    finished_seconds: float = 0.2,
    ttft_seconds: float = 0.1,
    latency_seconds: float = 0.2,
    generated_tokens: int = 1,
    generated_token_ids: list[int] | None = None,
    configured_max_tokens_per_turn: int = 1,
    effective_max_tokens: int = 1,
    is_resume: bool | None = None,
    finish_reason: str | None = None,
) -> TurnResult:
    return TurnResult(
        chat_id=chat_id,
        turn=turn,
        scheduled_send_seconds=scheduled_send_seconds,
        actual_send_seconds=actual_send_seconds,
        send_lateness_seconds=send_lateness_seconds,
        first_output_seconds=first_output_seconds,
        finished_seconds=finished_seconds,
        ttft_seconds=ttft_seconds,
        latency_seconds=latency_seconds,
        generated_tokens=generated_tokens,
        generated_token_ids=(
            [100 + turn] if generated_token_ids is None else generated_token_ids
        ),
        configured_max_tokens_per_turn=configured_max_tokens_per_turn,
        effective_max_tokens=effective_max_tokens,
        is_resume=turn > 1 if is_resume is None else is_resume,
        finish_reason=finish_reason,
    )


def make_session_result(root_chat_id: int, digest: str) -> SessionResult:
    return SessionResult(
        root_chat_id=root_chat_id,
        session_id=f"bailian-{root_chat_id}",
        turns=2,
        final_input_tokens=20,
        trace_output_tokens=10,
        scheduled_first_seconds=0.0,
        scheduled_last_seconds=1.0,
        generated_tokens=2,
        completed_turns=2,
        generated_token_sha256=digest,
        turn_results=[
            make_turn_result(
                chat_id=root_chat_id,
                turn=1,
                generated_token_ids=[101],
            ),
            make_turn_result(
                chat_id=root_chat_id + 1,
                turn=2,
                generated_token_ids=[102],
            ),
        ],
    )


def make_replay_args(**overrides):
    values = {
        "experiment_mode": "correctness",
        "kv_mode": "all-hot",
        "model": "Qwen/Qwen3-0.6B",
        "seed": 42,
        "time_scale": 0.05,
        "max_tokens_per_turn": 1,
        "demotion_start_utilization": None,
        "demotion_stop_utilization": None,
        "warm_pool_blocks": 0,
        "max_sessions": 5,
        "min_turns": 2,
        "max_input_length": 1024,
        "max_model_len": 2048,
        "max_num_seqs": 128,
        "gpu_memory_utilization": 0.6,
        "total_kv_budget_bytes": None,
    }
    values.update(overrides)
    if (
        values["experiment_mode"] == "performance"
        and "total_kv_budget_bytes" not in overrides
    ):
        values["total_kv_budget_bytes"] = 4 * 1024**3
    return SimpleNamespace(**values)


def make_comparison_result(**overrides):
    selection_sha256 = overrides.pop("selection_sha256", "selection")
    trace_sha256 = overrides.pop("trace_sha256", "trace")
    args = make_replay_args(**overrides)
    selection = {
        "selection_sha256": selection_sha256,
        "selected_session_count": 1,
        "selected_request_count": 2,
    }
    session = make_session_result(1, "digest")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_mode": args.experiment_mode,
        "experiment_config": build_experiment_config(args),
        "experiment_fingerprint": build_experiment_fingerprint(
            args,
            {"name": "trace.jsonl", "sha256": trace_sha256},
            selection,
        ),
        "sessions": [asdict(session)],
    }


def test_select_sessions_filters_whole_sessions_and_preserves_chains():
    text_root = make_record(
        chat_id=1,
        parent_chat_id=-1,
        timestamp=2.0,
        input_length=8,
        turn=1,
        hash_ids=(10,),
    )
    text_child = make_record(
        chat_id=2,
        parent_chat_id=1,
        timestamp=4.0,
        input_length=20,
        turn=2,
        hash_ids=(10, 11),
    )
    image_root = make_record(
        chat_id=3,
        parent_chat_id=-1,
        timestamp=0.0,
        input_length=8,
        turn=1,
        hash_ids=(30,),
        request_type="image",
    )
    image_child = make_record(
        chat_id=4,
        parent_chat_id=3,
        timestamp=1.0,
        input_length=20,
        turn=2,
        hash_ids=(30, 31),
        request_type="image",
    )

    selected = select_sessions(
        [image_child, text_child, image_root, text_root],
        request_type="text",
        min_turns=2,
        max_input_length=1024,
        max_sessions=None,
    )

    assert [record.chat_id for record in selected] == [1, 2]


def test_select_sessions_applies_limit_after_trace_time_sorting():
    late = make_record(
        chat_id=20,
        parent_chat_id=-1,
        timestamp=10.0,
        input_length=8,
        turn=1,
        hash_ids=(20,),
    )
    early = make_record(
        chat_id=10,
        parent_chat_id=-1,
        timestamp=1.0,
        input_length=8,
        turn=1,
        hash_ids=(10,),
    )

    selected = select_sessions(
        [late, early],
        request_type="text",
        min_turns=1,
        max_input_length=None,
        max_sessions=1,
    )

    assert [record.chat_id for record in selected] == [10]


def test_selection_metadata_is_stable_and_contains_counts():
    records = [
        make_record(
            chat_id=1,
            parent_chat_id=-1,
            timestamp=0.0,
            input_length=8,
            turn=1,
            hash_ids=(10,),
        ),
        make_record(
            chat_id=2,
            parent_chat_id=1,
            timestamp=1.0,
            input_length=16,
            turn=2,
            hash_ids=(10,),
        ),
    ]

    metadata = selection_metadata(records)

    assert metadata["selected_session_ids"] == [1]
    assert metadata["selected_session_count"] == 1
    assert metadata["selected_request_count"] == 2
    assert len(metadata["selection_sha256"]) == 64


def test_select_sessions_rejects_empty_selection():
    root = make_record(
        chat_id=1,
        parent_chat_id=-1,
        timestamp=0.0,
        input_length=8,
        turn=1,
        hash_ids=(10,),
    )

    with pytest.raises(ValueError, match="no Bailian sessions"):
        select_sessions(
            [root],
            request_type="text",
            min_turns=2,
            max_input_length=1024,
            max_sessions=None,
        )


def test_percentile_uses_sorted_nearest_rank_below():
    values = [0.4, 0.1, 0.3, 0.2]

    assert percentile(values, 0.50) == 0.2
    assert percentile(values, 0.95) == 0.3
    assert percentile([], 0.50) is None


def test_tokenizer_vocab_size_supports_property_and_len_fallback():
    assert tokenizer_vocab_size(SimpleNamespace(vocab_size=60000)) == 60000

    class LengthOnlyTokenizer:
        vocab_size = None

        def __len__(self):
            return 70000

    assert tokenizer_vocab_size(LengthOnlyTokenizer()) == 70000


def test_qwen_persistent_kv_byte_constants():
    assert hot_bytes_per_block() == 1_835_008
    assert warm_bytes_per_slot() == 946_176
    assert warm_pool_storage_bytes(128) == 121_110_528
    assert warm_slot_table_storage_bytes(2048, 128) == 65_536


def test_all_hot_explicit_budget_derivation():
    total = 4 * 1024**3
    budget = derive_persistent_kv_budget(
        make_replay_args(total_kv_budget_bytes=total)
    )
    expected_blocks = total // 1_835_008

    assert budget["derived_num_gpu_blocks"] == expected_blocks
    assert (
        budget["derived_hot_kv_budget_bytes"]
        == expected_blocks * 1_835_008
    )
    assert budget["derived_warm_kv_storage_bytes"] == 0
    assert budget["derived_hot_to_warm_map_storage_bytes"] == 0
    assert budget["derived_warm_slot_table_storage_bytes"] == 0
    assert budget["derived_actual_persistent_kv_bytes"] <= total


def test_mixed_explicit_budget_derivation_and_rounding():
    total = 4 * 1024**3
    args = make_replay_args(
        kv_mode="mixed",
        warm_pool_blocks=128,
        demotion_start_utilization=0.8,
        demotion_stop_utilization=0.65,
        total_kv_budget_bytes=total,
    )
    budget = derive_persistent_kv_budget(args)
    fixed_bytes = 121_110_528 + 65_536
    expected_blocks = (total - fixed_bytes) // (1_835_008 + 28 * 4)
    expected_maps = expected_blocks * 28 * 4
    expected_total = (
        expected_blocks * 1_835_008
        + expected_maps
        + fixed_bytes
    )

    assert budget["derived_num_gpu_blocks"] == expected_blocks
    assert budget["derived_hot_to_warm_map_storage_bytes"] == expected_maps
    assert budget["derived_actual_persistent_kv_bytes"] == expected_total
    assert expected_total <= total

    all_hot = derive_persistent_kv_budget(
        make_replay_args(total_kv_budget_bytes=total)
    )
    difference = abs(
        all_hot["derived_actual_persistent_kv_bytes"] - expected_total
    )
    assert difference <= max(
        all_hot["block_rounding_tolerance_bytes"],
        budget["block_rounding_tolerance_bytes"],
    )


def test_explicit_budget_engine_args_use_only_derived_hot_bytes():
    args = make_replay_args(
        experiment_mode="performance",
        max_tokens_per_turn=16,
        total_kv_budget_bytes=4 * 1024**3,
        gpu_memory_utilization=0.73,
        max_num_seqs=64,
    )
    budget = derive_persistent_kv_budget(args)

    kwargs = build_engine_args_kwargs(args)

    assert (
        kwargs["kv_cache_memory_bytes"]
        == budget["derived_hot_kv_budget_bytes"]
    )
    assert kwargs["gpu_memory_utilization"] == 0.73
    assert kwargs["max_num_seqs"] == 64


def test_correctness_without_explicit_budget_preserves_auto_sizing():
    args = make_replay_args()
    budget = derive_persistent_kv_budget(args)
    kwargs = build_engine_args_kwargs(args)

    assert budget["total_kv_budget_bytes"] is None
    assert budget["derived_num_gpu_blocks"] is None
    assert kwargs["kv_cache_memory_bytes"] is None
    assert kwargs["gpu_memory_utilization"] == args.gpu_memory_utilization


@pytest.mark.parametrize("total", [0, -1])
def test_non_positive_explicit_budget_is_rejected(total):
    with pytest.raises(ValueError, match="must be positive"):
        derive_persistent_kv_budget(
            make_replay_args(total_kv_budget_bytes=total)
        )


def test_too_small_mixed_budget_is_rejected():
    with pytest.raises(ValueError, match="must be less than"):
        derive_persistent_kv_budget(
            make_replay_args(
                kv_mode="mixed",
                warm_pool_blocks=128,
                total_kv_budget_bytes=121_176_064,
            )
        )


def test_unsupported_model_is_rejected_for_explicit_budget():
    with pytest.raises(ValueError, match="supports only"):
        derive_persistent_kv_budget(
            make_replay_args(
                model="other/model",
                total_kv_budget_bytes=4 * 1024**3,
            )
        )


def test_unique_storage_accounting_does_not_double_count_views():
    torch = pytest.importorskip("torch")
    hot = torch.zeros(32, dtype=torch.uint8)
    warm = torch.zeros(24, dtype=torch.uint8)
    mapping = torch.zeros(8, dtype=torch.int32)
    slot_table = torch.zeros(4, dtype=torch.int32)
    extension = HKVReplayWorkerExtension()
    extension.model_runner = SimpleNamespace(
        hkv_hot_kv_caches={"a": hot, "alias": hot.view(8, 4)},
        hkv_warm_kv_caches={"a": warm, "alias": warm.view(6, 4)},
        hkv_hot_to_warm_maps={"a": mapping, "alias": mapping.view(2, 4)},
        hkv_warm_slot_table=slot_table,
        hkv_warm_migration_manager=None,
        kv_cache_config=SimpleNamespace(num_blocks=8),
    )

    state = extension.inspect_hkv_replay()

    assert state["num_gpu_blocks"] == 8
    assert state["hot_kv_storage_bytes"] == 32
    assert state["warm_kv_storage_bytes"] == 24
    assert state["hot_to_warm_map_storage_bytes"] == 32
    assert state["warm_slot_table_storage_bytes"] == 16
    assert state["actual_persistent_kv_bytes"] == 104


def test_all_hot_runtime_accounting_uses_ordinary_kv_caches():
    torch = pytest.importorskip("torch")
    hot = torch.zeros(32, dtype=torch.uint8)
    extension = HKVReplayWorkerExtension()
    extension.model_runner = SimpleNamespace(
        kv_caches={
            "layers": [hot, hot.view(8, 4)],
            "nested": {"alias": hot.view(4, 8), "ignored": None},
            "scalars": {"count": 7, "label": "hot"},
        },
        kv_cache_config=SimpleNamespace(num_blocks=1),
        hkv_warm_migration_manager=None,
    )

    assert not hasattr(extension.model_runner, "hkv_hot_kv_caches")
    assert not hasattr(extension.model_runner, "hkv_warm_kv_caches")
    assert not hasattr(extension.model_runner, "hkv_hot_to_warm_maps")
    assert not hasattr(extension.model_runner, "hkv_warm_slot_table")

    state = extension.inspect_hkv_replay()

    assert state["hot_kv_storage_bytes"] == 32
    assert state["warm_kv_storage_bytes"] == 0
    assert state["hot_to_warm_map_storage_bytes"] == 0
    assert state["warm_slot_table_storage_bytes"] == 0
    assert state["actual_persistent_kv_bytes"] == 32


def test_empty_hkv_hot_kv_caches_falls_back_to_ordinary_kv_caches():
    torch = pytest.importorskip("torch")
    hot = torch.zeros(32, dtype=torch.uint8)
    extension = HKVReplayWorkerExtension()
    extension.model_runner = SimpleNamespace(
        hkv_hot_kv_caches={},
        kv_caches={"layer": hot},
        kv_cache_config=SimpleNamespace(num_blocks=1),
        hkv_warm_migration_manager=None,
    )

    state = extension.inspect_hkv_replay()

    assert state["hot_kv_storage_bytes"] == 32
    assert state["warm_kv_storage_bytes"] == 0
    assert state["hot_to_warm_map_storage_bytes"] == 0
    assert state["warm_slot_table_storage_bytes"] == 0
    assert state["actual_persistent_kv_bytes"] == 32


def test_hot_accounting_does_not_sum_aliased_hkv_and_kv_caches():
    torch = pytest.importorskip("torch")
    hot = torch.zeros(32, dtype=torch.uint8)
    extension = HKVReplayWorkerExtension()
    extension.model_runner = SimpleNamespace(
        hkv_hot_kv_caches={"a": hot, "alias": hot.view(8, 4)},
        kv_caches=[hot, hot.view(4, 8)],
        kv_cache_config=SimpleNamespace(num_blocks=1),
        hkv_warm_migration_manager=None,
    )

    state = extension.inspect_hkv_replay()

    assert state["hot_kv_storage_bytes"] == 32
    assert state["warm_kv_storage_bytes"] == 0
    assert state["hot_to_warm_map_storage_bytes"] == 0
    assert state["warm_slot_table_storage_bytes"] == 0
    assert state["actual_persistent_kv_bytes"] == 32


def test_compare_with_equivalent_baseline_accepts_ordered_tokens():
    result = make_comparison_result(kv_mode="mixed", warm_pool_blocks=128)
    baseline = make_comparison_result()

    comparison = compare_baseline(result, baseline)

    assert comparison["fingerprint_match"]
    assert comparison["ordered_turn_token_match"]
    assert comparison["token_comparison_role"] == "correctness_gate"
    assert comparison["policy_configuration"]["current"]["kv_mode"] == "mixed"
    assert comparison["policy_configuration"]["baseline"]["kv_mode"] == "all-hot"


def test_compare_with_baseline_reports_ordered_turn_mismatch():
    result = make_comparison_result()
    baseline = make_comparison_result()
    baseline["sessions"][0]["turn_results"][1]["generated_token_ids"] = [999]

    comparison = compare_baseline(result, baseline)

    assert not comparison["ordered_turn_token_match"]
    assert comparison["token_comparison_role"] == "correctness_gate"
    assert comparison["mismatched_turns"] == [
        {
            "root_chat_id": 1,
            "chat_id": 2,
            "turn": 2,
            "baseline_token_ids": [999],
            "current_token_ids": [102],
        }
    ]
    assert token_comparison_validation_errors("correctness", comparison)


def test_performance_token_mismatch_is_diagnostic_only():
    result = make_comparison_result(experiment_mode="performance")
    baseline = make_comparison_result(experiment_mode="performance")
    baseline["sessions"][0]["turn_results"][1]["generated_token_ids"] = [999]

    comparison = compare_baseline(result, baseline)

    assert not comparison["ordered_turn_token_match"]
    assert comparison["mismatched_turns"]
    assert comparison["token_comparison_role"] == "diagnostic_only"
    assert token_comparison_validation_errors("performance", comparison) == []


@pytest.mark.parametrize(
    ("field", "value", "difference"),
    [
        ("seed", 7, "seed"),
        ("model", "different/model", "model"),
        ("trace_sha256", "different-subset", "trace.sha256"),
        ("selection_sha256", "different-selection", "selection_sha256"),
        ("time_scale", 0.25, "time_scale"),
        ("max_tokens_per_turn", 4, "max_tokens_per_turn"),
        (
            "gpu_memory_utilization",
            0.8,
            "gpu_memory_configuration.gpu_memory_utilization",
        ),
    ],
)
def test_compare_rejects_mismatched_fingerprint(field, value, difference):
    result = make_comparison_result()
    baseline = make_comparison_result(**{field: value})

    with pytest.raises(ValueError, match=difference):
        compare_baseline(result, baseline)


@pytest.mark.parametrize("experiment_mode", ["correctness", "performance"])
def test_fingerprint_mismatch_is_rejected_in_both_modes(experiment_mode):
    result = make_comparison_result(experiment_mode=experiment_mode)
    baseline = make_comparison_result(
        experiment_mode=experiment_mode,
        seed=99,
    )

    with pytest.raises(ValueError, match="seed"):
        compare_baseline(result, baseline)


@pytest.mark.parametrize(
    ("field", "value", "difference"),
    [
        ("total_kv_budget_bytes", 3 * 1024**3, "total_kv_budget_bytes"),
        ("max_num_seqs", 64, "max_num_seqs"),
    ],
)
def test_fingerprint_rejects_mismatched_memory_budget(field, value, difference):
    result = make_comparison_result(
        experiment_mode="performance",
        max_tokens_per_turn=16,
    )
    baseline = make_comparison_result(
        experiment_mode="performance",
        max_tokens_per_turn=16,
        **{field: value},
    )

    with pytest.raises(ValueError, match=difference):
        compare_baseline(result, baseline)


def test_fingerprint_records_memory_formula_version():
    result = make_comparison_result()
    memory = result["experiment_fingerprint"]["fields"][
        "gpu_memory_configuration"
    ]

    assert memory["max_num_seqs"] == 128
    assert memory["kv_memory_formula_version"] == KV_MEMORY_FORMULA_VERSION


def test_legacy_baseline_is_rejected_clearly():
    with pytest.raises(ValueError, match="legacy baseline unsupported"):
        validate_baseline_schema(
            "performance",
            {"mode": "all-hot", "experiment_config": {}},
        )


def test_runtime_memory_validation_reports_exact_components():
    budget = derive_persistent_kv_budget(
        make_replay_args(total_kv_budget_bytes=4 * 1024**3)
    )
    runtime_state = {
        "num_gpu_blocks": budget["derived_num_gpu_blocks"],
        "hot_kv_storage_bytes": budget["derived_hot_kv_budget_bytes"] - 16,
        "warm_kv_storage_bytes": 0,
        "hot_to_warm_map_storage_bytes": 0,
        "warm_slot_table_storage_bytes": 0,
        "actual_persistent_kv_bytes": (
            budget["derived_actual_persistent_kv_bytes"] - 16
        ),
        "configured_total_kv_budget_bytes": budget[
            "total_kv_budget_bytes"
        ],
        "derived_hot_kv_budget_bytes": budget[
            "derived_hot_kv_budget_bytes"
        ],
        "budget_slack_bytes": budget["derived_budget_slack_bytes"] + 16,
    }
    runtime = build_runtime_memory_accounting(budget, runtime_state)

    errors = validate_runtime_memory_accounting(budget, runtime)

    assert len(errors) == 1
    assert "hot_kv_storage_bytes" in errors[0]
    assert "'expected':" in errors[0]
    assert "'actual':" in errors[0]


def test_turn_result_serialization():
    turn = make_turn_result(
        chat_id=101,
        turn=2,
        scheduled_send_seconds=1.5,
        actual_send_seconds=1.52,
        send_lateness_seconds=0.02,
        first_output_seconds=1.70,
        finished_seconds=1.75,
        ttft_seconds=0.18,
        latency_seconds=0.23,
    )
    serialized = asdict(turn)
    assert serialized == {
        "chat_id": 101,
        "turn": 2,
        "scheduled_send_seconds": 1.5,
        "actual_send_seconds": 1.52,
        "send_lateness_seconds": 0.02,
        "first_output_seconds": 1.70,
        "finished_seconds": 1.75,
        "ttft_seconds": 0.18,
        "latency_seconds": 0.23,
        "generated_tokens": 1,
        "generated_token_ids": [102],
        "configured_max_tokens_per_turn": 1,
        "effective_max_tokens": 1,
        "is_resume": True,
        "finish_reason": None,
    }
    loaded = json.loads(json.dumps(serialized))
    assert loaded["is_resume"] is True
    assert loaded["ttft_seconds"] == 0.18


def test_compute_turn_metrics_summary_aggregation_and_resume_separation():
    session1 = SessionResult(
        root_chat_id=1,
        session_id="bailian-1",
        turns=2,
        final_input_tokens=20,
        trace_output_tokens=10,
        scheduled_first_seconds=0.0,
        scheduled_last_seconds=1.0,
        generated_tokens=2,
        completed_turns=2,
        turn_results=[
            make_turn_result(chat_id=1, turn=1, ttft_seconds=0.20, latency_seconds=0.25),
            make_turn_result(chat_id=2, turn=2, ttft_seconds=0.40, latency_seconds=0.45),
        ],
    )
    session2 = SessionResult(
        root_chat_id=3,
        session_id="bailian-3",
        turns=2,
        final_input_tokens=25,
        trace_output_tokens=15,
        scheduled_first_seconds=0.5,
        scheduled_last_seconds=2.0,
        generated_tokens=2,
        completed_turns=2,
        turn_results=[
            make_turn_result(chat_id=3, turn=1, ttft_seconds=0.10, latency_seconds=0.15),
            make_turn_result(chat_id=4, turn=2, ttft_seconds=0.30, latency_seconds=0.35),
        ],
    )

    summary = compute_turn_metrics_summary([session1, session2])

    assert summary["all_turn_ttft_seconds"] == {
        "p50": 0.20,
        "p95": 0.30,
        "max": 0.40,
    }
    assert summary["resumed_turn_ttft_seconds"] == {
        "p50": 0.30,
        "p95": 0.30,
        "max": 0.40,
    }
    assert summary["all_turn_latency_seconds"] == {
        "p50": 0.25,
        "p95": 0.35,
        "max": 0.45,
    }
    assert summary["resumed_turn_latency_seconds"] == {
        "p50": 0.35,
        "p95": 0.35,
        "max": 0.45,
    }


def test_compute_turn_metrics_summary_empty_and_no_resumed():
    session = SessionResult(
        root_chat_id=1,
        session_id="bailian-1",
        turns=1,
        final_input_tokens=10,
        trace_output_tokens=5,
        scheduled_first_seconds=0.0,
        scheduled_last_seconds=0.0,
        generated_tokens=1,
        completed_turns=1,
        turn_results=[make_turn_result(chat_id=1, turn=1, ttft_seconds=0.15, latency_seconds=0.20)],
    )

    summary = compute_turn_metrics_summary([session])

    assert summary["all_turn_ttft_seconds"]["p50"] == 0.15
    assert summary["resumed_turn_ttft_seconds"] == {
        "p50": None,
        "p95": None,
        "max": None,
    }


def test_validate_timing_and_turns_valid_session():
    session = SessionResult(
        root_chat_id=1,
        session_id="bailian-1",
        turns=2,
        final_input_tokens=20,
        trace_output_tokens=10,
        scheduled_first_seconds=0.0,
        scheduled_last_seconds=1.0,
        generated_tokens=2,
        completed_turns=2,
        first_output_seconds=0.1,
        finished_seconds=1.2,
        turn_results=[
            make_turn_result(chat_id=1, turn=1, scheduled_send_seconds=0.0, actual_send_seconds=0.0, first_output_seconds=0.1, finished_seconds=0.2),
            make_turn_result(chat_id=2, turn=2, scheduled_send_seconds=1.0, actual_send_seconds=1.0, first_output_seconds=1.1, finished_seconds=1.2),
        ],
    )
    assert validate_timing_and_turns([session]) == []


def test_validate_timing_and_turns_detects_negative_timing():
    session = SessionResult(
        root_chat_id=1,
        session_id="bailian-1",
        turns=1,
        final_input_tokens=10,
        trace_output_tokens=5,
        scheduled_first_seconds=0.0,
        scheduled_last_seconds=0.0,
        generated_tokens=1,
        completed_turns=1,
        turn_results=[make_turn_result(chat_id=1, turn=1, ttft_seconds=-0.05)],
    )
    errors = validate_timing_and_turns([session])
    assert any("ttft_seconds" in err for err in errors)


def test_validate_timing_and_turns_detects_incomplete_and_missing_resumed_turns():
    session_incomplete = SessionResult(
        root_chat_id=1,
        session_id="bailian-1",
        turns=2,
        final_input_tokens=20,
        trace_output_tokens=10,
        scheduled_first_seconds=0.0,
        scheduled_last_seconds=1.0,
        generated_tokens=1,
        completed_turns=1,
        turn_results=[],
    )
    errors = validate_timing_and_turns([session_incomplete])
    assert any("completed 1/2 turns" in err for err in errors)
    assert any("missing per-turn metrics" in err for err in errors)

    session_bad_flag = SessionResult(
        root_chat_id=2,
        session_id="bailian-2",
        turns=1,
        final_input_tokens=10,
        trace_output_tokens=5,
        scheduled_first_seconds=1.0,
        scheduled_last_seconds=1.0,
        generated_tokens=1,
        completed_turns=1,
        turn_results=[make_turn_result(chat_id=2, turn=2, is_resume=False)],
    )
    errors_flag = validate_timing_and_turns([session_bad_flag])
    assert any("is_resume=False; expected True" in err for err in errors_flag)


def test_build_experiment_config_all_hot_and_mixed():
    args_all_hot = make_replay_args()
    cfg_all_hot = build_experiment_config(args_all_hot)
    assert cfg_all_hot["kv_mode"] == "all-hot"
    assert cfg_all_hot["warm_pool_blocks"] == 0
    assert cfg_all_hot["thresholds"]["hot_idle_threshold_seconds"] is None
    assert cfg_all_hot["max_tokens_per_turn"] == 1

    args_mixed = make_replay_args(
        experiment_mode="performance",
        kv_mode="mixed",
        max_tokens_per_turn=16,
        demotion_start_utilization=0.8,
        demotion_stop_utilization=0.65,
        warm_pool_blocks=256,
        max_sessions=10,
        min_turns=3,
        max_input_length=512,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
    )
    cfg_mixed = build_experiment_config(args_mixed)
    assert cfg_mixed["kv_mode"] == "mixed"
    assert cfg_mixed["warm_pool_blocks"] == 256
    assert cfg_mixed["thresholds"]["demotion_start_utilization"] == 0.8
    assert cfg_mixed["thresholds"]["demotion_stop_utilization"] == 0.65
    assert cfg_mixed["thresholds"]["hot_idle_threshold_seconds"] is None
    assert cfg_mixed["thresholds"]["cold_idle_threshold_seconds"] is None
    assert cfg_mixed["max_sessions"] == 10
    assert cfg_mixed["min_turns"] == 3
    assert cfg_mixed["gpu_memory_utilization"] == 0.8
    assert cfg_mixed["generation_policy"] == {
        "name": "final_turn_only_multi_token",
        "non_final_turn_max_tokens": 1,
        "final_turn_max_tokens": 16,
        "description": (
            "Non-final synthetic trace turns generate one discardable token; "
            "only the final turn uses the configured maximum."
        ),
    }
    fingerprint = build_experiment_fingerprint(
        args_mixed,
        {"name": "trace.jsonl", "sha256": "trace"},
        {
            "selection_sha256": "selection",
            "selected_session_count": 1,
            "selected_request_count": 2,
        },
    )
    assert (
        fingerprint["fields"]["generation_policy"]["name"]
        == "final_turn_only_multi_token"
    )


def test_correctness_mode_defaults_to_one_token():
    args = parse_args(
        [
            "--experiment-mode",
            "correctness",
            "--kv-mode",
            "all-hot",
            "--result-json",
            "result.json",
        ]
    )

    assert args.max_tokens_per_turn == 1
    assert build_experiment_config(args)["generation_policy"][
        "final_turn_max_tokens"
    ] == 1


def test_performance_mode_rejects_one_token():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--experiment-mode",
                "performance",
                "--kv-mode",
                "all-hot",
                "--result-json",
                "result.json",
                "--max-tokens-per-turn",
                "1",
            ]
        )


def test_performance_mode_rejects_missing_total_kv_budget():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--experiment-mode",
                "performance",
                "--kv-mode",
                "all-hot",
                "--result-json",
                "result.json",
                "--max-tokens-per-turn",
                "16",
            ]
        )


def test_pressure_mixed_mode_passes_thresholds_without_idle_thresholds():
    args = parse_args(
        [
            "--experiment-mode",
            "performance",
            "--kv-mode",
            "mixed",
            "--result-json",
            "mixed.json",
            "--baseline-json",
            "all-hot.json",
            "--max-tokens-per-turn",
            "16",
            "--demotion-start-utilization",
            "0.8",
            "--demotion-stop-utilization",
            "0.65",
            "--warm-pool-blocks",
            "128",
            "--total-kv-budget-bytes",
            str(4 * 1024**3),
        ]
    )

    kwargs = build_engine_args_kwargs(args)

    assert kwargs["kv_cache_demotion_start_utilization"] == 0.8
    assert kwargs["kv_cache_demotion_stop_utilization"] == 0.65
    assert kwargs["kv_cache_hot_idle_threshold_seconds"] is None
    assert kwargs["kv_cache_cold_idle_threshold_seconds"] is None
    assert kwargs["kv_cache_memory_bytes"] > 0
    assert kwargs["max_num_seqs"] == 128


@pytest.mark.parametrize(
    "threshold_args",
    [
        ["--demotion-start-utilization", "0.8"],
        ["--demotion-stop-utilization", "0.65"],
        [
            "--demotion-start-utilization",
            "0.6",
            "--demotion-stop-utilization",
            "0.8",
        ],
        [
            "--demotion-start-utilization",
            "nan",
            "--demotion-stop-utilization",
            "0.65",
        ],
    ],
)
def test_pressure_mixed_mode_rejects_invalid_thresholds(threshold_args):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--experiment-mode",
                "correctness",
                "--kv-mode",
                "mixed",
                "--result-json",
                "mixed.json",
                "--baseline-json",
                "all-hot.json",
                "--warm-pool-blocks",
                "128",
                *threshold_args,
            ]
        )


def test_service_metrics_exclude_slow_cleanup_and_shutdown():
    metrics = compute_run_metrics(
        requests=10,
        generated_tokens=100,
        workload_start=10.0,
        last_output_received=12.0,
        cleanup_start=12.0,
        shutdown_start=112.0,
        shutdown_complete=212.0,
    )

    assert metrics["service_window_duration_seconds"] == 2.0
    assert metrics["requests_per_second"] == 5.0
    assert metrics["output_tokens_per_second_service_window"] == 50.0
    assert metrics["cleanup_duration_seconds"] == 100.0
    assert metrics["shutdown_duration_seconds"] == 100.0


def test_reproducibility_metadata_contains_required_fields():
    args = make_replay_args(
        kv_mode="mixed",
        demotion_start_utilization=0.8,
        demotion_stop_utilization=0.65,
        warm_pool_blocks=128,
    )
    timing = {
        "replay_plan_ready": "2026-09-17T08:00:00+00:00",
        "workload_start": "2026-09-17T08:00:01+00:00",
        "first_request_sent": "2026-09-17T08:00:01+00:00",
        "last_output_received": "2026-09-17T08:00:03+00:00",
        "cleanup_start": "2026-09-17T08:00:03+00:00",
        "shutdown_complete": "2026-09-17T08:00:05+00:00",
        "start": "2026-09-17T07:59:59+00:00",
        "end": "2026-09-17T08:00:05+00:00",
        "service_window_duration_seconds": 2.0,
        "cleanup_duration_seconds": 1.0,
        "shutdown_duration_seconds": 1.0,
    }
    selection = {
        "selected_session_ids": [1],
        "selected_session_count": 1,
        "selected_request_count": 2,
        "selection_sha256": "selection",
    }

    metadata = build_reproducibility_metadata(
        args,
        {"name": "subset.jsonl", "sha256": "trace"},
        selection,
        timing,
    )

    required = {
        "schema_version",
        "experiment_mode",
        "kv_mode",
        "model",
        "seed",
        "trace",
        "selection",
        "time_scale",
        "max_tokens_per_turn",
        "generation_parameters",
        "generation_policy",
        "gpu_memory_utilization",
        "max_num_seqs",
        "persistent_kv_memory_budget",
        "configured_hkv",
        "warm_pool_blocks",
        "thresholds",
        "attention_backend",
        "dtype",
        "topology_assumptions",
        "git",
        "timing",
    }
    assert required <= metadata.keys()
    assert metadata["trace"] == {
        "name": "subset.jsonl",
        "sha256": "trace",
    }
    assert not str(metadata["trace"]).startswith("/")
    assert (
        metadata["generation_policy"]["name"]
        == "final_turn_only_multi_token"
    )


def test_run_session_with_fake_engine():
    async def _test():
        class FakeEngine:
            def __init__(self):
                self.received_inputs = []
                self.base_max_tokens = None

            async def generate(self, input_gen, base_params, session_id):
                self.base_max_tokens = base_params.max_tokens
                # 1. Pull turn 1 input
                inp1 = await anext(input_gen)
                self.received_inputs.append(inp1)

                # 2. Pull turn 2 input (queued before turn 1 completes)
                inp2 = await anext(input_gen)
                self.received_inputs.append(inp2)

                # 3. Turn 1: Emit first token in one chunk, finish reason in separate chunk
                yield SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=[101], finish_reason=None)]
                )
                await asyncio.sleep(0.01)
                yield SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=[], finish_reason="length")]
                )

                # 4. Turn 2: Emit first token in one chunk, finish reason in separate chunk
                yield SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=[201, 202], finish_reason=None)]
                )
                await asyncio.sleep(0.01)
                yield SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=[], finish_reason="length")]
                )

        turns = [
            ReplayTurn(
                session_id="bailian-10",
                root_chat_id=10,
                chat_id=10,
                parent_chat_id=-1,
                turn=1,
                send_at_seconds=0.0,
                input_length=4,
                trace_output_length=2,
                delta_token_ids=(10, 11, 12, 13),
            ),
            ReplayTurn(
                session_id="bailian-10",
                root_chat_id=10,
                chat_id=11,
                parent_chat_id=10,
                turn=2,
                send_at_seconds=0.0,
                input_length=8,
                trace_output_length=2,
                delta_token_ids=(20, 21, 22, 23),
            ),
        ]

        engine = FakeEngine()
        started = time.perf_counter()
        session_result, lateness_values = await run_session(
            engine, turns, started, seed=42, max_tokens_per_turn=2
        )

        assert len(engine.received_inputs) == 2
        assert engine.base_max_tokens == 2
        assert engine.received_inputs[0].sampling_params.max_tokens == 1
        assert engine.received_inputs[1].sampling_params.max_tokens == 2
        assert session_result.completed_turns == 2
        assert session_result.generated_tokens == 3
        assert len(session_result.turn_results) == 2

        # FIFO turn 1 mapping
        t1 = session_result.turn_results[0]
        assert t1.chat_id == 10
        assert t1.turn == 1
        assert t1.is_resume is False
        assert t1.generated_tokens == 1
        assert t1.generated_token_ids == [101]
        assert t1.configured_max_tokens_per_turn == 2
        assert t1.effective_max_tokens == 1
        assert t1.ttft_seconds >= 0.0
        assert t1.latency_seconds >= t1.ttft_seconds
        assert t1.first_output_seconds <= t1.finished_seconds
        assert t1.finish_reason == "length"

        # FIFO turn 2 mapping
        t2 = session_result.turn_results[1]
        assert t2.chat_id == 11
        assert t2.turn == 2
        assert t2.is_resume is True
        assert t2.generated_tokens == 2
        assert t2.generated_token_ids == [201, 202]
        assert t2.configured_max_tokens_per_turn == 2
        assert t2.effective_max_tokens == 2
        assert t2.ttft_seconds >= 0.0
        assert t2.latency_seconds >= t2.ttft_seconds
        assert t2.first_output_seconds <= t2.finished_seconds
        assert t2.finish_reason == "length"

        # Validate overall session timing
        assert validate_timing_and_turns([session_result]) == []

    asyncio.run(_test())


def test_single_turn_uses_configured_multi_token_limit():
    async def _test():
        class FakeEngine:
            def __init__(self):
                self.base_max_tokens = None
                self.input_max_tokens = None

            async def generate(self, input_gen, base_params, session_id):
                self.base_max_tokens = base_params.max_tokens
                streaming_input = await anext(input_gen)
                self.input_max_tokens = streaming_input.sampling_params.max_tokens
                yield SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            token_ids=[101, 102, 103, 104],
                            finish_reason="length",
                        )
                    ]
                )

        turn = ReplayTurn(
            session_id="bailian-10",
            root_chat_id=10,
            chat_id=10,
            parent_chat_id=-1,
            turn=1,
            send_at_seconds=0.0,
            input_length=4,
            trace_output_length=2,
            delta_token_ids=(10, 11, 12, 13),
        )
        engine = FakeEngine()
        session_result, _ = await run_session(
            engine,
            [turn],
            time.perf_counter(),
            seed=42,
            max_tokens_per_turn=4,
        )

        assert engine.base_max_tokens == 4
        assert engine.input_max_tokens == 4
        assert session_result.turn_results[0].effective_max_tokens == 4
        assert (
            session_result.turn_results[0].configured_max_tokens_per_turn
            == 4
        )

    asyncio.run(_test())


def test_run_session_final_turn_uses_configured_multi_token_limit():
    async def _test():
        class FakeEngine:
            def __init__(self):
                self.received_inputs = []
                self.base_max_tokens = None

            async def generate(self, input_gen, base_params, session_id):
                self.base_max_tokens = base_params.max_tokens
                collected = []
                for _ in range(3):
                    collected.append(await anext(input_gen))
                self.received_inputs.extend(collected)
                next_id = 101
                for inp in collected:
                    count = inp.sampling_params.max_tokens
                    token_ids = list(range(next_id, next_id + count))
                    next_id += 100
                    yield SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                token_ids=token_ids,
                                finish_reason=None,
                            )
                        ]
                    )
                    await asyncio.sleep(0.01)
                    yield SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                token_ids=[],
                                finish_reason="length",
                            )
                        ]
                    )

        turns = [
            ReplayTurn(
                session_id="bailian-10",
                root_chat_id=10,
                chat_id=10,
                parent_chat_id=-1,
                turn=1,
                send_at_seconds=0.0,
                input_length=4,
                trace_output_length=2,
                delta_token_ids=(10, 11, 12, 13),
            ),
            ReplayTurn(
                session_id="bailian-10",
                root_chat_id=10,
                chat_id=11,
                parent_chat_id=10,
                turn=2,
                send_at_seconds=0.0,
                input_length=8,
                trace_output_length=2,
                delta_token_ids=(20, 21, 22, 23),
            ),
            ReplayTurn(
                session_id="bailian-10",
                root_chat_id=10,
                chat_id=12,
                parent_chat_id=11,
                turn=3,
                send_at_seconds=0.0,
                input_length=12,
                trace_output_length=2,
                delta_token_ids=(30, 31, 32, 33),
            ),
        ]
        engine = FakeEngine()
        session_result, _ = await run_session(
            engine,
            turns,
            time.perf_counter(),
            seed=42,
            max_tokens_per_turn=16,
        )

        assert engine.base_max_tokens == 16
        assert engine.base_max_tokens != 1
        assert [
            inp.sampling_params.max_tokens for inp in engine.received_inputs
        ] == [1, 1, 16]
        assert [
            turn.effective_max_tokens for turn in session_result.turn_results
        ] == [1, 1, 16]
        assert session_result.generated_tokens == 18
        assert session_result.completed_turns == 3

        t1, t2, t3 = session_result.turn_results
        assert t1.generated_tokens == 1
        assert t1.generated_token_ids == [101]
        assert t1.effective_max_tokens == 1
        assert t1.is_resume is False
        assert t1.finish_reason == "length"

        assert t2.generated_tokens == 1
        assert t2.generated_token_ids == [201]
        assert t2.effective_max_tokens == 1
        assert t2.is_resume is True
        assert t2.finish_reason == "length"

        assert t3.generated_tokens == 16
        assert t3.generated_token_ids == list(range(301, 317))
        assert t3.effective_max_tokens == 16
        assert t3.configured_max_tokens_per_turn == 16
        assert t3.is_resume is True
        assert t3.finish_reason == "length"

        assert validate_timing_and_turns([session_result]) == []
        assert validate_final_turn_generation_limits([session_result]) == []

    asyncio.run(_test())


def _three_turn_session(
    *,
    final_generated_tokens: int,
    final_finish_reason: str | None,
    final_effective_max_tokens: int = 16,
) -> SessionResult:
    return SessionResult(
        root_chat_id=10,
        session_id="bailian-10",
        turns=3,
        final_input_tokens=12,
        trace_output_tokens=6,
        scheduled_first_seconds=0.0,
        scheduled_last_seconds=1.0,
        generated_tokens=2 + final_generated_tokens,
        completed_turns=3,
        turn_results=[
            make_turn_result(
                chat_id=10,
                turn=1,
                generated_tokens=1,
                generated_token_ids=[101],
                configured_max_tokens_per_turn=16,
                effective_max_tokens=1,
                finish_reason="length",
            ),
            make_turn_result(
                chat_id=11,
                turn=2,
                generated_tokens=1,
                generated_token_ids=[201],
                configured_max_tokens_per_turn=16,
                effective_max_tokens=1,
                finish_reason="length",
            ),
            make_turn_result(
                chat_id=12,
                turn=3,
                generated_tokens=final_generated_tokens,
                generated_token_ids=list(range(301, 301 + final_generated_tokens)),
                configured_max_tokens_per_turn=16,
                effective_max_tokens=final_effective_max_tokens,
                finish_reason=final_finish_reason,
            ),
        ],
    )


def test_validate_rejects_silently_capped_final_turns():
    errors = validate_final_turn_generation_limits(
        [
            _three_turn_session(
                final_generated_tokens=1,
                final_finish_reason="length",
            )
        ]
    )
    assert errors
    assert "silently limited to one token" in errors[0]


def test_validate_rejects_silently_capped_final_turns_without_finish_reason():
    errors = validate_final_turn_generation_limits(
        [
            _three_turn_session(
                final_generated_tokens=1,
                final_finish_reason=None,
            )
        ]
    )
    assert errors
    assert "silently limited to one token" in errors[0]


def test_validate_allows_final_turn_that_reaches_configured_maximum():
    assert (
        validate_final_turn_generation_limits(
            [
                _three_turn_session(
                    final_generated_tokens=16,
                    final_finish_reason="length",
                )
            ]
        )
        == []
    )


def test_validate_allows_legitimate_early_stop_on_final_turn():
    assert (
        validate_final_turn_generation_limits(
            [
                _three_turn_session(
                    final_generated_tokens=1,
                    final_finish_reason="abort",
                )
            ]
        )
        == []
    )
