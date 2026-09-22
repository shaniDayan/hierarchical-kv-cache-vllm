from __future__ import annotations

import asyncio
import math
import os
from argparse import Namespace
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.scripts.hkv_gsm8k import (
    BLOCK_SIZE,
    DEFAULT_BALLAST_SAFETY_MARGIN_BLOCKS,
    EXCLUSION_NO_RESUMED_MIXED_READ,
    EXCLUSION_NOT_WARM_BEFORE_RESUME,
    TURN2_TEXT,
    PromptPlan,
    SessionQuality,
    build_capacity_report,
    build_prompt_plans,
    complete_blocks_are_warm,
    divide_blocks_across_sessions,
    exact_length_token_ids,
    fingerprint_differences,
    get_answer_value,
    load_gsm8k_subset,
    mixed_exclusion_reason,
    normalize_token_ids,
    pair_quality,
    plan_ballast_allocation,
    prompt_fingerprint_fields,
    quality_summary,
    score_generated_text,
)
from experiments.scripts.run_hkv_gsm8k_resume_quality import (
    GSM8K_ASYNC_SCHEDULING,
    GSM8K_SEQUENTIAL_RESUME,
    RESUME_PROGRESS_DEBUG_ENV,
    EvaluatedSessionState,
    HKVGsm8kWorkerExtension,
    MixedDemotionBarrierTimeout,
    SequentialResumeTracker,
    abort_ballast,
    apply_hkv_environment,
    build_experiment_fingerprint,
    generation_parameters,
    gsm8k_engine_args_kwargs,
    parse_args,
    prepare_workload,
    record_pre_resume_inspect,
    resume_turn2_sequentially,
    run_ballast_session,
    run_evaluated_session,
    sample_hkv_observation,
    wait_mixed_demotion_barrier,
    _turn2_output_is_complete,
)
from experiments.scripts.run_qwen_bailian_replay import (
    derive_persistent_kv_budget,
    drain_warm_residency,
    update_observation,
)

FIXTURE_DIR = Path("experiments/tests/fixtures/gsm8k_tiny")


class StubTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        content = messages[0]["content"]
        length = 20 + (len(content) % 7)
        return list(range(length))

    def encode(self, text: str) -> list[int]:
        length = 5 + (len(text) % 5)
        return list(range(200, 200 + length))


def make_plan(
    *,
    question_index: int = 0,
    turn1_tokens: int = 37,
    gold_value: int = 18,
) -> PromptPlan:
    return PromptPlan(
        session_id=f"gsm8k-{question_index:04d}",
        question_index=question_index,
        gold_value=gold_value,
        turn1_text="question",
        turn2_text=TURN2_TEXT,
        turn1_token_ids=tuple(range(turn1_tokens)),
        turn2_token_ids=(1, 2, 3),
    )


def budget_args(**overrides) -> Namespace:
    values = dict(
        model="Qwen/Qwen3-0.6B",
        kv_mode="mixed",
        max_model_len=4096,
        max_num_seqs=128,
        total_kv_budget_bytes=4 * 1024**3,
        warm_pool_blocks=1024,
        gpu_memory_utilization=0.6,
        seed=0,
        demotion_start_utilization=0.8,
        demotion_stop_utilization=0.65,
    )
    values.update(overrides)
    return Namespace(**values)


def test_load_local_subset_does_not_require_download():
    subset = load_gsm8k_subset(
        FIXTURE_DIR,
        num_questions=4,
        num_shots=5,
    )
    assert len(subset.questions) == 4
    assert len(subset.shots) == 5
    assert subset.questions[0].gold_value == 18
    assert subset.shots[0].gold_value == 72


def test_missing_local_dataset_is_a_file_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not download"):
        load_gsm8k_subset(tmp_path, num_questions=1, num_shots=0)


def test_natural_prompts_are_not_block_padded():
    subset = load_gsm8k_subset(FIXTURE_DIR, num_questions=4, num_shots=1)
    plans = build_prompt_plans(subset, StubTokenizer())
    assert plans[0].turn2_token_ids == tuple(StubTokenizer().encode(TURN2_TEXT))
    for plan in plans:
        assert plan.turn1_token_count % BLOCK_SIZE != 0
        assert plan.complete_historical_blocks == plan.turn1_token_count // BLOCK_SIZE
        assert "Do not give the final numeric answer yet." in plan.turn1_text
        assert plan.turn2_text == TURN2_TEXT


def test_prompt_plans_are_identical_for_the_same_subset():
    subset = load_gsm8k_subset(FIXTURE_DIR, num_questions=4, num_shots=5)
    first = prompt_fingerprint_fields(build_prompt_plans(subset, StubTokenizer()))
    second = prompt_fingerprint_fields(build_prompt_plans(subset, StubTokenizer()))
    assert first == second


def test_normalize_token_ids_accepts_common_tokenizer_forms():
    torch = pytest.importorskip("torch")
    expected = (10, 11, 12)
    assert normalize_token_ids([10, 11, 12]) == expected
    assert normalize_token_ids({"input_ids": [10, 11, 12]}) == expected
    assert normalize_token_ids({"input_ids": [[10, 11, 12]]}) == expected
    assert normalize_token_ids(torch.tensor([10, 11, 12])) == expected
    assert normalize_token_ids(torch.tensor([[10, 11, 12]])) == expected


def test_normalize_token_ids_rejects_invalid_inputs():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="2 batch rows"):
        normalize_token_ids([[10, 11], [12, 13]])
    with pytest.raises(ValueError, match="2 batch rows"):
        normalize_token_ids(torch.tensor([[10, 11], [12, 13]]))
    with pytest.raises(ValueError, match="missing 'input_ids'"):
        normalize_token_ids({"token_ids": [10, 11, 12]})
    with pytest.raises(ValueError, match="not text"):
        normalize_token_ids("input_ids")
    with pytest.raises(ValueError, match="empty token sequence"):
        normalize_token_ids([])


def test_answer_extraction_and_parse_failure():
    assert get_answer_value("junk #### 18") == 18
    extracted, failed, match = score_generated_text("The answer is #### 18", 18)
    assert extracted == 18 and not failed and match
    extracted, failed, match = score_generated_text("no number here", 18)
    assert failed and not match


def test_complete_historical_warm_confirmation_ignores_incomplete_tail():
    residency = [
        {"key": ["gsm8k-0000", 0, 0]},
        {"key": ["gsm8k-0000", 0, 1]},
    ]
    assert complete_blocks_are_warm(
        complete_historical_blocks=2,
        residency=residency,
        block_ids=[10, 11, 12],
        request_id="gsm8k-0000",
    )
    assert not complete_blocks_are_warm(
        complete_historical_blocks=2,
        residency=[{"key": ["gsm8k-0000", 0, 0]}],
        block_ids=[10, 11, 12],
        request_id="gsm8k-0000",
    )
    assert not complete_blocks_are_warm(
        complete_historical_blocks=0,
        residency=[],
        block_ids=[3],
        request_id="gsm8k-0000",
    )


def _warm_entries(
    session_id: str,
    logical_end_inclusive: int,
    *,
    as_tuple: bool = False,
    kv_group: int = 0,
) -> list[dict]:
    entries = []
    for logical in range(logical_end_inclusive + 1):
        key: tuple | list
        key = (session_id, kv_group, logical) if as_tuple else [
            session_id,
            kv_group,
            logical,
        ]
        entries.append(
            {
                "key": key,
                "warm_slot_id": 1000 + logical,
                "temporary_shadow_hot_block_id": 2000 + logical,
            }
        )
    return entries


def test_observed_inspect_schema_confirms_complete_blocks_only():
    residency = _warm_entries("gsm8k-0000", 46)
    block_ids = list(range(50, 98))
    assert len(block_ids) == 48
    assert complete_blocks_are_warm(
        complete_historical_blocks=47,
        residency=residency,
        block_ids=block_ids,
        request_id="gsm8k-0000",
    )
    missing_23 = [item for item in residency if item["key"][2] != 23]
    assert not complete_blocks_are_warm(
        complete_historical_blocks=47,
        residency=missing_23,
        block_ids=block_ids,
        request_id="gsm8k-0000",
    )
    only_46 = _warm_entries("gsm8k-0000", 45)
    assert not complete_blocks_are_warm(
        complete_historical_blocks=47,
        residency=only_46,
        block_ids=block_ids,
        request_id="gsm8k-0000",
    )
    assert complete_blocks_are_warm(
        complete_historical_blocks=47,
        residency=residency,
        block_ids=block_ids,
        request_id="gsm8k-0000",
    )
    mixed_requests = residency + _warm_entries("gsm8k-9999", 46)
    assert complete_blocks_are_warm(
        complete_historical_blocks=47,
        residency=mixed_requests,
        block_ids=block_ids,
        request_id="gsm8k-0000",
    )
    assert not complete_blocks_are_warm(
        complete_historical_blocks=47,
        residency=_warm_entries("gsm8k-9999", 46),
        block_ids=block_ids,
        request_id="gsm8k-0000",
    )


def test_warm_residency_tuple_and_json_list_keys_both_work():
    block_ids = list(range(10, 58))
    listed = _warm_entries("gsm8k-0000", 46, as_tuple=False)
    tupled = _warm_entries("gsm8k-0000", 46, as_tuple=True)
    for residency in (listed, tupled):
        assert complete_blocks_are_warm(
            complete_historical_blocks=47,
            residency=residency,
            block_ids=block_ids,
            request_id="gsm8k-0000",
        )


def test_mixed_eligibility_requires_warm_and_mixed_read():
    assert mixed_exclusion_reason(
        complete_historical_blocks=2,
        pre_resume_warm_confirmed=False,
        attention_had_warm_slots=True,
    ) == EXCLUSION_NOT_WARM_BEFORE_RESUME
    assert mixed_exclusion_reason(
        complete_historical_blocks=2,
        pre_resume_warm_confirmed=True,
        attention_had_warm_slots=False,
    ) == EXCLUSION_NO_RESUMED_MIXED_READ
    assert (
        mixed_exclusion_reason(
            complete_historical_blocks=2,
            pre_resume_warm_confirmed=True,
            attention_had_warm_slots=True,
        )
        is None
    )


def test_quality_summary_excludes_unconfirmed_mixed_sessions():
    eligible = SessionQuality(
        session_id="gsm8k-0000",
        question_index=0,
        gold_value=18,
        final_text="#### 18",
        extracted_value=18,
        parse_failure=False,
        exact_match=True,
        complete_historical_blocks=2,
        pre_resume_warm_confirmed=True,
        attention_had_warm_slots=True,
        mixed_read_steps=3,
        warm_logical_blocks_observed=2,
    )
    excluded = SessionQuality(
        session_id="gsm8k-0001",
        question_index=1,
        gold_value=3,
        final_text="#### 3",
        extracted_value=3,
        parse_failure=False,
        exact_match=True,
        complete_historical_blocks=2,
        pre_resume_warm_confirmed=False,
        attention_had_warm_slots=False,
        mixed_read_steps=0,
        warm_logical_blocks_observed=0,
        exclusion_reason=EXCLUSION_NOT_WARM_BEFORE_RESUME,
    )
    summary = quality_summary([eligible, excluded], kv_mode="mixed")
    assert summary["n_eligible"] == 1
    assert summary["n_exact_match"] == 1
    assert summary["excluded_sessions"][0]["session_id"] == "gsm8k-0001"


def test_pair_quality_counts_divergences_on_eligible_mixed_only():
    all_hot = [
        SessionQuality(
            session_id="gsm8k-0000",
            question_index=0,
            gold_value=18,
            final_text="#### 18",
            extracted_value=18,
            parse_failure=False,
            exact_match=True,
            complete_historical_blocks=2,
            pre_resume_warm_confirmed=False,
            attention_had_warm_slots=False,
            mixed_read_steps=0,
            warm_logical_blocks_observed=0,
        ),
        SessionQuality(
            session_id="gsm8k-0001",
            question_index=1,
            gold_value=3,
            final_text="#### 4",
            extracted_value=4,
            parse_failure=False,
            exact_match=False,
            complete_historical_blocks=2,
            pre_resume_warm_confirmed=False,
            attention_had_warm_slots=False,
            mixed_read_steps=0,
            warm_logical_blocks_observed=0,
        ),
    ]
    mixed = [
        SessionQuality(
            session_id="gsm8k-0000",
            question_index=0,
            gold_value=18,
            final_text="#### 18",
            extracted_value=18,
            parse_failure=False,
            exact_match=True,
            complete_historical_blocks=2,
            pre_resume_warm_confirmed=True,
            attention_had_warm_slots=True,
            mixed_read_steps=2,
            warm_logical_blocks_observed=2,
        ),
        SessionQuality(
            session_id="gsm8k-0001",
            question_index=1,
            gold_value=3,
            final_text="#### 3",
            extracted_value=3,
            parse_failure=False,
            exact_match=True,
            complete_historical_blocks=2,
            pre_resume_warm_confirmed=False,
            attention_had_warm_slots=False,
            mixed_read_steps=0,
            warm_logical_blocks_observed=0,
            exclusion_reason=EXCLUSION_NOT_WARM_BEFORE_RESUME,
        ),
    ]
    paired = pair_quality(all_hot, mixed)
    assert paired["n_paired_eligible"] == 1
    assert paired["n_identical_extracted_answers"] == 1
    assert paired["n_both_correct"] == 1


def test_capacity_report_refuses_warm_pool_too_small():
    plans = [make_plan(turn1_tokens=64), make_plan(question_index=1, turn1_tokens=64)]
    budget = derive_persistent_kv_budget(budget_args(warm_pool_blocks=2))
    report = build_capacity_report(
        plans,
        kv_mode="mixed",
        memory_budget=budget,
        warm_pool_blocks=2,
        demotion_start_utilization=0.8,
        num_ballast_sessions=8,
        max_num_seqs=128,
        max_model_len=4096,
    )
    assert report["feasible"] is False
    assert report["can_satisfy_per_session_warm_confirmation"] is False
    assert any("WARM pool" in reason for reason in report["refusal_reasons"])


def test_capacity_report_refuses_zero_complete_blocks_for_mixed():
    plans = [make_plan(turn1_tokens=15)]
    budget = derive_persistent_kv_budget(budget_args())
    report = build_capacity_report(
        plans,
        kv_mode="mixed",
        memory_budget=budget,
        warm_pool_blocks=1024,
        demotion_start_utilization=0.8,
        num_ballast_sessions=8,
        max_num_seqs=128,
        max_model_len=4096,
    )
    assert report["feasible"] is False
    assert any(
        "no complete historical block" in reason for reason in report["refusal_reasons"]
    )


def test_capacity_report_smoke_four_questions_with_stub_tokenizer():
    subset = load_gsm8k_subset(FIXTURE_DIR, num_questions=4, num_shots=5)
    plans = build_prompt_plans(subset, StubTokenizer())
    args = budget_args()
    report = build_capacity_report(
        plans,
        kv_mode="mixed",
        memory_budget=derive_persistent_kv_budget(args),
        warm_pool_blocks=1024,
        demotion_start_utilization=0.8,
        num_ballast_sessions=8,
        max_num_seqs=128,
        max_model_len=4096,
    )
    assert report["feasible"] is True
    assert report["total_complete_historical_blocks"] == sum(
        plan.complete_historical_blocks for plan in plans
    )
    assert all(item["turn1_token_count"] > BLOCK_SIZE for item in report["questions"])


def test_all_hot_and_mixed_fingerprints_match_except_treatment():
    subset = load_gsm8k_subset(FIXTURE_DIR, num_questions=4, num_shots=5)
    plans = build_prompt_plans(subset, StubTokenizer())
    common = dict(
        dataset_dir=FIXTURE_DIR,
        model="Qwen/Qwen3-0.6B",
        seed=0,
        num_questions=4,
        num_shots=5,
        start_index=0,
        shot_start_index=0,
        max_tokens=64,
        gpu_memory_utilization=0.6,
        max_model_len=4096,
        max_num_seqs=128,
        num_ballast_sessions=8,
        total_kv_budget_bytes=4 * 1024**3,
    )
    all_hot = Namespace(
        kv_mode="all-hot",
        warm_pool_blocks=0,
        demotion_start_utilization=None,
        demotion_stop_utilization=None,
        **common,
    )
    mixed = Namespace(
        kv_mode="mixed",
        warm_pool_blocks=1024,
        demotion_start_utilization=0.8,
        demotion_stop_utilization=0.65,
        **common,
    )
    left = build_experiment_fingerprint(all_hot, subset, plans)
    right = build_experiment_fingerprint(mixed, subset, plans)
    assert left["sha256"] == right["sha256"]
    assert fingerprint_differences(left["fields"], right["fields"]) == []
    assert left["fields"]["async_scheduling"] is True
    assert left["fields"]["sequential_resume"] is True
    assert GSM8K_ASYNC_SCHEDULING is True
    assert GSM8K_SEQUENTIAL_RESUME is True


def test_parse_args_smoke_defaults_and_mixed_requirements():
    all_hot = parse_args(
        [
            "--kv-mode",
            "all-hot",
            "--preflight",
            "--total-kv-budget-bytes",
            str(4 * 1024**3),
        ]
    )
    assert all_hot.num_questions == 4
    assert all_hot.warm_pool_blocks == 0
    with pytest.raises(SystemExit):
        parse_args(["--kv-mode", "mixed", "--preflight"])


def test_prepare_workload_preflight_payload():
    args = parse_args(
        [
            "--kv-mode",
            "all-hot",
            "--preflight",
            "--dataset-dir",
            str(FIXTURE_DIR),
            "--num-questions",
            "4",
            "--num-shots",
            "5",
            "--total-kv-budget-bytes",
            str(4 * 1024**3),
        ]
    )
    subset, plans, budget, capacity = prepare_workload(args, StubTokenizer())
    assert len(plans) == 4
    assert budget["total_kv_budget_bytes"] == 4 * 1024**3
    assert capacity["feasible"] is True
    assert subset.questions[0].index == 0


def test_engine_args_disable_prefix_cache_and_use_gsm8k_extension():
    args = parse_args(
        [
            "--kv-mode",
            "all-hot",
            "--preflight",
            "--total-kv-budget-bytes",
            str(4 * 1024**3),
        ]
    )
    kwargs = gsm8k_engine_args_kwargs(args)
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["enforce_eager"] is True
    assert kwargs["async_scheduling"] is True
    assert kwargs["attention_backend"] == "TRITON_ATTN"
    assert kwargs["dtype"] == "float16"
    assert kwargs["worker_extension_cls"].endswith("HKVGsm8kWorkerExtension")
    assert generation_parameters(args)["temperature"] == 0.0
    assert generation_parameters(args)["turn2_ignore_eos"] is False


def test_gsm8k_keeps_async_scheduling_and_sequential_resume_for_both_modes():
    common = [
        "--preflight",
        "--total-kv-budget-bytes",
        str(4 * 1024**3),
        "--warm-pool-blocks",
        "1024",
        "--demotion-start-utilization",
        "0.8",
        "--demotion-stop-utilization",
        "0.65",
    ]
    all_hot = gsm8k_engine_args_kwargs(
        parse_args(["--kv-mode", "all-hot", *common[:3]])
    )
    mixed = gsm8k_engine_args_kwargs(
        parse_args(["--kv-mode", "mixed", *common])
    )
    assert all_hot["async_scheduling"] is True
    assert mixed["async_scheduling"] is True
    assert all_hot["async_scheduling"] == mixed["async_scheduling"]
    assert GSM8K_SEQUENTIAL_RESUME is True
    assert GSM8K_ASYNC_SCHEDULING is True


def test_apply_hkv_environment_disables_resume_progress_debug(monkeypatch):
    monkeypatch.setenv(RESUME_PROGRESS_DEBUG_ENV, "1")
    args = parse_args(
        [
            "--kv-mode",
            "all-hot",
            "--preflight",
            "--total-kv-budget-bytes",
            str(4 * 1024**3),
        ]
    )
    apply_hkv_environment(args)
    assert os.environ[RESUME_PROGRESS_DEBUG_ENV] == "0"


def test_inspect_hkv_request_is_read_only_and_includes_stats():
    pytest.importorskip("torch")

    class Indexable:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, _item):
            return self.value

    class CpuList(list):
        def cpu(self):
            return self

        def tolist(self):
            return list(self)

    extension = HKVGsm8kWorkerExtension()
    extension.model_runner = SimpleNamespace(
        hkv_warm_migration_manager=SimpleNamespace(
            warm_residency={
                ("gsm8k-0000", 0, 0): SimpleNamespace(
                    warm_slot_id=4,
                    temporary_shadow_hot_block_id=9,
                ),
                ("gsm8k-0000", 0, 1): SimpleNamespace(
                    warm_slot_id=4,
                    temporary_shadow_hot_block_id=10,
                ),
            },
            allocator=SimpleNamespace(
                lookup=lambda key: 4,
                num_owned_slots=2,
            ),
        ),
        req_states=SimpleNamespace(
            req_id_to_index={"gsm8k-0000": 0},
            num_computed_tokens_np=Indexable(37),
        ),
        block_tables=SimpleNamespace(
            num_blocks=SimpleNamespace(np=Indexable(2)),
            block_tables=[SimpleNamespace(gpu=Indexable(CpuList([0, 0])))],
        ),
        hkv_mixed_read_stats={
            "gsm8k-0000": {
                "mixed_read_steps": 2,
                "attention_had_warm_slots": True,
                "warm_logical_blocks_observed": 1,
            }
        },
    )
    state = extension.inspect_hkv_request("gsm8k-0000")
    assert state["block_ids"] == [0, 0]
    assert state["warm_residency"][0]["warm_slot_id"] == 4
    assert state["mixed_read_stats"]["mixed_read_steps"] == 2
    stats = extension.inspect_hkv_mixed_read_stats()
    assert stats["requests"]["gsm8k-0000"]["attention_had_warm_slots"] is True
    assert complete_blocks_are_warm(
        complete_historical_blocks=2,
        residency=state["warm_residency"],
        block_ids=state["block_ids"],
        request_id="gsm8k-0000",
    )


def test_peak_warm_alone_does_not_make_a_session_eligible():
    session = SessionQuality(
        session_id="gsm8k-0000",
        question_index=0,
        gold_value=18,
        final_text="#### 18",
        extracted_value=18,
        parse_failure=False,
        exact_match=True,
        complete_historical_blocks=2,
        pre_resume_warm_confirmed=False,
        attention_had_warm_slots=False,
        mixed_read_steps=0,
        warm_logical_blocks_observed=0,
        exclusion_reason=EXCLUSION_NOT_WARM_BEFORE_RESUME,
    )
    summary = quality_summary([session], kv_mode="mixed")
    assert summary["n_eligible"] == 0
    assert summary["accuracy_exact_match"] is None
    assert summary["n_confirmed_warm_or_mixed_read"] == 0


def test_exact_ballast_block_planning_matches_capacity_formula():
    usable = 1518
    start = 0.8
    turn1 = 186
    plan = plan_ballast_allocation(
        kv_mode="mixed",
        usable_hot_blocks=usable,
        demotion_start_utilization=start,
        total_hot_blocks_turn1=turn1,
        num_ballast_sessions=8,
        max_model_len=4096,
        safety_margin_blocks=16,
    )
    start_blocks = math.ceil(start * usable)
    assert start_blocks == 1215
    assert plan["demotion_start_hot_blocks"] == 1215
    assert plan["target_hot_blocks"] == 1215 + 16
    assert plan["required_ballast_blocks"] == 1215 + 16 - turn1
    assert plan["total_planned_ballast_blocks"] == plan["required_ballast_blocks"]
    assert sum(plan["planned_blocks_per_request"]) == plan["required_ballast_blocks"]
    for blocks, prompt_tokens, request in zip(
        plan["planned_blocks_per_request"],
        plan["prompt_tokens_per_request"],
        plan["requests"],
    ):
        assert prompt_tokens == blocks * BLOCK_SIZE
        assert len(request["prompt_token_ids"]) == prompt_tokens
        assert request["generation_tokens"] == 1
        assert request["total_tokens"] <= 4096


def test_ballast_blocks_are_divided_with_remainder_on_first_sessions():
    assert divide_blocks_across_sessions(10, 4) == (3, 3, 2, 2)
    assert divide_blocks_across_sessions(8, 8) == (1, 1, 1, 1, 1, 1, 1, 1)
    assert divide_blocks_across_sessions(5, 8) == (1, 1, 1, 1, 1, 0, 0, 0)
    plan = plan_ballast_allocation(
        kv_mode="mixed",
        usable_hot_blocks=100,
        demotion_start_utilization=0.8,
        total_hot_blocks_turn1=10,
        num_ballast_sessions=8,
        max_model_len=4096,
        safety_margin_blocks=16,
    )
    required = math.ceil(0.8 * 100) + 16 - 10
    assert plan["required_ballast_blocks"] == required
    assert plan["planned_blocks_per_request"] == list(
        divide_blocks_across_sessions(required, 8)
    )


def test_safety_margin_is_above_demotion_start_threshold():
    without_margin = plan_ballast_allocation(
        kv_mode="mixed",
        usable_hot_blocks=100,
        demotion_start_utilization=0.8,
        total_hot_blocks_turn1=10,
        num_ballast_sessions=4,
        max_model_len=4096,
        safety_margin_blocks=0,
    )
    with_margin = plan_ballast_allocation(
        kv_mode="mixed",
        usable_hot_blocks=100,
        demotion_start_utilization=0.8,
        total_hot_blocks_turn1=10,
        num_ballast_sessions=4,
        max_model_len=4096,
        safety_margin_blocks=DEFAULT_BALLAST_SAFETY_MARGIN_BLOCKS,
    )
    assert DEFAULT_BALLAST_SAFETY_MARGIN_BLOCKS >= 16
    assert with_margin["target_hot_blocks"] == (
        without_margin["target_hot_blocks"] + DEFAULT_BALLAST_SAFETY_MARGIN_BLOCKS
    )
    assert with_margin["required_ballast_blocks"] == (
        without_margin["required_ballast_blocks"]
        + DEFAULT_BALLAST_SAFETY_MARGIN_BLOCKS
    )


def test_capacity_report_refuses_ballast_longer_than_max_model_len():
    plans = [make_plan(turn1_tokens=32)]
    budget = derive_persistent_kv_budget(budget_args())
    report = build_capacity_report(
        plans,
        kv_mode="mixed",
        memory_budget=budget,
        warm_pool_blocks=1024,
        demotion_start_utilization=0.8,
        num_ballast_sessions=1,
        max_num_seqs=128,
        max_model_len=64,
        safety_margin_blocks=16,
    )
    assert report["feasible"] is False
    assert any("max-model-len" in reason for reason in report["refusal_reasons"])
    tokens = report["ballast_plan"]["max_request_tokens"]
    assert tokens > 64


def test_capacity_report_refuses_zero_ballast_sessions_in_mixed():
    plans = [make_plan(turn1_tokens=64)]
    budget = derive_persistent_kv_budget(budget_args())
    report = build_capacity_report(
        plans,
        kv_mode="mixed",
        memory_budget=budget,
        warm_pool_blocks=1024,
        demotion_start_utilization=0.8,
        num_ballast_sessions=0,
        max_num_seqs=128,
        max_model_len=4096,
    )
    assert report["feasible"] is False
    assert any(
        "num-ballast-sessions" in reason for reason in report["refusal_reasons"]
    )


def test_exact_length_token_ids_are_deterministic_and_exact():
    ids = exact_length_token_ids(20, fill_token_id=7)
    assert ids == (7,) * 20
    assert exact_length_token_ids(0) == ()


def test_capacity_preflight_reports_runtime_ballast_plan():
    subset = load_gsm8k_subset(FIXTURE_DIR, num_questions=4, num_shots=5)
    plans = build_prompt_plans(subset, StubTokenizer())
    report = build_capacity_report(
        plans,
        kv_mode="mixed",
        memory_budget=derive_persistent_kv_budget(budget_args()),
        warm_pool_blocks=1024,
        demotion_start_utilization=0.8,
        num_ballast_sessions=8,
        max_num_seqs=128,
        max_model_len=4096,
        safety_margin_blocks=16,
    )
    peak = report["estimated_peak_requirement"]
    ballast = report["ballast_plan"]
    assert ballast["safety_margin_blocks"] == 16
    assert peak["prompt_tokens_per_ballast_request"] == ballast[
        "prompt_tokens_per_request"
    ]
    assert peak["generation_tokens_per_ballast_request"] == ballast[
        "generation_tokens_per_request"
    ]
    assert peak["planned_blocks_per_ballast_request"] == ballast[
        "planned_blocks_per_request"
    ]
    assert peak["total_planned_ballast_blocks"] == ballast[
        "total_planned_ballast_blocks"
    ]
    assert peak["estimated_peak_hot_blocks"] == ballast["estimated_peak_hot_blocks"]
    assert report["feasible"] is True


def test_mixed_demotion_barrier_timeout_raises_instead_of_returning(monkeypatch):
    async def fake_inspect(_engine, request_id):
        return {
            "request_id": request_id,
            "block_ids": [3, 4],
            "warm_residency": [],
            "allocator_matches_residency": True,
            "ownership_count_matches": True,
        }

    async def fake_worker(_engine):
        return {
            "warm_blocks": 0,
            "warm_requests": 0,
            "owned_warm_slots": 0,
            "allocator_consistent": True,
            "num_gpu_blocks": 100,
        }

    monkeypatch.setattr(
        "experiments.scripts.run_hkv_gsm8k_resume_quality.inspect_request",
        fake_inspect,
    )
    monkeypatch.setattr(
        "experiments.scripts.run_hkv_gsm8k_resume_quality.inspect_worker",
        fake_worker,
    )

    async def _run():
        state = EvaluatedSessionState(plan=make_plan(turn1_tokens=32))
        ballast_plan = {
            "required_ballast_blocks": 40,
            "planned_blocks_per_request": [20, 20],
            "safety_margin_blocks": 16,
            "total_planned_ballast_blocks": 40,
            "estimated_peak_hot_blocks": 42,
        }

        async def forever():
            await asyncio.Event().wait()

        tasks = [asyncio.create_task(forever()) for _ in range(2)]
        try:
            with pytest.raises(MixedDemotionBarrierTimeout, match="timed out") as caught:
                await wait_mixed_demotion_barrier(
                    engine=object(),
                    states=[state],
                    timeout=0.05,
                    poll_interval=0.01,
                    ballast_plan=ballast_plan,
                    ballast_ids=["gsm8k-ballast-000", "gsm8k-ballast-001"],
                    ballast_tasks=tasks,
                )
            diagnostics = caught.value.diagnostics
            assert diagnostics["required_ballast_blocks"] == 40
            assert diagnostics["planned_blocks_per_request"] == [20, 20]
            assert diagnostics["evaluated_sessions"][0][
                "complete_historical_blocks"
            ] == 2
            assert diagnostics["evaluated_sessions"][0]["block_ids"] == [3, 4]
            assert diagnostics["ballast_tasks"][0]["running"] is True
            assert diagnostics["allocator_consistent"] is True
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(_run())


def test_ballast_tasks_stay_alive_until_barrier_succeeds(monkeypatch):
    class FakeParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeStreamingInput:
        def __init__(self, payload, params):
            self.payload = payload
            self.params = params

    monkeypatch.setattr(
        "experiments.scripts.run_hkv_gsm8k_resume_quality._load_sampling_types",
        lambda: (FakeParams, FakeStreamingInput, SimpleNamespace(DELTA="delta")),
    )
    polls = {"count": 0}

    async def fake_inspect(_engine, request_id):
        polls["count"] += 1
        warm = polls["count"] >= 3
        return {
            "request_id": request_id,
            "block_ids": [0, 0],
            "warm_residency": (
                [{"key": [request_id, 0, 0]}, {"key": [request_id, 0, 1]}]
                if warm
                else []
            ),
            "allocator_matches_residency": True,
            "ownership_count_matches": True,
        }

    monkeypatch.setattr(
        "experiments.scripts.run_hkv_gsm8k_resume_quality.inspect_request",
        fake_inspect,
    )

    class FakeEngine:
        def __init__(self):
            self.aborted = []

        async def generate(self, prompt, params, request_id):
            agen = prompt
            await agen.__anext__()
            yield SimpleNamespace(outputs=[SimpleNamespace(finish_reason="length")])
            try:
                await agen.__anext__()
            except StopAsyncIteration:
                return

        async def abort(self, request_ids):
            self.aborted.extend(request_ids)

    async def _run():
        engine = FakeEngine()
        hold = asyncio.Event()
        task = asyncio.create_task(
            run_ballast_session(
                engine,
                "gsm8k-ballast-000",
                exact_length_token_ids(32),
                seed=0,
                max_tokens=1,
                hold_event=hold,
            )
        )
        state = EvaluatedSessionState(plan=make_plan(turn1_tokens=32))
        latest = await wait_mixed_demotion_barrier(
            engine,
            [state],
            timeout=1.0,
            poll_interval=0.01,
            ballast_plan={
                "required_ballast_blocks": 2,
                "planned_blocks_per_request": [2],
                "safety_margin_blocks": 16,
            },
            ballast_ids=["gsm8k-ballast-000"],
            ballast_tasks=[task],
        )
        assert task.done() is False
        assert latest[state.plan.session_id]["block_ids"] == [0, 0]
        await abort_ballast(engine, ["gsm8k-ballast-000"], [task])
        assert engine.aborted == ["gsm8k-ballast-000"]
        assert task.done() is True

    asyncio.run(_run())
    assert polls["count"] >= 3


def test_observed_four_session_layouts_satisfy_mixed_demotion_barrier(monkeypatch):
    layouts = {
        "gsm8k-0000": (47, 48, True),
        "gsm8k-0001": (44, 45, False),
        "gsm8k-0002": (46, 47, True),
        "gsm8k-0003": (45, 46, False),
    }
    snapshots = {}
    for session_id, (complete, n_block_ids, as_tuple) in layouts.items():
        snapshots[session_id] = {
            "request_id": session_id,
            "block_ids": list(range(100, 100 + n_block_ids)),
            "warm_residency": _warm_entries(
                session_id,
                complete - 1,
                as_tuple=as_tuple,
            ),
            "allocator_matches_residency": True,
            "ownership_count_matches": True,
        }

    async def fake_inspect(_engine, request_id):
        return snapshots[request_id]

    monkeypatch.setattr(
        "experiments.scripts.run_hkv_gsm8k_resume_quality.inspect_request",
        fake_inspect,
    )

    async def _run():
        states = [
            EvaluatedSessionState(
                plan=make_plan(
                    question_index=index,
                    turn1_tokens=complete * BLOCK_SIZE + 1,
                )
            )
            for index, (complete, _n_ids, _as_tuple) in enumerate(
                (
                    layouts["gsm8k-0000"],
                    layouts["gsm8k-0001"],
                    layouts["gsm8k-0002"],
                    layouts["gsm8k-0003"],
                )
            )
        ]
        for state in states:
            complete = layouts[state.plan.session_id][0]
            assert state.plan.complete_historical_blocks == complete
            assert complete_blocks_are_warm(
                complete_historical_blocks=complete,
                residency=snapshots[state.plan.session_id]["warm_residency"],
                block_ids=snapshots[state.plan.session_id]["block_ids"],
                request_id=state.plan.session_id,
            )

        async def forever():
            await asyncio.Event().wait()

        task = asyncio.create_task(forever())
        try:
            latest = await wait_mixed_demotion_barrier(
                engine=object(),
                states=states,
                timeout=0.5,
                poll_interval=0.01,
                ballast_plan={
                    "required_ballast_blocks": 1045,
                    "planned_blocks_per_request": [131] * 8,
                    "safety_margin_blocks": 16,
                },
                ballast_ids=["gsm8k-ballast-000"],
                ballast_tasks=[task],
            )
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert set(latest) == set(layouts)
        for session_id, snapshot in latest.items():
            complete = layouts[session_id][0]
            assert len(snapshot["block_ids"]) == complete + 1
            assert len(snapshot["warm_residency"]) == complete

    asyncio.run(_run())


def _empty_hkv_observation() -> dict:
    return {
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


def _worker_snapshot(
    *,
    warm_blocks: int,
    warm_requests: int,
    owned_warm_slots: int,
) -> dict:
    return {
        "warm_blocks": warm_blocks,
        "warm_requests": warm_requests,
        "owned_warm_slots": owned_warm_slots,
        "allocator_consistent": True,
        "max_gpu_allocated_bytes": 1_000,
        "max_gpu_reserved_bytes": 2_000,
        "num_gpu_blocks": 1519,
        "hot_kv_storage_bytes": 10,
        "warm_kv_storage_bytes": 20,
        "hot_to_warm_map_storage_bytes": 30,
        "warm_slot_table_storage_bytes": 40,
        "actual_persistent_kv_bytes": 100,
        "configured_total_kv_budget_bytes": 3758096384,
        "derived_hot_kv_budget_bytes": 50,
        "budget_slack_bytes": 5,
    }


def test_barrier_worker_sample_updates_all_peak_fields(monkeypatch):
    observation = _empty_hkv_observation()
    barrier = _worker_snapshot(
        warm_blocks=313,
        warm_requests=5,
        owned_warm_slots=313,
    )

    async def fake_inspect(_engine):
        return barrier

    monkeypatch.setattr(
        "experiments.scripts.run_hkv_gsm8k_resume_quality.inspect_worker",
        fake_inspect,
    )

    recorded = asyncio.run(sample_hkv_observation(object(), observation))
    assert recorded["warm_blocks"] == 313
    assert observation["samples"] == 1
    assert observation["warm_observed"] is True
    assert observation["peak_warm_blocks"] == 313
    assert observation["peak_warm_requests"] == 5
    assert observation["peak_owned_warm_slots"] == 313
    assert observation["final_warm_blocks"] is None
    assert observation["cleanup_complete"] is None


def test_cleanup_zero_sample_preserves_peaks_and_final_warm_zero(monkeypatch):
    observation = _empty_hkv_observation()
    update_observation(
        observation,
        _worker_snapshot(
            warm_blocks=313,
            warm_requests=5,
            owned_warm_slots=313,
        ),
    )

    async def fake_inspect(_engine):
        return _worker_snapshot(
            warm_blocks=0,
            warm_requests=0,
            owned_warm_slots=0,
        )

    monkeypatch.setattr(
        "experiments.scripts.run_qwen_bailian_replay.inspect_worker",
        fake_inspect,
    )

    final_state = asyncio.run(
        drain_warm_residency(object(), observation, cleanup_timeout=0.2)
    )
    assert final_state["warm_blocks"] == 0
    assert observation["samples"] == 2
    assert observation["warm_observed"] is True
    assert observation["peak_warm_blocks"] == 313
    assert observation["peak_warm_requests"] == 5
    assert observation["peak_owned_warm_slots"] == 313
    assert observation["final_warm_blocks"] == 0
    assert observation["cleanup_complete"] is True


def _warm_snapshot(session_id: str) -> dict:
    return {
        "request_id": session_id,
        "block_ids": [0, 0],
        "warm_residency": [
            {"key": [session_id, 0, 0]},
            {"key": [session_id, 0, 1]},
        ],
        "mixed_read_stats": {"mixed_read_steps": 0},
    }


def test_record_pre_resume_inspect_is_the_immediate_warm_proof():
    state = EvaluatedSessionState(plan=make_plan(turn1_tokens=32))
    errors: list[str] = []
    snapshot = record_pre_resume_inspect(
        state,
        _warm_snapshot(state.plan.session_id),
        kv_mode="mixed",
        validation_errors=errors,
    )
    assert errors == []
    assert snapshot["pre_resume_warm_confirmed"] is True
    assert state.pre_resume_inspect["pre_resume_warm_confirmed"] is True

    all_hot = EvaluatedSessionState(plan=make_plan(question_index=1))
    hot_errors: list[str] = []
    hot_snapshot = record_pre_resume_inspect(
        all_hot,
        {
            "request_id": all_hot.plan.session_id,
            "block_ids": [3, 4],
            "warm_residency": [],
            "mixed_read_stats": {"mixed_read_steps": 0},
        },
        kv_mode="all-hot",
        validation_errors=hot_errors,
    )
    assert hot_errors == []
    assert hot_snapshot["pre_resume_warm_confirmed"] is False


def test_resume_turn2_sequentially_cannot_overlap_and_uses_question_order():
    async def _run():
        states = [
            EvaluatedSessionState(plan=make_plan(question_index=index))
            for index in range(3)
        ]
        tracker = SequentialResumeTracker()
        timeline: list[str] = []

        async def session_body(state: EvaluatedSessionState) -> None:
            await state.resume_event.wait()
            tracker.start_turn2(state.plan.session_id)
            state.turn2_start_event.set()
            timeline.append(f"start:{state.plan.session_id}")
            await asyncio.sleep(0.02)
            timeline.append(f"finish:{state.plan.session_id}")
            tracker.finish_turn2(state.plan.session_id)
            state.completed_turns = 2
            state.turn2_done.set()

        async def inspect_before_resume(state: EvaluatedSessionState):
            timeline.append(f"inspect:{state.plan.session_id}")
            assert tracker.active == set()
            return _warm_snapshot(state.plan.session_id)

        tasks = [
            asyncio.create_task(session_body(state)) for state in states
        ]
        errors: list[str] = []
        order = await resume_turn2_sequentially(
            states,
            inspect_before_resume=inspect_before_resume,
            kv_mode="mixed",
            timeout=1.0,
            validation_errors=errors,
        )
        await asyncio.gather(*tasks)
        expected = [state.plan.session_id for state in states]
        assert order == expected
        assert tracker.order == expected
        assert tracker.max_active == 1
        assert tracker.overlap_events == []
        assert errors == []
        assert timeline == [
            "inspect:gsm8k-0000",
            "start:gsm8k-0000",
            "finish:gsm8k-0000",
            "inspect:gsm8k-0001",
            "start:gsm8k-0001",
            "finish:gsm8k-0001",
            "inspect:gsm8k-0002",
            "start:gsm8k-0002",
            "finish:gsm8k-0002",
        ]
        assert all(
            state.pre_resume_inspect["pre_resume_warm_confirmed"]
            for state in states
        )

    asyncio.run(_run())


def test_concurrent_turn2_resume_is_detected_as_overlap():
    async def _run():
        states = [
            EvaluatedSessionState(plan=make_plan(question_index=index))
            for index in range(3)
        ]
        tracker = SequentialResumeTracker()
        release = asyncio.Event()

        async def session_body(state: EvaluatedSessionState) -> None:
            await state.resume_event.wait()
            tracker.start_turn2(state.plan.session_id)
            state.turn2_start_event.set()
            await release.wait()
            tracker.finish_turn2(state.plan.session_id)
            state.completed_turns = 2
            state.turn2_done.set()

        tasks = [asyncio.create_task(session_body(state)) for state in states]
        for state in states:
            state.resume_event.set()
        await asyncio.sleep(0.05)
        assert tracker.max_active == 3
        assert tracker.overlap_events
        release.set()
        await asyncio.gather(*tasks)
        assert tracker.active == set()

    asyncio.run(_run())


class _FakeCompletion:
    def __init__(self, text="", token_ids=None, finish_reason=None):
        self.text = text
        self.token_ids = list(token_ids or [])
        self.finish_reason = finish_reason


class _FakeOutput:
    def __init__(self, *, text="", token_ids=None, finish_reason=None):
        self.outputs = [_FakeCompletion(text, token_ids, finish_reason)]
        self.finished = False


class _FakeStreamingEngine:
    """Mimic AsyncLLM: a separate handle_inputs task pulls StreamingInput."""

    _STREAM_END = object()

    def __init__(self):
        self.timeline: list[str] = []
        self.turn2_active: set[str] = set()
        self.max_turn2 = 0
        self.alive_generate: set[str] = set()

    async def generate(self, input_stream, params, request_id):
        self.alive_generate.add(request_id)
        queue: asyncio.Queue = asyncio.Queue()

        async def handle_inputs():
            turn = 0
            async for _chunk in input_stream:
                turn += 1
                if turn == 1:
                    await queue.put(
                        _FakeOutput(
                            text="t",
                            token_ids=[1],
                            finish_reason="length",
                        )
                    )
                    await queue.put(
                        _FakeOutput(text="", token_ids=[], finish_reason="length")
                    )
                    continue
                self.timeline.append(f"start:{request_id}")
                self.turn2_active.add(request_id)
                self.max_turn2 = max(self.max_turn2, len(self.turn2_active))
                await asyncio.sleep(0.03)
                await queue.put(
                    _FakeOutput(text="42", token_ids=[42, 43], finish_reason="stop")
                )
                self.turn2_active.discard(request_id)
                self.timeline.append(f"finish:{request_id}")
            await queue.put(self._STREAM_END)

        reader = asyncio.create_task(handle_inputs())
        try:
            while True:
                out = await queue.get()
                if out is self._STREAM_END:
                    break
                yield out
        finally:
            if not reader.done():
                reader.cancel()
                with suppress(asyncio.CancelledError):
                    await reader
            self.alive_generate.discard(request_id)


def test_sequential_turn2_retains_finished_sessions_and_ignores_turn1_finish_dupes(
    monkeypatch,
):
    class FakeParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeStreamingInput:
        def __init__(self, payload, params):
            self.payload = payload
            self.params = params

    monkeypatch.setattr(
        "experiments.scripts.run_hkv_gsm8k_resume_quality._load_sampling_types",
        lambda: (FakeParams, FakeStreamingInput, SimpleNamespace(DELTA="delta")),
    )

    async def _run():
        engine = _FakeStreamingEngine()
        states = [
            EvaluatedSessionState(plan=make_plan(question_index=index))
            for index in range(3)
        ]
        retain_event = asyncio.Event()
        tracker = SequentialResumeTracker()
        tasks = [
            asyncio.create_task(
                run_evaluated_session(
                    engine,
                    state,
                    seed=0,
                    max_tokens=8,
                    retain_event=retain_event,
                    tracker=tracker,
                )
            )
            for state in states
        ]
        await asyncio.wait_for(
            asyncio.gather(*(state.turn1_done.wait() for state in states)),
            timeout=1,
        )
        assert [state.completed_turns for state in states] == [1, 1, 1]
        assert all(not state.turn2_done.is_set() for state in states)
        assert all(not state.turn2_token_ids for state in states)

        async def inspect(_state):
            return {
                "request_id": _state.plan.session_id,
                "block_ids": [3, 4],
                "warm_residency": [],
                "mixed_read_stats": {"mixed_read_steps": 0},
            }

        errors: list[str] = []
        order = await resume_turn2_sequentially(
            states,
            inspect_before_resume=inspect,
            kv_mode="all-hot",
            timeout=1.0,
            validation_errors=errors,
            tasks=tasks,
        )
        expected = [state.plan.session_id for state in states]
        assert order == expected
        assert engine.timeline == [
            "start:gsm8k-0000",
            "finish:gsm8k-0000",
            "start:gsm8k-0001",
            "finish:gsm8k-0001",
            "start:gsm8k-0002",
            "finish:gsm8k-0002",
        ]
        assert engine.max_turn2 == 1
        assert tracker.max_active == 1
        assert tracker.overlap_events == []
        assert engine.alive_generate == set(expected)
        assert all(not task.done() for task in tasks)
        assert [state.completed_turns for state in states] == [2, 2, 2]
        assert all(state.turn2_token_ids == [42, 43] for state in states)
        retain_event.set()
        await asyncio.gather(*tasks)
        assert engine.alive_generate == set()

    asyncio.run(_run())


def test_turn2_completes_on_eos_or_empty_finish_chunk():
    eos = frozenset({151645})
    with_eos = _FakeOutput(text="#### 3", token_ids=[18, 151645], finish_reason=None)
    assert _turn2_output_is_complete(with_eos, with_eos.outputs[0], eos) is True
    empty_finish = _FakeOutput(text="", token_ids=[], finish_reason="stop")
    assert (
        _turn2_output_is_complete(empty_finish, empty_finish.outputs[0], eos) is True
    )
    mid_sentence = _FakeOutput(text="The", token_ids=[791], finish_reason=None)
    assert _turn2_output_is_complete(mid_sentence, mid_sentence.outputs[0], eos) is False
    empty_no_finish = _FakeOutput(text="", token_ids=[], finish_reason=None)
    empty_no_finish.outputs = []
    assert _turn2_output_is_complete(empty_no_finish, None, eos) is False


def test_sequential_turn2_completes_if_eos_arrives_without_finish_reason(monkeypatch):
    class FakeParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeStreamingInput:
        def __init__(self, payload, params):
            self.payload = payload
            self.params = params

    monkeypatch.setattr(
        "experiments.scripts.run_hkv_gsm8k_resume_quality._load_sampling_types",
        lambda: (FakeParams, FakeStreamingInput, SimpleNamespace(DELTA="delta")),
    )
    eos_id = 151645

    class HangAfterEosEngine(_FakeStreamingEngine):
        async def generate(self, input_stream, params, request_id):
            self.alive_generate.add(request_id)
            queue: asyncio.Queue = asyncio.Queue()

            async def handle_inputs():
                turn = 0
                async for _chunk in input_stream:
                    turn += 1
                    if turn == 1:
                        await queue.put(
                            _FakeOutput(
                                text="t",
                                token_ids=[1],
                                finish_reason="length",
                            )
                        )
                        continue
                    self.timeline.append(f"start:{request_id}")
                    self.turn2_active.add(request_id)
                    self.max_turn2 = max(self.max_turn2, len(self.turn2_active))
                    await queue.put(
                        _FakeOutput(
                            text="#### 3",
                            token_ids=[18, eos_id],
                            finish_reason=None,
                        )
                    )
                    self.turn2_active.discard(request_id)
                    self.timeline.append(f"finish:{request_id}")
                await queue.put(self._STREAM_END)

            reader = asyncio.create_task(handle_inputs())
            try:
                while True:
                    out = await queue.get()
                    if out is self._STREAM_END:
                        break
                    yield out
            finally:
                if not reader.done():
                    reader.cancel()
                    with suppress(asyncio.CancelledError):
                        await reader
                self.alive_generate.discard(request_id)

    async def _run():
        engine = HangAfterEosEngine()
        states = [
            EvaluatedSessionState(plan=make_plan(question_index=index))
            for index in range(3)
        ]
        retain_event = asyncio.Event()
        tracker = SequentialResumeTracker()
        tasks = [
            asyncio.create_task(
                run_evaluated_session(
                    engine,
                    state,
                    seed=0,
                    max_tokens=8,
                    retain_event=retain_event,
                    tracker=tracker,
                    eos_token_ids=frozenset({eos_id}),
                )
            )
            for state in states
        ]
        await asyncio.wait_for(
            asyncio.gather(*(state.turn1_done.wait() for state in states)),
            timeout=1,
        )

        async def inspect(_state):
            return {
                "request_id": _state.plan.session_id,
                "block_ids": [3],
                "warm_residency": [],
                "mixed_read_stats": {"mixed_read_steps": 0},
            }

        errors: list[str] = []
        order = await resume_turn2_sequentially(
            states,
            inspect_before_resume=inspect,
            kv_mode="all-hot",
            timeout=1.0,
            validation_errors=errors,
            tasks=tasks,
        )
        assert order == [state.plan.session_id for state in states]
        assert engine.timeline == [
            "start:gsm8k-0000",
            "finish:gsm8k-0000",
            "start:gsm8k-0001",
            "finish:gsm8k-0001",
            "start:gsm8k-0002",
            "finish:gsm8k-0002",
        ]
        assert engine.max_turn2 == 1
        assert [state.completed_turns for state in states] == [2, 2, 2]
        retain_event.set()
        await asyncio.gather(*tasks)

    asyncio.run(_run())

