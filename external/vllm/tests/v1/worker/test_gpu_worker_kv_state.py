# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_state import (
    KVBlockState,
    KVCacheBlockTransition,
    KVCacheStateTransition,
    KVCacheTransitionResult,
    KVCacheTransitionStatus,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu.hkv_migration import (
    HKVWarmCapacityError,
    HKVWarmMigrationManager,
    HKVWarmResidency,
    HKVWarmStaleValidationError,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker


def make_transition(
    transition_id: int = 0,
    request_id: str = "request",
    previous_state: KVBlockState = KVBlockState.HOT,
    new_state: KVBlockState = KVBlockState.WARM,
    changed_blocks: tuple[list[KVCacheBlockTransition], ...] | None = None,
) -> KVCacheStateTransition:
    if changed_blocks is None:
        changed_blocks = ([KVCacheBlockTransition(0, 1)],)
    return KVCacheStateTransition(
        transition_id=transition_id,
        request_id=request_id,
        previous_state=previous_state,
        new_state=new_state,
        changed_blocks=changed_blocks,
    )


def make_worker(use_v2_model_runner: bool, events: list[object]) -> Worker:
    worker = Worker.__new__(Worker)
    worker._pp_send_work = []
    worker.profiler = None
    worker.use_v2_model_runner = use_v2_model_runner
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )

    def execute_model(_scheduler_output, _intermediate_tensors):
        events.append("runner")
        return None

    worker.model_runner = SimpleNamespace(
        execute_model=execute_model,
        is_pooling_model=False,
    )
    return worker


def make_mock_model_runner(
    migration_manager: object = None,
    block_tables_data: list[list[int]] | None = None,
    device: str = "cpu",
) -> SimpleNamespace:
    if block_tables_data is None:
        block_tables_data = [[0, 1, 7, 3, 4, 11]]
    num_blocks = len(block_tables_data[0])
    return SimpleNamespace(
        hkv_warm_migration_manager=migration_manager,
        block_tables=SimpleNamespace(
            blocks_per_kv_block=[1],
            block_tables=[
                SimpleNamespace(gpu=torch.tensor(block_tables_data))
            ],
            num_blocks=SimpleNamespace(np=torch.tensor([[num_blocks]])),
        ),
        req_states=SimpleNamespace(
            req_id_to_index={"request": 0, "request-a": 0, "request-b": 0}
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        dp_size=1,
        device=torch.device(device),
        _pending_kv_transition_results=[],
        _take_kv_transition_results=(
            lambda self=None: GPUModelRunner._take_kv_transition_results(
                self or runner
            )
        ),
    )


def test_disabled_migration_keeps_validation_only_behavior(monkeypatch):
    monkeypatch.delenv("HKV_ENABLE_MULTI_BLOCK_WARM_MIGRATION", raising=False)
    worker = Worker.__new__(Worker)
    worker.use_v2_model_runner = True
    worker.model_runner = SimpleNamespace(
        handle_kv_cache_state_transitions=MagicMock()
    )
    worker._handle_kv_cache_state_transitions([make_transition()])

    worker.model_runner.handle_kv_cache_state_transitions.assert_not_called()


def test_enabled_v2_receives_transitions_before_zero_token_runner(monkeypatch):
    monkeypatch.setenv("HKV_ENABLE_MULTI_BLOCK_WARM_MIGRATION", "1")
    events = []
    worker = make_worker(True, events)

    def handle_transitions(transitions):
        events.append(("migration", transitions))

    worker.model_runner.handle_kv_cache_state_transitions = handle_transitions
    output = SchedulerOutput.make_empty()
    output.kv_cache_state_transitions = [make_transition()]

    Worker.execute_model(worker, output)

    assert events == [
        ("migration", output.kv_cache_state_transitions),
        "runner",
    ]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("legacy", "legacy/MRV1"),
        ("multiple_groups", "exactly one"),
        ("block_expansion", "blocks_per_kv_block"),
        ("hot_to_cold", "only HOT->WARM"),
        ("warm_to_cold", "only HOT->WARM"),
    ],
)
def test_enabled_migration_rejects_unsupported_configurations(
    monkeypatch,
    case: str,
    message: str,
):
    monkeypatch.setenv("HKV_ENABLE_MULTI_BLOCK_WARM_MIGRATION", "1")
    transition = make_transition()
    if case == "legacy":
        target = Worker.__new__(Worker)
        target.use_v2_model_runner = False
        call = partial(target._handle_kv_cache_state_transitions, [transition])
    else:
        target = make_mock_model_runner(
            migration_manager=SimpleNamespace(migrate=MagicMock())
        )
        if case == "multiple_groups":
            transition = make_transition(
                changed_blocks=(
                    [KVCacheBlockTransition(0, 1)],
                    [KVCacheBlockTransition(0, 2)],
                )
            )
        elif case == "block_expansion":
            target.block_tables.blocks_per_kv_block = [2]
        elif case == "hot_to_cold":
            transition = make_transition(new_state=KVBlockState.COLD)
        else:
            transition = make_transition(
                previous_state=KVBlockState.WARM,
                new_state=KVBlockState.COLD,
            )
        call = partial(
            GPUModelRunner.handle_kv_cache_state_transitions,
            target,
            [transition],
        )

    with pytest.raises(ValueError, match=message):
        call()


def test_shadow_migration_preserves_logical_and_cache_group_identity():
    migration_manager = SimpleNamespace(migrate=MagicMock(return_value=False))
    target = make_mock_model_runner(migration_manager=migration_manager)
    transition = make_transition(
        changed_blocks=(
            [
                KVCacheBlockTransition(2, 7),
                KVCacheBlockTransition(5, 11),
            ],
        )
    )

    results = GPUModelRunner.handle_kv_cache_state_transitions(target, [transition])

    migration_manager.migrate.assert_called_once_with(
        "request",
        transition.changed_blocks,
        ((0, 1, 7, 3, 4, 11),),
    )
    assert len(results) == 1
    assert results[0].status is KVCacheTransitionStatus.SUCCESS
    assert results[0].transition_id == 0


def test_logical_warm_table_rebuild_clears_reused_request_row():
    warm_slot_table = torch.full((3, 4), 9, dtype=torch.int32)
    warm_residency = {
        ("request-a", 0, 2): HKVWarmResidency(
            warm_slot_id=5,
            temporary_shadow_hot_block_id=7,
        )
    }
    residency_items = MagicMock(wraps=warm_residency.items)
    migration_manager = SimpleNamespace(
        warm_residency=SimpleNamespace(items=residency_items),
        warm_residency_revision=1,
    )
    block_tables = SimpleNamespace(
        gather_block_tables=MagicMock(
            return_value=(torch.zeros((3, 4), dtype=torch.int32),)
        ),
        compute_slot_mappings=MagicMock(return_value=torch.zeros((1, 1))),
    )
    target = SimpleNamespace(
        hkv_warm_slot_table=warm_slot_table,
        hkv_warm_migration_manager=migration_manager,
        _hkv_warm_slot_table_req_ids=None,
        _hkv_warm_slot_table_revision=-1,
        _hkv_warm_slot_table_num_reqs_after_padding=-1,
        block_tables=block_tables,
    )
    input_batch = SimpleNamespace(
        req_ids=["other-request", "request-a"],
        num_reqs_after_padding=2,
        idx_mapping=torch.tensor([0, 1]),
        query_start_loc=torch.tensor([0, 1, 2]),
        positions=torch.tensor([0, 0]),
        num_tokens_after_padding=2,
    )

    GPUModelRunner.prepare_attn(target, input_batch)
    assert warm_slot_table[:2].tolist() == [
        [-1, -1, -1, -1],
        [-1, -1, 5, -1],
    ]
    assert residency_items.call_count == 1

    warm_slot_table[0, 0] = 8
    GPUModelRunner.prepare_attn(target, input_batch)
    assert warm_slot_table[0, 0].item() == 8
    assert residency_items.call_count == 1

    warm_residency[("other-request", 0, 1)] = HKVWarmResidency(
        warm_slot_id=3,
        temporary_shadow_hot_block_id=4,
    )
    migration_manager.warm_residency_revision += 1
    GPUModelRunner.prepare_attn(target, input_batch)
    assert warm_slot_table[:2].tolist() == [
        [-1, 3, -1, -1],
        [-1, -1, 5, -1],
    ]
    assert residency_items.call_count == 2

    input_batch.req_ids = ["request-a", "request-b"]
    GPUModelRunner.prepare_attn(target, input_batch)
    assert warm_slot_table[:2].tolist() == [
        [-1, -1, 5, -1],
        [-1, -1, -1, -1],
    ]
    assert residency_items.call_count == 3

    warm_slot_table[2].fill_(6)
    input_batch.num_reqs_after_padding = 3
    GPUModelRunner.prepare_attn(target, input_batch)
    assert warm_slot_table[2].tolist() == [-1, -1, -1, -1]
    assert residency_items.call_count == 4


def test_insufficient_warm_capacity_leaves_state_unchanged():
    hot_to_warm_maps = {}
    manager = HKVWarmMigrationManager(
        warm_capacity=1,
        hot_kv_caches={},
        warm_kv_caches={},
        hot_to_warm_maps=hot_to_warm_maps,
        device="cpu",
    )

    with pytest.raises(HKVWarmCapacityError):
        manager.migrate(
            "request",
            ([KVCacheBlockTransition(0, 2), KVCacheBlockTransition(1, 7)],),
            ((2, 7),),
        )

    assert hot_to_warm_maps == {}
    assert manager.allocator.num_owned_slots == 0
    assert manager.warm_residency == {}


@pytest.mark.parametrize(
    ("finished_req_ids", "preempted_req_ids"),
    [
        ({"request"}, None),
        (set(), {"request"}),
        ({"request"}, {"request"}),
    ],
    ids=("finished", "preempted", "finished-and-preempted"),
)
def test_request_releases_warm_state_before_runner_removal(
    finished_req_ids: set[str],
    preempted_req_ids: set[str] | None,
):
    events = []
    target = SimpleNamespace(
        hkv_warm_migration_manager=SimpleNamespace(
            release_request=lambda req_id: events.append(("release", req_id))
        ),
        _remove_request=lambda req_id: events.append(("remove", req_id)),
    )
    output = SchedulerOutput.make_empty()
    output.finished_req_ids = finished_req_ids
    output.preempted_req_ids = preempted_req_ids

    GPUModelRunner.finish_requests(target, output)

    assert events == [
        ("release", "request"),
        ("remove", "request"),
    ]


# ==============================================================================
# M3C1 Completion Protocol & Acknowledgment Transport Tests
# ==============================================================================

def test_cuda_stream_synchronize_called_exactly_once_for_new_work(monkeypatch):
    sync_mock = MagicMock()
    mock_stream = SimpleNamespace(synchronize=sync_mock)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _dev: mock_stream)

    migration_manager = SimpleNamespace(migrate=MagicMock(return_value=True))
    target = make_mock_model_runner(
        migration_manager=migration_manager, device="cuda:0"
    )

    t1 = make_transition(transition_id=0, request_id="request-a")
    t2 = make_transition(transition_id=1, request_id="request-b")

    results = GPUModelRunner.handle_kv_cache_state_transitions(target, [t1, t2])

    assert sync_mock.call_count == 1
    assert len(results) == 2
    assert results[0].status is KVCacheTransitionStatus.SUCCESS
    assert results[0].transition_id == 0
    assert results[1].status is KVCacheTransitionStatus.SUCCESS
    assert results[1].transition_id == 1


def test_cuda_stream_synchronize_not_called_for_idempotent_repeat(monkeypatch):
    sync_mock = MagicMock()
    mock_stream = SimpleNamespace(synchronize=sync_mock)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _dev: mock_stream)

    migration_manager = SimpleNamespace(migrate=MagicMock(return_value=False))
    target = make_mock_model_runner(
        migration_manager=migration_manager, device="cuda:0"
    )

    t1 = make_transition(transition_id=5, request_id="request-a")
    results = GPUModelRunner.handle_kv_cache_state_transitions(target, [t1])

    sync_mock.assert_not_called()
    assert len(results) == 1
    assert results[0].status is KVCacheTransitionStatus.SUCCESS
    assert results[0].transition_id == 5


def test_cuda_stream_synchronize_not_called_for_capacity_nack(monkeypatch):
    sync_mock = MagicMock()
    mock_stream = SimpleNamespace(synchronize=sync_mock)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _dev: mock_stream)

    def raise_capacity(*args, **kwargs):
        raise HKVWarmCapacityError("cannot fit in WARM pool")

    migration_manager = SimpleNamespace(migrate=raise_capacity)
    target = make_mock_model_runner(
        migration_manager=migration_manager, device="cuda:0"
    )

    t1 = make_transition(transition_id=10, request_id="request-a")
    results = GPUModelRunner.handle_kv_cache_state_transitions(target, [t1])

    sync_mock.assert_not_called()
    assert len(results) == 1
    assert results[0].status is KVCacheTransitionStatus.RETRYABLE_CAPACITY
    assert results[0].transition_id == 10
    assert "cannot fit" in (results[0].error_message or "")


def test_cuda_stream_synchronize_not_called_for_stale_validation(monkeypatch):
    sync_mock = MagicMock()
    mock_stream = SimpleNamespace(synchronize=sync_mock)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _dev: mock_stream)

    def raise_stale(*args, **kwargs):
        raise HKVWarmStaleValidationError("source HOT block mismatch")

    migration_manager = SimpleNamespace(migrate=raise_stale)
    target = make_mock_model_runner(
        migration_manager=migration_manager, device="cuda:0"
    )

    t1 = make_transition(transition_id=12, request_id="request-a")
    results = GPUModelRunner.handle_kv_cache_state_transitions(target, [t1])

    sync_mock.assert_not_called()
    assert len(results) == 1
    assert results[0].status is KVCacheTransitionStatus.STALE_VALIDATION
    assert results[0].transition_id == 12


def test_success_appears_only_after_fence(monkeypatch):
    state_at_sync = {}

    def do_sync():
        state_at_sync["pending_count"] = len(target._pending_kv_transition_results)

    mock_stream = SimpleNamespace(synchronize=do_sync)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _dev: mock_stream)

    migration_manager = SimpleNamespace(migrate=MagicMock(return_value=True))
    target = make_mock_model_runner(
        migration_manager=migration_manager, device="cuda:0"
    )

    t1 = make_transition(transition_id=0)
    results = GPUModelRunner.handle_kv_cache_state_transitions(target, [t1])

    assert state_at_sync["pending_count"] == 0
    assert len(results) == 1
    assert results[0].status is KVCacheTransitionStatus.SUCCESS


def test_cuda_synchronize_failure_propagates(monkeypatch):
    def fail_sync():
        raise RuntimeError("CUDA device error during stream sync")

    mock_stream = SimpleNamespace(synchronize=fail_sync)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _dev: mock_stream)

    migration_manager = SimpleNamespace(migrate=MagicMock(return_value=True))
    target = make_mock_model_runner(
        migration_manager=migration_manager, device="cuda:0"
    )

    t1 = make_transition(transition_id=0)
    with pytest.raises(RuntimeError, match="CUDA device error"):
        GPUModelRunner.handle_kv_cache_state_transitions(target, [t1])


def test_zero_token_result_reaches_scheduler():
    runner = SimpleNamespace(
        update_pp_decode_requests=MagicMock(),
        finish_requests=MagicMock(),
        free_states=MagicMock(),
        add_requests=MagicMock(),
        update_requests=MagicMock(),
        block_tables=SimpleNamespace(apply_staged_writes=MagicMock()),
        kv_connector=SimpleNamespace(
            no_forward=MagicMock(return_value=EMPTY_MODEL_RUNNER_OUTPUT)
        ),
        _pending_kv_transition_results=[],
        _take_kv_transition_results=lambda: [
            make_transition(transition_id=42).to_result(
                KVCacheTransitionStatus.SUCCESS
            )
        ],
        _attach_kv_transition_results=lambda out: (
            GPUModelRunner._attach_kv_transition_results(runner, out)
        ),
    )

    sched_out = SchedulerOutput.make_empty()
    sched_out.total_num_scheduled_tokens = 0
    sched_out.kv_cache_state_transitions = [make_transition(transition_id=42)]

    out = GPUModelRunner.execute_model(runner, sched_out)

    assert isinstance(out, ModelRunnerOutput)
    assert out is not EMPTY_MODEL_RUNNER_OUTPUT
    assert len(out.kv_cache_transition_results) == 1
    assert out.kv_cache_transition_results[0].transition_id == 42
    assert (
        out.kv_cache_transition_results[0].status
        is KVCacheTransitionStatus.SUCCESS
    )

    # Scheduler validation accepts this
    sched = Scheduler.__new__(Scheduler)
    t = sched_out.kv_cache_state_transitions[0]
    sched._pending_kv_transitions = {t.request_id: t}
    sched.requests = {t.request_id: SimpleNamespace(kv_cache_state=KVBlockState.HOT)}
    sched.kv_cache_manager = SimpleNamespace(
        commit_request_kv_transition=MagicMock()
    )
    Scheduler._validate_kv_cache_transition_results(sched, sched_out, out)


def test_scheduler_validates_transition_results_exact_match():
    sched = Scheduler.__new__(Scheduler)
    t = make_transition(transition_id=1, request_id="r1")
    sched._pending_kv_transitions = {"r1": t}
    sched.requests = {"r1": SimpleNamespace(kv_cache_state=KVBlockState.HOT)}
    sched.kv_cache_manager = SimpleNamespace(
        commit_request_kv_transition=MagicMock()
    )
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    res = t.to_result(KVCacheTransitionStatus.SUCCESS)
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res]
    )

    # Should not raise and commits transition
    Scheduler._validate_kv_cache_transition_results(sched, sched_out, mr_out)
    assert sched.requests["r1"].kv_cache_state is KVBlockState.WARM
    assert "r1" not in sched._pending_kv_transitions


def test_scheduler_rejects_missing_result():
    sched = Scheduler.__new__(Scheduler)
    t = make_transition(transition_id=1, request_id="r1")
    sched._pending_kv_transitions = {"r1": t}
    sched.requests = {"r1": SimpleNamespace(kv_cache_state=KVBlockState.HOT)}
    sched.kv_cache_manager = SimpleNamespace(
        commit_request_kv_transition=MagicMock()
    )
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[]
    )
    with pytest.raises(
        ValueError, match="Mismatch in KV transition result count: expected 1"
    ):
        Scheduler._validate_kv_cache_transition_results(sched, sched_out, mr_out)


def test_scheduler_rejects_extra_result():
    sched = Scheduler.__new__(Scheduler)
    t = make_transition(transition_id=1, request_id="r1")
    sched._pending_kv_transitions = {"r1": t}
    sched.requests = {"r1": SimpleNamespace(kv_cache_state=KVBlockState.HOT)}
    sched.kv_cache_manager = SimpleNamespace(
        commit_request_kv_transition=MagicMock()
    )
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    res1 = t.to_result(KVCacheTransitionStatus.SUCCESS)
    res2 = make_transition(transition_id=2).to_result(
        KVCacheTransitionStatus.SUCCESS
    )
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res1, res2]
    )

    with pytest.raises(
        ValueError, match="Mismatch in KV transition result count: expected 1"
    ):
        Scheduler._validate_kv_cache_transition_results(sched, sched_out, mr_out)


def test_scheduler_rejects_duplicate_result():
    sched = Scheduler.__new__(Scheduler)
    t1 = make_transition(transition_id=1, request_id="r1")
    t2 = make_transition(transition_id=2, request_id="r2")
    sched._pending_kv_transitions = {"r1": t1, "r2": t2}
    sched.requests = {
        "r1": SimpleNamespace(kv_cache_state=KVBlockState.HOT),
        "r2": SimpleNamespace(kv_cache_state=KVBlockState.HOT),
    }
    sched.kv_cache_manager = SimpleNamespace(
        commit_request_kv_transition=MagicMock()
    )
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t1, t2]

    res1 = t1.to_result(KVCacheTransitionStatus.SUCCESS)
    # create duplicate transition_id=1
    res2 = KVCacheTransitionResult(
        transition_id=1,
        request_id="r2",
        previous_state=KVBlockState.HOT,
        new_state=KVBlockState.WARM,
        changed_blocks=t2.signature[4],
        status=KVCacheTransitionStatus.SUCCESS,
    )
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res1, res2]
    )

    with pytest.raises(
        ValueError, match="Duplicate KV transition result for transition_id=1"
    ):
        Scheduler._validate_kv_cache_transition_results(sched, sched_out, mr_out)


def test_scheduler_rejects_mismatched_transition_id():
    sched = Scheduler.__new__(Scheduler)
    t = make_transition(transition_id=1, request_id="r1")
    sched._pending_kv_transitions = {"r1": t}
    sched.requests = {"r1": SimpleNamespace(kv_cache_state=KVBlockState.HOT)}
    sched.kv_cache_manager = SimpleNamespace(
        commit_request_kv_transition=MagicMock()
    )
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    res = KVCacheTransitionResult(
        transition_id=99,
        request_id="r1",
        previous_state=KVBlockState.HOT,
        new_state=KVBlockState.WARM,
        changed_blocks=t.signature[4],
        status=KVCacheTransitionStatus.SUCCESS,
    )
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res]
    )

    with pytest.raises(
        ValueError, match="KV transition ID mismatch: expected 1, got 99"
    ):
        Scheduler._validate_kv_cache_transition_results(sched, sched_out, mr_out)


def test_scheduler_rejects_mismatched_signature():
    sched = Scheduler.__new__(Scheduler)
    t = make_transition(
        transition_id=1,
        request_id="r1",
        changed_blocks=([KVCacheBlockTransition(0, 5)],),
    )
    sched._pending_kv_transitions = {"r1": t}
    sched.requests = {"r1": SimpleNamespace(kv_cache_state=KVBlockState.HOT)}
    sched.kv_cache_manager = SimpleNamespace(
        commit_request_kv_transition=MagicMock()
    )
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    # result with different block source ID
    res = KVCacheTransitionResult(
        transition_id=1,
        request_id="r1",
        previous_state=KVBlockState.HOT,
        new_state=KVBlockState.WARM,
        changed_blocks=((0, 0, 999),),
        status=KVCacheTransitionStatus.SUCCESS,
    )
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res]
    )

    with pytest.raises(
        ValueError,
        match="KV transition signature mismatch for transition_id=1",
    ):
        Scheduler._validate_kv_cache_transition_results(sched, sched_out, mr_out)


def test_scheduler_rejects_missing_pending_transition():
    sched = Scheduler.__new__(Scheduler)
    t = make_transition(transition_id=1, request_id="r1")
    sched._pending_kv_transitions = {}  # Empty pending transitions
    sched.requests = {"r1": SimpleNamespace(kv_cache_state=KVBlockState.HOT)}
    sched.kv_cache_manager = SimpleNamespace(
        commit_request_kv_transition=MagicMock()
    )
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    res = t.to_result(KVCacheTransitionStatus.SUCCESS)
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res]
    )

    with pytest.raises(
        ValueError,
        match="No matching pending KV transition for request r1",
    ):
        Scheduler._validate_kv_cache_transition_results(sched, sched_out, mr_out)


def _make_mock_scheduler_for_commit(
    request_id: str = "r1",
    request_state: KVBlockState = KVBlockState.HOT,
    block_states: list[KVBlockState] | None = None,
    block_ids: list[int] | None = None,
) -> tuple[Scheduler, list[SimpleNamespace]]:
    if block_states is None:
        block_states = [KVBlockState.HOT, KVBlockState.HOT]
    if block_ids is None:
        block_ids = [10, 20]
    blocks = [
        SimpleNamespace(block_id=bid, hierarchy_state=bstate)
        for bid, bstate in zip(block_ids, block_states, strict=True)
    ]
    request = SimpleNamespace(
        request_id=request_id, kv_cache_state=request_state
    )

    def commit_transition(transition):
        req_blocks = (blocks,)
        for group_idx, group_transitions in enumerate(
            transition.changed_blocks
        ):
            if group_idx >= len(req_blocks):
                raise ValueError(
                    f"Cache group index {group_idx} out of range during "
                    f"commit for request {transition.request_id}"
                )
            current_group = req_blocks[group_idx]
            for block_trans in group_transitions:
                logical_idx = block_trans.logical_block_index
                expected_hot_id = block_trans.source_hot_block_id
                if logical_idx >= len(current_group):
                    raise ValueError(
                        f"Logical block index {logical_idx} out of range "
                        f"during commit for request {transition.request_id}"
                    )
                block = current_group[logical_idx]
                if block.block_id != expected_hot_id:
                    raise ValueError(
                        f"Physical block ID mismatch during commit for "
                        f"request {transition.request_id} group {group_idx} "
                        f"logical block {logical_idx}: expected block_id "
                        f"{expected_hot_id}, found {block.block_id}"
                    )
                block.hierarchy_state = transition.new_state

    kv_cache_manager = SimpleNamespace(
        get_blocks=lambda _req: SimpleNamespace(blocks=(blocks,)),
        commit_request_kv_transition=commit_transition,
    )
    sched = Scheduler.__new__(Scheduler)
    sched._pending_kv_transitions = {}
    sched.requests = {request_id: request}
    sched.kv_cache_manager = kv_cache_manager
    return sched, blocks


def test_scheduler_retryable_capacity_clears_pending_and_leaves_hot():
    sched, blocks = _make_mock_scheduler_for_commit(
        block_states=[KVBlockState.HOT, KVBlockState.HOT, KVBlockState.HOT],
        block_ids=[10, 20, 30],
    )
    t = make_transition(
        transition_id=1,
        request_id="r1",
        changed_blocks=(
            [
                KVCacheBlockTransition(0, 10),
                KVCacheBlockTransition(1, 20),
            ],
        ),
    )
    sched._pending_kv_transitions = {"r1": t}
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    res = t.to_result(
        KVCacheTransitionStatus.RETRYABLE_CAPACITY, "no warm slots"
    )
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res]
    )

    Scheduler._validate_kv_cache_transition_results(sched, sched_out, mr_out)

    assert sched.requests["r1"].kv_cache_state is KVBlockState.HOT
    assert blocks[0].hierarchy_state is KVBlockState.HOT
    assert blocks[1].hierarchy_state is KVBlockState.HOT
    assert blocks[2].hierarchy_state is KVBlockState.HOT
    assert "r1" not in sched._pending_kv_transitions


def test_scheduler_stale_validation_clears_pending_and_leaves_hot():
    sched, blocks = _make_mock_scheduler_for_commit(
        block_states=[KVBlockState.HOT],
        block_ids=[10],
    )
    t = make_transition(
        transition_id=1,
        request_id="r1",
        changed_blocks=([KVCacheBlockTransition(0, 10)],),
    )
    sched._pending_kv_transitions = {"r1": t}
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    res = t.to_result(
        KVCacheTransitionStatus.STALE_VALIDATION, "stale block mismatch"
    )
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res]
    )

    Scheduler._validate_kv_cache_transition_results(sched, sched_out, mr_out)

    assert sched.requests["r1"].kv_cache_state is KVBlockState.HOT
    assert blocks[0].hierarchy_state is KVBlockState.HOT
    assert "r1" not in sched._pending_kv_transitions


def test_scheduler_success_commits_warm_state():
    sched, blocks = _make_mock_scheduler_for_commit(
        block_states=[KVBlockState.HOT, KVBlockState.HOT],
        block_ids=[10, 20],
    )
    t = make_transition(
        transition_id=1,
        request_id="r1",
        changed_blocks=(
            [
                KVCacheBlockTransition(0, 10),
                KVCacheBlockTransition(1, 20),
            ],
        ),
    )
    sched._pending_kv_transitions = {"r1": t}
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    res = t.to_result(KVCacheTransitionStatus.SUCCESS)
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res]
    )

    Scheduler._validate_kv_cache_transition_results(sched, sched_out, mr_out)

    assert sched.requests["r1"].kv_cache_state is KVBlockState.WARM
    assert blocks[0].hierarchy_state is KVBlockState.WARM
    assert blocks[1].hierarchy_state is KVBlockState.WARM
    assert "r1" not in sched._pending_kv_transitions


def test_scheduler_success_does_not_modify_untransitioned_blocks():
    # block 0 was transitioned; block 1 was untransitioned and remained HOT
    sched, blocks = _make_mock_scheduler_for_commit(
        block_states=[KVBlockState.HOT, KVBlockState.HOT],
        block_ids=[10, 20],
    )
    t = make_transition(
        transition_id=1,
        request_id="r1",
        changed_blocks=([KVCacheBlockTransition(0, 10)],),
    )
    sched._pending_kv_transitions = {"r1": t}
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    res = t.to_result(KVCacheTransitionStatus.SUCCESS)
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res]
    )

    Scheduler._validate_kv_cache_transition_results(sched, sched_out, mr_out)

    assert blocks[0].hierarchy_state is KVBlockState.WARM
    assert blocks[1].hierarchy_state is KVBlockState.HOT
    assert "r1" not in sched._pending_kv_transitions


def test_scheduler_success_fails_closed_on_physical_block_mismatch():
    # current block 0 has block_id=999 instead of 10
    sched, blocks = _make_mock_scheduler_for_commit(
        block_states=[KVBlockState.HOT],
        block_ids=[999],
    )
    t = make_transition(
        transition_id=1,
        request_id="r1",
        changed_blocks=([KVCacheBlockTransition(0, 10)],),
    )
    sched._pending_kv_transitions = {"r1": t}
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    res = t.to_result(KVCacheTransitionStatus.SUCCESS)
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res]
    )

    with pytest.raises(
        ValueError,
        match="Physical block ID mismatch during commit for request r1",
    ):
        Scheduler._validate_kv_cache_transition_results(
            sched, sched_out, mr_out
        )


def test_scheduler_resume_while_pending_is_deferred_and_proceeds_after_ack():
    from tests.v1.streaming_input.test_scheduler_streaming import (
        DummyRequest,
        create_scheduler,
    )

    scheduler = create_scheduler(hot_threshold=10.0, cold_threshold=20.0)
    session = DummyRequest(
        request_id="session",
        prompt_token_ids=list(range(20)),
        arrival_time=100.0,
    )
    scheduler.add_request(session)
    scheduler.kv_cache_manager.allocate_slots(session, 20)
    session.num_computed_tokens = 20
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ

    # Step 1: Planning creates pending transition
    transitions = scheduler._classify_idle_kv_sessions(current_time=110.0)
    assert len(transitions) == 1
    assert "session" in scheduler._pending_kv_transitions

    # Step 2: Incoming prompt chunk arrives while transition is pending
    # Must NOT raise RuntimeError; must queue into streaming_queue
    next_chunk = DummyRequest(
        request_id="session",
        prompt_token_ids=list(range(20, 30)),
        arrival_time=111.0,
    )
    scheduler.add_request(next_chunk)
    assert len(session.streaming_queue) == 1
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ

    # Step 3: Worker ACK arrives
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = transitions
    mr_out = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        kv_cache_transition_results=[
            transitions[0].to_result(KVCacheTransitionStatus.SUCCESS)
        ],
    )
    scheduler._validate_kv_cache_transition_results(sched_out, mr_out)

    # Transition committed, pending cleared, queued update processed
    assert "session" not in scheduler._pending_kv_transitions
    assert len(session.streaming_queue) == 0
    assert session.status == RequestStatus.WAITING
    assert session.kv_cache_state is KVBlockState.HOT
    assert session.num_computed_tokens == 20
    block_groups = scheduler.kv_cache_manager.get_blocks(
        session.request_id
    ).blocks
    assert block_groups[0][0].is_null is True
    assert block_groups[0][1].hierarchy_state is KVBlockState.HOT


def test_scheduler_abort_while_pending_is_deferred_and_propagates_worker_cleanup():
    from tests.v1.streaming_input.test_scheduler_streaming import (
        DummyRequest,
        create_scheduler,
    )

    scheduler = create_scheduler(hot_threshold=10.0, cold_threshold=20.0)
    session = DummyRequest(
        request_id="session",
        prompt_token_ids=list(range(20)),
        arrival_time=100.0,
    )
    scheduler.add_request(session)
    scheduler.kv_cache_manager.allocate_slots(session, 20)
    session.num_computed_tokens = 20
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ

    # Step 1: Planning creates pending transition
    transitions = scheduler._classify_idle_kv_sessions(current_time=110.0)
    assert len(transitions) == 1
    assert "session" in scheduler._pending_kv_transitions

    # Step 2: Client aborts while pending
    # Must NOT raise RuntimeError; must record deferred abort
    aborted = scheduler.finish_requests(
        ["session"], RequestStatus.FINISHED_ABORTED
    )
    assert aborted == [("session", 0)]
    assert scheduler._pending_finish_requests.get("session") == (
        RequestStatus.FINISHED_ABORTED
    )
    # Blocks must NOT be freed yet while migration is pending
    assert "session" in scheduler.requests
    assert "session" not in scheduler.finished_req_ids

    # Step 3: Worker ACK arrives
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = transitions
    mr_out = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        kv_cache_transition_results=[
            transitions[0].to_result(KVCacheTransitionStatus.SUCCESS)
        ],
    )
    scheduler._validate_kv_cache_transition_results(sched_out, mr_out)

    # Pending cleared, deferred abort executed, blocks freed
    assert "session" not in scheduler._pending_kv_transitions
    assert "session" not in scheduler._pending_finish_requests
    assert "session" not in scheduler.requests
    assert "session" in scheduler.finished_req_ids

    # Step 4: Next schedule() produces finished_req_ids for worker cleanup
    migration_manager = SimpleNamespace(
        release_request=MagicMock(),
    )
    runner = SimpleNamespace(
        hkv_warm_migration_manager=migration_manager,
        _remove_request=MagicMock(),
    )
    next_sched_out = scheduler.schedule()
    assert "session" in next_sched_out.finished_req_ids

    GPUModelRunner.finish_requests(runner, next_sched_out)
    migration_manager.release_request.assert_called_once_with("session")
    runner._remove_request.assert_called_once_with("session")


def test_scheduler_queued_finish_sentinel_cleans_up_after_ack():
    from tests.v1.streaming_input.test_scheduler_streaming import (
        DummyRequest,
        create_scheduler,
    )

    scheduler = create_scheduler(hot_threshold=10.0, cold_threshold=20.0)
    session = DummyRequest(
        request_id="session",
        prompt_token_ids=list(range(20)),
        arrival_time=100.0,
    )
    scheduler.add_request(session)
    scheduler.kv_cache_manager.allocate_slots(session, 20)
    session.num_computed_tokens = 20
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ

    # Step 1: Planning creates pending transition
    transitions = scheduler._classify_idle_kv_sessions(current_time=110.0)
    assert len(transitions) == 1
    assert "session" in scheduler._pending_kv_transitions

    # Step 2: Streaming finished sentinel (None update) arrives via add_request
    finish_sentinel = DummyRequest(
        request_id="session",
        resumable=False,
        arrival_time=111.0,
    )
    scheduler.add_request(finish_sentinel)
    assert len(session.streaming_queue) == 1
    assert session.streaming_queue[0] is None

    # Step 3: Worker ACK arrives
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = transitions
    mr_out = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        kv_cache_transition_results=[
            transitions[0].to_result(KVCacheTransitionStatus.SUCCESS)
        ],
    )
    scheduler._validate_kv_cache_transition_results(sched_out, mr_out)

    # Sentinel popped and finished_requests executed
    assert "session" not in scheduler._pending_kv_transitions
    assert "session" not in scheduler.requests
    assert "session" in scheduler.finished_req_ids


def test_scheduler_lifecycle_guards_while_pending():
    t = make_transition(transition_id=1, request_id="r1")
    sched = Scheduler.__new__(Scheduler)
    sched._pending_kv_transitions = {"r1": t}
    sched._pending_finish_requests = {}
    sched.requests = {
        "r1": SimpleNamespace(
            request_id="r1",
            status=RequestStatus.WAITING_FOR_STREAMING_REQ,
            is_finished=lambda: False,
        )
    }

    # Direct resume call rejected while pending
    session = SimpleNamespace(request_id="r1")
    with pytest.raises(
        RuntimeError, match="Cannot resume request r1 while a KV cache"
    ):
        sched._update_request_as_session(session, SimpleNamespace())

    # Direct free rejected while pending
    req = SimpleNamespace(
        request_id="r1",
        is_finished=lambda: True,
    )
    with pytest.raises(
        RuntimeError, match="Cannot free request r1 while a KV cache"
    ):
        sched._free_request(req)

    # Preempt rejected while pending
    running_req = SimpleNamespace(
        request_id="r1",
        status=RequestStatus.RUNNING,
    )
    with pytest.raises(
        RuntimeError, match="Cannot preempt request r1 while a KV cache"
    ):
        sched._preempt_request(running_req, 100.0)


def test_scheduler_fresh_retry_receives_new_transition_id():
    from tests.v1.streaming_input.test_scheduler_streaming import (
        DummyRequest,
        create_scheduler,
    )

    scheduler = create_scheduler(hot_threshold=10.0, cold_threshold=20.0)
    session = DummyRequest(
        request_id="session",
        prompt_token_ids=list(range(20)),
        arrival_time=100.0,
    )
    scheduler.add_request(session)
    scheduler.kv_cache_manager.allocate_slots(session, 20)
    session.num_computed_tokens = 20
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ

    # Step 1: Initial planning
    transitions1 = scheduler._classify_idle_kv_sessions(current_time=110.0)
    assert len(transitions1) == 1
    assert transitions1[0].transition_id == 0
    assert scheduler._pending_kv_transitions[session.request_id] == transitions1[0]
    assert session.kv_cache_state is KVBlockState.HOT

    # Duplicate planning is suppressed
    assert scheduler._classify_idle_kv_sessions(current_time=110.0) == []

    # Worker reports RETRYABLE_CAPACITY
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = transitions1
    mr_out = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        kv_cache_transition_results=[
            transitions1[0].to_result(
                KVCacheTransitionStatus.RETRYABLE_CAPACITY, "capacity full"
            )
        ],
    )
    scheduler._validate_kv_cache_transition_results(sched_out, mr_out)

    assert session.kv_cache_state is KVBlockState.HOT
    assert session.request_id not in scheduler._pending_kv_transitions

    # Step 2: Fresh classification after cleared pending receives new transition_id=1
    transitions2 = scheduler._classify_idle_kv_sessions(current_time=110.0)
    assert len(transitions2) == 1
    assert transitions2[0].transition_id == 1
    assert scheduler._pending_kv_transitions[session.request_id] == transitions2[0]
