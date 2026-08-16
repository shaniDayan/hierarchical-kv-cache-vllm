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
    Scheduler._validate_kv_cache_transition_results(sched, sched_out, out)


def test_scheduler_validates_transition_results_exact_match():
    sched = Scheduler.__new__(Scheduler)
    t = make_transition(transition_id=1, request_id="r1")
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    res = t.to_result(KVCacheTransitionStatus.SUCCESS)
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res]
    )

    # Should not raise
    Scheduler._validate_kv_cache_transition_results(sched, sched_out, mr_out)


def test_scheduler_rejects_missing_result():
    sched = Scheduler.__new__(Scheduler)
    t = make_transition(transition_id=1, request_id="r1")
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


def _make_mock_scheduler_for_rollback(
    request_id: str = "r1",
    request_state: KVBlockState = KVBlockState.WARM,
    block_states: list[KVBlockState] | None = None,
    block_ids: list[int] | None = None,
) -> tuple[Scheduler, list[SimpleNamespace]]:
    if block_states is None:
        block_states = [KVBlockState.WARM, KVBlockState.WARM]
    if block_ids is None:
        block_ids = [10, 20]
    blocks = [
        SimpleNamespace(block_id=bid, hierarchy_state=bstate)
        for bid, bstate in zip(block_ids, block_states, strict=True)
    ]
    request = SimpleNamespace(
        request_id=request_id, kv_cache_state=request_state
    )
    kv_cache_manager = SimpleNamespace(
        get_blocks=lambda _req: SimpleNamespace(blocks=(blocks,))
    )
    sched = Scheduler.__new__(Scheduler)
    sched.requests = {request_id: request}
    sched.kv_cache_manager = kv_cache_manager
    return sched, blocks


def test_scheduler_rollback_retryable_capacity_rolls_back_warm_to_hot():
    sched, blocks = _make_mock_scheduler_for_rollback(
        block_states=[KVBlockState.WARM, KVBlockState.WARM, KVBlockState.HOT],
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


def test_scheduler_rollback_stale_validation_rolls_back_matching_identity():
    sched, blocks = _make_mock_scheduler_for_rollback(
        block_states=[KVBlockState.WARM],
        block_ids=[10],
    )
    t = make_transition(
        transition_id=1,
        request_id="r1",
        changed_blocks=([KVCacheBlockTransition(0, 10)],),
    )
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


def test_scheduler_success_preserves_warm_state():
    sched, blocks = _make_mock_scheduler_for_rollback(
        block_states=[KVBlockState.WARM, KVBlockState.WARM],
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


def test_scheduler_rollback_does_not_modify_untransitioned_blocks():
    # block 0 was transitioned; block 1 was untransitioned and remained in custom state
    sched, blocks = _make_mock_scheduler_for_rollback(
        block_states=[KVBlockState.WARM, KVBlockState.WARM],
        block_ids=[10, 20],
    )
    t = make_transition(
        transition_id=1,
        request_id="r1",
        changed_blocks=([KVCacheBlockTransition(0, 10)],),
    )
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    res = t.to_result(
        KVCacheTransitionStatus.RETRYABLE_CAPACITY, "no warm slots"
    )
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res]
    )

    Scheduler._validate_kv_cache_transition_results(sched, sched_out, mr_out)

    # Block 0 rolled back to HOT, block 1 untouched
    assert blocks[0].hierarchy_state is KVBlockState.HOT
    assert blocks[1].hierarchy_state is KVBlockState.WARM


def test_scheduler_rollback_fails_closed_on_physical_block_mismatch():
    # current block 0 has block_id=999 instead of 10
    sched, blocks = _make_mock_scheduler_for_rollback(
        block_states=[KVBlockState.WARM],
        block_ids=[999],
    )
    t = make_transition(
        transition_id=1,
        request_id="r1",
        changed_blocks=([KVCacheBlockTransition(0, 10)],),
    )
    sched_out = SchedulerOutput.make_empty()
    sched_out.kv_cache_state_transitions = [t]

    res = t.to_result(
        KVCacheTransitionStatus.STALE_VALIDATION, "stale block mismatch"
    )
    mr_out = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, kv_cache_transition_results=[res]
    )

    with pytest.raises(
        ValueError,
        match="Physical block ID mismatch during rollback for request r1",
    ):
        Scheduler._validate_kv_cache_transition_results(
            sched, sched_out, mr_out
        )
