# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_state import (
    KVBlockState,
    KVCacheBlockTransition,
    KVCacheStateTransition,
)
from vllm.v1.worker.gpu.hkv_migration import (
    HKVWarmCapacityError,
    HKVWarmMigrationManager,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker


def make_transition(
    previous_state: KVBlockState = KVBlockState.HOT,
    new_state: KVBlockState = KVBlockState.WARM,
    changed_blocks: tuple[list[KVCacheBlockTransition], ...] | None = None,
) -> KVCacheStateTransition:
    if changed_blocks is None:
        changed_blocks = ([KVCacheBlockTransition(0, 1)],)
    return KVCacheStateTransition(
        request_id="request",
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
        target = SimpleNamespace(
            hkv_warm_migration_manager=SimpleNamespace(migrate=MagicMock()),
            block_tables=SimpleNamespace(blocks_per_kv_block=[1]),
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


def test_shadow_migration_receives_source_hot_block_ids():
    migration_manager = SimpleNamespace(migrate=MagicMock())
    target = SimpleNamespace(
        hkv_warm_migration_manager=migration_manager,
        block_tables=SimpleNamespace(blocks_per_kv_block=[1]),
    )
    transition = make_transition(
        changed_blocks=(
            [
                KVCacheBlockTransition(2, 7),
                KVCacheBlockTransition(5, 11),
            ],
        )
    )

    GPUModelRunner.handle_kv_cache_state_transitions(target, [transition])

    migration_manager.migrate.assert_called_once_with("request", [7, 11])


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
        manager.migrate("request", [2, 7])

    assert hot_to_warm_maps == {}
    assert manager.allocator.num_owned_slots == 0


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
