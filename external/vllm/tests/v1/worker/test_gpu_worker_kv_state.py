# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_state import KVBlockState, KVCacheStateTransition
from vllm.v1.worker.gpu_worker import Worker


def make_transition(
    previous_state: KVBlockState = KVBlockState.HOT,
    new_state: KVBlockState = KVBlockState.WARM,
    changed_block_ids: tuple[list[int], ...] | None = None,
) -> KVCacheStateTransition:
    if changed_block_ids is None:
        changed_block_ids = ([1], [], [2])
    return KVCacheStateTransition(
        request_id="request",
        previous_state=previous_state,
        new_state=new_state,
        changed_block_ids=changed_block_ids,
    )


def make_bare_worker(use_v2_model_runner: bool, events: list[str]) -> Worker:
    worker = Worker.__new__(Worker)
    worker._pp_send_work = []
    worker.profiler = None
    worker.use_v2_model_runner = use_v2_model_runner
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )

    def execute_model(scheduler_output, intermediate_tensors):
        events.append("runner")
        return None

    worker.model_runner = SimpleNamespace(
        execute_model=execute_model,
        is_pooling_model=False,
    )
    return worker


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_worker_handles_transitions_before_zero_token_runner(use_v2_model_runner):
    events = []
    worker = make_bare_worker(use_v2_model_runner, events)
    worker._handle_kv_cache_state_transitions = MagicMock(
        side_effect=lambda transitions: events.append("handler")
    )
    output = SchedulerOutput.make_empty()
    output.kv_cache_state_transitions = [make_transition()]

    Worker.execute_model(worker, output)

    assert events == ["handler", "runner"]
    worker._handle_kv_cache_state_transitions.assert_called_once_with(
        output.kv_cache_state_transitions
    )


def test_worker_skips_handler_for_empty_transition_list():
    events = []
    worker = make_bare_worker(False, events)
    worker._handle_kv_cache_state_transitions = MagicMock()

    Worker.execute_model(worker, SchedulerOutput.make_empty())

    worker._handle_kv_cache_state_transitions.assert_not_called()
    assert events == ["runner"]


def test_worker_rejects_negative_transition_block_id():
    worker = Worker.__new__(Worker)

    with pytest.raises(ValueError, match="non-negative integers"):
        worker._handle_kv_cache_state_transitions(
            [make_transition(changed_block_ids=([1], [], [-1]))]
        )


@pytest.mark.parametrize(
    ("previous_state", "new_state"),
    [
        (KVBlockState.HOT, KVBlockState.WARM),
        (KVBlockState.WARM, KVBlockState.COLD),
        (KVBlockState.HOT, KVBlockState.COLD),
    ],
)
def test_worker_accepts_supported_transition(previous_state, new_state):
    worker = Worker.__new__(Worker)

    worker._handle_kv_cache_state_transitions(
        [make_transition(previous_state, new_state, ([1], [], [2]))]
    )


def test_worker_rejects_unsupported_hotter_transition():
    worker = Worker.__new__(Worker)

    with pytest.raises(ValueError, match="Unsupported KV-cache state transition"):
        worker._handle_kv_cache_state_transitions(
            [make_transition(KVBlockState.COLD, KVBlockState.HOT)]
        )
