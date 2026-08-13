# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import pickle

import pytest

from vllm.v1.kv_cache_state import (
    KVBlockState,
    KVCacheStateTransition,
    classify_request_kv_state,
)

HOT_THRESHOLD = 10.0
COLD_THRESHOLD = 20.0


def test_kv_cache_state_transition_equality_and_pickle_round_trip():
    transition = KVCacheStateTransition(
        request_id="request",
        previous_state=KVBlockState.HOT,
        new_state=KVBlockState.WARM,
        changed_block_ids=([1, 2], [], [3]),
    )

    equivalent = KVCacheStateTransition(
        request_id="request",
        previous_state=KVBlockState.HOT,
        new_state=KVBlockState.WARM,
        changed_block_ids=([1, 2], [], [3]),
    )
    restored = pickle.loads(pickle.dumps(transition))

    assert transition == equivalent
    assert restored == transition
    assert restored.changed_block_ids == ([1, 2], [], [3])


def test_classify_request_kv_state_hot_below_boundary():
    state = classify_request_kv_state(
        idle_time=HOT_THRESHOLD - 1.0,
        hot_threshold=HOT_THRESHOLD,
        cold_threshold=COLD_THRESHOLD,
    )

    assert state is KVBlockState.HOT


def test_classify_request_kv_state_warm_at_hot_boundary():
    state = classify_request_kv_state(
        idle_time=HOT_THRESHOLD,
        hot_threshold=HOT_THRESHOLD,
        cold_threshold=COLD_THRESHOLD,
    )

    assert state is KVBlockState.WARM


def test_classify_request_kv_state_warm_below_cold_boundary():
    state = classify_request_kv_state(
        idle_time=math.nextafter(COLD_THRESHOLD, -math.inf),
        hot_threshold=HOT_THRESHOLD,
        cold_threshold=COLD_THRESHOLD,
    )

    assert state is KVBlockState.WARM


def test_classify_request_kv_state_cold_at_boundary():
    state = classify_request_kv_state(
        idle_time=COLD_THRESHOLD,
        hot_threshold=HOT_THRESHOLD,
        cold_threshold=COLD_THRESHOLD,
    )

    assert state is KVBlockState.COLD


@pytest.mark.parametrize(
    ("idle_time", "hot_threshold", "cold_threshold"),
    [
        (-1.0, HOT_THRESHOLD, COLD_THRESHOLD),
        (0.0, -1.0, COLD_THRESHOLD),
        (0.0, HOT_THRESHOLD, HOT_THRESHOLD),
        (0.0, HOT_THRESHOLD, HOT_THRESHOLD - 1.0),
    ],
)
def test_classify_request_kv_state_rejects_invalid_inputs(
    idle_time: float,
    hot_threshold: float,
    cold_threshold: float,
):
    with pytest.raises(ValueError):
        classify_request_kv_state(idle_time, hot_threshold, cold_threshold)
