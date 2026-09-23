# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest

from vllm.v1.kv_cache_state import (
    KVBlockState,
    classify_request_kv_state,
)

HOT_THRESHOLD = 10.0
COLD_THRESHOLD = 20.0


@pytest.mark.parametrize(
    ("idle_time", "expected"),
    [
        (HOT_THRESHOLD - 1.0, KVBlockState.HOT),
        (HOT_THRESHOLD, KVBlockState.WARM),
        (math.nextafter(COLD_THRESHOLD, -math.inf), KVBlockState.WARM),
        (COLD_THRESHOLD, KVBlockState.COLD),
    ],
)
def test_classify_request_kv_state_boundaries(idle_time, expected):
    assert (
        classify_request_kv_state(idle_time, HOT_THRESHOLD, COLD_THRESHOLD)
        is expected
    )


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
