# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum


class KVBlockState(str, Enum):
    """Hierarchical state of KV-cache data."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class KVCacheStateTransition:
    """A request KV-state transition and its affected private blocks.

    ``changed_block_ids`` preserves cache-group structure and contains only
    private eligible blocks whose metadata changed to ``new_state``.
    """

    request_id: str
    previous_state: KVBlockState
    new_state: KVBlockState
    changed_block_ids: tuple[list[int], ...]


def classify_request_kv_state(
    idle_time: float,
    hot_threshold: float,
    cold_threshold: float,
) -> KVBlockState:
    """Classify request KV state from its idle time."""
    if idle_time < 0:
        raise ValueError("idle_time must be non-negative")
    if hot_threshold < 0:
        raise ValueError("hot_threshold must be non-negative")
    if cold_threshold <= hot_threshold:
        raise ValueError("cold_threshold must be greater than hot_threshold")

    if idle_time < hot_threshold:
        return KVBlockState.HOT
    if idle_time < cold_threshold:
        return KVBlockState.WARM
    return KVBlockState.COLD
