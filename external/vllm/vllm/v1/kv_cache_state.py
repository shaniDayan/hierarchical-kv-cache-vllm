# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class KVBlockState(str, Enum):
    """Hierarchical state of KV-cache data."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class KVCacheTransitionStatus(str, Enum):
    """Execution status of a KV-cache state transition on the worker."""

    SUCCESS = "success"
    RETRYABLE_CAPACITY = "retryable_capacity"
    STALE_VALIDATION = "stale_validation"
    FATAL = "fatal"


@dataclass(frozen=True, slots=True)
class KVCacheBlockTransition:
    """One changed request block and its current physical HOT source."""

    logical_block_index: int
    source_hot_block_id: int


def normalize_transition_blocks(
    changed_blocks: tuple[Sequence[KVCacheBlockTransition], ...],
) -> tuple[tuple[int, int, int], ...]:
    """Normalize hierarchical changed blocks to flat (group, logical, hot_id)."""
    return tuple(
        (group_idx, block.logical_block_index, block.source_hot_block_id)
        for group_idx, group in enumerate(changed_blocks)
        for block in group
    )


KVCacheTransitionSignature = tuple[
    int,
    str,
    KVBlockState,
    KVBlockState,
    tuple[tuple[int, int, int], ...],
]


@dataclass(frozen=True, slots=True)
class KVCacheTransitionResult:
    """Worker completion result for one request KV-cache state transition."""

    transition_id: int
    request_id: str
    previous_state: KVBlockState
    new_state: KVBlockState
    changed_blocks: tuple[tuple[int, int, int], ...]
    status: KVCacheTransitionStatus
    error_message: str | None = None

    @property
    def signature(self) -> KVCacheTransitionSignature:
        return (
            self.transition_id,
            self.request_id,
            self.previous_state,
            self.new_state,
            self.changed_blocks,
        )


@dataclass
class KVCacheStateTransition:
    """A request KV-state transition and its affected private blocks.

    ``changed_blocks`` preserves cache-group structure. Each entry identifies
    both the block's logical position in the request and the physical HOT block
    containing its current KV data.
    """

    transition_id: int
    request_id: str
    previous_state: KVBlockState
    new_state: KVBlockState
    changed_blocks: tuple[list[KVCacheBlockTransition], ...]

    @property
    def signature(self) -> KVCacheTransitionSignature:
        return (
            self.transition_id,
            self.request_id,
            self.previous_state,
            self.new_state,
            normalize_transition_blocks(self.changed_blocks),
        )

    def to_result(
        self,
        status: KVCacheTransitionStatus,
        error_message: str | None = None,
    ) -> KVCacheTransitionResult:
        return KVCacheTransitionResult(
            transition_id=self.transition_id,
            request_id=self.request_id,
            previous_state=self.previous_state,
            new_state=self.new_state,
            changed_blocks=normalize_transition_blocks(self.changed_blocks),
            status=status,
            error_message=error_message,
        )


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
