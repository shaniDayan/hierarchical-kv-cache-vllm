# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.gpu.hkv_migration import (
    HKVBlockSource,
    HKVWarmCapacityError,
    HKVWarmOwnershipError,
    HKVWarmSlotAllocator,
)


def source(group: int, block: int) -> HKVBlockSource:
    return HKVBlockSource(group, block)


def test_allocator_determinism_identity_and_reuse():
    allocator = HKVWarmSlotAllocator(4)
    first = source(0, 9)
    same_id_other_group = source(1, 9)
    third = source(0, 2)

    initial = allocator.reserve_many(
        (first, same_id_other_group, first, third),
        owner_token="request-a",
    )
    assert initial.mappings == (
        (first, 0),
        (same_id_other_group, 1),
        (third, 2),
    )
    assert initial.newly_allocated == (first, same_id_other_group, third)
    allocator.commit(initial)

    fourth = source(0, 31)
    mixed = allocator.reserve_many(
        (same_id_other_group, fourth),
        owner_token="request-a",
    )
    assert mixed.mappings == ((same_id_other_group, 1), (fourth, 3))
    assert mixed.existing == (same_id_other_group,)
    allocator.commit(mixed)

    assert allocator.release_source(same_id_other_group) == 1
    replacement = source(2, 4)
    reused = allocator.reserve_many((replacement,), owner_token="request-b")
    assert reused.mappings == ((replacement, 1),)
    allocator.commit(reused)
    allocator.validate_invariants()


def test_allocator_failures_are_atomic_and_preserve_existing_mappings():
    allocator = HKVWarmSlotAllocator(2)
    existing = source(0, 1)
    initial = allocator.reserve_many((existing,), owner_token="request-a")
    allocator.commit(initial)

    with pytest.raises(HKVWarmCapacityError, match="only 1 available"):
        allocator.reserve_many(
            (existing, source(0, 2), source(0, 3)),
            owner_token="request-a",
        )
    assert allocator.lookup(existing) == 0
    assert allocator.lookup(source(0, 2)) is None
    assert allocator.num_owned_slots == 1

    candidate = source(0, 4)
    reservation = allocator.reserve_many(
        (existing, candidate),
        owner_token="request-a",
    )
    allocator.rollback(reservation)
    assert allocator.lookup(existing) == 0
    assert allocator.lookup(candidate) is None

    with pytest.raises(HKVWarmOwnershipError, match="request-a"):
        allocator.reserve_many((existing,), owner_token="request-b")
    assert allocator.lookup(existing) == 0
    assert allocator.owner_token_of(existing) == "request-a"
    allocator.validate_invariants()
