# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker.gpu.hkv_migration import (
    HKVBlockSource,
    HKVWarmCapacityError,
    HKVWarmOwnershipError,
    HKVWarmReservationError,
    HKVWarmSlotAllocator,
)


def source(group: int, block: int) -> HKVBlockSource:
    return HKVBlockSource(group, block)


def assert_valid(allocator: HKVWarmSlotAllocator) -> None:
    allocator.validate_invariants()


def test_allocates_lowest_slots_for_noncontiguous_hot_blocks():
    allocator = HKVWarmSlotAllocator(5)
    sources = (source(0, 9), source(0, 2), source(0, 31))

    reservation = allocator.reserve_many(sources, owner_token="request-a")

    assert reservation.mappings == tuple(zip(sources, (0, 1, 2), strict=True))
    assert reservation.existing == ()
    assert reservation.newly_allocated == sources
    assert [allocator.lookup(item) for item in sources] == [0, 1, 2]
    assert [allocator.owner_of(slot) for slot in range(3)] == list(sources)
    assert_valid(allocator)

    allocator.commit(reservation)
    assert_valid(allocator)


def test_cache_group_is_part_of_source_identity():
    allocator = HKVWarmSlotAllocator(2)
    first = source(0, 7)
    second = source(1, 7)

    reservation = allocator.reserve_many(
        (first, second), owner_token="request-a"
    )

    assert reservation.mappings == ((first, 0), (second, 1))
    allocator.commit(reservation)
    assert_valid(allocator)


def test_duplicate_sources_are_deduplicated():
    allocator = HKVWarmSlotAllocator(2)
    first = source(0, 7)
    second = source(0, 8)

    reservation = allocator.reserve_many(
        (first, first, second, first), owner_token="request-a"
    )

    assert reservation.mappings == ((first, 0), (second, 1))
    assert reservation.newly_allocated == (first, second)
    assert allocator.num_owned_slots == 2
    assert_valid(allocator)

    allocator.commit(reservation)
    assert_valid(allocator)


def test_mixed_reservation_reuses_existing_and_allocates_new_source():
    allocator = HKVWarmSlotAllocator(4)
    existing = source(0, 1)
    initial = allocator.reserve_many((existing,), owner_token="request-a")
    allocator.commit(initial)
    assert_valid(allocator)

    new_source = source(0, 9)
    reservation = allocator.reserve_many(
        (existing, new_source), owner_token="request-a"
    )

    assert reservation.mappings == ((existing, 0), (new_source, 1))
    assert reservation.existing == (existing,)
    assert reservation.newly_allocated == (new_source,)
    assert_valid(allocator)

    allocator.commit(reservation)
    assert allocator.lookup(existing) == 0
    assert allocator.lookup(new_source) == 1
    assert_valid(allocator)


def test_insufficient_capacity_is_atomic():
    allocator = HKVWarmSlotAllocator(2)
    existing = source(0, 1)
    initial = allocator.reserve_many((existing,), owner_token="request-a")
    allocator.commit(initial)
    assert_valid(allocator)

    with pytest.raises(HKVWarmCapacityError, match="only 1 available"):
        allocator.reserve_many(
            (existing, source(0, 2), source(0, 3)),
            owner_token="request-a",
        )

    assert allocator.lookup(existing) == 0
    assert allocator.lookup(source(0, 2)) is None
    assert allocator.lookup(source(0, 3)) is None
    assert allocator.num_owned_slots == 1
    assert allocator.num_free_slots == 1
    assert_valid(allocator)


def test_rollback_releases_only_newly_reserved_slots():
    allocator = HKVWarmSlotAllocator(3)
    existing = source(0, 1)
    initial = allocator.reserve_many((existing,), owner_token="request-a")
    allocator.commit(initial)

    new_source = source(0, 2)
    reservation = allocator.reserve_many(
        (existing, new_source), owner_token="request-a"
    )
    assert allocator.lookup(new_source) == 1
    assert_valid(allocator)

    allocator.rollback(reservation)

    assert allocator.lookup(existing) == 0
    assert allocator.lookup(new_source) is None
    assert allocator.num_owned_slots == 1
    assert allocator.num_free_slots == 2
    assert_valid(allocator)


def test_commit_preserves_new_mappings():
    allocator = HKVWarmSlotAllocator(2)
    first = source(0, 3)
    second = source(0, 4)
    reservation = allocator.reserve_many(
        (first, second), owner_token="request-a"
    )

    allocator.commit(reservation)

    assert allocator.lookup(first) == 0
    assert allocator.lookup(second) == 1
    assert allocator.num_free_slots == 0
    assert_valid(allocator)


def test_release_is_idempotent_and_reuses_lowest_slot():
    allocator = HKVWarmSlotAllocator(3)
    first = source(0, 1)
    second = source(0, 2)
    initial = allocator.reserve_many((first, second), owner_token="request-a")
    allocator.commit(initial)

    assert allocator.release_source(first) == 0
    assert allocator.release_source(first) is None
    assert allocator.release_source(source(0, 99)) is None
    assert_valid(allocator)

    replacement = source(0, 8)
    reservation = allocator.reserve_many(
        (replacement,), owner_token="request-b"
    )

    assert reservation.mappings == ((replacement, 0),)
    allocator.commit(reservation)
    assert_valid(allocator)


def test_release_sources_deduplicates_and_preserves_reverse_ownership():
    allocator = HKVWarmSlotAllocator(3)
    first = source(0, 1)
    second = source(0, 2)
    initial = allocator.reserve_many((first, second), owner_token="request-a")
    allocator.commit(initial)

    released = allocator.release_sources((second, first, second))

    assert released == (1, 0)
    assert allocator.owner_of(0) is None
    assert allocator.owner_of(1) is None
    assert allocator.num_free_slots == 3
    assert_valid(allocator)


def test_clear_and_reset_restore_initial_state():
    allocator = HKVWarmSlotAllocator(3)
    reservation = allocator.reserve_many(
        (source(0, 1), source(0, 2)), owner_token="request-a"
    )
    assert_valid(allocator)

    allocator.clear()

    assert allocator.num_owned_slots == 0
    assert allocator.num_free_slots == 3
    assert allocator.lookup(source(0, 1)) is None
    assert_valid(allocator)
    with pytest.raises(HKVWarmReservationError, match="not active"):
        allocator.commit(reservation)

    replacement = allocator.reserve_many((source(0, 8),), owner_token="request-b")
    allocator.commit(replacement)
    allocator.reset()
    assert allocator.num_owned_slots == 0
    assert allocator.num_free_slots == 3
    assert_valid(allocator)


@pytest.mark.parametrize("capacity", [0, -1])
def test_rejects_invalid_capacity(capacity: int):
    with pytest.raises(ValueError, match="greater than zero"):
        HKVWarmSlotAllocator(capacity)


def test_rejects_non_integer_capacity_and_owner_token():
    with pytest.raises(TypeError, match="capacity must be an integer"):
        HKVWarmSlotAllocator(True)

    allocator = HKVWarmSlotAllocator(1)
    with pytest.raises(ValueError, match="must not be None"):
        allocator.reserve_many((source(0, 1),), owner_token=None)
    with pytest.raises(TypeError, match="must be hashable"):
        allocator.reserve_many((source(0, 1),), owner_token=[])
    assert_valid(allocator)


@pytest.mark.parametrize(
    "invalid_source",
    [
        (-1, 0),
        (0, -1),
    ],
)
def test_rejects_negative_source_ids(invalid_source: tuple[int, int]):
    with pytest.raises(ValueError, match="must be non-negative"):
        HKVBlockSource(*invalid_source)


def test_conflicting_logical_owner_is_rejected_until_invalidation():
    allocator = HKVWarmSlotAllocator(2)
    block = source(0, 5)
    initial = allocator.reserve_many((block,), owner_token=("request-a", 0))
    allocator.commit(initial)
    assert_valid(allocator)

    with pytest.raises(HKVWarmOwnershipError, match="request-a"):
        allocator.reserve_many((block,), owner_token=("request-b", 0))
    with pytest.raises(HKVWarmOwnershipError, match="request-a"):
        allocator.invalidate_source(block, owner_token=("request-b", 0))
    assert allocator.lookup(block) == 0
    assert allocator.owner_token_of(block) == ("request-a", 0)
    assert_valid(allocator)

    assert allocator.invalidate_source(block, owner_token=("request-a", 0)) == 0
    replacement = allocator.reserve_many(
        (block,), owner_token=("request-b", 0)
    )
    assert replacement.mappings == ((block, 0),)
    allocator.commit(replacement)
    assert allocator.owner_token_of(block) == ("request-b", 0)
    assert_valid(allocator)


def test_active_reservation_must_be_closed_before_other_lifecycle_operations():
    allocator = HKVWarmSlotAllocator(2)
    reservation = allocator.reserve_many((source(0, 1),), owner_token="request-a")
    assert_valid(allocator)

    with pytest.raises(HKVWarmReservationError, match="must be committed"):
        allocator.reserve_many((source(0, 2),), owner_token="request-a")
    with pytest.raises(HKVWarmReservationError, match="reservation is active"):
        allocator.release_source(source(0, 1))

    allocator.rollback(reservation)
    assert_valid(allocator)
