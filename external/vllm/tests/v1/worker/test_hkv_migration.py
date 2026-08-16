# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.hkv_migration import (
    HKVBlockSource,
    HKVWarmCapacityError,
    HKVWarmMigrationManager,
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


def test_release_request_invalidates_owned_maps_and_reuses_slots():
    hot_to_warm_map = torch.full((16,), -1, dtype=torch.int32)
    manager = HKVWarmMigrationManager(
        warm_capacity=3,
        hot_kv_caches={},
        warm_kv_caches={},
        hot_to_warm_maps={"layer": hot_to_warm_map, "alias": hot_to_warm_map},
        device="cpu",
    )
    request_sources = (source(0, 3), source(0, 7))
    request_reservation = manager.allocator.reserve_many(
        request_sources, owner_token="request-a"
    )
    manager.allocator.commit(request_reservation)
    other_source = source(0, 11)
    other_reservation = manager.allocator.reserve_many(
        (other_source,), owner_token="request-b"
    )
    manager.allocator.commit(other_reservation)
    published_mappings = request_reservation.mappings + other_reservation.mappings
    for block_source, slot in published_mappings:
        hot_to_warm_map[block_source.kernel_hot_block_id] = slot

    assert manager.release_request("request-a") == (0, 1)
    assert hot_to_warm_map[3].item() == -1
    assert hot_to_warm_map[7].item() == -1
    assert hot_to_warm_map[11].item() == 2
    assert all(
        manager.allocator.lookup(source_) is None for source_ in request_sources
    )
    assert manager.allocator.owner_token_of(other_source) == "request-b"

    assert manager.release_request("request-a") == ()
    replacement_sources = (source(0, 5), source(0, 13))
    replacement = manager.allocator.reserve_many(
        replacement_sources, owner_token="request-c"
    )
    assert replacement.mappings == (
        (replacement_sources[0], 0),
        (replacement_sources[1], 1),
    )
    manager.allocator.commit(replacement)
    manager.allocator.validate_invariants()
