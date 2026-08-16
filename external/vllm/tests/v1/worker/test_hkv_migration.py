# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.kv_cache_state import KVCacheBlockTransition
from vllm.v1.worker.gpu import attn_utils
from vllm.v1.worker.gpu.hkv_migration import (
    HKVWarmCapacityError,
    HKVWarmMigrationManager,
    HKVWarmSlotAllocator,
    HKVWarmStaleValidationError,
)


def key(request_id: str, group: int, logical_block: int) -> tuple[str, int, int]:
    return request_id, group, logical_block


def transition(logical_block: int, hot_block: int) -> KVCacheBlockTransition:
    return KVCacheBlockTransition(logical_block, hot_block)


def test_allocator_determinism_identity_and_reuse():
    allocator = HKVWarmSlotAllocator(4)
    first = key("request-a", 0, 9)
    same_index_other_group = key("request-a", 1, 9)
    third = key("request-a", 0, 2)

    initial = allocator.reserve_many((first, same_index_other_group, first, third))
    assert initial.mappings == (
        (first, 0),
        (same_index_other_group, 1),
        (third, 2),
    )
    assert initial.newly_allocated == (first, same_index_other_group, third)
    allocator.commit(initial)

    fourth = key("request-b", 0, 31)
    mixed = allocator.reserve_many((same_index_other_group, fourth))
    assert mixed.mappings == ((same_index_other_group, 1), (fourth, 3))
    assert mixed.existing == (same_index_other_group,)
    allocator.commit(mixed)

    assert allocator.release_key(same_index_other_group) == 1
    replacement = key("request-b", 2, 4)
    reused = allocator.reserve_many((replacement,))
    assert reused.mappings == ((replacement, 1),)
    allocator.commit(reused)
    allocator.validate_invariants()


def test_allocator_failures_are_atomic_and_preserve_existing_mappings():
    allocator = HKVWarmSlotAllocator(2)
    existing = key("request-a", 0, 1)
    initial = allocator.reserve_many((existing,))
    allocator.commit(initial)

    with pytest.raises(HKVWarmCapacityError, match="only 1 available"):
        allocator.reserve_many(
            (existing, key("request-a", 0, 2), key("request-a", 0, 3)),
        )
    assert allocator.lookup(existing) == 0
    assert allocator.lookup(key("request-a", 0, 2)) is None
    assert allocator.num_owned_slots == 1

    candidate = key("request-a", 0, 4)
    reservation = allocator.reserve_many((existing, candidate))
    allocator.rollback(reservation)
    assert allocator.lookup(existing) == 0
    assert allocator.lookup(candidate) is None

    invalid_keys = (
        (("request-a", 0), TypeError),
        ((1, 0, 0), TypeError),
        (("request-a", True, 0), TypeError),
        (("request-a", -1, 0), ValueError),
        (("request-a", 0, True), TypeError),
        (("request-a", 0, -1), ValueError),
    )
    for invalid_key, error in invalid_keys:
        with pytest.raises(error):
            allocator.reserve_many((invalid_key,))  # type: ignore[arg-type]
    assert allocator.lookup(existing) == 0
    allocator.validate_invariants()


def test_logical_residency_is_idempotent_and_released(monkeypatch):
    hot_to_warm_map = torch.full((16,), -1, dtype=torch.int32)
    hot_kv_caches = {"layer": object()}
    manager = HKVWarmMigrationManager(
        warm_capacity=3,
        hot_kv_caches=hot_kv_caches,
        warm_kv_caches={},
        hot_to_warm_maps={"layer": hot_to_warm_map, "alias": hot_to_warm_map},
        device="cpu",
    )

    quantized = []

    def quantize(**kwargs):
        quantized.append((kwargs["hot_block_ids"], kwargs["warm_slot_ids"]))
        hot_to_warm_map[list(kwargs["hot_block_ids"])] = torch.tensor(
            kwargs["warm_slot_ids"], dtype=torch.int32
        )

    monkeypatch.setattr(attn_utils, "quantize_hkv_blocks_to_warm", quantize)
    changed = ([transition(0, 3), transition(1, 7)],)
    manager.migrate("request-a", changed, ((3, 7),))
    assert manager.warm_residency_revision == 1
    manager.migrate("request-a", changed, ((3, 7),))
    assert manager.warm_residency_revision == 1
    manager.migrate("request-b", ([transition(0, 11)],), ((11,),))
    assert manager.warm_residency_revision == 2

    assert quantized == [((3, 7), (0, 1)), ((11,), (2,))]
    assert manager.warm_residency[("request-a", 0, 0)].warm_slot_id == 0
    assert (
        manager.warm_residency[("request-a", 0, 1)]
        .temporary_shadow_hot_block_id
        == 7
    )
    assert manager.hot_kv_caches is hot_kv_caches

    assert manager.release_request("request-a") == (0, 1)
    assert manager.warm_residency_revision == 3
    assert hot_to_warm_map[3].item() == -1
    assert hot_to_warm_map[7].item() == -1
    assert hot_to_warm_map[11].item() == 2
    assert not any(key[0] == "request-a" for key in manager.warm_residency)
    assert manager.allocator.lookup(key("request-b", 0, 0)) == 2

    assert manager.release_request("request-a") == ()
    assert manager.warm_residency_revision == 3
    manager.migrate(
        "request-c",
        ([transition(0, 3), transition(1, 13)],),
        ((3, 13),),
    )
    assert manager.warm_residency_revision == 4
    assert manager.warm_residency[("request-c", 0, 0)].warm_slot_id == 0
    assert manager.warm_residency[("request-c", 0, 1)].warm_slot_id == 1
    manager.allocator.validate_invariants()


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("stale_source", HKVWarmStaleValidationError, "does not match"),
        ("conflicting_duplicate", ValueError, "conflicting HOT sources"),
        ("capacity", HKVWarmCapacityError, "only 1 available"),
        ("quantization", RuntimeError, "quantization failed"),
    ],
)
def test_population_failures_preserve_existing_residency(
    monkeypatch, case, error, message
):
    hot_to_warm_map = torch.full((16,), -1, dtype=torch.int32)
    manager = HKVWarmMigrationManager(
        warm_capacity=2,
        hot_kv_caches={},
        warm_kv_caches={},
        hot_to_warm_maps={"layer": hot_to_warm_map},
        device="cpu",
    )
    fail_quantization = False

    def quantize(**kwargs):
        hot_to_warm_map[list(kwargs["hot_block_ids"])] = torch.tensor(
            kwargs["warm_slot_ids"], dtype=torch.int32
        )
        if fail_quantization:
            raise RuntimeError("quantization failed")

    monkeypatch.setattr(attn_utils, "quantize_hkv_blocks_to_warm", quantize)
    manager.migrate("request-a", ([transition(0, 3)],), ((3,),))
    existing_revision = manager.warm_residency_revision
    existing_residency = manager.warm_residency.copy()
    existing_projection = hot_to_warm_map.clone()

    if case == "stale_source":
        changed, block_table = ([transition(0, 7)],), ((8,),)
    elif case == "conflicting_duplicate":
        changed = ([transition(0, 7), transition(0, 8)],)
        block_table = ((7,),)
    elif case == "capacity":
        changed = ([transition(0, 7), transition(1, 8)],)
        block_table = ((7, 8),)
    else:
        changed, block_table = ([transition(0, 7)],), ((7,),)
        fail_quantization = True

    with pytest.raises(error, match=message):
        manager.migrate("request-b", changed, block_table)

    assert manager.warm_residency == existing_residency
    assert manager.warm_residency_revision == existing_revision
    assert torch.equal(hot_to_warm_map, existing_projection)
    assert manager.allocator.num_owned_slots == 1
    assert manager.allocator.lookup(key("request-a", 0, 0)) == 0
    assert manager.allocator.lookup(key("request-b", 0, 0)) is None
    manager.allocator.validate_invariants()
