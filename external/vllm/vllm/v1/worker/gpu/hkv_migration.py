# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.v1.kv_cache_state import KVCacheBlockTransition


class HKVWarmAllocatorError(RuntimeError):
    """Base error for WARM-slot allocation failures."""


class HKVWarmCapacityError(HKVWarmAllocatorError):
    """Raised when a reservation cannot fit in the WARM pool."""


class HKVWarmOwnershipError(HKVWarmAllocatorError):
    """Raised when WARM-slot ownership is inconsistent."""


class HKVWarmReservationError(HKVWarmAllocatorError):
    """Raised for invalid reservation lifecycle operations."""


def is_hkv_multi_block_warm_migration_enabled() -> bool:
    return os.getenv("HKV_ENABLE_MULTI_BLOCK_WARM_MIGRATION") == "1"


HKVWarmLogicalKey = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class HKVWarmReservation:
    """One transactional WARM-slot reservation.

    Ownership is visible through the allocator immediately after reservation.
    Commit validates and closes the transaction. Rollback removes only the
    mappings listed in ``newly_allocated``.
    """

    mappings: tuple[tuple[HKVWarmLogicalKey, int], ...]
    existing: tuple[HKVWarmLogicalKey, ...]
    newly_allocated: tuple[HKVWarmLogicalKey, ...]
    _reservation_id: int = field(repr=False, compare=False)
    _allocator_token: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class HKVWarmResidency:
    warm_slot_id: int
    temporary_shadow_hot_block_id: int


class HKVWarmSlotAllocator:
    """Deterministic transactional allocator for WARM KV-cache slots."""

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")

        self.capacity = capacity
        self._allocator_token = object()
        self._next_reservation_id = 0
        self._active_reservation: HKVWarmReservation | None = None
        self._key_to_slot: dict[HKVWarmLogicalKey, int] = {}
        self._slot_to_key: dict[int, HKVWarmLogicalKey] = {}
        self._free_slots = list(range(capacity))
        self._free_slot_set = set(self._free_slots)
        heapq.heapify(self._free_slots)

    @property
    def num_owned_slots(self) -> int:
        return len(self._key_to_slot)

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def lookup(self, key: HKVWarmLogicalKey) -> int | None:
        """Return the WARM slot owned by ``key``, if any."""
        self._validate_logical_key(key)
        return self._key_to_slot.get(key)

    def owner_of(self, slot: int) -> HKVWarmLogicalKey | None:
        """Return the logical key owning ``slot``, if any."""
        self._validate_slot(slot)
        return self._slot_to_key.get(slot)

    def reserve_many(
        self,
        keys: Iterable[HKVWarmLogicalKey],
    ) -> HKVWarmReservation:
        """Reserve slots for all unique logical keys or none of them."""
        if self._active_reservation is not None:
            raise HKVWarmReservationError(
                "the active reservation must be committed or rolled back first"
            )

        keys = tuple(keys)
        for key in keys:
            self._validate_logical_key(key)
        unique_keys = tuple(dict.fromkeys(keys))
        existing: list[HKVWarmLogicalKey] = []
        new_keys: list[HKVWarmLogicalKey] = []
        for key in unique_keys:
            if key in self._key_to_slot:
                existing.append(key)
            else:
                new_keys.append(key)

        if len(new_keys) > self.num_free_slots:
            raise HKVWarmCapacityError(
                f"cannot reserve {len(new_keys)} new WARM slots with "
                f"only {self.num_free_slots} available"
            )

        for key in new_keys:
            slot = heapq.heappop(self._free_slots)
            self._free_slot_set.remove(slot)
            self._key_to_slot[key] = slot
            self._slot_to_key[slot] = key

        mappings = tuple(
            (key, self._key_to_slot[key]) for key in unique_keys
        )
        reservation = HKVWarmReservation(
            mappings=mappings,
            existing=tuple(existing),
            newly_allocated=tuple(new_keys),
            _reservation_id=self._next_reservation_id,
            _allocator_token=self._allocator_token,
        )
        self._next_reservation_id += 1
        self._active_reservation = reservation
        return reservation

    def commit(self, reservation: HKVWarmReservation) -> None:
        """Validate and close a successful reservation."""
        self._validate_active_reservation(reservation)
        self._active_reservation = None

    def rollback(self, reservation: HKVWarmReservation) -> None:
        """Remove only mappings newly created by ``reservation``."""
        self._validate_active_reservation(reservation)
        for key in reservation.newly_allocated:
            self._release_key(key)
        self._active_reservation = None

    def release_key(self, key: HKVWarmLogicalKey) -> int | None:
        """Release ``key`` and return its former slot, if present."""
        self._require_no_active_reservation()
        self._validate_logical_key(key)
        return self._release_key(key)

    def release_keys(
        self, keys: Iterable[HKVWarmLogicalKey]
    ) -> tuple[int, ...]:
        """Release multiple keys, ignoring duplicates and unknown keys."""
        self._require_no_active_reservation()
        keys = tuple(keys)
        for key in keys:
            self._validate_logical_key(key)
        released_slots: list[int] = []
        for key in dict.fromkeys(keys):
            slot = self._release_key(key)
            if slot is not None:
                released_slots.append(slot)
        return tuple(released_slots)

    def clear(self) -> None:
        """Clear ownership and restore every WARM slot to the free heap."""
        self._active_reservation = None
        self._key_to_slot.clear()
        self._slot_to_key.clear()
        self._free_slots = list(range(self.capacity))
        self._free_slot_set = set(self._free_slots)
        heapq.heapify(self._free_slots)

    reset = clear

    def validate_invariants(self) -> None:
        """Assert allocator map, ownership, and free-slot consistency."""
        free_slots = set(self._free_slots)
        owned_slots = set(self._slot_to_key)
        all_slots = set(range(self.capacity))

        assert len(self._free_slots) == len(free_slots)
        assert free_slots == self._free_slot_set
        assert free_slots.isdisjoint(owned_slots)
        assert free_slots | owned_slots == all_slots
        assert len(owned_slots) + len(free_slots) == self.capacity
        assert len(self._key_to_slot) == len(self._slot_to_key)

        for key, slot in self._key_to_slot.items():
            assert 0 <= slot < self.capacity
            assert self._slot_to_key[slot] == key
        for slot, key in self._slot_to_key.items():
            assert 0 <= slot < self.capacity
            assert self._key_to_slot[key] == slot

        if self._active_reservation is not None:
            reservation = self._active_reservation
            for key, slot in reservation.mappings:
                assert self._key_to_slot[key] == slot

    def _release_key(self, key: HKVWarmLogicalKey) -> int | None:
        slot = self._key_to_slot.get(key)
        if slot is None:
            return None
        reverse_key = self._slot_to_key.get(slot)
        if reverse_key != key:
            raise HKVWarmOwnershipError(
                f"inconsistent reverse ownership for logical key {key!r}"
            )
        if slot in self._free_slot_set:
            raise HKVWarmOwnershipError(f"WARM slot {slot} is already free")

        del self._key_to_slot[key]
        del self._slot_to_key[slot]
        heapq.heappush(self._free_slots, slot)
        self._free_slot_set.add(slot)
        return slot

    def _validate_active_reservation(
        self, reservation: HKVWarmReservation
    ) -> None:
        if reservation._allocator_token is not self._allocator_token:
            raise HKVWarmReservationError(
                "reservation belongs to a different allocator"
            )
        if self._active_reservation is not reservation:
            raise HKVWarmReservationError("reservation is not active")
        for key, slot in reservation.mappings:
            if (
                self._key_to_slot.get(key) != slot
                or self._slot_to_key.get(slot) != key
            ):
                raise HKVWarmOwnershipError(
                    f"ownership changed during reservation for logical key {key!r}"
                )

    def _require_no_active_reservation(self) -> None:
        if self._active_reservation is not None:
            raise HKVWarmReservationError(
                "cannot modify ownership while a reservation is active"
            )

    def _validate_slot(self, slot: int) -> None:
        if not isinstance(slot, int) or isinstance(slot, bool):
            raise TypeError("slot must be an integer")
        if slot < 0 or slot >= self.capacity:
            raise ValueError(f"slot must be in [0, {self.capacity})")

    @staticmethod
    def _validate_logical_key(key: HKVWarmLogicalKey) -> None:
        if not isinstance(key, tuple) or len(key) != 3:
            raise TypeError("logical key must be a three-element tuple")
        request_id, cache_group_index, logical_block_index = key
        if not isinstance(request_id, str):
            raise TypeError("logical key request_id must be a string")
        for name, value in (
            ("cache_group_index", cache_group_index),
            ("logical_block_index", logical_block_index),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"logical key {name} must be an integer")
            if value < 0:
                raise ValueError(f"logical key {name} must be non-negative")


class HKVWarmMigrationManager:
    def __init__(
        self,
        *,
        warm_capacity: int,
        hot_kv_caches: dict[str, Any],
        warm_kv_caches: dict[str, Any],
        hot_to_warm_maps: dict[str, Any],
        device: Any,
    ) -> None:
        self.allocator = HKVWarmSlotAllocator(warm_capacity)
        self.hot_kv_caches = hot_kv_caches
        self.warm_kv_caches = warm_kv_caches
        self.hot_to_warm_maps = hot_to_warm_maps
        self.device = device
        self.warm_residency: dict[tuple[str, int, int], HKVWarmResidency] = {}
        self.warm_residency_revision = 0

    def migrate(
        self,
        request_id: str,
        changed_blocks: Sequence[Sequence[KVCacheBlockTransition]],
        request_block_table: Sequence[Sequence[int]],
    ) -> None:
        """Populate logical WARM residency from validated HOT blocks."""
        if len(changed_blocks) != len(request_block_table):
            raise ValueError("transition and block-table groups must match")

        normalized: dict[HKVWarmLogicalKey, int] = {}
        for cache_group_index, group in enumerate(changed_blocks):
            current_group = request_block_table[cache_group_index]
            for block in group:
                if not isinstance(block, KVCacheBlockTransition):
                    raise TypeError("invalid KV-cache block transition")
                if (
                    not isinstance(block.logical_block_index, int)
                    or isinstance(block.logical_block_index, bool)
                    or block.logical_block_index < 0
                ):
                    raise ValueError(
                        "logical block index must be a non-negative integer"
                    )
                key = (request_id, cache_group_index, block.logical_block_index)
                source_hot_block_id = block.source_hot_block_id
                if (
                    not isinstance(source_hot_block_id, int)
                    or isinstance(source_hot_block_id, bool)
                    or source_hot_block_id < 0
                ):
                    raise ValueError(
                        "source HOT block ID must be a non-negative integer"
                    )
                previous_source = normalized.get(key)
                if (
                    previous_source is not None
                    and previous_source != source_hot_block_id
                ):
                    raise ValueError(
                        f"logical key {key!r} has conflicting HOT sources"
                    )
                normalized[key] = source_hot_block_id

                if block.logical_block_index >= len(current_group):
                    raise ValueError(
                        f"logical block index {block.logical_block_index} is "
                        f"outside current block-table group {cache_group_index}"
                    )
                current_hot_block_id = current_group[block.logical_block_index]
                if current_hot_block_id != block.source_hot_block_id:
                    raise ValueError(
                        f"source HOT block {block.source_hot_block_id} does not "
                        f"match current block-table value {current_hot_block_id} "
                        f"for {key!r}"
                    )
                existing = self.warm_residency.get(key)
                if (
                    existing is not None
                    and existing.temporary_shadow_hot_block_id
                    != block.source_hot_block_id
                ):
                    raise ValueError(
                        f"logical key {key!r} already shadows "
                        f"HOT block {existing.temporary_shadow_hot_block_id}"
                    )

        new_entries = {
            key: hot_block_id
            for key, hot_block_id in normalized.items()
            if key not in self.warm_residency
        }
        if not new_entries:
            return

        reservation = self.allocator.reserve_many(new_entries)
        mappings = dict(reservation.mappings)
        new_hot_block_ids = tuple(
            new_entries[key] for key in reservation.newly_allocated
        )
        new_warm_slot_ids = tuple(
            mappings[key] for key in reservation.newly_allocated
        )
        projection_snapshots = []
        try:
            if new_hot_block_ids:
                unique_maps = {
                    (
                        hot_to_warm_map.device,
                        hot_to_warm_map.untyped_storage().data_ptr(),
                    ): hot_to_warm_map
                    for hot_to_warm_map in self.hot_to_warm_maps.values()
                }
                projection_snapshots = [
                    (
                        hot_to_warm_map,
                        hot_to_warm_map[list(new_hot_block_ids)].clone(),
                    )
                    for hot_to_warm_map in unique_maps.values()
                ]
                from vllm.v1.worker.gpu.attn_utils import (
                    quantize_hkv_blocks_to_warm,
                )

                quantize_hkv_blocks_to_warm(
                    hot_kv_caches=self.hot_kv_caches,
                    warm_kv_caches=self.warm_kv_caches,
                    hot_to_warm_maps=self.hot_to_warm_maps,
                    hot_block_ids=new_hot_block_ids,
                    warm_slot_ids=new_warm_slot_ids,
                    device=self.device,
                )
        except Exception:
            try:
                for hot_to_warm_map, previous in projection_snapshots:
                    hot_to_warm_map[list(new_hot_block_ids)] = previous
            finally:
                self.allocator.rollback(reservation)
            raise
        self.allocator.commit(reservation)
        for key, hot_block_id in new_entries.items():
            self.warm_residency[key] = HKVWarmResidency(
                warm_slot_id=mappings[key],
                temporary_shadow_hot_block_id=hot_block_id,
            )
        self.warm_residency_revision += 1

    def release_request(self, request_id: str) -> tuple[int, ...]:
        """Invalidate and release all WARM mappings owned by a request."""
        request_entries = [
            (key, entry)
            for key, entry in self.warm_residency.items()
            if key[0] == request_id
        ]
        if not request_entries:
            return ()
        if any(key[1] != 0 for key, _ in request_entries):
            raise ValueError("WARM migration cleanup supports only cache group 0")

        hot_block_ids = list(
            dict.fromkeys(
                entry.temporary_shadow_hot_block_id
                for _, entry in request_entries
            )
        )
        unique_maps = {}
        for hot_to_warm_map in self.hot_to_warm_maps.values():
            storage_key = (
                hot_to_warm_map.device,
                hot_to_warm_map.untyped_storage().data_ptr(),
            )
            unique_maps[storage_key] = hot_to_warm_map
        for hot_to_warm_map in unique_maps.values():
            hot_to_warm_map[hot_block_ids] = -1

        released_slots = self.allocator.release_keys(
            key for key, _ in request_entries
        )
        for key, _ in request_entries:
            del self.warm_residency[key]
        self.warm_residency_revision += 1
        return released_slots
