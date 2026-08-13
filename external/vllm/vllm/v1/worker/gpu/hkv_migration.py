# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field


class HKVWarmAllocatorError(RuntimeError):
    """Base error for WARM-slot allocation failures."""


class HKVWarmCapacityError(HKVWarmAllocatorError):
    """Raised when a reservation cannot fit in the WARM pool."""


class HKVWarmOwnershipError(HKVWarmAllocatorError):
    """Raised when a physical source has a conflicting logical owner."""


class HKVWarmReservationError(HKVWarmAllocatorError):
    """Raised for invalid reservation lifecycle operations."""


@dataclass(frozen=True, slots=True, order=True)
class HKVBlockSource:
    """Physical HOT block identity within a KV-cache group."""

    cache_group_index: int
    kernel_hot_block_id: int

    def __post_init__(self) -> None:
        for name, value in (
            ("cache_group_index", self.cache_group_index),
            ("kernel_hot_block_id", self.kernel_hot_block_id),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class HKVWarmReservation:
    """One transactional WARM-slot reservation.

    Ownership is visible through the allocator immediately after reservation.
    Commit validates and closes the transaction. Rollback removes only the
    mappings listed in ``newly_allocated``.
    """

    mappings: tuple[tuple[HKVBlockSource, int], ...]
    existing: tuple[HKVBlockSource, ...]
    newly_allocated: tuple[HKVBlockSource, ...]
    owner_token: Hashable
    _reservation_id: int = field(repr=False, compare=False)
    _allocator_token: object = field(repr=False, compare=False)


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
        self._source_to_slot: dict[HKVBlockSource, int] = {}
        self._slot_to_source: dict[int, HKVBlockSource] = {}
        self._source_to_owner_token: dict[HKVBlockSource, Hashable] = {}
        self._free_slots = list(range(capacity))
        self._free_slot_set = set(self._free_slots)
        heapq.heapify(self._free_slots)

    @property
    def num_owned_slots(self) -> int:
        return len(self._source_to_slot)

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def lookup(self, source: HKVBlockSource) -> int | None:
        """Return the WARM slot owned by ``source``, if any."""
        return self._source_to_slot.get(source)

    def owner_of(self, slot: int) -> HKVBlockSource | None:
        """Return the source owning ``slot``, if any."""
        self._validate_slot(slot)
        return self._slot_to_source.get(slot)

    def owner_token_of(self, source: HKVBlockSource) -> Hashable | None:
        """Return the logical owner token associated with ``source``."""
        return self._source_to_owner_token.get(source)

    def reserve_many(
        self,
        sources: Iterable[HKVBlockSource],
        *,
        owner_token: Hashable,
    ) -> HKVWarmReservation:
        """Reserve slots for all unique sources or none of them."""
        if self._active_reservation is not None:
            raise HKVWarmReservationError(
                "the active reservation must be committed or rolled back first"
            )
        self._validate_owner_token(owner_token)

        unique_sources = tuple(dict.fromkeys(sources))
        existing: list[HKVBlockSource] = []
        new_sources: list[HKVBlockSource] = []
        for source in unique_sources:
            if not isinstance(source, HKVBlockSource):
                raise TypeError("sources must contain HKVBlockSource values")
            if source in self._source_to_slot:
                existing_owner = self._source_to_owner_token[source]
                if existing_owner != owner_token:
                    raise HKVWarmOwnershipError(
                        f"source {source!r} is owned by {existing_owner!r}, "
                        f"not {owner_token!r}"
                    )
                existing.append(source)
            else:
                new_sources.append(source)

        if len(new_sources) > self.num_free_slots:
            raise HKVWarmCapacityError(
                f"cannot reserve {len(new_sources)} new WARM slots with "
                f"only {self.num_free_slots} available"
            )

        for source in new_sources:
            slot = heapq.heappop(self._free_slots)
            self._free_slot_set.remove(slot)
            self._source_to_slot[source] = slot
            self._slot_to_source[slot] = source
            self._source_to_owner_token[source] = owner_token

        mappings = tuple(
            (source, self._source_to_slot[source]) for source in unique_sources
        )
        reservation = HKVWarmReservation(
            mappings=mappings,
            existing=tuple(existing),
            newly_allocated=tuple(new_sources),
            owner_token=owner_token,
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
        for source in reservation.newly_allocated:
            self._release_source(source)
        self._active_reservation = None

    def release_source(self, source: HKVBlockSource) -> int | None:
        """Release ``source`` and return its former slot, if present."""
        self._require_no_active_reservation()
        return self._release_source(source)

    def release_sources(
        self, sources: Iterable[HKVBlockSource]
    ) -> tuple[int, ...]:
        """Release multiple sources, ignoring duplicates and unknown sources."""
        self._require_no_active_reservation()
        released_slots: list[int] = []
        for source in dict.fromkeys(sources):
            slot = self._release_source(source)
            if slot is not None:
                released_slots.append(slot)
        return tuple(released_slots)

    def invalidate_source(
        self,
        source: HKVBlockSource,
        *,
        owner_token: Hashable,
    ) -> int | None:
        """Release a reused source only when its logical owner matches."""
        self._require_no_active_reservation()
        self._validate_owner_token(owner_token)
        existing_owner = self._source_to_owner_token.get(source)
        if existing_owner is None:
            return None
        if existing_owner != owner_token:
            raise HKVWarmOwnershipError(
                f"source {source!r} is owned by {existing_owner!r}, "
                f"not {owner_token!r}"
            )
        return self._release_source(source)

    def clear(self) -> None:
        """Clear ownership and restore every WARM slot to the free heap."""
        self._active_reservation = None
        self._source_to_slot.clear()
        self._slot_to_source.clear()
        self._source_to_owner_token.clear()
        self._free_slots = list(range(self.capacity))
        self._free_slot_set = set(self._free_slots)
        heapq.heapify(self._free_slots)

    reset = clear

    def validate_invariants(self) -> None:
        """Assert allocator map, ownership, and free-slot consistency."""
        free_slots = set(self._free_slots)
        owned_slots = set(self._slot_to_source)
        all_slots = set(range(self.capacity))

        assert len(self._free_slots) == len(free_slots)
        assert free_slots == self._free_slot_set
        assert free_slots.isdisjoint(owned_slots)
        assert free_slots | owned_slots == all_slots
        assert len(owned_slots) + len(free_slots) == self.capacity
        assert len(self._source_to_slot) == len(self._slot_to_source)
        assert self._source_to_owner_token.keys() == self._source_to_slot.keys()

        for source, slot in self._source_to_slot.items():
            assert 0 <= slot < self.capacity
            assert self._slot_to_source[slot] == source
        for slot, source in self._slot_to_source.items():
            assert 0 <= slot < self.capacity
            assert self._source_to_slot[source] == slot

        if self._active_reservation is not None:
            reservation = self._active_reservation
            for source, slot in reservation.mappings:
                assert self._source_to_slot[source] == slot
                assert self._source_to_owner_token[source] == reservation.owner_token

    def _release_source(self, source: HKVBlockSource) -> int | None:
        slot = self._source_to_slot.get(source)
        if slot is None:
            return None
        reverse_source = self._slot_to_source.get(slot)
        if reverse_source != source:
            raise HKVWarmOwnershipError(
                f"inconsistent reverse ownership for source {source!r}"
            )
        if slot in self._free_slot_set:
            raise HKVWarmOwnershipError(f"WARM slot {slot} is already free")

        del self._source_to_slot[source]
        del self._slot_to_source[slot]
        del self._source_to_owner_token[source]
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
        for source, slot in reservation.mappings:
            if (
                self._source_to_slot.get(source) != slot
                or self._slot_to_source.get(slot) != source
                or self._source_to_owner_token.get(source)
                != reservation.owner_token
            ):
                raise HKVWarmOwnershipError(
                    f"ownership changed during reservation for source {source!r}"
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
    def _validate_owner_token(owner_token: Hashable) -> None:
        if owner_token is None:
            raise ValueError("owner_token must not be None")
        try:
            hash(owner_token)
        except TypeError as exc:
            raise TypeError("owner_token must be hashable") from exc
