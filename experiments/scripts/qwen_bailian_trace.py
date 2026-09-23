"""Qwen-Bailian trace parsing and deterministic token reconstruction.

This module turns the anonymized request trace into append-only replay turns
that can be fed to vLLM's streaming-input API. It deliberately has no vLLM or
GPU dependencies so its behavior can be unit-tested on CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BLOCK_SIZE = 16


@dataclass(frozen=True, slots=True)
class BailianRecord:
    chat_id: int
    parent_chat_id: int
    timestamp: float
    input_length: int
    output_length: int
    request_type: str
    turn: int
    hash_ids: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "BailianRecord":
        record = cls(
            chat_id=int(value["chat_id"]),
            parent_chat_id=int(value["parent_chat_id"]),
            timestamp=float(value["timestamp"]),
            input_length=int(value["input_length"]),
            output_length=int(value["output_length"]),
            request_type=str(value["type"]),
            turn=int(value["turn"]),
            hash_ids=tuple(int(item) for item in value["hash_ids"]),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.chat_id < 0:
            raise ValueError("chat_id must be non-negative")
        if self.parent_chat_id < -1:
            raise ValueError("parent_chat_id must be -1 or non-negative")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if self.input_length <= 0:
            raise ValueError("input_length must be positive")
        if self.output_length < 0:
            raise ValueError("output_length must be non-negative")
        if self.turn <= 0:
            raise ValueError("turn must be positive")
        expected_hashes = math.ceil(self.input_length / BLOCK_SIZE)
        if len(self.hash_ids) != expected_hashes:
            raise ValueError(
                f"chat_id {self.chat_id} has {len(self.hash_ids)} hashes; "
                f"expected {expected_hashes} for input_length={self.input_length}"
            )


@dataclass(frozen=True, slots=True)
class ReplayTurn:
    session_id: str
    root_chat_id: int
    chat_id: int
    parent_chat_id: int
    turn: int
    send_at_seconds: float
    input_length: int
    trace_output_length: int
    delta_token_ids: tuple[int, ...]


def load_bailian_records(path: str | Path) -> list[BailianRecord]:
    records: list[BailianRecord] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(BailianRecord.from_mapping(value))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid Bailian record on line {line_number}") from exc
    if not records:
        raise ValueError("Bailian trace is empty")
    return records


def group_linear_sessions(
    records: Iterable[BailianRecord],
) -> dict[int, list[BailianRecord]]:
    records = list(records)
    by_id: dict[int, BailianRecord] = {}
    for record in records:
        if record.chat_id in by_id:
            raise ValueError(f"duplicate chat_id {record.chat_id}")
        by_id[record.chat_id] = record

    root_cache: dict[int, int] = {}

    def find_root(chat_id: int) -> int:
        cached = root_cache.get(chat_id)
        if cached is not None:
            return cached

        current_id = chat_id
        path: list[int] = []
        seen: set[int] = set()
        while True:
            if current_id in seen:
                raise ValueError(f"cycle detected at chat_id {current_id}")
            seen.add(current_id)
            path.append(current_id)

            current = by_id.get(current_id)
            if current is None:
                raise ValueError(f"missing parent chat_id {current_id}")
            if current.parent_chat_id == -1:
                root_id = current.chat_id
                break
            current_id = current.parent_chat_id

        for path_id in path:
            root_cache[path_id] = root_id
        return root_id

    grouped: dict[int, list[BailianRecord]] = defaultdict(list)
    for record in records:
        grouped[find_root(record.chat_id)].append(record)

    result: dict[int, list[BailianRecord]] = {}
    for root_id, session in grouped.items():
        session.sort(key=lambda item: (item.turn, item.timestamp, item.chat_id))
        if session[0].chat_id != root_id or session[0].parent_chat_id != -1:
            raise ValueError(f"session {root_id} does not start with its root")
        for previous, current in zip(session, session[1:]):
            if current.parent_chat_id != previous.chat_id:
                raise ValueError(
                    f"session {root_id} branches or skips a parent: "
                    f"turn {current.turn} points to {current.parent_chat_id}, "
                    f"expected {previous.chat_id}"
                )
            if current.turn != previous.turn + 1:
                raise ValueError(
                    f"session {root_id} has non-consecutive turns "
                    f"{previous.turn}->{current.turn}"
                )
            if current.timestamp < previous.timestamp:
                raise ValueError(f"session {root_id} timestamps move backwards")
        result[root_id] = session
    return result


class HashTokenBlockStore:
    """Assign one stable 16-token block to every anonymized block hash."""

    def __init__(
        self,
        *,
        vocab_size: int,
        seed: int = 0,
        minimum_token_id: int = 1000,
        maximum_token_id: int = 50000,
    ) -> None:
        upper_bound = min(vocab_size, maximum_token_id)
        if minimum_token_id < 0 or upper_bound <= minimum_token_id:
            raise ValueError("token ID range is empty")
        self.seed = seed
        self.minimum_token_id = minimum_token_id
        self.upper_bound = upper_bound
        self._blocks: dict[int, tuple[int, ...]] = {}

    def _token_id(self, hash_id: int, offset: int) -> int:
        payload = f"{self.seed}:{hash_id}:{offset}".encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        return self.minimum_token_id + value % (
            self.upper_bound - self.minimum_token_id
        )

    def get_block(
        self,
        hash_id: int,
        *,
        required_prefix: tuple[int, ...] = (),
    ) -> tuple[int, ...]:
        if len(required_prefix) >= BLOCK_SIZE:
            raise ValueError("required block prefix must be shorter than 16 tokens")

        existing = self._blocks.get(hash_id)
        if existing is not None:
            if existing[: len(required_prefix)] != required_prefix:
                raise ValueError(
                    f"hash_id {hash_id} conflicts with an existing token block"
                )
            return existing

        suffix = tuple(
            self._token_id(hash_id, offset)
            for offset in range(len(required_prefix), BLOCK_SIZE)
        )
        block = required_prefix + suffix
        self._blocks[hash_id] = block
        return block


def _extend_session_tokens(
    *,
    record: BailianRecord,
    previous: BailianRecord | None,
    session_tokens: list[int],
    block_store: HashTokenBlockStore,
) -> tuple[int, ...]:
    previous_length = len(session_tokens)
    if previous is None:
        if record.parent_chat_id != -1 or record.turn != 1:
            raise ValueError("first replay record is not a root turn")
    else:
        if record.parent_chat_id != previous.chat_id:
            raise ValueError("record does not continue the previous turn")
        if previous_length != previous.input_length:
            raise ValueError("current session length does not match previous trace input")
        if record.input_length < previous.input_length:
            raise ValueError("Bailian session input length shrank")

        complete_parent_blocks = previous.input_length // BLOCK_SIZE
        if (
            record.hash_ids[:complete_parent_blocks]
            != previous.hash_ids[:complete_parent_blocks]
        ):
            raise ValueError(
                f"chat_id {record.chat_id} does not preserve its parent's "
                "complete-block prefix"
            )

    while len(session_tokens) < record.input_length:
        block_index = len(session_tokens) // BLOCK_SIZE
        block_offset = len(session_tokens) % BLOCK_SIZE
        hash_id = record.hash_ids[block_index]
        block_start = block_index * BLOCK_SIZE
        required_prefix = tuple(session_tokens[block_start:]) if block_offset else ()
        block = block_store.get_block(hash_id, required_prefix=required_prefix)
        remaining = record.input_length - len(session_tokens)
        take = min(BLOCK_SIZE - block_offset, remaining)
        session_tokens.extend(block[block_offset : block_offset + take])

    if len(session_tokens) != record.input_length:
        raise AssertionError("reconstructed prompt length mismatch")
    return tuple(session_tokens[previous_length:])


def build_replay_plan(
    records: Iterable[BailianRecord],
    *,
    vocab_size: int,
    time_scale: float,
    seed: int = 0,
) -> dict[int, list[ReplayTurn]]:
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")

    sessions = group_linear_sessions(records)
    trace_start = min(record.timestamp for session in sessions.values() for record in session)
    block_store = HashTokenBlockStore(vocab_size=vocab_size, seed=seed)
    plan: dict[int, list[ReplayTurn]] = {}

    for root_id, session in sessions.items():
        previous: BailianRecord | None = None
        session_tokens: list[int] = []
        turns: list[ReplayTurn] = []
        for record in session:
            delta = _extend_session_tokens(
                record=record,
                previous=previous,
                session_tokens=session_tokens,
                block_store=block_store,
            )
            turns.append(
                ReplayTurn(
                    session_id=f"bailian-{root_id}",
                    root_chat_id=root_id,
                    chat_id=record.chat_id,
                    parent_chat_id=record.parent_chat_id,
                    turn=record.turn,
                    send_at_seconds=(record.timestamp - trace_start) * time_scale,
                    input_length=record.input_length,
                    trace_output_length=record.output_length,
                    delta_token_ids=delta,
                )
            )
            previous = record
        plan[root_id] = turns
    return plan


def flatten_replay_plan(
    plan: dict[int, list[ReplayTurn]],
) -> list[ReplayTurn]:
    return sorted(
        (turn for session in plan.values() for turn in session),
        key=lambda turn: (turn.send_at_seconds, turn.root_chat_id, turn.turn),
    )