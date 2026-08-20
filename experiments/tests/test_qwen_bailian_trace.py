from __future__ import annotations

import json

import pytest

from experiments.scripts.qwen_bailian_trace import (
    BLOCK_SIZE,
    BailianRecord,
    HashTokenBlockStore,
    build_replay_plan,
    flatten_replay_plan,
    group_linear_sessions,
    load_bailian_records,
)


def make_record(
    *,
    chat_id: int,
    parent_chat_id: int,
    timestamp: float,
    input_length: int,
    turn: int,
    hash_ids: tuple[int, ...],
    output_length: int = 10,
) -> BailianRecord:
    return BailianRecord(
        chat_id=chat_id,
        parent_chat_id=parent_chat_id,
        timestamp=timestamp,
        input_length=input_length,
        output_length=output_length,
        request_type="text",
        turn=turn,
        hash_ids=hash_ids,
    )


def test_load_bailian_records_reads_jsonl_and_ignores_blank_lines(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "chat_id": 7,
                "parent_chat_id": -1,
                "timestamp": 1.25,
                "input_length": 17,
                "output_length": 9,
                "type": "text",
                "turn": 1,
                "hash_ids": [100, 101],
            }
        )
        + "\n\n",
        encoding="utf-8",
    )

    records = load_bailian_records(trace_path)

    assert records == [
        make_record(
            chat_id=7,
            parent_chat_id=-1,
            timestamp=1.25,
            input_length=17,
            output_length=9,
            turn=1,
            hash_ids=(100, 101),
        )
    ]


def test_load_bailian_records_reports_invalid_line_number(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("\n{not-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_bailian_records(trace_path)


def test_record_rejects_wrong_number_of_hash_blocks():
    record = make_record(
        chat_id=1,
        parent_chat_id=-1,
        timestamp=0.0,
        input_length=BLOCK_SIZE + 1,
        turn=1,
        hash_ids=(10,),
    )

    with pytest.raises(ValueError, match="expected 2"):
        record.validate()


def test_group_linear_sessions_orders_each_parent_chain():
    root = make_record(
        chat_id=10,
        parent_chat_id=-1,
        timestamp=0.0,
        input_length=8,
        turn=1,
        hash_ids=(1,),
    )
    child = make_record(
        chat_id=11,
        parent_chat_id=10,
        timestamp=5.0,
        input_length=20,
        turn=2,
        hash_ids=(1, 2),
    )
    other_root = make_record(
        chat_id=20,
        parent_chat_id=-1,
        timestamp=1.0,
        input_length=4,
        turn=1,
        hash_ids=(3,),
    )

    sessions = group_linear_sessions([child, other_root, root])

    assert list(sessions) == [10, 20]
    assert [record.chat_id for record in sessions[10]] == [10, 11]
    assert [record.chat_id for record in sessions[20]] == [20]


def test_group_linear_sessions_rejects_a_branch():
    root = make_record(
        chat_id=1,
        parent_chat_id=-1,
        timestamp=0.0,
        input_length=8,
        turn=1,
        hash_ids=(1,),
    )
    first_child = make_record(
        chat_id=2,
        parent_chat_id=1,
        timestamp=1.0,
        input_length=17,
        turn=2,
        hash_ids=(1, 2),
    )
    second_child = make_record(
        chat_id=3,
        parent_chat_id=1,
        timestamp=2.0,
        input_length=18,
        turn=2,
        hash_ids=(1, 3),
    )

    with pytest.raises(ValueError, match="branches or skips a parent"):
        group_linear_sessions([root, first_child, second_child])


def test_build_replay_plan_preserves_partial_parent_block_and_emits_only_delta():
    root = make_record(
        chat_id=1,
        parent_chat_id=-1,
        timestamp=10.0,
        input_length=19,
        turn=1,
        hash_ids=(100, 101),
    )
    child = make_record(
        chat_id=2,
        parent_chat_id=1,
        timestamp=40.0,
        input_length=40,
        turn=2,
        # The incomplete parent boundary block may receive a new trace hash.
        hash_ids=(100, 200, 201),
    )

    turns = build_replay_plan(
        [child, root], vocab_size=60000, time_scale=0.01, seed=3
    )[1]

    assert [turn.session_id for turn in turns] == ["bailian-1", "bailian-1"]
    assert [len(turn.delta_token_ids) for turn in turns] == [19, 21]
    assert [turn.input_length for turn in turns] == [19, 40]
    assert turns[0].send_at_seconds == pytest.approx(0.0)
    assert turns[1].send_at_seconds == pytest.approx(0.3)

    reconstructed_child = turns[0].delta_token_ids + turns[1].delta_token_ids
    assert len(reconstructed_child) == 40
    assert reconstructed_child[:19] == turns[0].delta_token_ids


def test_build_replay_plan_rejects_changed_complete_parent_prefix():
    root = make_record(
        chat_id=1,
        parent_chat_id=-1,
        timestamp=0.0,
        input_length=20,
        turn=1,
        hash_ids=(10, 11),
    )
    child = make_record(
        chat_id=2,
        parent_chat_id=1,
        timestamp=1.0,
        input_length=32,
        turn=2,
        hash_ids=(999, 12),
    )

    with pytest.raises(ValueError, match="complete-block prefix"):
        build_replay_plan([root, child], vocab_size=60000, time_scale=1.0)


def test_token_reconstruction_is_deterministic_for_the_same_seed():
    root = make_record(
        chat_id=1,
        parent_chat_id=-1,
        timestamp=0.0,
        input_length=32,
        turn=1,
        hash_ids=(10, 11),
    )

    first = build_replay_plan(
        [root], vocab_size=60000, time_scale=1.0, seed=42
    )[1][0]
    second = build_replay_plan(
        [root], vocab_size=60000, time_scale=1.0, seed=42
    )[1][0]

    assert first.delta_token_ids == second.delta_token_ids
    assert all(1000 <= token_id < 50000 for token_id in first.delta_token_ids)


def test_hash_token_block_store_rejects_conflicting_required_prefix():
    store = HashTokenBlockStore(vocab_size=60000, seed=1)
    original = store.get_block(55)
    conflicting_prefix = (original[0] + 1,)

    with pytest.raises(ValueError, match="conflicts"):
        store.get_block(55, required_prefix=conflicting_prefix)


def test_flatten_replay_plan_sorts_sessions_by_scaled_send_time():
    late_root = make_record(
        chat_id=20,
        parent_chat_id=-1,
        timestamp=20.0,
        input_length=4,
        turn=1,
        hash_ids=(2,),
    )
    early_root = make_record(
        chat_id=10,
        parent_chat_id=-1,
        timestamp=5.0,
        input_length=4,
        turn=1,
        hash_ids=(1,),
    )

    plan = build_replay_plan(
        [late_root, early_root], vocab_size=60000, time_scale=0.1
    )

    assert [turn.chat_id for turn in flatten_replay_plan(plan)] == [10, 20]