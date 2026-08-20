from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.scripts.qwen_bailian_trace import BailianRecord
from experiments.scripts.run_qwen_bailian_replay import (
    SessionResult,
    compare_baseline,
    percentile,
    select_sessions,
    tokenizer_vocab_size,
)


def make_record(
    *,
    chat_id: int,
    parent_chat_id: int,
    timestamp: float,
    input_length: int,
    turn: int,
    hash_ids: tuple[int, ...],
    request_type: str = "text",
) -> BailianRecord:
    return BailianRecord(
        chat_id=chat_id,
        parent_chat_id=parent_chat_id,
        timestamp=timestamp,
        input_length=input_length,
        output_length=10,
        request_type=request_type,
        turn=turn,
        hash_ids=hash_ids,
    )


def make_session_result(root_chat_id: int, digest: str) -> SessionResult:
    return SessionResult(
        root_chat_id=root_chat_id,
        session_id=f"bailian-{root_chat_id}",
        turns=2,
        final_input_tokens=20,
        trace_output_tokens=10,
        scheduled_first_seconds=0.0,
        scheduled_last_seconds=1.0,
        generated_tokens=2,
        generated_token_sha256=digest,
    )


def test_select_sessions_filters_whole_sessions_and_preserves_chains():
    text_root = make_record(
        chat_id=1,
        parent_chat_id=-1,
        timestamp=2.0,
        input_length=8,
        turn=1,
        hash_ids=(10,),
    )
    text_child = make_record(
        chat_id=2,
        parent_chat_id=1,
        timestamp=4.0,
        input_length=20,
        turn=2,
        hash_ids=(10, 11),
    )
    image_root = make_record(
        chat_id=3,
        parent_chat_id=-1,
        timestamp=0.0,
        input_length=8,
        turn=1,
        hash_ids=(30,),
        request_type="image",
    )
    image_child = make_record(
        chat_id=4,
        parent_chat_id=3,
        timestamp=1.0,
        input_length=20,
        turn=2,
        hash_ids=(30, 31),
        request_type="image",
    )

    selected = select_sessions(
        [image_child, text_child, image_root, text_root],
        request_type="text",
        min_turns=2,
        max_input_length=1024,
        max_sessions=None,
    )

    assert [record.chat_id for record in selected] == [1, 2]


def test_select_sessions_applies_limit_after_trace_time_sorting():
    late = make_record(
        chat_id=20,
        parent_chat_id=-1,
        timestamp=10.0,
        input_length=8,
        turn=1,
        hash_ids=(20,),
    )
    early = make_record(
        chat_id=10,
        parent_chat_id=-1,
        timestamp=1.0,
        input_length=8,
        turn=1,
        hash_ids=(10,),
    )

    selected = select_sessions(
        [late, early],
        request_type="text",
        min_turns=1,
        max_input_length=None,
        max_sessions=1,
    )

    assert [record.chat_id for record in selected] == [10]


def test_select_sessions_rejects_empty_selection():
    root = make_record(
        chat_id=1,
        parent_chat_id=-1,
        timestamp=0.0,
        input_length=8,
        turn=1,
        hash_ids=(10,),
    )

    with pytest.raises(ValueError, match="no Bailian sessions"):
        select_sessions(
            [root],
            request_type="text",
            min_turns=2,
            max_input_length=1024,
            max_sessions=None,
        )


def test_percentile_uses_sorted_nearest_rank_below():
    values = [0.4, 0.1, 0.3, 0.2]

    assert percentile(values, 0.50) == 0.2
    assert percentile(values, 0.95) == 0.3
    assert percentile([], 0.50) is None


def test_tokenizer_vocab_size_supports_property_and_len_fallback():
    assert tokenizer_vocab_size(SimpleNamespace(vocab_size=60000)) == 60000

    class LengthOnlyTokenizer:
        vocab_size = None

        def __len__(self):
            return 70000

    assert tokenizer_vocab_size(LengthOnlyTokenizer()) == 70000


def test_compare_with_baseline_reports_session_digest_mismatch():
    sessions = [make_session_result(1, "same"), make_session_result(2, "actual")]
    baseline = {
        "sessions": [
            {
                "root_chat_id": 1,
                "generated_tokens": 2,
                "generated_token_sha256": "same",
            },
            {
                "root_chat_id": 2,
                "generated_tokens": 2,
                "generated_token_sha256": "expected",
            },
        ]
    }

    comparison = compare_baseline(sessions, baseline)

    assert not comparison["exact_session_output_match"]
    assert comparison["mismatched_root_chat_ids"] == [2]
