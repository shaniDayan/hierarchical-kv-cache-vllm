"""GSM8K helpers for the hierarchical KV resume-quality experiment.

This module has no vLLM or GPU dependency so CPU tests can exercise prompt
construction, extraction, pairing, and capacity preflight.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

BLOCK_SIZE = 16
INVALID_ANSWER = -9999999
RESULT_SCHEMA_VERSION = "hkv-gsm8k-resume-quality-1.0"
DEFAULT_BALLAST_SAFETY_MARGIN_BLOCKS = 16
DEFAULT_BALLAST_GENERATION_TOKENS = 1
DEFAULT_BALLAST_FILL_TOKEN_ID = 1
TURN1_INSTRUCTION = (
    "You will later solve the last problem. Read it carefully. "
    "Do not give the final numeric answer yet."
)
TURN2_TEXT = (
    "Now solve the last problem. Show brief reasoning and put the numeric "
    "answer on its own line as #### <number>."
)
EXCLUSION_NO_COMPLETE_BLOCKS = "no_complete_historical_blocks"
EXCLUSION_NOT_WARM_BEFORE_RESUME = (
    "complete_historical_blocks_not_warm_before_resume"
)
EXCLUSION_NO_RESUMED_MIXED_READ = "no_resumed_mixed_read_step"


@dataclass(frozen=True, slots=True)
class Gsm8kExample:
    index: int
    question: str
    answer: str
    gold_value: int


@dataclass(frozen=True, slots=True)
class Gsm8kSubset:
    train_path: Path
    test_path: Path
    train_sha256: str
    test_sha256: str
    shots: tuple[Gsm8kExample, ...]
    questions: tuple[Gsm8kExample, ...]


@dataclass(frozen=True, slots=True)
class PromptPlan:
    session_id: str
    question_index: int
    gold_value: int
    turn1_text: str
    turn2_text: str
    turn1_token_ids: tuple[int, ...]
    turn2_token_ids: tuple[int, ...]

    @property
    def turn1_token_count(self) -> int:
        return len(self.turn1_token_ids)

    @property
    def complete_historical_blocks(self) -> int:
        return self.turn1_token_count // BLOCK_SIZE

    @property
    def hot_blocks_turn1(self) -> int:
        if self.turn1_token_count <= 0:
            return 0
        return math.ceil(self.turn1_token_count / BLOCK_SIZE)


@dataclass(slots=True)
class SessionQuality:
    session_id: str
    question_index: int
    gold_value: int
    final_text: str
    extracted_value: int
    parse_failure: bool
    exact_match: bool
    complete_historical_blocks: int
    pre_resume_warm_confirmed: bool
    attention_had_warm_slots: bool
    mixed_read_steps: int
    warm_logical_blocks_observed: int
    exclusion_reason: str | None = None
    generated_token_ids: list[int] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_gsm8k_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid GSM8K JSON on line {line_number} of {path}"
                ) from exc
            if "question" not in value or "answer" not in value:
                raise ValueError(
                    f"GSM8K record on line {line_number} of {path} "
                    "is missing question or answer"
                )
            records.append(value)
    if not records:
        raise ValueError(f"GSM8K file is empty: {path}")
    return records


def get_answer_value(answer_str: str) -> int:
    """Extract the last integer from a GSM8K-style answer string."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID_ANSWER
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID_ANSWER


def extract_final_response(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        return text.rsplit("</think>", maxsplit=1)[-1].strip()
    return text


def _examples_from_records(
    records: Sequence[dict[str, Any]],
    *,
    start: int,
    limit: int,
    label: str,
) -> tuple[Gsm8kExample, ...]:
    selected = records[start : start + limit]
    if len(selected) < limit:
        raise ValueError(
            f"{label} needs {limit} examples starting at {start}; "
            f"only {len(records)} records are available"
        )
    examples: list[Gsm8kExample] = []
    for offset, record in enumerate(selected):
        gold = get_answer_value(str(record["answer"]))
        if gold == INVALID_ANSWER:
            raise ValueError(
                f"{label} example {start + offset} has no numeric gold answer"
            )
        examples.append(
            Gsm8kExample(
                index=start + offset,
                question=str(record["question"]),
                answer=str(record["answer"]),
                gold_value=gold,
            )
        )
    return tuple(examples)


def load_gsm8k_subset(
    dataset_dir: Path,
    *,
    num_questions: int,
    num_shots: int,
    start_index: int = 0,
    shot_start_index: int = 0,
) -> Gsm8kSubset:
    if num_questions <= 0:
        raise ValueError("num_questions must be positive")
    if num_shots < 0:
        raise ValueError("num_shots must be non-negative")
    if start_index < 0 or shot_start_index < 0:
        raise ValueError("dataset start indices must be non-negative")

    train_path = dataset_dir / "train.jsonl"
    test_path = dataset_dir / "test.jsonl"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "local GSM8K JSONL was not found; expected "
            f"{train_path} and {test_path}. This harness does not download data."
        )
    train_records = load_gsm8k_jsonl(train_path)
    test_records = load_gsm8k_jsonl(test_path)
    shots = (
        _examples_from_records(
            train_records,
            start=shot_start_index,
            limit=num_shots,
            label="train shots",
        )
        if num_shots
        else ()
    )
    questions = _examples_from_records(
        test_records,
        start=start_index,
        limit=num_questions,
        label="test questions",
    )
    return Gsm8kSubset(
        train_path=train_path,
        test_path=test_path,
        train_sha256=sha256_file(train_path),
        test_sha256=sha256_file(test_path),
        shots=shots,
        questions=questions,
    )


def render_turn1_user_message(
    example: Gsm8kExample,
    shots: Sequence[Gsm8kExample],
) -> str:
    parts: list[str] = []
    for shot in shots:
        parts.append(f"Question: {shot.question}\nAnswer: {shot.answer}\n")
    parts.append(TURN1_INSTRUCTION)
    parts.append(f"Question: {example.question}")
    return "\n".join(parts)


def session_id_for_index(question_index: int) -> str:
    return f"gsm8k-{question_index:04d}"


def _to_nested_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise ValueError(
        f"unsupported tokenizer tensor type: {type(value).__name__}"
    )


def _is_tensor_like(value: Any) -> bool:
    if isinstance(value, (list, tuple, Mapping, str, bytes, bytearray)):
        return False
    shape = getattr(value, "shape", None)
    return shape is not None and hasattr(value, "tolist")


def _as_token_int(token: Any) -> int:
    if isinstance(token, (bool, str, bytes, bytearray)):
        raise ValueError(
            f"tokenizer produced a non-integral token value: {token!r}"
        )
    try:
        as_int = int(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"tokenizer produced a non-integral token value: {token!r}"
        ) from exc
    if as_int != token:
        raise ValueError(
            f"tokenizer produced a non-integral token value: {token!r}"
        )
    return as_int


def _row_to_token_tuple(row: Any) -> tuple[int, ...]:
    if isinstance(row, Mapping):
        raise ValueError(
            "tokenizer output row is a mapping; expected a sequence of token ids"
        )
    if isinstance(row, (str, bytes, bytearray)):
        raise ValueError("tokenizer output must be token ids, not text")
    if _is_tensor_like(row):
        shape = tuple(int(dim) for dim in row.shape)
        if len(shape) != 1:
            raise ValueError(
                "tokenizer output is a nested batch; expected one sequence"
            )
        row = _to_nested_list(row)
    if not isinstance(row, (list, tuple)):
        raise ValueError(
            f"unsupported tokenizer output type: {type(row).__name__}"
        )
    if not row:
        raise ValueError("tokenizer produced an empty token sequence")
    if row and isinstance(row[0], (list, tuple, Mapping)):
        raise ValueError(
            "tokenizer output is a nested batch; expected one sequence"
        )
    if row and _is_tensor_like(row[0]):
        raise ValueError(
            "tokenizer output is a nested batch; expected one sequence"
        )
    return tuple(_as_token_int(token) for token in row)


def normalize_token_ids(value: Any) -> tuple[int, ...]:
    """Normalize tokenizer output to a single sequence of token ids."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("tokenizer output mapping is missing 'input_ids'")
        value = value["input_ids"]
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("tokenizer output must be token ids, not text")
    if value is None:
        raise ValueError("tokenizer produced an empty token sequence")

    if _is_tensor_like(value):
        shape = tuple(int(dim) for dim in value.shape)
        if len(shape) == 0:
            raise ValueError("tokenizer output is a scalar, not a token sequence")
        if len(shape) > 2:
            raise ValueError(
                f"tokenizer output has rank {len(shape)}; expected a single "
                "sequence or a batch of shape [1, seq]"
            )
        if len(shape) == 2 and shape[0] != 1:
            raise ValueError(
                f"tokenizer output contains {shape[0]} batch rows; "
                "expected one sequence"
            )
        nested = _to_nested_list(value)
        if len(shape) == 1:
            return _row_to_token_tuple(nested)
        return _row_to_token_tuple(nested[0])

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("tokenizer produced an empty token sequence")
        first = value[0]
        batched = (
            isinstance(first, (list, tuple, Mapping))
            or _is_tensor_like(first)
        )
        if batched:
            if len(value) != 1:
                raise ValueError(
                    f"tokenizer output contains {len(value)} batch rows; "
                    "expected one sequence"
                )
            return _row_to_token_tuple(value[0])
        return _row_to_token_tuple(value)

    raise ValueError(
        f"unsupported tokenizer output type: {type(value).__name__}"
    )


def _tokenize_chat_user(tokenizer: Any, user_text: str) -> tuple[int, ...]:
    apply = getattr(tokenizer, "apply_chat_template", None)
    messages = [{"role": "user", "content": user_text}]
    if callable(apply):
        try:
            token_ids = apply(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            token_ids = apply(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        return normalize_token_ids(token_ids)
    return normalize_token_ids(tokenizer.encode(user_text))


def _tokenize_plain(tokenizer: Any, text: str) -> tuple[int, ...]:
    return normalize_token_ids(tokenizer.encode(text))


def build_prompt_plans(
    subset: Gsm8kSubset,
    tokenizer: Any,
) -> list[PromptPlan]:
    plans: list[PromptPlan] = []
    for example in subset.questions:
        turn1_text = render_turn1_user_message(example, subset.shots)
        plans.append(
            PromptPlan(
                session_id=session_id_for_index(example.index),
                question_index=example.index,
                gold_value=example.gold_value,
                turn1_text=turn1_text,
                turn2_text=TURN2_TEXT,
                turn1_token_ids=_tokenize_chat_user(tokenizer, turn1_text),
                turn2_token_ids=_tokenize_plain(tokenizer, TURN2_TEXT),
            )
        )
    return plans


def complete_historical_indices(complete_historical_blocks: int) -> list[int]:
    return list(range(complete_historical_blocks))


def residency_entry_key(item: Any) -> tuple[str, int, int] | None:
    """Parse inspect_hkv_request residency keys in tuple or JSON-list form.

    Expected key: (request_id, kv_group, logical_block_index).
    """
    key = item.get("key") if isinstance(item, dict) else item
    if not isinstance(key, (list, tuple)) or len(key) < 3:
        return None
    try:
        return str(key[0]), int(key[1]), int(key[2])
    except (TypeError, ValueError):
        return None


def warm_logical_indices(
    residency: Iterable[dict[str, Any]],
    *,
    request_id: str | None = None,
    kv_group: int = 0,
) -> set[int]:
    indices: set[int] = set()
    expected_request = None if request_id is None else str(request_id)
    for item in residency:
        parsed = residency_entry_key(item)
        if parsed is None:
            continue
        item_request, item_group, logical_index = parsed
        if item_group != int(kv_group):
            continue
        if expected_request is not None and item_request != expected_request:
            continue
        indices.add(logical_index)
    return indices


def complete_blocks_are_warm(
    *,
    complete_historical_blocks: int,
    residency: Iterable[dict[str, Any]],
    block_ids: Sequence[int] | None = None,
    request_id: str | None = None,
    kv_group: int = 0,
) -> bool:
    """Confirm every complete historical logical block is in WARM residency.

    Physical HOT ``block_ids``, ``warm_slot_id``, and
    ``temporary_shadow_hot_block_id`` are not logical indices. The incomplete
    tail block may remain HOT.
    """
    if complete_historical_blocks <= 0:
        return False
    expected = set(complete_historical_indices(complete_historical_blocks))
    observed = warm_logical_indices(
        residency,
        request_id=request_id,
        kv_group=kv_group,
    )
    if not expected.issubset(observed):
        return False
    if block_ids is None:
        return True
    return len(block_ids) >= complete_historical_blocks


def mixed_exclusion_reason(
    *,
    complete_historical_blocks: int,
    pre_resume_warm_confirmed: bool,
    attention_had_warm_slots: bool,
) -> str | None:
    if complete_historical_blocks <= 0:
        return EXCLUSION_NO_COMPLETE_BLOCKS
    if not pre_resume_warm_confirmed:
        return EXCLUSION_NOT_WARM_BEFORE_RESUME
    if not attention_had_warm_slots:
        return EXCLUSION_NO_RESUMED_MIXED_READ
    return None


def score_generated_text(text: str, gold_value: int) -> tuple[int, bool, bool]:
    extracted = get_answer_value(extract_final_response(text))
    parse_failure = extracted == INVALID_ANSWER
    exact_match = (not parse_failure) and extracted == gold_value
    return extracted, parse_failure, exact_match


def quality_summary(
    sessions: Sequence[SessionQuality],
    *,
    kv_mode: str,
) -> dict[str, Any]:
    completed = list(sessions)
    if kv_mode == "mixed":
        eligible = [
            session for session in completed if session.exclusion_reason is None
        ]
    else:
        eligible = completed
    parse_failures = sum(1 for session in eligible if session.parse_failure)
    exact = sum(1 for session in eligible if session.exact_match)
    confirmed = sum(
        1
        for session in completed
        if session.pre_resume_warm_confirmed and session.attention_had_warm_slots
    )
    n_eligible = len(eligible)
    return {
        "n_selected": len(completed),
        "n_completed": len(completed),
        "n_eligible": n_eligible,
        "n_excluded": len(completed) - n_eligible,
        "n_confirmed_warm_or_mixed_read": confirmed,
        "fraction_confirmed_warm_or_mixed_read": (
            confirmed / len(completed) if completed else 0.0
        ),
        "n_parse_failures": parse_failures,
        "n_exact_match": exact,
        "accuracy_exact_match": (
            exact / n_eligible if n_eligible else None
        ),
        "excluded_sessions": [
            {
                "session_id": session.session_id,
                "question_index": session.question_index,
                "reason": session.exclusion_reason,
            }
            for session in completed
            if session.exclusion_reason is not None
        ],
    }


def pair_quality(
    all_hot: Sequence[SessionQuality],
    mixed: Sequence[SessionQuality],
) -> dict[str, Any]:
    mixed_by_id = {session.session_id: session for session in mixed}
    paired: list[tuple[SessionQuality, SessionQuality]] = []
    missing: list[str] = []
    for session in all_hot:
        other = mixed_by_id.get(session.session_id)
        if other is None:
            missing.append(session.session_id)
            continue
        if other.exclusion_reason is not None:
            continue
        paired.append((session, other))

    identical = 0
    divergent = 0
    only_all_hot = 0
    only_mixed = 0
    both_correct = 0
    both_wrong = 0
    for left, right in paired:
        if left.extracted_value == right.extracted_value:
            identical += 1
        else:
            divergent += 1
        if left.exact_match and right.exact_match:
            both_correct += 1
        elif left.exact_match and not right.exact_match:
            only_all_hot += 1
        elif right.exact_match and not left.exact_match:
            only_mixed += 1
        else:
            both_wrong += 1

    return {
        "n_paired_eligible": len(paired),
        "n_missing_from_mixed": missing,
        "n_identical_extracted_answers": identical,
        "n_divergent_extracted_answers": divergent,
        "n_only_all_hot_correct": only_all_hot,
        "n_only_mixed_correct": only_mixed,
        "n_both_correct": both_correct,
        "n_both_wrong": both_wrong,
    }


def prompt_fingerprint_fields(plans: Sequence[PromptPlan]) -> dict[str, Any]:
    return {
        "session_ids": [plan.session_id for plan in plans],
        "question_indices": [plan.question_index for plan in plans],
        "turn1_token_sha256": sha256_json(
            [list(plan.turn1_token_ids) for plan in plans]
        ),
        "turn2_token_sha256": sha256_json(
            [list(plan.turn2_token_ids) for plan in plans]
        ),
        "turn2_text": TURN2_TEXT,
    }


def exact_length_token_ids(
    length: int,
    *,
    fill_token_id: int = DEFAULT_BALLAST_FILL_TOKEN_ID,
) -> tuple[int, ...]:
    """Build a deterministic token sequence of exactly ``length`` ids."""
    if length < 0:
        raise ValueError("token length must be non-negative")
    return (int(fill_token_id),) * int(length)


def divide_blocks_across_sessions(
    total_blocks: int,
    num_sessions: int,
) -> tuple[int, ...]:
    """Spread ``total_blocks`` across sessions, giving remainder to the first."""
    if num_sessions < 0:
        raise ValueError("num_sessions must be non-negative")
    if total_blocks < 0:
        raise ValueError("total_blocks must be non-negative")
    if num_sessions == 0:
        return ()
    base, remainder = divmod(int(total_blocks), int(num_sessions))
    return tuple(base + (1 if index < remainder else 0) for index in range(num_sessions))


def plan_ballast_allocation(
    *,
    kv_mode: str,
    usable_hot_blocks: int | None,
    demotion_start_utilization: float | None,
    total_hot_blocks_turn1: int,
    num_ballast_sessions: int,
    max_model_len: int,
    safety_margin_blocks: int = DEFAULT_BALLAST_SAFETY_MARGIN_BLOCKS,
    generation_tokens: int = DEFAULT_BALLAST_GENERATION_TOKENS,
    fill_token_id: int = DEFAULT_BALLAST_FILL_TOKEN_ID,
) -> dict[str, Any]:
    """Plan exact Mixed-mode ballast HOT blocks, tokens, and refusal reasons."""
    if safety_margin_blocks < 0:
        raise ValueError("safety_margin_blocks must be non-negative")
    if generation_tokens < 0:
        raise ValueError("generation_tokens must be non-negative")
    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")

    pressure_target = (
        None
        if usable_hot_blocks is None or demotion_start_utilization is None
        else math.ceil(demotion_start_utilization * usable_hot_blocks)
    )
    target_hot_blocks = (
        None
        if pressure_target is None
        else pressure_target + int(safety_margin_blocks)
    )
    required_blocks = 0
    if (
        kv_mode == "mixed"
        and target_hot_blocks is not None
        and total_hot_blocks_turn1 < target_hot_blocks
    ):
        required_blocks = target_hot_blocks - total_hot_blocks_turn1

    blocks_per_request = divide_blocks_across_sessions(
        required_blocks,
        num_ballast_sessions if kv_mode == "mixed" else 0,
    )
    prompt_tokens_per_request = tuple(
        blocks * BLOCK_SIZE for blocks in blocks_per_request
    )
    generation_tokens_per_request = tuple(
        int(generation_tokens) if blocks > 0 else 0
        for blocks in blocks_per_request
    )
    requests = [
        {
            "request_id": f"gsm8k-ballast-{index:03d}",
            "planned_blocks": blocks,
            "prompt_tokens": prompt_tokens,
            "generation_tokens": gen_tokens,
            "prompt_token_ids": exact_length_token_ids(
                prompt_tokens,
                fill_token_id=fill_token_id,
            ),
            "total_tokens": prompt_tokens + gen_tokens,
        }
        for index, (blocks, prompt_tokens, gen_tokens) in enumerate(
            zip(
                blocks_per_request,
                prompt_tokens_per_request,
                generation_tokens_per_request,
            )
        )
    ]
    launched = [item for item in requests if item["planned_blocks"] > 0]
    max_request_tokens = max(
        (item["total_tokens"] for item in launched),
        default=0,
    )
    reasons: list[str] = []
    if kv_mode == "mixed" and num_ballast_sessions <= 0:
        reasons.append("mixed mode requires a positive --num-ballast-sessions")
    if launched and max_request_tokens > max_model_len:
        reasons.append(
            "required per-request ballast length "
            f"({max_request_tokens} tokens) exceeds --max-model-len "
            f"({max_model_len})"
        )
    remaining_hot = (
        None
        if usable_hot_blocks is None
        else max(usable_hot_blocks - total_hot_blocks_turn1, 0)
    )
    if (
        kv_mode == "mixed"
        and remaining_hot is not None
        and required_blocks > remaining_hot
    ):
        reasons.append(
            "remaining HOT after evaluated turn 1 "
            f"({remaining_hot}) cannot hold planned ballast "
            f"({required_blocks} HOT blocks, including safety margin "
            f"{safety_margin_blocks})"
        )
    total_planned = sum(blocks_per_request)
    estimated_peak = total_hot_blocks_turn1 + total_planned
    return {
        "safety_margin_blocks": int(safety_margin_blocks),
        "demotion_start_hot_blocks": pressure_target,
        "target_hot_blocks": target_hot_blocks,
        "required_ballast_blocks": required_blocks,
        "num_ballast_sessions": num_ballast_sessions,
        "planned_blocks_per_request": list(blocks_per_request),
        "prompt_tokens_per_request": list(prompt_tokens_per_request),
        "generation_tokens_per_request": list(generation_tokens_per_request),
        "generation_tokens": int(generation_tokens),
        "total_planned_ballast_blocks": total_planned,
        "estimated_peak_hot_blocks": estimated_peak,
        "max_request_tokens": max_request_tokens,
        "fill_token_id": int(fill_token_id),
        "requests": requests,
        "launched_request_ids": [item["request_id"] for item in launched],
        "refusal_reasons": reasons,
    }


def build_capacity_report(
    plans: Sequence[PromptPlan],
    *,
    kv_mode: str,
    memory_budget: dict[str, Any],
    warm_pool_blocks: int,
    demotion_start_utilization: float | None,
    num_ballast_sessions: int,
    max_num_seqs: int,
    max_model_len: int,
    safety_margin_blocks: int = DEFAULT_BALLAST_SAFETY_MARGIN_BLOCKS,
    ballast_generation_tokens: int = DEFAULT_BALLAST_GENERATION_TOKENS,
) -> dict[str, Any]:
    per_question = [
        {
            "session_id": plan.session_id,
            "question_index": plan.question_index,
            "turn1_token_count": plan.turn1_token_count,
            "turn2_token_count": len(plan.turn2_token_ids),
            "complete_historical_blocks": plan.complete_historical_blocks,
            "hot_blocks_turn1": plan.hot_blocks_turn1,
            "exceeds_max_model_len": (
                plan.turn1_token_count + len(plan.turn2_token_ids)
                > max_model_len
            ),
        }
        for plan in plans
    ]
    total_complete = sum(plan.complete_historical_blocks for plan in plans)
    total_hot_turn1 = sum(plan.hot_blocks_turn1 for plan in plans)
    derived_hot = memory_budget.get("derived_num_gpu_blocks")
    usable_hot = (
        None if derived_hot is None else max(derived_hot - 1, 0)
    )
    start = demotion_start_utilization
    ballast_plan = plan_ballast_allocation(
        kv_mode=kv_mode,
        usable_hot_blocks=usable_hot,
        demotion_start_utilization=start,
        total_hot_blocks_turn1=total_hot_turn1,
        num_ballast_sessions=num_ballast_sessions,
        max_model_len=max_model_len,
        safety_margin_blocks=safety_margin_blocks,
        generation_tokens=ballast_generation_tokens,
    )
    pressure_target = ballast_plan["demotion_start_hot_blocks"]
    ballast_hot_needed = 0
    if (
        kv_mode == "mixed"
        and pressure_target is not None
        and total_hot_turn1 < pressure_target
    ):
        ballast_hot_needed = pressure_target - total_hot_turn1

    reasons: list[str] = []
    if any(item["exceeds_max_model_len"] for item in per_question):
        reasons.append("at least one prompt exceeds --max-model-len")
    if any(plan.turn1_token_count <= 0 for plan in plans):
        reasons.append("a turn-1 prompt tokenized to zero tokens")
    if kv_mode == "mixed":
        if any(plan.complete_historical_blocks <= 0 for plan in plans):
            reasons.append(
                "a scored session has no complete historical block, so WARM "
                "confirmation of history is impossible"
            )
        if derived_hot is None:
            reasons.append(
                "explicit --total-kv-budget-bytes is required to assess HOT "
                "capacity"
            )
        else:
            assert usable_hot is not None
            if total_hot_turn1 > usable_hot:
                reasons.append(
                    "evaluated turn-1 HOT blocks "
                    f"({total_hot_turn1}) exceed usable HOT capacity "
                    f"({usable_hot})"
                )
            if total_complete > warm_pool_blocks:
                reasons.append(
                    "total complete historical blocks "
                    f"({total_complete}) exceed WARM pool ({warm_pool_blocks})"
                )
            too_large = [
                plan.session_id
                for plan in plans
                if plan.complete_historical_blocks > warm_pool_blocks
            ]
            if too_large:
                reasons.append(
                    "a session's complete historical blocks exceed the WARM "
                    f"pool: {too_large}"
                )
        required_seqs = len(plans) + max(num_ballast_sessions, 0)
        if required_seqs > max_num_seqs:
            reasons.append(
                "evaluated plus ballast sessions "
                f"({required_seqs}) exceed --max-num-seqs ({max_num_seqs})"
            )
        reasons.extend(ballast_plan["refusal_reasons"])
    elif derived_hot is not None and usable_hot is not None:
        if total_hot_turn1 > usable_hot:
            reasons.append(
                "evaluated turn-1 HOT blocks "
                f"({total_hot_turn1}) exceed usable HOT capacity "
                f"({usable_hot})"
            )

    peak = {
        "evaluated_hot_blocks_after_turn1": total_hot_turn1,
        "evaluated_complete_historical_blocks": total_complete,
        "ballast_hot_blocks_to_reach_demotion_start": ballast_hot_needed,
        "ballast_safety_margin_blocks": ballast_plan["safety_margin_blocks"],
        "total_planned_ballast_blocks": ballast_plan[
            "total_planned_ballast_blocks"
        ],
        "planned_blocks_per_ballast_request": ballast_plan[
            "planned_blocks_per_request"
        ],
        "prompt_tokens_per_ballast_request": ballast_plan[
            "prompt_tokens_per_request"
        ],
        "generation_tokens_per_ballast_request": ballast_plan[
            "generation_tokens_per_request"
        ],
        "estimated_peak_hot_blocks": ballast_plan["estimated_peak_hot_blocks"],
        "estimated_peak_warm_blocks": (
            total_complete if kv_mode == "mixed" else 0
        ),
    }
    public_ballast_plan = {
        key: value
        for key, value in ballast_plan.items()
        if key != "requests"
    }
    public_ballast_plan["requests"] = [
        {
            "request_id": item["request_id"],
            "planned_blocks": item["planned_blocks"],
            "prompt_tokens": item["prompt_tokens"],
            "generation_tokens": item["generation_tokens"],
            "total_tokens": item["total_tokens"],
        }
        for item in ballast_plan["requests"]
    ]
    feasible = not reasons
    return {
        "block_size": BLOCK_SIZE,
        "kv_mode": kv_mode,
        "questions": per_question,
        "total_complete_historical_blocks": total_complete,
        "total_hot_blocks_turn1": total_hot_turn1,
        "derived_hot_block_capacity": derived_hot,
        "usable_hot_blocks": usable_hot,
        "configured_warm_slot_capacity": (
            warm_pool_blocks if kv_mode == "mixed" else 0
        ),
        "demotion_start_utilization": start,
        "num_ballast_sessions": num_ballast_sessions,
        "ballast_plan": public_ballast_plan,
        "estimated_peak_requirement": peak,
        "can_satisfy_per_session_warm_confirmation": (
            feasible if kv_mode == "mixed" else True
        ),
        "feasible": feasible,
        "refusal_reasons": reasons,
    }


def fingerprint_differences(
    current: Any,
    baseline: Any,
    prefix: str = "",
) -> list[str]:
    if isinstance(current, dict) and isinstance(baseline, dict):
        keys = sorted(set(current) | set(baseline))
        differences: list[str] = []
        for key in keys:
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in current:
                differences.append(f"missing {child}")
                continue
            if key not in baseline:
                differences.append(f"extra {child}")
                continue
            differences.extend(
                fingerprint_differences(current[key], baseline[key], child)
            )
        return differences
    return [] if current == baseline else [prefix or "value"]
