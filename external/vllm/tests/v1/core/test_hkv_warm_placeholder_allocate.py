# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WARM null-placeholder accounting vs resumed HOT allocation.

gsm8k-0002 Mixed resume: 750 historical tokens (46 complete blocks + 14-token
HOT tail), then 23 turn-2 tokens. After WARM reclaim the scheduler table keeps
null placeholders in the complete logical slots. The next physical HOT block
must still be allocated when ``cdiv(num_tokens, 16)`` exceeds that logical
length — the placeholders must not be treated as usable HOT capacity.
"""

from __future__ import annotations

from math import ceil

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.kv_cache_state import KVBlockState, KVCacheStateTransition
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
# gsm8k-0002 preflight layout.
TURN1_TOKENS = 750
TURN2_TOKENS = 23
COMPLETE_HISTORICAL_BLOCKS = TURN1_TOKENS // BLOCK_SIZE  # 46
HOT_BLOCKS_TURN1 = ceil(TURN1_TOKENS / BLOCK_SIZE)  # 47


def _make_manager() -> KVCacheManager:
    init_none_hash(sha256)
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    return KVCacheManager(
        config,
        max_model_len=8192,
        enable_caching=False,
        hash_block_size=BLOCK_SIZE,
        scheduler_block_size=BLOCK_SIZE,
    )


def _make_request(num_tokens: int, request_id: str = "gsm8k-0002") -> Request:
    sampling_params = SamplingParams(max_tokens=512)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _demote_complete_blocks(manager: KVCacheManager, request: Request) -> None:
    planned = manager.plan_request_kv_state(
        request.request_id,
        KVBlockState.WARM,
        num_computed_tokens=request.num_computed_tokens,
    )
    assert len(planned[0]) == COMPLETE_HISTORICAL_BLOCKS
    manager.commit_request_kv_transition(
        KVCacheStateTransition(
            transition_id=1,
            request_id=request.request_id,
            previous_state=KVBlockState.HOT,
            new_state=KVBlockState.WARM,
            changed_blocks=planned,
        )
    )


def test_warm_placeholders_are_not_physical_hot_and_new_hot_appends():
    manager = _make_manager()
    pool = manager.block_pool
    null_block = pool.null_block
    request = _make_request(TURN1_TOKENS)

    computed_blocks, num_computed = manager.get_computed_blocks(request)
    assert num_computed == 0
    allocated = manager.allocate_slots(
        request, TURN1_TOKENS, num_computed, computed_blocks
    )
    assert allocated is not None
    request.num_computed_tokens = TURN1_TOKENS

    before = manager.get_blocks(request.request_id).blocks[0]
    assert len(before) == HOT_BLOCKS_TURN1
    assert all(not blk.is_null for blk in before)
    tail = before[COMPLETE_HISTORICAL_BLOCKS]
    reclaimed_ids = {blk.block_id for blk in before[:COMPLETE_HISTORICAL_BLOCKS]}
    free_before_commit = pool.get_num_free_blocks()

    _demote_complete_blocks(manager, request)

    after = manager.get_blocks(request.request_id).blocks[0]
    assert len(after) == HOT_BLOCKS_TURN1
    assert all(after[i] is null_block for i in range(COMPLETE_HISTORICAL_BLOCKS))
    assert after[COMPLETE_HISTORICAL_BLOCKS] is tail
    assert tail.is_null is False
    # Reclaimed HOT blocks are free, not still allocated to this request.
    assert pool.get_num_free_blocks() == free_before_commit + COMPLETE_HISTORICAL_BLOCKS
    for block_id in reclaimed_ids:
        assert pool.blocks[block_id].ref_cnt == 0
        assert pool.blocks[block_id] not in after

    # Next write is the incomplete HOT tail, not a WARM/null slot.
    next_logical = request.num_computed_tokens // BLOCK_SIZE
    assert next_logical == COMPLETE_HISTORICAL_BLOCKS
    assert after[next_logical] is tail
    assert after[next_logical].is_null is False
    assert sum(blk.is_null for blk in after) == COMPLETE_HISTORICAL_BLOCKS
    assert sum(not blk.is_null for blk in after) == 1

    # Logical nulls still occupy table slots, so filling the tail needs 0 new
    # physical blocks until cdiv exceeds len(req_blocks).
    num_to_alloc_same_len = manager.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=TURN1_TOKENS,
        new_computed_blocks=manager.empty_kv_cache_blocks.blocks,
        num_encoder_tokens=0,
        total_computed_tokens=TURN1_TOKENS,
        num_tokens_main_model=TURN1_TOKENS,
    )
    assert num_to_alloc_same_len == 0

    turn2 = manager.allocate_slots(request, TURN2_TOKENS)
    assert turn2 is not None
    turn2_ids = turn2.get_block_ids()[0]
    assert len(turn2_ids) == 2
    assert all(block_id != null_block.block_id for block_id in turn2_ids)
    request.append_output_token_ids(list(range(TURN2_TOKENS)))
    request.num_computed_tokens = TURN1_TOKENS + TURN2_TOKENS

    after_turn2 = manager.get_blocks(request.request_id).blocks[0]
    assert len(after_turn2) == 49
    assert sum(1 for blk in after_turn2 if blk.is_null) == COMPLETE_HISTORICAL_BLOCKS
    assert sum(1 for blk in after_turn2 if not blk.is_null) == 3

    # First extra HOT block after the remaining tail + turn-2 prefill fills.
    # 16 * 49 = 784, so token 785 needs logical block index 49.
    tokens_at_boundary = 49 * BLOCK_SIZE
    already = request.num_computed_tokens
    assert already == 773
    fill = tokens_at_boundary - already
    filled = manager.allocate_slots(request, fill)
    assert filled is not None
    assert filled.get_block_ids(allow_none=True) is None
    request.num_computed_tokens = tokens_at_boundary

    crossing = manager.allocate_slots(request, 1)
    assert crossing is not None
    crossing_ids = crossing.get_block_ids()[0]
    assert len(crossing_ids) == 1
    assert crossing_ids[0] != null_block.block_id
    new_block = manager.get_blocks(request.request_id).blocks[0][-1]
    assert new_block.is_null is False
    assert new_block.block_id == crossing_ids[0]
    assert len(manager.get_blocks(request.request_id).blocks[0]) == 50
