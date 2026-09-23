# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_state import KVBlockState, KVCacheBlockTransition


def make_manager(
    block_groups: tuple[list[KVCacheBlock], ...],
    block_sizes: tuple[int, ...],
    *,
    num_gpu_blocks: int = 64,
    enable_caching: bool = False,
) -> KVCacheManager:
    from vllm.v1.core.block_pool import BlockPool

    manager = KVCacheManager.__new__(KVCacheManager)
    manager.get_blocks = Mock(return_value=KVCacheBlocks(block_groups))
    manager.block_sizes = block_sizes
    manager.block_pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=enable_caching,
        hash_block_size=block_sizes[0],
    )
    manager.enable_caching = enable_caching
    return manager


def make_block(
    block_id: int,
    *,
    state: KVBlockState = KVBlockState.HOT,
    ref_cnt: int = 1,
    is_null: bool = False,
) -> KVCacheBlock:
    return KVCacheBlock(
        block_id=block_id,
        hierarchy_state=state,
        ref_cnt=ref_cnt,
        is_null=is_null,
    )


def test_apply_request_state_private_shared_and_idempotent():
    private = make_block(0)
    unchanged = make_block(1, state=KVBlockState.WARM)
    shared = make_block(2, state=KVBlockState.WARM, ref_cnt=2)
    null = make_block(3, is_null=True)
    unreferenced = make_block(4, ref_cnt=0)
    groups = ([private, unchanged, shared, null, unreferenced], [make_block(10)])
    manager = make_manager(groups, (4, 4))

    assert manager.apply_request_kv_state("request", KVBlockState.WARM) == (
        [KVCacheBlockTransition(logical_block_index=0, source_hot_block_id=0)],
        [KVCacheBlockTransition(logical_block_index=0, source_hot_block_id=10)],
    )
    assert private.hierarchy_state is KVBlockState.WARM
    assert unchanged.hierarchy_state is KVBlockState.WARM
    assert shared.hierarchy_state is KVBlockState.HOT
    assert null.hierarchy_state is KVBlockState.HOT
    assert unreferenced.hierarchy_state is KVBlockState.HOT
    assert manager.apply_request_kv_state("request", KVBlockState.WARM) == (
        [],
        [],
    )


@pytest.mark.parametrize(
    ("block_sizes", "num_computed_tokens", "expected"),
    [
        ((4,), 0, ([],)),
        ((4,), 3, ([],)),
        ((4,), 4, ([(0, 0)],)),
        ((4,), 5, ([(0, 0)],)),
        ((4,), 8, ([(0, 0), (1, 1)],)),
        ((4, 8), 8, ([(0, 0), (1, 1)], [(0, 10)])),
    ],
)
def test_apply_demotion_uses_complete_blocks_per_group(
    block_sizes: tuple[int, ...],
    num_computed_tokens: int,
    expected: tuple[list[tuple[int, int]], ...],
):
    groups = tuple(
        [make_block(group * 10 + index) for index in range(2)]
        for group in range(len(block_sizes))
    )
    manager = make_manager(groups, block_sizes)

    changed = manager.apply_request_kv_state(
        "request",
        KVBlockState.COLD,
        num_computed_tokens=num_computed_tokens,
    )

    expected_transitions = tuple(
        [KVCacheBlockTransition(index, block_id) for index, block_id in group]
        for group in expected
    )
    assert changed == expected_transitions
    for group, changed_group in zip(groups, expected_transitions, strict=True):
        changed_ids = {block.source_hot_block_id for block in changed_group}
        assert [block.hierarchy_state for block in group] == [
            KVBlockState.COLD
            if block.block_id in changed_ids
            else KVBlockState.HOT
            for block in group
        ]


def test_plan_request_kv_state_non_mutating():
    private = make_block(0)
    unchanged = make_block(1, state=KVBlockState.WARM)
    shared = make_block(2, state=KVBlockState.HOT, ref_cnt=2)
    null = make_block(3, is_null=True)
    unreferenced = make_block(4, ref_cnt=0)
    groups = ([private, unchanged, shared, null, unreferenced], [make_block(10)])
    manager = make_manager(groups, (4, 4))

    planned = manager.plan_request_kv_state("request", KVBlockState.WARM)
    assert planned == (
        [KVCacheBlockTransition(logical_block_index=0, source_hot_block_id=0)],
        [KVCacheBlockTransition(logical_block_index=0, source_hot_block_id=10)],
    )
    # Planning must NOT mutate block hierarchy states
    assert private.hierarchy_state is KVBlockState.HOT
    assert unchanged.hierarchy_state is KVBlockState.WARM
    assert shared.hierarchy_state is KVBlockState.HOT
    assert null.hierarchy_state is KVBlockState.HOT
    assert unreferenced.hierarchy_state is KVBlockState.HOT


def test_commit_request_kv_transition_success_reclaims_hot_blocks():
    from vllm.v1.kv_cache_state import KVCacheStateTransition

    manager = make_manager(([],), (4,), num_gpu_blocks=16)
    pool = manager.block_pool
    blk0, blk1, blk2 = pool.get_new_blocks(3)
    groups = ([blk0, blk1, blk2],)
    manager.get_blocks = Mock(return_value=KVCacheBlocks(groups))
    free_before = pool.get_num_free_blocks()

    transition = KVCacheStateTransition(
        transition_id=1,
        request_id="request",
        previous_state=KVBlockState.HOT,
        new_state=KVBlockState.WARM,
        changed_blocks=([
            KVCacheBlockTransition(0, blk0.block_id),
            KVCacheBlockTransition(1, blk1.block_id),
        ],),
    )

    manager.commit_request_kv_transition(transition)

    # Migrated logical entries become null_block placeholders
    assert groups[0][0] is pool.null_block
    assert groups[0][0].is_null is True
    assert groups[0][1] is pool.null_block
    assert groups[0][1].is_null is True

    # Tail/unmigrated block remains untouched HOT block
    assert groups[0][2] is blk2
    assert blk2.is_null is False
    assert blk2.ref_cnt == 1
    assert blk2.hierarchy_state is KVBlockState.HOT

    # Reclaimed physical blocks returned to pool
    assert blk0.ref_cnt == 0
    assert blk1.ref_cnt == 0
    assert pool.get_num_free_blocks() == free_before + 2

    # Physical blocks are immediately reusable
    reused = pool.get_new_blocks(2)
    assert {b.block_id for b in reused} == {blk0.block_id, blk1.block_id}
    assert all(b.ref_cnt == 1 and b.hierarchy_state is KVBlockState.HOT for b in reused)


def test_commit_request_kv_transition_fail_closed_validations():
    from vllm.v1.kv_cache_state import KVCacheStateTransition

    manager = make_manager(([],), (4,), num_gpu_blocks=16)
    pool = manager.block_pool
    blk0, blk1 = pool.get_new_blocks(2)
    shared_blk = pool.get_new_blocks(1)[0]
    shared_blk.ref_cnt = 2
    groups = ([blk0, blk1, shared_blk],)
    manager.get_blocks = Mock(return_value=KVCacheBlocks(groups))
    free_before = pool.get_num_free_blocks()

    # 1. Stale physical block ID mismatch
    with pytest.raises(ValueError, match="Physical block ID mismatch"):
        manager.commit_request_kv_transition(
            KVCacheStateTransition(
                transition_id=1,
                request_id="request",
                previous_state=KVBlockState.HOT,
                new_state=KVBlockState.WARM,
                changed_blocks=([KVCacheBlockTransition(0, 999)],),
            )
        )

    # 2. Out of range logical block index
    with pytest.raises(ValueError, match="Logical block index 5 out of range"):
        manager.commit_request_kv_transition(
            KVCacheStateTransition(
                transition_id=2,
                request_id="request",
                previous_state=KVBlockState.HOT,
                new_state=KVBlockState.WARM,
                changed_blocks=([KVCacheBlockTransition(5, blk0.block_id)],),
            )
        )

    # 3. Out of range cache group index
    with pytest.raises(ValueError, match="Transition has 2 groups but request"):
        manager.commit_request_kv_transition(
            KVCacheStateTransition(
                transition_id=3,
                request_id="request",
                previous_state=KVBlockState.HOT,
                new_state=KVBlockState.WARM,
                changed_blocks=([], [KVCacheBlockTransition(0, blk0.block_id)]),
            )
        )

    # 4. Reclaiming a shared block (ref_cnt > 1) is rejected
    with pytest.raises(ValueError, match="Cannot reclaim shared or unreferenced block"):
        manager.commit_request_kv_transition(
            KVCacheStateTransition(
                transition_id=4,
                request_id="request",
                previous_state=KVBlockState.HOT,
                new_state=KVBlockState.WARM,
                changed_blocks=([KVCacheBlockTransition(2, shared_blk.block_id)],),
            )
        )

    # All failures leave state completely unmodified
    assert groups[0][0] is blk0 and blk0.ref_cnt == 1
    assert groups[0][1] is blk1 and blk1.ref_cnt == 1
    assert groups[0][2] is shared_blk and shared_blk.ref_cnt == 2
    assert pool.get_num_free_blocks() == free_before


def test_commit_request_kv_transition_evicts_prefix_cache_before_reclaim():
    from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
    from vllm.v1.kv_cache_state import KVCacheStateTransition

    manager = make_manager(([],), (4,), num_gpu_blocks=16, enable_caching=True)
    pool = manager.block_pool
    blk0 = pool.get_new_blocks(1)[0]
    groups = ([blk0],)
    manager.get_blocks = Mock(return_value=KVCacheBlocks(groups))

    # Set up a prefix-cache hash entry on the block
    hash_key = make_block_hash_with_group_id(BlockHash(b"hkv_hash_key_1234"), 0)
    pool._insert_block_hash(hash_key, blk0, num_tokens=4)
    assert pool.cached_block_hash_to_block.contain(hash_key, blk0.block_id)
    assert blk0.block_hash == hash_key

    transition = KVCacheStateTransition(
        transition_id=1,
        request_id="request",
        previous_state=KVBlockState.HOT,
        new_state=KVBlockState.WARM,
        changed_blocks=([KVCacheBlockTransition(0, blk0.block_id)],),
    )

    manager.commit_request_kv_transition(transition)

    # Hash entry is evicted from APC lookup table
    assert not pool.cached_block_hash_to_block.contain(hash_key, blk0.block_id)
    assert blk0.block_hash is None
    assert groups[0][0] is pool.null_block
    assert blk0.ref_cnt == 0


def test_finish_mixed_null_and_hot_blocks_no_double_free_or_ref_cnt_corruption():
    from vllm.v1.kv_cache_state import KVCacheStateTransition

    manager = make_manager(([],), (4,), num_gpu_blocks=16)
    pool = manager.block_pool
    blk0, blk1, tail_blk = pool.get_new_blocks(3)
    groups = ([blk0, blk1, tail_blk],)
    manager.get_blocks = Mock(return_value=KVCacheBlocks(groups))

    # 1. Commit migration of blk0 and blk1 to WARM
    transition = KVCacheStateTransition(
        transition_id=1,
        request_id="request",
        previous_state=KVBlockState.HOT,
        new_state=KVBlockState.WARM,
        changed_blocks=([
            KVCacheBlockTransition(0, blk0.block_id),
            KVCacheBlockTransition(1, blk1.block_id),
        ],),
    )
    manager.commit_request_kv_transition(transition)

    assert groups[0][0] is pool.null_block
    assert groups[0][1] is pool.null_block
    assert groups[0][2] is tail_blk
    free_after_migration = pool.get_num_free_blocks()
    null_ref_before = pool.null_block.ref_cnt

    # 2. Free request blocks on finish
    req_blocks = groups[0]
    pool.free_blocks(reversed(req_blocks))

    # Only tail_blk is freed; blk0 and blk1 are not freed a second time
    assert tail_blk.ref_cnt == 0
    assert pool.get_num_free_blocks() == free_after_migration + 1

    # null_block.ref_cnt must not be decremented / corrupted
    assert pool.null_block.ref_cnt == null_ref_before
