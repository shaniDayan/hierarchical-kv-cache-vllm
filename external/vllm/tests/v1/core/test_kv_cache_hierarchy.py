# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_state import KVBlockState


def make_manager(
    block_groups: tuple[list[KVCacheBlock], ...],
) -> KVCacheManager:
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.get_blocks = Mock(return_value=KVCacheBlocks(block_groups))
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


def test_apply_warm_across_groups_and_skip_ineligible_blocks():
    warm_block = make_block(1, state=KVBlockState.WARM)
    null_block = make_block(2, is_null=True)
    unreferenced_block = make_block(3, ref_cnt=0)
    shared_warm_block = make_block(4, state=KVBlockState.WARM, ref_cnt=2)
    shared_hot_block = make_block(5, ref_cnt=2)
    groups = (
        [
            make_block(0),
            warm_block,
            null_block,
            unreferenced_block,
            shared_warm_block,
            shared_hot_block,
        ],
        [make_block(10)],
        [make_block(20, state=KVBlockState.WARM, ref_cnt=0)],
    )
    manager = make_manager(groups)

    changed = manager.apply_request_kv_state("request", KVBlockState.WARM)

    assert changed == ([0, 4], [10], [])
    assert groups[0][0].hierarchy_state is KVBlockState.WARM
    assert warm_block.hierarchy_state is KVBlockState.WARM
    assert null_block.hierarchy_state is KVBlockState.HOT
    assert unreferenced_block.hierarchy_state is KVBlockState.HOT
    assert shared_warm_block.hierarchy_state is KVBlockState.HOT
    assert shared_hot_block.hierarchy_state is KVBlockState.HOT
    assert groups[1][0].hierarchy_state is KVBlockState.WARM
    manager.get_blocks.assert_called_once_with("request")


def test_apply_cold_changes_all_eligible_blocks():
    groups = ([make_block(0), make_block(1)], [make_block(10)])
    manager = make_manager(groups)

    changed = manager.apply_request_kv_state("request", KVBlockState.COLD)

    assert changed == ([0, 1], [10])
    assert all(
        block.hierarchy_state is KVBlockState.COLD
        for group in groups
        for block in group
    )


def test_apply_state_is_idempotent_and_hot_promotes_eligible_blocks():
    groups = ([make_block(0), make_block(1)], [make_block(10)])
    manager = make_manager(groups)

    first_changed = manager.apply_request_kv_state("request", KVBlockState.WARM)
    second_changed = manager.apply_request_kv_state("request", KVBlockState.WARM)
    promoted = manager.apply_request_kv_state("request", KVBlockState.HOT)

    assert first_changed == ([0, 1], [10])
    assert second_changed == ([], [])
    assert promoted == ([0, 1], [10])
    assert all(
        block.hierarchy_state is KVBlockState.HOT for group in groups for block in group
    )
