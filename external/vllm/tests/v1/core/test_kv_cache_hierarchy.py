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
) -> KVCacheManager:
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.get_blocks = Mock(return_value=KVCacheBlocks(block_groups))
    manager.block_sizes = block_sizes
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
