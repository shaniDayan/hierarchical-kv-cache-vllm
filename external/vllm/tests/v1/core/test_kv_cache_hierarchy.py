# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_state import KVBlockState


def make_manager(
    block_groups: tuple[list[KVCacheBlock], ...],
    block_sizes: tuple[int, ...] | None = None,
) -> KVCacheManager:
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.get_blocks = Mock(return_value=KVCacheBlocks(block_groups))
    manager.block_sizes = block_sizes or (16,) * len(block_groups)
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

    assert changed == ([0], [10], [])
    assert groups[0][0].hierarchy_state is KVBlockState.WARM
    assert warm_block.hierarchy_state is KVBlockState.WARM
    assert null_block.hierarchy_state is KVBlockState.HOT
    assert unreferenced_block.hierarchy_state is KVBlockState.HOT
    assert shared_warm_block.hierarchy_state is KVBlockState.HOT
    assert shared_hot_block.hierarchy_state is KVBlockState.HOT
    assert 4 not in changed[0]
    assert 5 not in changed[0]
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


@pytest.mark.parametrize("state", [KVBlockState.WARM, KVBlockState.COLD])
@pytest.mark.parametrize(
    ("num_computed_tokens", "expected_changed"),
    [
        (0, []),
        (3, []),
        (4, [0]),
        (5, [0]),
        (8, [0, 1]),
    ],
)
def test_apply_demotion_only_changes_complete_blocks(
    state: KVBlockState,
    num_computed_tokens: int,
    expected_changed: list[int],
):
    groups = ([make_block(0), make_block(1)],)
    manager = make_manager(groups, block_sizes=(4,))

    changed = manager.apply_request_kv_state(
        "request",
        state,
        num_computed_tokens=num_computed_tokens,
    )

    assert changed == (expected_changed,)
    for block in groups[0]:
        expected_state = (
            state if block.block_id in expected_changed else KVBlockState.HOT
        )
        assert block.hierarchy_state is expected_state


def test_apply_demotion_keeps_complete_shared_block_hot():
    private_block = make_block(0)
    shared_block = make_block(1, state=KVBlockState.WARM, ref_cnt=2)
    groups = ([private_block, shared_block],)
    manager = make_manager(groups, block_sizes=(4,))

    changed = manager.apply_request_kv_state(
        "request",
        KVBlockState.COLD,
        num_computed_tokens=8,
    )

    assert changed == ([0],)
    assert private_block.hierarchy_state is KVBlockState.COLD
    assert shared_block.hierarchy_state is KVBlockState.HOT


def test_apply_hot_is_not_filtered_by_computed_token_boundary():
    groups = (
        [
            make_block(0, state=KVBlockState.WARM),
            make_block(1, state=KVBlockState.COLD),
        ],
    )
    manager = make_manager(groups, block_sizes=(4,))

    changed = manager.apply_request_kv_state(
        "request",
        KVBlockState.HOT,
        num_computed_tokens=0,
    )

    assert changed == ([0, 1],)
    assert all(block.hierarchy_state is KVBlockState.HOT for block in groups[0])


def test_apply_demotion_uses_each_cache_group_block_size():
    groups = (
        [make_block(0), make_block(1), make_block(2)],
        [make_block(10), make_block(11)],
    )
    manager = make_manager(groups, block_sizes=(4, 8))

    changed = manager.apply_request_kv_state(
        "request",
        KVBlockState.WARM,
        num_computed_tokens=8,
    )

    assert changed == ([0, 1], [10])
    assert [block.hierarchy_state for block in groups[0]] == [
        KVBlockState.WARM,
        KVBlockState.WARM,
        KVBlockState.HOT,
    ]
    assert [block.hierarchy_state for block in groups[1]] == [
        KVBlockState.WARM,
        KVBlockState.HOT,
    ]
