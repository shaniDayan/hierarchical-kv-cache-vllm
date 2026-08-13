# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_state import KVBlockState
from vllm.v1.request import RequestStatus


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


def make_allocation_manager(
    request_blocks: tuple[list[KVCacheBlock], ...],
    new_blocks: tuple[list[KVCacheBlock], ...],
    *,
    num_free_blocks: int = 10,
    num_blocks_to_allocate: int = 1,
) -> KVCacheManager:
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.max_model_len = 32
    manager.watermark_blocks = 0
    manager.enable_caching = False
    manager.empty_kv_cache_blocks = KVCacheBlocks(((),))
    manager.block_pool = Mock()
    manager.block_pool.get_num_free_blocks.return_value = num_free_blocks
    manager.coordinator = Mock()
    manager.coordinator.get_num_blocks_to_allocate.return_value = num_blocks_to_allocate
    manager.coordinator.allocate_new_blocks.return_value = new_blocks
    manager.coordinator.get_blocks.return_value = request_blocks
    manager.apply_request_kv_state = Mock(wraps=manager.apply_request_kv_state)
    return manager


def make_request(request_id: str, state: KVBlockState):
    return SimpleNamespace(
        request_id=request_id,
        kv_cache_state=state,
        num_computed_tokens=0,
        num_tokens=5,
        status=RequestStatus.WAITING,
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


def test_allocate_slots_applies_request_state_to_new_and_prefix_hit_blocks():
    prefix_block = make_block(0)
    new_block = make_block(1)
    manager = make_allocation_manager(
        request_blocks=([prefix_block, new_block],),
        new_blocks=([new_block],),
    )
    request = make_request("request", KVBlockState.WARM)

    allocated = manager.allocate_slots(
        request,
        num_new_tokens=1,
        num_new_computed_tokens=4,
        new_computed_blocks=KVCacheBlocks(([prefix_block],)),
    )

    assert allocated is not None
    assert allocated.get_block_ids() == ([1],)
    manager.apply_request_kv_state.assert_called_once_with("request", KVBlockState.WARM)
    assert prefix_block.hierarchy_state is KVBlockState.WARM
    assert new_block.hierarchy_state is KVBlockState.WARM


def test_allocate_slots_failure_does_not_apply_request_state():
    block = make_block(0)
    manager = make_allocation_manager(
        request_blocks=([block],),
        new_blocks=([],),
        num_free_blocks=0,
        num_blocks_to_allocate=1,
    )
    request = make_request("request", KVBlockState.COLD)

    allocated = manager.allocate_slots(request, num_new_tokens=1)

    assert allocated is None
    manager.apply_request_kv_state.assert_not_called()
    assert block.hierarchy_state is KVBlockState.HOT
