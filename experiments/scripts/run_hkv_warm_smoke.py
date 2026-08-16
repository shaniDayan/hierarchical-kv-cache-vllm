import argparse
import asyncio
import json
import os
import time
from contextlib import suppress
from pathlib import Path

class HKVSmokeWorkerExtension:
    def inspect_hkv_smoke(self, request_id, action="inspect"):
        import torch
        runner = self.model_runner
        manager = runner.hkv_warm_migration_manager
        if action == "repeat":
            from vllm.v1.kv_cache_state import KVCacheBlockTransition
            from vllm.v1.worker.gpu import attn_utils
            before = self.inspect_hkv_smoke(request_id)
            index = runner.req_states.req_id_to_index[request_id]
            groups = [[] for _ in runner.block_tables.block_tables]
            for item in before["warm_residency"]:
                _, group, logical = item["key"]
                groups[group].append(KVCacheBlockTransition(
                    logical, item["temporary_shadow_hot_block_id"]))
            current = tuple(tuple(table.gpu[index, :int(count)].tolist())
                            for table, count in zip(runner.block_tables.block_tables,
                            runner.block_tables.num_blocks.np[:, index], strict=True))
            real_quantize = attn_utils.quantize_hkv_blocks_to_warm
            quantize_calls = 0
            def counted_quantize(**kwargs):
                nonlocal quantize_calls
                quantize_calls += 1
                return real_quantize(**kwargs)
            attn_utils.quantize_hkv_blocks_to_warm = counted_quantize
            try:
                manager.migrate(request_id, tuple(groups), current)
            finally:
                attn_utils.quantize_hkv_blocks_to_warm = real_quantize
            after = self.inspect_hkv_smoke(request_id)
            slots_unchanged = before["request_slot_ownership"] == after["request_slot_ownership"]
            residency_unchanged = before["warm_residency"] == after["warm_residency"]
            after["repeat_idempotency"] = {
                "quantization_calls": quantize_calls, "slot_ownership_unchanged": slots_unchanged,
                "residency_unchanged": residency_unchanged,
                "passed": quantize_calls == 0 and slots_unchanged and residency_unchanged,
            }
            return after
        if action == "release":
            before = self.inspect_hkv_smoke(request_id)
            released_hot_ids = sorted(item["temporary_shadow_hot_block_id"] for item in before["warm_residency"])
            released_slots = manager.release_request(request_id)
            after = self.inspect_hkv_smoke(request_id)
            projection = next(iter(runner.hkv_hot_to_warm_maps.values()))
            projection_values = projection[released_hot_ids].cpu().tolist()
            after["release_result"] = {
                "released_slots": list(released_slots),
                "released_projection_values": list(zip(released_hot_ids, projection_values, strict=True)),
                "no_residency": not after["warm_residency"],
                "no_owned_slots": not after["request_slot_ownership"],
                "projections_invalidated": all(value == -1 for value in projection_values),
            }
            release = after["release_result"]
            release["passed"] = all(release[key] for key in ("no_residency", "no_owned_slots", "projections_invalidated"))
            return after
        mixed_read_enabled = (
            os.getenv("HKV_DEBUG_MIXED_READ", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        maps = [value.cpu().tolist() for value in runner.hkv_hot_to_warm_maps.values()]
        assert not maps or all(value == maps[0] for value in maps)
        values = maps[0] if maps else []
        allocator = manager.allocator if manager is not None else None
        residency_items = () if manager is None else sorted(manager.warm_residency.items())
        allocator_mappings = [] if allocator is None else [
            [*key, allocator.lookup(key)] for key, _ in residency_items]
        index = runner.req_states.req_id_to_index.get(request_id)
        blocks, computed = [], None
        if index is not None:
            count = int(runner.block_tables.num_blocks.np[0, index])
            table = runner.block_tables.block_tables[0].gpu
            blocks = table[index, :count].cpu().tolist()
            computed = int(runner.req_states.num_computed_tokens_np[index])
        residency = [] if manager is None else [
            {"key": list(key), "warm_slot_id": entry.warm_slot_id,
             "temporary_shadow_hot_block_id": entry.temporary_shadow_hot_block_id}
            for key, entry in residency_items
            if key[0] == request_id
        ]
        request_slots = [] if allocator is None else [
            [key[1], key[2], allocator.lookup(key)]
            for key, _ in residency_items if key[0] == request_id]
        allocator_matches = allocator is None or all(
            allocator.lookup(key) == entry.warm_slot_id for key, entry in residency_items)
        ownership_count_matches = allocator is None or (
            allocator.num_owned_slots == len(residency_items))
        hot_ids = sorted(item["temporary_shadow_hot_block_id"] for item in residency)
        hot_cache_checks = []
        for name, cache in runner.hkv_hot_kv_caches.items():
            in_range = all(block < cache.shape[0] for block in hot_ids)
            samples = cache[hot_ids, 0, 0, 0, 0].cpu().tolist() if in_range else []
            hot_cache_checks.append({"name": name, "storage_bytes": cache.untyped_storage().nbytes(),
                                     "ids_in_range": in_range, "readable_samples": samples})
        projection_agrees = all(values[item["temporary_shadow_hot_block_id"]] == item["warm_slot_id"] for item in residency)
        residency_matches_blocks = all(item["key"][1] == 0 and blocks[item["key"][2]] == item["temporary_shadow_hot_block_id"] for item in residency)
        return {
            "request_id": request_id, "num_computed_tokens": computed,
            "block_ids": blocks,
            "mixed_read_enabled": mixed_read_enabled,
            "warm_mappings": [[i, slot] for i, slot in enumerate(values) if slot >= 0],
            "allocator_mappings": allocator_mappings,
            "request_slot_ownership": request_slots,
            "allocator_matches_residency": allocator_matches,
            "ownership_count_matches": ownership_count_matches,
            "warm_residency": residency,
            "projection_matches_residency": projection_agrees,
            "residency_matches_block_table": residency_matches_blocks,
            "hot_copy_retained": bool(hot_ids) and all(block in blocks for block in hot_ids)
                                 and all(check["storage_bytes"] > 0 and check["ids_in_range"] for check in hot_cache_checks),
            "hot_cache_checks": hot_cache_checks,
            "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
async def run(args):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    engine_args = AsyncEngineArgs(
        model=args.model, dtype="float16", enforce_eager=True, seed=0,
        max_model_len=1024, gpu_memory_utilization=0.6, block_size=16,
        attention_backend="TRITON_ATTN",
        worker_extension_cls=(
            "experiments.scripts.run_hkv_warm_smoke.HKVSmokeWorkerExtension"
        ),
        kv_cache_hot_idle_threshold_seconds=0.05,
        kv_cache_cold_idle_threshold_seconds=3600.0,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    tokens = engine.get_tokenizer().encode(
        "Time-aware KV caches preserve useful history. " * 12
    )
    history, continuation = tokens[:37], tokens[37:57]
    historical_token_count, continuation_token_count = len(history), len(continuation)
    assert historical_token_count == 37 and continuation_token_count == 20
    common = dict(temperature=0.0, seed=0, ignore_eos=True,
                  output_kind=RequestOutputKind.DELTA)
    first = SamplingParams(max_tokens=1, **common)
    second = SamplingParams(max_tokens=16, **common)
    keep_params = SamplingParams(max_tokens=512, **common)
    resume, turn_two_observed, turn = asyncio.Event(), asyncio.Event(), 1
    first_yielded, continuation_yielded, output_counts = False, False, {1: 0, 2: 0}
    async def inputs():
        nonlocal turn, first_yielded, continuation_yielded
        first_yielded = True
        yield StreamingInput({"prompt_token_ids": list(history)}, first)
        await resume.wait()
        turn = 2
        continuation_yielded = True
        yield StreamingInput({"prompt_token_ids": list(continuation)}, second)
        await turn_two_observed.wait()
    async def drain(generator):
        async for _ in generator:
            pass
    keep = asyncio.create_task(drain(engine.generate(
        "Keep scheduling while the session is idle.", keep_params, "hkv-keepalive"
    )))
    token_ids, text, demotion, resumed = [], "", None, None
    repeat_check, before_release, after_release = None, None, None
    started, idle_elapsed = time.perf_counter(), 0.0
    try:
        async for output in engine.generate(inputs(), first, "hkv-target"):
            if not output.outputs:
                continue
            output_counts[turn] += 1
            completion = output.outputs[0]
            token_ids.extend(completion.token_ids)
            text += completion.text
            if turn == 1 and completion.finish_reason:
                idle_start = time.perf_counter()
                await asyncio.sleep(0.25)
                idle_elapsed += time.perf_counter() - idle_start
                demotion = (await engine.engine_core.collective_rpc_async(
                    "inspect_hkv_smoke", args=("hkv-target",)))[0]
                resume.set()
            elif turn == 2:
                if resumed is None:
                    resumed = (await engine.engine_core.collective_rpc_async(
                        "inspect_hkv_smoke", args=("hkv-target",)))[0]
                if completion.finish_reason and before_release is None:
                    if args.mode == "mixed":
                        repeat_check = (await engine.engine_core.collective_rpc_async(
                            "inspect_hkv_smoke", args=("hkv-target", "repeat")))[0]
                        before_release = repeat_check
                        after_release = (await engine.engine_core.collective_rpc_async(
                            "inspect_hkv_smoke", args=("hkv-target", "release")))[0]
                    turn_two_observed.set()
    finally:
        keep.cancel()
        with suppress(asyncio.CancelledError):
            await keep
        engine.shutdown()
    if not (demotion and resumed and first_yielded and continuation_yielded
            and output_counts[2]):
        raise RuntimeError(
            "incomplete streaming lifecycle: "
            f"demotion_present={demotion is not None}, "
            f"resumed_present={resumed is not None}, "
            f"first_chunk_yielded={first_yielded}, "
            f"continuation_yielded={continuation_yielded}, current_turn={turn}, "
            f"turn_1_outputs={output_counts[1]}, turn_2_outputs={output_counts[2]}")
    mappings = dict(demotion["warm_mappings"])
    resumed_mappings = dict(resumed["warm_mappings"])
    mapped_block_ids_at_resume = sorted(
        block_id
        for block_id in resumed["block_ids"]
        if block_id in resumed_mappings
    )
    hot_block_ids_at_resume = sorted(
        block_id
        for block_id in resumed["block_ids"]
        if block_id not in resumed_mappings
    )
    pre_resume_mappings_persisted = bool(mappings) and all(
        resumed_mappings.get(block_id) == slot
        for block_id, slot in mappings.items()
    )
    expected_complete_historical_blocks = historical_token_count // 16  # Excludes first generated final token.
    assert expected_complete_historical_blocks >= 2
    if args.mode == "mixed":
        assert demotion["mixed_read_enabled"] and resumed["mixed_read_enabled"]
        assert len(mappings) >= 2 and len(set(mappings.values())) == len(mappings)
        assert pre_resume_mappings_persisted
        assert set(mappings).issubset(mapped_block_ids_at_resume)
        assert hot_block_ids_at_resume
        assert demotion["projection_matches_residency"] and demotion["residency_matches_block_table"] and demotion["hot_copy_retained"] and demotion["allocator_matches_residency"] and demotion["ownership_count_matches"]
        assert repeat_check["repeat_idempotency"]["passed"] and repeat_check["allocator_matches_residency"] and repeat_check["ownership_count_matches"] and before_release["projection_matches_residency"]
        assert before_release["hot_copy_retained"] and before_release["allocator_matches_residency"] and before_release["ownership_count_matches"] and after_release["release_result"]["passed"] and after_release["allocator_matches_residency"] and after_release["ownership_count_matches"]
    else:
        assert not demotion["mixed_read_enabled"]
        assert not resumed["mixed_read_enabled"]
        assert not mappings and not resumed_mappings
    elapsed = time.perf_counter() - started - idle_elapsed
    result = {
        "mode": args.mode, "request_id": "hkv-target", "transition": "HOT->WARM",
        "historical_token_count": historical_token_count, "continuation_token_count": continuation_token_count, "expected_complete_historical_blocks": expected_complete_historical_blocks,
        "demotion": demotion, "observed_migrated_block_ids": sorted(mappings),
        "migration_evidence": "post-migration GPU HOT-to-WARM map",
        "migrated_block_slots": demotion["allocator_mappings"],
        "warm_quantization_and_mapping_observed": bool(mappings),
        "mixed_attention_read_enabled": demotion["mixed_read_enabled"],
        "mixed_attention_read_evidence": {
            "pre_resume_mappings_persisted": pre_resume_mappings_persisted,
            "warm_block_ids_in_resumed_block_table": mapped_block_ids_at_resume,
            "hot_block_ids_in_resumed_block_table": hot_block_ids_at_resume,
        },
        "before_release": before_release,
        "repeat_idempotency": None if repeat_check is None else repeat_check["repeat_idempotency"],
        "after_release": after_release,
        "partial_tail_evidence": "Scheduler first transitioned two complete blocks; a third block transitioned only after continuation (verified in scheduler log).",
        "after_resume": resumed, "generated_token_ids": token_ids,
        "generated_text": text, "generation_time_seconds": elapsed,
        "tokens_per_second": len(token_ids) / elapsed,
        "peak_allocated_bytes": resumed["max_gpu_allocated_bytes"], "peak_reserved_bytes": resumed["max_gpu_reserved_bytes"],
        "physical_migration_enabled": args.mode == "mixed", "migration_retains_hot_copy": bool(before_release and before_release["hot_copy_retained"]), "first_chunk_yielded": first_yielded, "continuation_yielded": continuation_yielded, "turn_output_counts": output_counts,
    }
    if args.baseline_json:
        expected = json.loads(Path(args.baseline_json).read_text())["generated_token_ids"]
        aligned = list(zip(expected, token_ids))
        matching = sum(baseline == mixed for baseline, mixed in aligned)
        first_mismatch = next((i for i, pair in enumerate(aligned)
                               if pair[0] != pair[1]), None)
        if first_mismatch is None and len(expected) != len(token_ids):
            first_mismatch = len(aligned)
        result.update({
            "exact_token_match": expected == token_ids,
            "matching_aligned_tokens": matching, "compared_aligned_tokens": len(aligned),
            "token_match_percent": 100 * matching / len(aligned) if aligned else 0.0,
            "first_mismatch_position": first_mismatch,
            "baseline_token_count": len(expected), "mixed_token_count": len(token_ids),
        })
        assert result["exact_token_match"]
    path = Path(args.result_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all-hot", "mixed"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--baseline-json")
    args = parser.parse_args()
    if args.mode == "mixed" and not args.baseline_json:
        parser.error("--baseline-json is required for Mixed mode")
    os.environ.update({
        "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "HKV_ENABLE_PHYSICAL_TIERS": "1", "HKV_WARM_POOL_BLOCKS": "16",
        "HKV_DEBUG_DEMOTE_ONE_BLOCK": "0",
        "HKV_ENABLE_MULTI_BLOCK_WARM_MIGRATION": "1" if args.mode == "mixed" else "0",
        "HKV_DEBUG_MIXED_READ": "1" if args.mode == "mixed" else "0",
    })
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
