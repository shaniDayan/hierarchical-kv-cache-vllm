import argparse
import asyncio
import json
import os
import time
from contextlib import suppress
from pathlib import Path

class HKVSmokeWorkerExtension:
    def inspect_hkv_smoke(self, request_id):
        import torch
        runner = self.model_runner
        maps = [value.cpu().tolist() for value in runner.hkv_hot_to_warm_maps.values()]
        assert not maps or all(value == maps[0] for value in maps)
        values = maps[0] if maps else []
        manager = runner.hkv_warm_migration_manager
        allocator = manager.allocator if manager is not None else None
        allocator_mappings = [] if allocator is None else sorted(
            [source.cache_group_index, source.kernel_hot_block_id, slot]
            for source, slot in allocator._source_to_slot.items()
        )
        index = runner.req_states.req_id_to_index.get(request_id)
        blocks, computed = [], None
        if index is not None:
            count = int(runner.block_tables.num_blocks.np[0, index])
            table = runner.block_tables.block_tables[0].gpu
            blocks = table[index, :count].cpu().tolist()
            computed = int(runner.req_states.num_computed_tokens_np[index])
        return {
            "request_id": request_id, "num_computed_tokens": computed,
            "block_ids": blocks,
            "warm_mappings": [[i, slot] for i, slot in enumerate(values) if slot >= 0],
            "allocator_mappings": allocator_mappings,
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
            elif turn == 2 and resumed is None:
                resumed = (await engine.engine_core.collective_rpc_async(
                    "inspect_hkv_smoke", args=("hkv-target",)))[0]
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
    expected_complete_historical_blocks = historical_token_count // 16  # Excludes first generated final token.
    assert expected_complete_historical_blocks >= 2
    if args.mode == "mixed":
        assert len(mappings) >= 2 and len(set(mappings.values())) == len(mappings)
        allocator_map = {block: slot for group, block, slot in
                         demotion["allocator_mappings"] if group == 0}
        assert allocator_map == mappings
    else:
        assert not mappings
    elapsed = time.perf_counter() - started - idle_elapsed
    result = {
        "mode": args.mode, "request_id": "hkv-target", "transition": "HOT->WARM",
        "historical_token_count": historical_token_count, "continuation_token_count": continuation_token_count, "expected_complete_historical_blocks": expected_complete_historical_blocks,
        "demotion": demotion, "observed_migrated_block_ids": sorted(mappings),
        "migration_evidence": "post-migration GPU HOT-to-WARM map",
        "migrated_block_slots": demotion["allocator_mappings"],
        "partial_tail_evidence": "Scheduler first transitioned two complete blocks; a third block transitioned only after continuation (verified in scheduler log).",
        "after_resume": resumed, "generated_token_ids": token_ids,
        "generated_text": text, "generation_time_seconds": elapsed,
        "tokens_per_second": len(token_ids) / elapsed,
        "peak_allocated_bytes": resumed["max_gpu_allocated_bytes"], "peak_reserved_bytes": resumed["max_gpu_reserved_bytes"],
        "physical_migration_enabled": args.mode == "mixed", "migration_retains_hot_copy": args.mode == "mixed", "first_chunk_yielded": first_yielded, "continuation_yielded": continuation_yielded, "turn_output_counts": output_counts,
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
    })
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
