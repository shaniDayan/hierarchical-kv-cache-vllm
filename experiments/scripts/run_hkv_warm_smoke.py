import argparse
import asyncio
import json
import os
import time
from contextlib import suppress
from pathlib import Path


class HKVSmokeWorkerExtension:
    """Read-only worker extension to observe KV cache hierarchy state."""

    def inspect_hkv_smoke(self, request_id: str) -> dict:
        import torch

        runner = self.model_runner
        manager = getattr(runner, "hkv_warm_migration_manager", None)
        mixed_read_enabled = (
            os.getenv("HKV_DEBUG_MIXED_READ", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        allocator = manager.allocator if manager is not None else None
        residency_items = (
            () if manager is None else sorted(manager.warm_residency.items())
        )
        allocator_mappings = (
            []
            if allocator is None
            else [[*key, allocator.lookup(key)] for key, _ in residency_items]
        )
        index = runner.req_states.req_id_to_index.get(request_id)
        blocks, computed = [], None
        if index is not None:
            count = int(runner.block_tables.num_blocks.np[0, index])
            table = runner.block_tables.block_tables[0].gpu
            blocks = table[index, :count].cpu().tolist()
            computed = int(runner.req_states.num_computed_tokens_np[index])

        residency = (
            []
            if manager is None
            else [
                {
                    "key": list(key),
                    "warm_slot_id": entry.warm_slot_id,
                    "temporary_shadow_hot_block_id": entry.temporary_shadow_hot_block_id,
                }
                for key, entry in residency_items
                if key[0] == request_id
            ]
        )
        request_slots = (
            []
            if allocator is None
            else [
                [key[1], key[2], allocator.lookup(key)]
                for key, _ in residency_items
                if key[0] == request_id
            ]
        )
        allocator_matches = allocator is None or all(
            allocator.lookup(key) == entry.warm_slot_id
            for key, entry in residency_items
        )
        ownership_count_matches = allocator is None or (
            allocator.num_owned_slots == len(residency_items)
        )

        return {
            "request_id": request_id,
            "num_computed_tokens": computed,
            "block_ids": blocks,
            "mixed_read_enabled": mixed_read_enabled,
            "allocator_mappings": allocator_mappings,
            "request_slot_ownership": request_slots,
            "allocator_matches_residency": allocator_matches,
            "ownership_count_matches": ownership_count_matches,
            "warm_residency": residency,
            "max_gpu_allocated_bytes": (
                torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else 0
            ),
            "max_gpu_reserved_bytes": (
                torch.cuda.max_memory_reserved()
                if torch.cuda.is_available()
                else 0
            ),
        }


async def poll_hkv_state(
    engine,
    request_id: str,
    predicate,
    timeout: float = 2.0,
    poll_interval: float = 0.05,
) -> dict:
    """Poll the worker extension passively until a predicate is satisfied."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        state = (
            await engine.engine_core.collective_rpc_async(
                "inspect_hkv_smoke", args=(request_id,)
            )
        )[0]
        if predicate(state):
            return state
        await asyncio.sleep(poll_interval)
    return (
        await engine.engine_core.collective_rpc_async(
            "inspect_hkv_smoke", args=(request_id,)
        )
    )[0]


async def run(args):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    hot_idle_threshold = 0.05 if args.mode == "mixed" else None
    cold_idle_threshold = 3600.0 if args.mode == "mixed" else None

    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="float16",
        enforce_eager=True,
        seed=0,
        max_model_len=1024,
        gpu_memory_utilization=0.6,
        block_size=16,
        attention_backend="TRITON_ATTN",
        worker_extension_cls=(
            "experiments.scripts.run_hkv_warm_smoke.HKVSmokeWorkerExtension"
        ),
        kv_cache_hot_idle_threshold_seconds=hot_idle_threshold,
        kv_cache_cold_idle_threshold_seconds=cold_idle_threshold,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    tokens = engine.get_tokenizer().encode(
        "Time-aware KV caches preserve useful history. " * 12
    )
    history, continuation, followup = tokens[:37], tokens[37:57], tokens[57:60]
    historical_token_count, continuation_token_count, followup_token_count = (
        len(history),
        len(continuation),
        len(followup),
    )
    assert (
        historical_token_count == 37
        and continuation_token_count == 20
        and followup_token_count == 3
    )
    common = dict(
        temperature=0.0,
        seed=0,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )
    first = SamplingParams(max_tokens=1, **common)
    second = SamplingParams(max_tokens=16, **common)
    third = SamplingParams(max_tokens=1, **common)
    keep_params = SamplingParams(max_tokens=512, **common)
    resume = asyncio.Event()
    resume_third = asyncio.Event()
    turn_three_observed = asyncio.Event()
    turn = 1
    first_yielded = False
    continuation_yielded = False
    followup_yielded = False
    output_counts = {1: 0, 2: 0, 3: 0}

    async def inputs():
        nonlocal turn, first_yielded, continuation_yielded, followup_yielded
        first_yielded = True
        yield StreamingInput({"prompt_token_ids": list(history)}, first)
        await resume.wait()
        turn = 2
        continuation_yielded = True
        yield StreamingInput({"prompt_token_ids": list(continuation)}, second)
        await resume_third.wait()
        turn = 3
        followup_yielded = True
        yield StreamingInput({"prompt_token_ids": list(followup)}, third)
        await turn_three_observed.wait()

    async def drain(generator):
        async for _ in generator:
            pass

    keep = asyncio.create_task(
        drain(
            engine.generate(
                "Keep scheduling while the session is idle.",
                keep_params,
                "hkv-keepalive",
            )
        )
    )
    token_ids, text, demotion, resumed = [], "", None, None
    second_demotion, second_resumed, after_release = None, None, None
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
                if args.mode == "mixed":
                    demotion = await poll_hkv_state(
                        engine,
                        "hkv-target",
                        predicate=lambda s: (
                            len(s["warm_residency"]) >= 2
                            and s["allocator_matches_residency"]
                            and s["ownership_count_matches"]
                        ),
                        timeout=2.0,
                    )
                else:
                    await asyncio.sleep(0.1)
                    demotion = (
                        await engine.engine_core.collective_rpc_async(
                            "inspect_hkv_smoke", args=("hkv-target",)
                        )
                    )[0]
                idle_elapsed += time.perf_counter() - idle_start
                resume.set()
            elif turn == 2:
                if resumed is None:
                    resumed = (
                        await engine.engine_core.collective_rpc_async(
                            "inspect_hkv_smoke", args=("hkv-target",)
                        )
                    )[0]
                if completion.finish_reason:
                    idle_start = time.perf_counter()
                    if args.mode == "mixed":
                        second_demotion = await poll_hkv_state(
                            engine,
                            "hkv-target",
                            predicate=lambda s: (
                                len(s["warm_residency"])
                                > len(demotion.get("warm_residency", []))
                                and s["allocator_matches_residency"]
                                and s["ownership_count_matches"]
                            ),
                            timeout=2.0,
                        )
                    else:
                        await asyncio.sleep(0.1)
                        second_demotion = (
                            await engine.engine_core.collective_rpc_async(
                                "inspect_hkv_smoke", args=("hkv-target",)
                            )
                        )[0]
                    idle_elapsed += time.perf_counter() - idle_start
                    resume_third.set()
            elif turn == 3:
                if second_resumed is None:
                    second_resumed = (
                        await engine.engine_core.collective_rpc_async(
                            "inspect_hkv_smoke", args=("hkv-target",)
                        )
                    )[0]
                if completion.finish_reason:
                    turn_three_observed.set()

    finally:
        keep.cancel()
        with suppress(asyncio.CancelledError):
            await keep

        # Passively observe that normal request finish cleans up WARM residency
        if args.mode == "mixed":
            after_release = await poll_hkv_state(
                engine,
                "hkv-target",
                predicate=lambda s: (
                    not s["warm_residency"]
                    and not s["request_slot_ownership"]
                ),
                timeout=2.0,
            )
        else:
            after_release = (
                await engine.engine_core.collective_rpc_async(
                    "inspect_hkv_smoke", args=("hkv-target",)
                )
            )[0]

        engine.shutdown()

    if not (
        demotion
        and resumed
        and second_demotion
        and second_resumed
        and first_yielded
        and continuation_yielded
        and followup_yielded
        and output_counts[3]
    ):
        raise RuntimeError(
            "incomplete streaming lifecycle: "
            f"demotion_present={demotion is not None}, "
            f"resumed_present={resumed is not None}, "
            f"second_demotion_present={second_demotion is not None}, "
            f"second_resumed_present={second_resumed is not None}, "
            f"first_chunk_yielded={first_yielded}, "
            f"continuation_yielded={continuation_yielded}, "
            f"followup_yielded={followup_yielded}, current_turn={turn}, "
            f"turn_1_outputs={output_counts[1]}, "
            f"turn_2_outputs={output_counts[2]}, "
            f"turn_3_outputs={output_counts[3]}"
        )

    expected_complete_historical_blocks = historical_token_count // 16
    assert expected_complete_historical_blocks >= 2

    if args.mode == "mixed":
        assert demotion["mixed_read_enabled"] and resumed["mixed_read_enabled"]
        assert (
            second_demotion["mixed_read_enabled"]
            and second_resumed["mixed_read_enabled"]
        )
        assert len(demotion["warm_residency"]) >= 2
        assert (
            demotion["allocator_matches_residency"]
            and demotion["ownership_count_matches"]
        )
        first_keys = {tuple(item["key"]) for item in demotion["warm_residency"]}
        first_migrated_indices = sorted(key[2] for key in first_keys)
        assert all(
            item.get("temporary_shadow_hot_block_id") is not None
            and item["temporary_shadow_hot_block_id"] > 0
            for item in demotion["warm_residency"]
        )
        hot_before_second = {
            idx
            for idx, block_id in enumerate(resumed["block_ids"])
            if block_id > 0
        }
        # First demotion reclamation: all historical positions are strictly null-backed
        assert all(
            resumed["block_ids"][idx] == 0 for idx in first_migrated_indices
        )
        assert len(resumed["block_ids"]) > len(first_migrated_indices)

        second_keys = {
            tuple(item["key"]) for item in second_demotion["warm_residency"]
        }
        newly_migrated = second_keys - first_keys
        assert bool(newly_migrated)
        newly_migrated_indices = sorted(key[2] for key in newly_migrated)
        assert set(newly_migrated_indices).issubset(hot_before_second)

        assert len(second_demotion["warm_residency"]) > len(
            demotion["warm_residency"]
        )
        assert (
            second_demotion["allocator_matches_residency"]
            and second_demotion["ownership_count_matches"]
        )
        assert all(
            item in second_demotion["warm_residency"]
            for item in demotion["warm_residency"]
        )
        assert all(
            second_demotion["block_ids"][idx] == 0
            for idx in first_migrated_indices
        )

        # After second resume synchronization point:
        all_warm_indices = {key[2] for key in second_keys}
        assert all(
            second_resumed["block_ids"][idx] == 0
            for idx in newly_migrated_indices
        )
        assert all(
            second_resumed["block_ids"][idx] == 0
            for idx in first_migrated_indices
        )
        assert all(
            item in second_resumed["warm_residency"]
            for item in demotion["warm_residency"]
        )
        assert all(
            second_resumed["block_ids"][idx] > 0
            for idx in range(len(second_resumed["block_ids"]))
            if idx not in all_warm_indices
        )

        assert not after_release["warm_residency"]
        assert not after_release["request_slot_ownership"]
    else:
        assert not demotion["mixed_read_enabled"]
        assert not resumed["mixed_read_enabled"]
        assert not second_demotion["mixed_read_enabled"]
        assert not second_resumed["mixed_read_enabled"]
        assert not demotion["warm_residency"]
        assert not demotion["request_slot_ownership"]
        assert not demotion["allocator_mappings"]
        assert not resumed["warm_residency"]
        assert not resumed["request_slot_ownership"]
        assert not second_demotion["warm_residency"]
        assert not second_resumed["warm_residency"]

    first_source_hot_ids = {
        item["temporary_shadow_hot_block_id"]
        for item in (demotion["warm_residency"] if demotion else [])
        if item.get("temporary_shadow_hot_block_id") is not None
    }
    live_hot_blocks_after_first = {
        b for b in (resumed["block_ids"] if resumed else []) if b > 0
    } | {
        b for b in (second_resumed["block_ids"] if second_resumed else []) if b > 0
    }
    reclaimed_hot_reuse_observed = bool(
        first_source_hot_ids & live_hot_blocks_after_first
    )

    elapsed = time.perf_counter() - started - idle_elapsed
    result = {
        "mode": args.mode,
        "request_id": "hkv-target",
        "transition": "HOT->WARM" if args.mode == "mixed" else "NONE",
        "historical_token_count": historical_token_count,
        "continuation_token_count": continuation_token_count,
        "followup_token_count": followup_token_count,
        "expected_complete_historical_blocks": expected_complete_historical_blocks,
        "demotion": demotion,
        "migrated_warm_residency": demotion["warm_residency"],
        "migrated_block_slots": demotion["allocator_mappings"],
        "warm_quantization_observed": bool(demotion["warm_residency"]),
        "warm_slot_ownership_observed": bool(demotion["request_slot_ownership"]),
        "first_demotion_hot_blocks_reclaimed": (
            all(resumed["block_ids"][idx] == 0 for idx in first_migrated_indices)
            if args.mode == "mixed" and resumed
            else False
        ),
        "second_demotion_hot_blocks_reclaimed": (
            all(
                second_resumed["block_ids"][idx] == 0
                for idx in newly_migrated_indices
            )
            if args.mode == "mixed" and second_resumed
            else False
        ),
        "reclaimed_hot_reuse_observed": reclaimed_hot_reuse_observed,
        "mixed_attention_read_enabled": demotion["mixed_read_enabled"],
        "second_demotion": second_demotion,
        "after_second_resume": second_resumed,
        "after_release": after_release,
        "after_resume": resumed,
        "generated_token_ids": token_ids,
        "generated_text": text,
        "generation_time_seconds": elapsed,
        "tokens_per_second": len(token_ids) / elapsed,
        "peak_allocated_bytes": resumed["max_gpu_allocated_bytes"],
        "peak_reserved_bytes": resumed["max_gpu_reserved_bytes"],
        "physical_migration_enabled": args.mode == "mixed",
        "first_chunk_yielded": first_yielded,
        "continuation_yielded": continuation_yielded,
        "followup_yielded": followup_yielded,
        "turn_output_counts": output_counts,
    }
    if args.baseline_json:
        expected = json.loads(Path(args.baseline_json).read_text())[
            "generated_token_ids"
        ]
        aligned = list(zip(expected, token_ids))
        matching = sum(baseline == mixed for baseline, mixed in aligned)
        first_mismatch = next(
            (i for i, pair in enumerate(aligned) if pair[0] != pair[1]), None
        )
        if first_mismatch is None and len(expected) != len(token_ids):
            first_mismatch = len(aligned)
        result.update({
            "exact_token_match": expected == token_ids,
            "matching_aligned_tokens": matching,
            "compared_aligned_tokens": len(aligned),
            "token_match_percent": 100 * matching / len(aligned) if aligned else 0.0,
            "first_mismatch_position": first_mismatch,
            "baseline_token_count": len(expected),
            "mixed_token_count": len(token_ids),
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
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "HKV_ENABLE_PHYSICAL_TIERS": "1" if args.mode == "mixed" else "0",
        "HKV_WARM_POOL_BLOCKS": "16" if args.mode == "mixed" else "0",
        "HKV_DEBUG_DEMOTE_ONE_BLOCK": "0",
        "HKV_ENABLE_MULTI_BLOCK_WARM_MIGRATION": (
            "1" if args.mode == "mixed" else "0"
        ),
        "HKV_DEBUG_MIXED_READ": "1" if args.mode == "mixed" else "0",
    })
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
