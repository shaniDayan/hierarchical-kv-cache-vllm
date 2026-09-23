# Pressure-Aware Hierarchical KV Cache Quantization for Concurrent LLM Inference

Research prototype built on [vLLM](external/vllm) for retaining conversational KV-cache state under GPU memory pressure. The system keeps active and newly generated KV blocks in an FP16 **HOT** pool and can move complete, privately owned blocks from inactive sessions to a GPU-resident INT8 **WARM** pool. A resumed session reads its WARM history and newly allocated HOT blocks in the same Triton attention operation.

The prototype and experiments are described in the accompanying paper, *Pressure-Aware Hierarchical KV Cache Quantization for Concurrent LLM Inference*.

## Motivation

In multi-turn serving, a session can be idle between user messages while its KV cache remains useful for the next turn. Retaining every idle history in FP16 limits the memory available for new and active requests. The hierarchy compresses eligible idle blocks while keeping their state available for resumption, without recomputing the entire history or expanding the persistent KV-memory budget.

## Design

| Tier | Data | Location | Use |
| --- | --- | --- | --- |
| HOT | FP16 KV blocks | GPU | Active requests, new tokens, incomplete or shared blocks |
| WARM | INT8 KV blocks with per-token, per-head FP32 scales | GPU | Eligible complete blocks of inactive resumable sessions |

The scheduler begins demotion when HOT-block utilization reaches a configurable start threshold and plans migrations until projected utilization reaches a lower stop threshold. It visits eligible inactive sessions in least-recently-active order. The two thresholds provide hysteresis. Blocks already awaiting a worker result count toward projected releases and are not scheduled a second time.

For each migration, the worker validates the request and physical mapping, reserves WARM slots, quantizes and writes the blocks, then acknowledges completion. The scheduler releases the original HOT blocks only after a successful acknowledgment. Capacity failures and stale plans leave HOT authoritative. A capacity failure also triggers bounded retry backoff. WARM ownership is keyed by `(request_id, cache_group_index, logical_block_index)`, so recycling a physical HOT block cannot change the identity of the saved history.

On resume, a logical residency table tells the Triton attention kernel whether each block lives in HOT or WARM. WARM values are dequantized during attention; newly processed tokens remain in HOT. Finishing, aborting, and preempting requests release their associated tier resources. In-flight transitions are resolved before conflicting lifecycle operations continue.

**Scope:** This is a two-tier, GPU-resident prototype. CPU COLD offload is not implemented. The evaluated configuration uses the vLLM V2 model runner, Triton attention, FP16 HOT storage, one KV-cache group, one physical block per logical block, and tensor/pipeline/data parallel sizes of one. Shared prefix blocks are kept HOT; prefix caching was disabled for the reported experiments. The WARM-pool capacity is fixed at engine startup.

## Results

The capacity-pressure replay used Qwen3-0.6B on one NVIDIA GeForce RTX 2080 Ti (11 GiB), a common **3.5 GiB persistent KV-memory budget**, and Qwen-BaiLian Trace A sessions with deterministic reconstructed token inputs. All-HOT allocated 2,048 FP16 blocks (3,758,096,384 bytes). Mixed allocated 1,517 FP16 HOT blocks, 1,024 INT8 WARM slots, and persistent mapping metadata (3,757,479,856 bytes). Mixed used 616,528 fewer persistent bytes than All-HOT.

| Selected sessions | Requests | All-HOT completed | Mixed completed | Outcome |
| ---: | ---: | ---: | ---: | --- |
| 100 | 225 | 225 | 225 | Both completed; service windows 151.39 s and 151.72 s, respectively |
| 125 | 275 | 158 | 275 | All-HOT timed out; Mixed completed |
| 150 | 330 | 158 | 190 | Both timed out |

The 125-session experiment was repeated three times. Mixed completed **275/275 requests** in each run; All-HOT completed **158, 154, and 147** requests before the 600-second timeout. Mixed peaked at 1,020–1,023 occupied WARM slots across these repetitions. In the full-trace overload test (16,328 sessions, 30,830 requests), both modes timed out: All-HOT completed 60 sessions and 124 requests, while Mixed completed 91 sessions and 177 requests. These timed-out counts show capacity behavior under overload; they are **not complete-workload throughput comparisons**.

### GSM8K resume quality

The separate [GSM8K resume-quality harness](experiments/scripts/run_hkv_gsm8k_resume_quality.py) evaluates natural-language math questions after a two-turn session resumes. The analyzed `final_n100_batches5_cap512_seq_v3` run contains 100 paired questions (indices 0–99), split into 20 batches of five. Both modes use Qwen/Qwen3-0.6B with seed 0, five-shot prompts, and a 512-token response cap. All 100 questions completed in each mode; neither mode recorded a failed or timed-out question.

| Metric | All-HOT | Mixed |
| --- | ---: | ---: |
| GSM8K exact matches | 52/100 (52%) | 54/100 (54%) |
| Successfully extracted answers | 99/100 | 99/100 |
| Generated tokens | 15,669 | 15,728 |
| Process wall-clock time | 904 s | 1,007 s |
| Peak GPU allocated | 4.862 GiB | 4.885 GiB |
| Peak GPU reserved | 5.006 GiB | 5.072 GiB |

Paired outcomes were 49 questions correct in both modes, 42 incorrect in both, three correct only in All-HOT, and five correct only in Mixed. Generated token sequences matched exactly on 49 questions; extracted numerical answers matched on 81 and differed on 18. The remaining question, `gsm8k-0005`, reached the 512-token cap in both modes and had no extractable answer. Thus the two-point exact-match difference is an observation on this sample, **not evidence that quantization improves reasoning accuracy**. The response cap also limits interpretation of the absolute accuracy numbers.

Every evaluated Mixed history was INT8 WARM-resident before resume (peak WARM occupancy: 352–364 blocks), providing direct evidence that the experiment exercises mixed-tier resumption. The recorded INT8 WARM storage is 0.902 GiB; All-HOT has no WARM pool. The reported peak allocated/reserved CUDA memory also includes model and runtime allocations and should not be read as a measurement of persistent KV-cache savings. Mixed took about 11.4% longer in process wall-clock time across these sequential batches; this experiment measures quality and resumption behavior, not a controlled throughput comparison.

The results and per-question comparisons are in [`experiments/hkv_pressure_study/tables/`](experiments/hkv_pressure_study/tables/), with analysis in [`analyze_gsm8k_resume_quality.ipynb`](experiments/hkv_pressure_study/notebooks/analyze_gsm8k_resume_quality.ipynb). The two earlier failed attempts produced no result JSON and are excluded from these 100-question totals.

## Running the experiments

Use the modified vLLM checkout at `external/vllm` and an environment with its GPU dependencies installed. The replay scripts import `experiments.*`; run them from the repository root using the Python environment that loads this checkout. The Qwen-BaiLian trace and full GSM8K dataset are external inputs and are not supplied by the example commands below. Download or prepare the datasets separately.

The following paired run illustrates the 125-session capacity experiment. Replace `TRACE` with the path to the prepared Qwen-BaiLian Trace A JSONL. Run All-HOT first; Mixed uses its JSON output to check that the paired workload matches. The script sets the required HKV runtime environment for each mode.

```bash
TRACE=/path/to/qwen_traceA.jsonl
BUDGET=3758096384
mkdir -p results

python experiments/scripts/run_qwen_bailian_replay.py \
  --experiment-mode performance --kv-mode all-hot \
  --trace "$TRACE" --result-json results/all_hot_125.json \
  --total-kv-budget-bytes "$BUDGET" \
  --max-sessions 125 --min-turns 1 \
  --max-input-length 8176 --max-model-len 9216 \
  --max-num-seqs 2048 --time-scale 0.05 \
  --max-tokens-per-turn 32 --timeout 600

python experiments/scripts/run_qwen_bailian_replay.py \
  --experiment-mode performance --kv-mode mixed \
  --trace "$TRACE" --result-json results/mixed_125.json \
  --baseline-json results/all_hot_125.json \
  --total-kv-budget-bytes "$BUDGET" \
  --warm-pool-blocks 1024 \
  --demotion-start-utilization 0.80 \
  --demotion-stop-utilization 0.65 \
  --max-sessions 125 --min-turns 1 \
  --max-input-length 8176 --max-model-len 9216 \
  --max-num-seqs 2048 --time-scale 0.05 \
  --max-tokens-per-turn 32 --timeout 600
```

The runner also provides a `correctness` experiment mode. Performance mode requires an explicit positive persistent KV budget and `--max-tokens-per-turn` greater than one. Mixed runs require both pressure thresholds, a positive WARM-pool size, and the paired `--baseline-json`. For a natural-text quality evaluation, see the CLI options in `run_hkv_gsm8k_resume_quality.py` and use the same dataset and question indices for the two modes.

## Repository map

| Path | Contents |
| --- | --- |
| [`external/vllm/`](external/vllm) | Modified vLLM scheduler, KV manager, GPU worker, migration code, and Triton attention |
| [`experiments/scripts/run_qwen_bailian_replay.py`](experiments/scripts/run_qwen_bailian_replay.py) | Paired trace replay, persistent budget derivation, metrics and validation |
| [`experiments/scripts/run_hkv_gsm8k_resume_quality.py`](experiments/scripts/run_hkv_gsm8k_resume_quality.py) | Two-turn GSM8K resume-quality experiment and WARM-read evidence |
| [`experiments/scripts/qwen_bailian_trace.py`](experiments/scripts/qwen_bailian_trace.py) | Trace parsing and session construction |
| [`experiments/tests/`](experiments/tests) | Experiment and replay tests |
| [`experiments/notebooks/`](experiments/notebooks) | Exploratory analyses and evaluation notebooks |
| [`experiments/hkv_pressure_study/`](experiments/hkv_pressure_study) | Pressure-study notebooks, plots, and 100-question GSM8K tables |
| [`docs/hierarchical_kv_cache.md`](docs/hierarchical_kv_cache.md) | Implementation-level guide; some historical time-based descriptions predate the pressure policy |

## Limitations

The experiments use one model and one GPU, with a fixed WARM pool and a Triton-specific attention path. When both configurations exceed capacity, neither completes the full trace. The quality experiment and the capacity replay answer different questions: replay inputs are deterministically reconstructed from trace hashes and do not measure natural-language answer quality. The GSM8K sample is limited to 100 questions and a 512-token output cap. Extending the design to CPU storage, other attention backends, distributed execution, and migration of shared prefix blocks remains future work.
