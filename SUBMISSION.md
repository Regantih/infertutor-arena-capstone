# InferTutor Arena — Capstone Submission

**Track 1 (Multimodal / Mixed)** · Model: Qwen/Qwen3-VL-4B-Instruct · Engine: vLLM 0.21.0 on Modal (H100)
Composite scorer: `score = goodput · users / (TTFT_p95_s · ITL_p95_s · total_GPUs)`, `goodput = chunks/s · (1 − error_rate)`

## Final score

| Result | Config | GPUs | Score |
|---|---|---:|---:|
| **Official — within Track 1's 4×H100 budget** | `mix-4r-300u` | **4** | **149,776,620** |
| Optional Boss Fight (8×H100, spec-allowed tier) | `mix-8r-460u` | 8 | 189,183,025 |
| Baseline (1×H100, default/unoptimized) | `mix-1r-base-80u` | 1 | 319,183 |

The optimized + scaled pipeline is **~469×** the unoptimized 1-GPU baseline.

## The 8 required answers
1. **Final score:** 149,776,620 (4 GPU). Boss fight: 189,183,025 (8 GPU).
2. **Best TTFT p95:** 803 ms (8r/460u); 1029 ms at the 4-GPU headline.
3. **Best ITL p95:** 5.7 ms (4r/300u); 6.6 ms at the boss fight.
4. **Best goodput:** 17,425 chunks/s (8r/460u); 11,649 at the 4-GPU headline.
5. **Total GPU count:** 4 (official); 8 (optional boss fight).
6. **Helped most:** (a) co-locating the load client inside Modal (kills residential-uplink bufferbloat that pins TTFT p95), (b) compiled mode `--no-fast-boot` (ITL 38.8 ms eager → 5.7 ms), (c) `max_num_batched_tokens` 4096→16384 (unblocks prefill admission → lower ITL/TTFT).
7. **Failed / surprised:** prefix caching ON regressed 150M→34M (TTFT 1029→2990 ms); chunked-prefill OFF regressed 150M→75M. Most surprising: the 8-GPU error cliff — 460u (0.5% err)=189M but 500u (3.3% err)=150M; staying in the near-zero-error regime beats chasing raw user count.
8. **Try next:** the ~17.4k c/s aggregate ceiling is a single Modal web-endpoint chokepoint (scaling is sub-linear: 4→8 GPUs gives 1.5× throughput, not 2×). Beating it needs multiple load-balanced endpoints — architectural, not a server flag. Also: a quality-gated run, since the official score multiplies by `quality_pass_rate` (the local harness assumes 1.0).

## Exact final-run command (official 4-GPU result)
```bash
set PYTHONUTF8=1
# Deploy: 4xH100, compiled, seq32, max_batch_tokens 16384, no prefix cache, chunked prefill on, ci128
python run_infertutor_experiment.py --label mix-4r --gpu-type H100 --replicas 4 \
  --no-fast-boot --no-prefix-cache --max-seqs 32 --max-batch-tokens 16384 \
  --concurrent-inputs 128 --mode mixed --deploy-only
# Load from INSIDE Modal (co-located client, no laptop in the path)
modal run load_client_modal.py --url https://<you>--infertutor-mix-4r-serve.modal.run \
  --label mix-4r-300u --mode mixed --users 300 --shards 16 --duration 150 --ramp-up 75 --total-gpus 4
# Score
python score_infertutor.py results_infertutor/mix-4r-300u_mixed_300u_<ts>.json
```

## Table of experiments (≥5; full matrix in experiments.md)

| label | gpus | users | prefix | chunked | err% | TTFT p95 | ITL p95 | chunks/s | score |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| **mix-4r-300u (official)** | 4 | 300 | off | on | 0.0 | 1029 ms | 5.7 ms | 11649 | **149,776,620** |
| mix-4r-240u | 4 | 240 | off | on | 0.0 | 785 ms | 6.1 ms | 10827 | 136,280,000 |
| mix-4r-400u (past knee) | 4 | 400 | off | on | 6.4 | 3581 ms | 6.1 ms | 7923 | 33,950,000 |
| mix-4r-pc-300u (prefix on) | 4 | 300 | on | on | 0.3 | 2990 ms | 7.0 ms | 9437 | 33,719,000 |
| mix-4r-ncp-300u (chunked off) | 4 | 300 | off | off | 1.0 | 1498 ms | 6.5 ms | 9897 | 75,122,000 |
| mix-1r-base-80u (baseline) | 1 | 80 | on | on | 0.0 | 4994 ms | 38.8 ms | 773 | 319,183 |
| mix-8r-460u (boss fight) | 8 | 460 | off | on | 0.5 | 803 ms | 6.6 ms | 17425 | 189,183,025 |
| mix-8r-500u (error cliff) | 8 | 500 | off | on | 3.3 | 1047 ms | 6.6 ms | 17089 | 149,740,000 |

## Best config — short explanation
4 replicas (4×H100), compiled CUDA graphs, `max_num_seqs=32`, `max_num_batched_tokens=16384`, prefix cache **off**, chunked prefill **on**, 128 concurrent inputs, driven by a Modal-co-located load client at 300 users. This holds the prefill queue shallow enough for **TTFT p95 1029 ms / ITL 5.7 ms with 0 errors** at the throughput knee, which is where the composite peaks within the 4-GPU budget.

## Attached files
- `mix-4r-300u_mixed_300u_1781402073.json` — **final benchmark JSON (official, 4-GPU)**
- `mix-8r-460u_mixed_460u_1781404394.json` — boss-fight benchmark JSON (8-GPU)
- `report.md` — one-page engineering report
- `experiments.md` — full experiment log (both tracks)

> Scorer note: scores use `score_infertutor.py`, which omits `quality_pass_rate` (assumes 1.0); the official leaderboard may include it.
