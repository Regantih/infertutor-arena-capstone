# InferTutor Arena — Experiment Log

**Track 1 (Multimodal / Mixed) is the submitted track — jump to the bottom section.** The text-track log below is the speed-track R&D that *discovered and validated the levers* (Modal-co-located client, compiled mode, `max_batch_tokens` 16384) which were then applied to the mixed workload.

Model: Qwen/Qwen3-VL-4B-Instruct · GPU: H100 · compiled (`--no-fast-boot`) · `--no-prefix-cache` · max-tokens=96
Text-track reference target (score_infertutor.py composite): **219,235,337**

---

## Text-track R&D (mode=text, duration=90s) — how the levers were found

| label | replicas | max_seqs | max_batch_tokens | users | ramp_up | err% | TTFT_p95 | ITL_p95 | throughput | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| repro-seq32-4r-400u | 4 | 32 | 4096 | 400 | 40 | 0.0 | 1731 ms | 7.8 ms | 6118.3 | 45,292,077 |
| clean-seq32-4r-400u | 4 | 32 | 4096 | 400 | 40 | 0.0 | 1902 ms | 7.9 ms | 6054.1 | 40,311,033 |
| baseline_text_50u | 1 | 32 | 4096 | 50 | 15 | 0.0 | 636 ms | 6.1 ms | 2454.9 | 31,708,473 |
| seq64-4r-400u | 4 | 64 | 8192 | 400 | 40 | 0.0 | 2026 ms | 7.8 ms | 4949.1 | 31,345,035 |
| seq64-4r-400u-ramp60 | 4 | 64 | 8192 | 400 | 60 | 0.0 | 2216 ms | 7.7 ms | 5357.6 | 31,202,135 |
| seq64-4r-400u-ramp60 (outlier run) | 4 | 64 | 8192 | 400 | 60 | 0.0 | 11343 ms | 7.4 ms | 2815.5 | 3,335,677 |
| seq64-8r-600u-ramp60 | 8 | 64 | 8192 | 600 | 60 | 0.0 | 2348 ms | 8.2 ms | 4053.0 | 15,745,695 |
| compiled-8r-600u-ramp90 (ci128) | 8 | 32 | 4096 | 600 | 90 | 0.0 | 2654 ms | 8.3 ms | 3739.8 | 12,759,438 |
| compiled-2r-300u-ramp40 (ci128) | 2 | 32 | 4096 | 300 | 40 | 0.0 | 2352 ms | 7.7 ms | 4211.5 | 34,707,551 |
| compiled-4r-400u-ramp40 (single client) | 4 | 32 | 4096 | 400 | 40 | 0.0 | 2034 ms | 8.1 ms | 4044.4 | 24,560,489 |
| sharded-4r-400u (8 proc) | 4 | 32 | 4096 | 400 | 30 | 0.0 | 3124 ms | 8.7 ms | 8399.4 | 30,740,719 |
| sharded-10r-600u cold (10 proc) | 10 | 32 | 4096 | 600 | 40 | 1.5 | 5502 ms | 6.1 ms | 12792.7 | 22,710,121 |
| sharded-10r-600u warm (10 proc) | 10 | 32 | 4096 | 600 | 20 | 1.5 | 3419 ms | 6.1 ms | 11692.8 | 33,413,017 |
| sharded-10r-400u warm (10 proc) | 10 | 32 | 4096 | 400 | 20 | 0.0 | 3043 ms | 5.4 ms | 13244.9 | 32,124,575 |
| sharded-10r-200u warm (10 proc) | 10 | 32 | 4096 | 200 | 15 | 0.0 | 2832 ms | 5.2 ms | 7779.8 | 10,573,947 |

## Modal-co-located client (`modal run load_client_modal.py`, no laptop in the path)

The decisive architecture change: the load generator now runs as a CPU-only `modal.Function`
**inside Modal's network**, co-located with the vLLM endpoint, instead of on the residential
laptop. This removes the home-uplink bufferbloat that had pinned TTFT p95 at ~3 s. Same server
configs as above; only the client moved.

| label | gpu | replicas | max_seqs | max_batch | users | shards | ci | ramp_up | err% | TTFT_p95 | ITL_p95 | throughput | score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| modal-4r-400u | H100 | 4 | 32 | 4096 | 400 | 8 | 64 | 40 | 0.0 | 905 ms | 7.6 ms | 6700 | 114,117,836 |
| modal-8r-600u-ci128 | H100 | 8 | 32 | 4096 | 600 | 16 | 128 | 90 | 9.0 | 1180 ms | 6.4 ms | 17500 | 163,191,045 |
| modal-10r-600u-ci128 | H100 | 10 | 32 | 4096 | 600 | 20 | 128 | 120 | 1.5 | 1240 ms | 6.1 ms | 14800 | 102,988,121 |
| **modal-10r-1000u-ci128** | H100 | 10 | 32 | 4096 | 1000 | 20 | 128 | 120 | 23.7 | 1365 ms | 6.4 ms | 18898 | **165,744,108** |
| modal-10r-1000u-ci64 | H100 | 10 | 32 | 4096 | 1000 | 20 | 64 | 120 | 20.4 | 1728 ms | 6.4 ms | 18600 | 141,157,407 |
| modal-h200-8r-600u | H200 | 8 | 32 | 4096 | 600 | 16 | 128 | 90 | 8.9 | 1463 ms | 5.4 ms | 17984 | 154,592,167 |

## BREAKTHROUGH: `max_num_batched_tokens` 4096 -> 16384 (the lever the old notes had locked out)

The earlier "proven fact" to freeze `--max-batch-tokens 4096` was derived on the **laptop/bufferbloat** client, where throughput never mattered (TTFT was network-bound). With the Modal-co-located client, the binding constraint became **prefill admission** — diagnostic: at 18.6k c/s the server ran only ~117 of its 320 decode slots full. Raising the prefill batch unblocked admission and, crucially, made decode steps more regular, **lowering ITL and TTFT** (the score's denominator). Server: H100, compiled, `--no-prefix-cache`, ci128 unless noted.

| label | replicas | GPUs | max_seqs | max_batch | users | err% | TTFT_p95 | ITL_p95 | throughput | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **prefill-8r-600u** | 8 | 8 | 32 | **16384** | 600 | 11.1 | 1080 ms | 5.5 ms | 18454 | **206,048,509** |
| prefill32k-8r-600u | 8 | 8 | 32 | 32768 | 600 | 10.0 | 1158 ms | 6.1 ms | 18273 | 174,424,837 |
| seq64-16k-8r-600u | 8 | 8 | 64 | 16384 | 600 | 5.8 | 1627 ms | 6.6 ms | 18083 | 118,469,546 |
| pc-16k-8r-600u (prefix-cache ON) | 8 | 8 | 32 | 16384 | 600 | 9.6 | 1201 ms | 6.3 ms | 18198 | 163,892,353 |
| r10-16k-1000u | 10 | 10 | 32 | 16384 | 1000 | 23.9 | 1329 ms | 6.2 ms | 18933 | 174,063,323 |
| r4-16k-400u | 4 | 4 | 32 | 16384 | 400 | 0.0 | 1718 ms | 6.4 ms | 14679 | 133,440,207 |
| r5-16k-600u | 5 | 5 | 32 | 16384 | 600 | 0.0 | 2282 ms | 6.4 ms | 16515 | 136,531,846 |
| zeropause-10r-500u (min/max pause 0/0.05) | 10 | 10 | 32 | 4096 | 500 | 11.6 | 1289 ms | 6.3 ms | 18631 | 101,595,627 |

Lever findings (all under the Modal-co-located client):
- **`max_batch_tokens` 16384 is the optimum.** 4096->16384 lifted 8r/600u from ~163M to **206M** (ITL 6.4->5.5, TTFT 1180->1080). 32768 *regressed* (175M) — prefill steals decode cycles, ITL/TTFT rise. The curve peaks at 16384.
- **`max_seqs` 32 still beats 64**, even here: seq64 cut errors (11%->5.8%) but bigger decode batches blew TTFT up (1080->1627) and ITL (5.5->6.6) -> net 119M. Keep 32.
- **Prefix caching is a net negative for the text track** (163M vs 206M): short shared prompt, so the block-hash/KV-contention overhead raised ITL/TTFT more than it saved on prefill. `--no-prefix-cache` confirmed correct in the co-located regime too.
- **Zero pause does NOT raise throughput** (18631 ~= the 18898 of the paused 1000u run): throughput is pinned at a ~18.6k c/s **aggregate ceiling** (single Modal web endpoint), not client idle time.
- **Fewer GPUs does NOT win, despite the smaller score divisor.** 16384 raised per-GPU throughput to ~3675 c/s (4r) so I tested 5r hoping to hit the ceiling with a ÷5 divisor. But concentrating 600 users onto 5 replicas oversubscribes the 160 decode slots 3.75x -> prefill queue deepens -> **TTFT 2282** -> 136M. 8 replicas is the genuine optimum: enough to keep oversubscription ~2.3x (TTFT 1080) while still reaching the throughput ceiling.
- **More users hits error-saturation, not more score.** 10r/1000u/16384 = 174M (23.9% errors) — the 16384 lever can't help once the system is past the saturation knee.

**Best overall this project: prefill-8r-600u = 206,048,509** (8xH100, compiled, seq32, max_batch_tokens 16384, no prefix cache, 600 users). 94% of the 219,235,337 target; 4.5x the original laptop baseline.

Notes:
- **Best score overall: repro-seq32-4r-400u = 45,292,077** (single-process client, 4×H100, compiled).
- **Client bottleneck found, then fixed:** the stock single-process load tester reached only **232/400 active users** on compiled-4r (4044 c/s) — vs a competitor's identical config at 14,446 c/s. A single asyncio event loop saturates: the ramp coroutine starves. Fix = shard the client across OS processes (`sharded_load_test.py`, one event loop + GIL per process, raw TTFT/ITL samples pooled across shards). Sharding doubled-to-tripled throughput (4044 → 8399 → 13244 c/s) with 0 errors.
- **But sharding LOWERED the score.** Root cause = a deeper bottleneck the higher throughput exposed: **home-network bufferbloat**. TTFT p50 stays low (~450–950 ms) but the p95/p99 tail explodes with aggregate download throughput (200u→2832 ms, 600u→3419 ms p95; p99 ~10 s). The residential downlink can't carry 13k chunks/s of concurrent SSE without queueing first-token packets. Because the composite score divides by **p95 TTFT**, the tail penalty outweighs the throughput gain.
- **Implication:** on a bandwidth-limited client, throughput and TTFT are coupled — pushing throughput past ~6k c/s inflates p95 TTFT faster than goodput rises. The naive single-process client accidentally self-limits at the score-optimal operating point (~6k c/s, TTFT ~1.7 s). GPU scaling (4→8→10) and client sharding cannot beat it because the binding constraint is the client's network, not the GPUs.
- **Verdict vs 219,235,337 target: NOT beaten.** Competitor scores (219M / 565.8M / 601.8M) require ~300–1100 ms p95 TTFT at 14k+ c/s simultaneously — only achievable from a low-latency, high-bandwidth (co-located/cloud) load generator. From this residential client the p95-TTFT floor caps the composite near ~45M regardless of server config. Server health itself is excellent throughout: 0–1.5% errors, ITL p95 5.2–6.1 ms, TTFT p50 ~0.5 s.
- seq64/b8192 underperformed seq32/b4096 at 4 replicas (lower throughput, higher TTFT).
- Composite is dominated by throughput × users / (TTFT × ITL × GPUs). High-concurrency TTFT is the main score killer.

### Modal-co-located client results (the architecture fix)
- **Moving the client into Modal raised the best score 3.7×: 45,292,077 → 165,744,108.** Exactly as predicted — TTFT p95 collapsed from ~3000 ms (laptop) to ~900–1400 ms (Modal) for the same server, because the residential downlink was no longer carrying the SSE chunk volume. p95/p50 TTFT ratio fell from ~6× to ~1.5×.
- **Best Modal-client score: modal-10r-1000u-ci128 = 165,744,108** (10×H100, 100 users/GPU, 18,898 c/s).
- **H100 throughput ceiling ≈ 18.9k c/s aggregate.** Per-GPU throughput is sub-linear (4r≈3336 c/s/gpu, 8r≈2299, 10r≈1679), implying a shared bottleneck — most likely the single Modal web-endpoint proxy in front of the replicas, not the GPUs.
- **Errors are genuine saturation, not a ci artifact.** ci64 did NOT cure the 10r/1000u errors (20.4% vs 23.7%) and actually *raised* TTFT (1365→1728 ms), scoring worse (141M). High user counts overrun the queue: 0% @ 400u, 9% @ 600u/8r, 23.7% @ 1000u/10r.
- **H200 lowered ITL (6.4→5.4 ms, decode is bandwidth-bound) but not enough.** Its run came in with higher TTFT (1463 ms), netting 154.6M — below the H100 best.
- **Verdict vs 219,235,337: NOT beaten. Best = 165,744,108 (≈1.32× short).** The remaining gap is not a server knob: competitor 219M reports ~1128 ms TTFT p95 **with 0 errors at 14k+ c/s** simultaneously. From this setup, pushing to 18.9k c/s forces either error-saturation (23.7%) or higher TTFT — the H100 + single-proxy path tops out near ~165M. The composite is now limited by the throughput ceiling and error onset, not by client bufferbloat (fixed) or compiled mode (on).
- Best overall this project: **modal-10r-1000u-ci128 = 165,744,108**.

---

# Track 1 — Multimodal / Mixed (the submitted track)

Model: Qwen/Qwen3-VL-4B-Instruct · GPU: H100 · mode=`mixed` (25% image / 20% long / 55% text, same workload mix as the harness) · Modal-co-located client (`load_client_modal.py`, extended this track to emit the deterministic diagram PNG + mixed request mix) · compiled (`--no-fast-boot`) · `--no-prefix-cache` · chunked-prefill ON · max-tokens=96 · duration=150s.

**Track 1 budget = 4 H100.** The official submission is the best **4-GPU** run; the **8-GPU** runs are the optional boss-fight tier (spec allows up to 8).

| label | gpus | replicas | max_seqs | max_batch | prefix | chunked | users | err% | TTFT_p95 | ITL_p95 | throughput | score |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| **mix-4r-300u (OFFICIAL, 4-GPU)** | 4 | 4 | 32 | 16384 | off | on | 300 | 0.0 | 1029 ms | 5.7 ms | 11649 | **149,776,620** |
| mix-4r-340u | 4 | 4 | 32 | 16384 | off | on | 340 | 0.0 | 1260 ms | 5.8 ms | 12301 | 142,810,000 |
| mix-4r-240u | 4 | 4 | 32 | 16384 | off | on | 240 | 0.0 | 785 ms | 6.1 ms | 10827 | 136,280,000 |
| mix-4r-ncp-300u (chunked **off**) | 4 | 4 | 32 | 16384 | off | **off** | 300 | 1.0 | 1498 ms | 6.5 ms | 9897 | 75,122,000 |
| mix-4r-400u (past knee) | 4 | 4 | 32 | 16384 | off | on | 400 | 6.4 | 3581 ms | 6.1 ms | 7923 | 33,950,000 |
| mix-4r-pc-300u (prefix **on**) | 4 | 4 | 32 | 16384 | **on** | on | 300 | 0.3 | 2990 ms | 7.0 ms | 9437 | 33,719,000 |
| mix-1r-base-80u (**baseline**, defaults: eager/prefix-on/b4096) | 1 | 1 | 32 | 4096 | on | on | 80 | 0.0 | 4994 ms | 38.8 ms | 773 | 319,183 |
| **mix-8r-460u (BOSS FIGHT, 8-GPU)** | 8 | 8 | 32 | 16384 | off | on | 460 | 0.5 | 803 ms | 6.6 ms | 17425 | **189,183,025** |
| mix-8r-480u | 8 | 8 | 32 | 16384 | off | on | 480 | 2.9 | 841 ms | 6.2 ms | 16732 | 186,770,000 |
| mix-8r-500u | 8 | 8 | 32 | 16384 | off | on | 500 | 3.3 | 1047 ms | 6.6 ms | 17089 | 149,740,000 |
| mix-8r-600u (error cliff) | 8 | 8 | 32 | 16384 | off | on | 600 | 8.7 | 1207 ms | 6.7 ms | 16937 | 143,580,000 |

Mixed-track findings:
- **The three text-track levers transfer directly.** Compiled mode (baseline eager ITL 38.8 ms → 5.7 ms under load), `max_batch_tokens` 16384, and the Modal-co-located client are what move a run from the ~0.3M baseline class to the ~150M class.
- **4-GPU knee = 300 users** (149.8M, 0 errors). 240u under-loads; 340u→400u inflate TTFT faster than throughput until errors appear.
- **Prefix caching ON regresses hard:** 149.8M → 33.7M (TTFT 1029→2990 ms). Even with a shared system prompt and image/long traffic, block-hash/KV contention beats the prompt-sharing saving for short outputs. `--no-prefix-cache` confirmed correct for mixed.
- **Chunked prefill OFF regresses:** 149.8M → 75.1M (TTFT 1029→1498, errors appear). Chunked prefill ON is needed to interleave heavy image/long prefill with decode.
- **Scale-out 1→4→8 and the error cliff:** 8 GPUs reach 17.4k c/s and TTFT 803 ms, but only win when kept in the **near-zero-error** regime — 460u (0.5% err) = **189M**, but 500u (3.3% err) drops to 150M. Staying below the error knee matters more than raw user count.
- **Sub-linear scaling:** 8 GPUs give ~17.4k c/s vs 4 GPUs ~11.6k (1.5×, not 2×), so the ÷8 divisor nearly cancels the extra throughput — same single-Modal-endpoint ceiling seen on the text track. Beating it is architectural (multiple load-balanced endpoints), not a server flag.

**Track 1 official (4-GPU): mix-4r-300u = 149,776,620.  Optional boss fight (8-GPU): mix-8r-460u = 189,183,025.**

> Scorer caveat: the official formula multiplies by `quality_pass_rate`; the local harness (`score_infertutor.py`) assumes it = 1.0, so all scores above are the throughput/latency composite under that assumption.

## Quality probe — is compiled-on-mixed safe? (closing the one open assumption)

The spec warns compiled mode "performed poorly on mixed multimodal traffic" and penalizes optimizations that "break multimodal traffic." Since our headline keeps compiled **on**, we measured quality directly (`probe_quality.py`): the official prompts (4 image w/ the harness 256×192 PNG, 2 long, 4 text), non-streaming, against two 1×H100 endpoints differing only in execution mode.

| endpoint | cases | flagged | image coherent | latency (image/text) | artifact |
|---|---:|---:|---:|---|---|
| compiled-mixed (`--no-fast-boot`) | 10 | 0 | 4/4 | ~0.5 s / ~1.4 s | `quality_compiled-mixed_*.json` |
| eager-mixed (default) | 10 | 0 | 4/4 | ~2.0 s / ~4.5 s | `quality_eager-mixed_*.json` |

Both modes return coherent, on-topic tutoring answers and the model genuinely reads the diagram (decode-heavy vs prefill-heavy, replicas vs TP). Image answers are essentially **identical in content** between modes — compiled is just **3–4× faster per request**. So the dry-run's "poorly" was a latency/throughput observation, not output-correctness: **compiled-on-mixed does not degrade `quality_pass_rate`**, and is strictly better here.
