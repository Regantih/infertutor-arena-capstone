# InferTutor Arena — Engineering Report (Track 1: Multimodal / Mixed)

**Model:** Qwen/Qwen3-VL-4B-Instruct · **GPU:** H100 · **Mode:** `mixed` (25% image / 20% long / 55% text) · **Engine:** vLLM 0.21.0 (OpenAI-compatible)
**Composite scorer:** `score = goodput · users / (TTFT_p95_s · ITL_p95_s · total_GPUs)`, `goodput = chunks/s · (1 − error_rate)`

| Result | Config | GPUs | Score |
|---|---|---:|---:|
| **Track 1 — official (within 4×H100 budget)** | `mix-4r-300u`: 4×H100, compiled, seq32, batch 16384, no-prefix-cache | **4** | **149,776,620** |
| Optional Boss Fight (8×H100) | `mix-8r-460u`: same config, scaled out | 8 | **189,183,025** |
| Baseline (unoptimized starting point) | `mix-1r-base-80u`: 1×H100, defaults (eager, prefix-on, batch 4096) | 1 | 319,183 |

> The official Track 1 submission is the **4-GPU** result (`mix-4r-300u` = **149.8M**), because Track 1's budget is **4 H100**. The **8-GPU** run (`mix-8r-460u` = **189.2M**) is reported as the **optional boss-fight tier** explicitly allowed by the spec (up to 8 H100), not as the in-budget result. Relative to the 1-GPU default baseline (0.32M), the optimized + scaled pipeline is a **~469× improvement**.

---

## The 8 required questions (answered up-front)

1. **Final score:** **149,776,620** (Track 1, 4×H100). Optional boss fight: 189,183,025 (8×H100).
2. **Best TTFT p95:** **803 ms** (boss fight 8r/460u); **1029 ms** at the 4-GPU headline.
3. **Best ITL p95:** **5.7 ms** (4r/300u headline); 6.6 ms at the boss fight.
4. **Best throughput (goodput):** **17,425 chunks/s** (8r/460u); 11,649 chunks/s at the 4-GPU headline.
5. **Total GPU count:** **4** for the official Track 1 result; 8 for the optional boss fight.
6. **Optimization that helped most:** moving the load client **into Modal** (co-located, kills residential bufferbloat) + `max_num_batched_tokens` **4096→16384** (unblocks prefill admission → lower ITL/TTFT). Together these are the difference between a client-bottlenecked ~0.3M-class run and a real 150M-class server.
7. **What failed / surprised:** **prefix caching ON regressed hard** in mixed (TTFT 1029→2990 ms, 150M→34M) — even with a shared system prompt, block-hash/KV contention costs more than it saves for these short-output requests. **Chunked prefill OFF also regressed** (150M→75M). Most surprising: the 8-GPU **error cliff** — 460u (0.5% err) scores **189M** but 500u (3.3% err) collapses to 150M; staying in the near-zero-error regime matters more than raw user count.
8. **What to try next:** the aggregate-throughput ceiling (~17.4k c/s) is a **single Modal web-endpoint** chokepoint. Beating it is architectural — **multiple independent load-balanced endpoints** so throughput scales without piling concurrency onto one proxy and inflating the prefill queue. Also worth: a quality-gated submission (the official score multiplies by `quality_pass_rate`, which the local harness assumes = 1.0).

---

## 1. Results table (Track 1 mixed — best of each lever)

| label | gpus | replicas | users | max_seqs | max_batch | prefix | chunked | err% | TTFT p95 | ITL p95 | chunks/s | score |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| **mix-4r-300u (official)** | 4 | 4 | 300 | 32 | 16384 | off | on | 0.0 | 1029 ms | 5.7 ms | 11649 | **149,776,620** |
| mix-4r-340u | 4 | 4 | 340 | 32 | 16384 | off | on | 0.0 | 1260 ms | 5.8 ms | 12301 | 142,810,000 |
| mix-4r-240u | 4 | 4 | 240 | 32 | 16384 | off | on | 0.0 | 785 ms | 6.1 ms | 10827 | 136,280,000 |
| mix-4r-ncp-300u (chunked **off**) | 4 | 4 | 300 | 32 | 16384 | off | **off** | 1.0 | 1498 ms | 6.5 ms | 9897 | 75,122,000 |
| mix-4r-400u (past knee) | 4 | 4 | 400 | 32 | 16384 | off | on | 6.4 | 3581 ms | 6.1 ms | 7923 | 33,950,000 |
| mix-4r-pc-300u (prefix **on**) | 4 | 4 | 300 | 32 | 16384 | **on** | on | 0.3 | 2990 ms | 7.0 ms | 9437 | 33,719,000 |
| mix-1r-base-80u (**baseline**, defaults) | 1 | 1 | 80 | 32 | 4096 | on | on | 0.0 | 4994 ms | 38.8 ms | 773 | 319,183 |
| — *optional boss fight (8 H100)* — | | | | | | | | | | | | |
| **mix-8r-460u (boss)** | 8 | 8 | 460 | 32 | 16384 | off | on | 0.5 | 803 ms | 6.6 ms | 17425 | **189,183,025** |
| mix-8r-480u | 8 | 8 | 480 | 32 | 16384 | off | on | 2.9 | 841 ms | 6.2 ms | 16732 | 186,770,000 |
| mix-8r-500u | 8 | 8 | 500 | 32 | 16384 | off | on | 3.3 | 1047 ms | 6.6 ms | 17089 | 149,740,000 |
| mix-8r-600u (error cliff) | 8 | 8 | 600 | 32 | 16384 | off | on | 8.7 | 1207 ms | 6.7 ms | 16937 | 143,580,000 |

(Text-track sweep — the prior speed-track work that established the levers — is in `experiments.md`.)

---

## 2. The engineering story

The score rewards *high throughput and low TTFT/ITL at once*, divided by GPU count. Three levers, found and validated on the text speed-track first, transfer directly to the mixed workload:

### Lever 1 — Co-locate the load client inside Modal (the foundational fix)
Running the load generator on the laptop let the residential uplink buffer at high SSE volume: TTFT p50 stayed low but p95/p99 exploded (~3 s / ~10 s) as first-token packets queued behind bulk traffic. Since the score divides by p95 TTFT, that alone pinned every result near the floor. `load_client_modal.py` runs the sharded generator as a CPU-only `modal.Function` **co-located with the endpoint** (launched with `modal run`), removing the laptop from the path entirely. For this track it was extended to emit the **full mixed workload** — it generates the same deterministic diagram PNG the harness uses and mixes image/long/text requests 25/20/55.

### Lever 2 — Compiled mode (`--no-fast-boot`)
CUDA-graph capture is the single biggest server knob: the baseline's eager mode runs at **ITL 38.8 ms**, while compiled mode holds **ITL ~5.7 ms** under load. Cold replicas are warmed by a short client warmup pass before measurement so the ramp isn't served in eager mode.

### Lever 3 — `max_num_batched_tokens` 4096 → 16384 (the decisive server find)
With bufferbloat gone, the bottleneck became **prefill admission** — requests weren't reaching decode fast enough, so decode slots sat idle. Raising the batch-token budget to **16384** unblocks admission and regularizes decode steps, pulling ITL and TTFT down — both directly on the score's denominator. (Tested past it: 32768 regresses as prefill steals decode cycles. 16384 is the peak.)

### Tuning the mixed workload on top of those levers
- **User sweep / the knee.** 4-GPU mixed peaks at **300 users** (149.8M, 0 errors). Below that (240u) leaves throughput on the table; above it (340u→400u) TTFT climbs faster than throughput until the queue saturates and errors appear.
- **Scale-out 1→4→8.** 1 replica (baseline) → 4 replicas (official) → 8 replicas (boss). The 8-GPU tier reaches **17.4k c/s and TTFT 803 ms** but the ÷8 divisor means it only wins when kept in the **near-zero-error** regime (460u). Push to 500u and errors hit 3.3%, collapsing the score back to the 4-GPU level — the clearest lesson of the sweep.

### What was measured and rejected (disciplined, not assumed)
- **Prefix caching ON:** 150M → **34M**. TTFT blew to 2990 ms — block-hash/KV contention outweighs any prompt-sharing benefit for short-output requests. `--no-prefix-cache` confirmed correct for mixed too.
- **Chunked prefill OFF:** 150M → **75M** (TTFT 1029→1498, errors appear). Chunked prefill ON is required to interleave image/long prefill with decode.
- **Over-subscription (4r/400u, 8r/600u):** TTFT and error rate spike together once the prefill queue saturates; the user-count numerator can't outrun it.

---

## 3. Why the ceiling sits where it does (honest accounting)

Both 8-replica and 4-replica runs plateau in aggregate throughput (~17.4k c/s on 8 GPUs, ~11.6k on 4 — i.e. **1.5×, not 2×**, for double the GPUs). That sub-linear scaling points to a **single Modal web-endpoint proxy** as the shared chokepoint: concurrency piles onto one proxy, deepening the prefill queue and coupling throughput to TTFT. Within that topology the knobs are exhausted — compiled mode (on), batch tokens (16384), seqs (32), prefix cache (off), chunked prefill (on), and replica/user balance are all at their measured optima. Going further is **architectural** (multiple independent load-balanced endpoints), which is outside the in-scope deploy knobs.

One scoring caveat: the official formula multiplies by `quality_pass_rate`, which the local harness assumes = 1.0. All scores here are throughput/latency-composite under that assumption.

---

## 4. Final configuration and exact commands

**Official Track 1 result — `mix-4r-300u` = 149,776,620 (4×H100).**

```bash
set PYTHONUTF8=1

# 1) Deploy: 4xH100, compiled, seq32, max_batch_tokens 16384, no prefix cache, chunked prefill on, ci128
python run_infertutor_experiment.py ^
  --label mix-4r ^
  --gpu-type H100 ^
  --replicas 4 ^
  --no-fast-boot ^
  --no-prefix-cache ^
  --max-seqs 32 ^
  --max-batch-tokens 16384 ^
  --concurrent-inputs 128 ^
  --mode mixed ^
  --deploy-only

# 2) Load from INSIDE Modal (co-located client, no laptop in the path)
modal run load_client_modal.py ^
  --url https://<you>--infertutor-mix-4r-serve.modal.run ^
  --label mix-4r-300u ^
  --mode mixed --users 300 --shards 16 --duration 150 --ramp-up 75 --total-gpus 4

# 3) Score (read the FULL integer from the JSON)
python score_infertutor.py results_infertutor\mix-4r-300u_mixed_300u_<ts>.json
```

**Optional boss fight — `mix-8r-460u` = 189,183,025 (8×H100):** identical flags with `--replicas 8`, then `modal run ... --users 460 --shards 20 --ramp-up 90 --total-gpus 8`.

---

## 5. Cleanup

All `infertutor-*` and `arena-loadgen` apps **stopped** after each run (`modal app stop ... --yes`); final `modal app list` shows **0 running**. The load-client app is serverless (CPU functions exit when the run completes). Nothing is billing.

| What | Value |
|---|---|
| **Track 1 official score (4×H100)** | **149,776,620** (`mix-4r-300u`: compiled, seq32, batch 16384, no-prefix-cache, 300 users, 0 errors) |
| Optional boss-fight score (8×H100) | 189,183,025 (`mix-8r-460u`: same config, 460 users, 0.5% errors) |
| Baseline (1×H100 default) | 319,183 (`mix-1r-base-80u`) → optimized+scaled is ~469× |
| Best TTFT p95 / ITL p95 | 803 ms / 5.7 ms |
| Best goodput | 17,425 chunks/s |
| Decisive levers | (1) load client into Modal; (2) compiled mode; (3) `max_num_batched_tokens` 16384 |
| Rejected (measured) | prefix-cache ON (34M), chunked-prefill OFF (75M), over-subscription past the knee |
| Root cause of residual ceiling | Single Modal-endpoint throughput ceiling (~17.4k c/s); scaling is sub-linear (÷GPUs grows faster than throughput). Closing it is a multi-endpoint architecture change |
| Live deployments | 0 (all stopped) |
