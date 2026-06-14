"""Sharded (multi-process) load tester for InferTutor Arena.

A single asyncio event loop saturates well before it can drive several hundred
concurrent streaming users: the ramp coroutine gets starved and per-request
TTFT/ITL get inflated by client-side scheduling delay. This driver shards the
user pool across N OS processes (each with its own event loop + GIL), pools the
RAW ttft/itl/latency samples from every shard, recomputes true percentiles, and
writes ONE merged JSON in the exact schema score_infertutor.py reads.

Nothing about the server config changes -- this only fixes the client so the
measured server metrics are real instead of client-bottlenecked.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import random
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
PROMPTS = json.loads((ROOT / "prompts.json").read_text())


def choose_messages(mode: str) -> list[dict]:
    system = {"role": "system", "content": PROMPTS["system_prompt"]}
    if mode == "long":
        return [system, {"role": "user", "content": random.choice(PROMPTS["long"])}]
    # text (default for these runs)
    return [system, {"role": "user", "content": random.choice(PROMPTS["text"])}]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p / 100), len(ordered) - 1)]


async def user_loop(args: dict, samples: dict, stop_event: asyncio.Event):
    async with httpx.AsyncClient(timeout=args["request_timeout"]) as client:
        while not stop_event.is_set():
            payload = {
                "model": args["model"],
                "messages": choose_messages(args["mode"]),
                "max_tokens": args["max_tokens"],
                "temperature": 0.2,
                "stream": True,
            }
            request_start = time.perf_counter()
            first_chunk_at = None
            chunk_times = []
            chunks = 0
            try:
                async with client.stream(
                    "POST",
                    f"{args['url'].rstrip('/')}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status_code != 200:
                        samples["requests"] += 1
                        samples["errors"] += 1
                        await resp.aread()
                        continue
                    async for line in resp.aiter_lines():
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                        try:
                            chunk = json.loads(line)
                            content = chunk["choices"][0]["delta"].get("content", "")
                        except Exception:
                            continue
                        if content:
                            now = time.perf_counter()
                            first_chunk_at = first_chunk_at or now
                            chunk_times.append(now)
                            chunks += 1
                request_end = time.perf_counter()
                if first_chunk_at is None or chunks == 0:
                    samples["requests"] += 1
                    samples["errors"] += 1
                    continue
                gaps = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
                ttft = (first_chunk_at - request_start) * 1000
                itl = (sum(gaps) / len(gaps) * 1000) if gaps else 0.0
                latency = (request_end - request_start) * 1000
                samples["requests"] += 1
                samples["chunks"] += chunks
                samples["ttft"].append(ttft)
                samples["itl"].append(itl)
                samples["latency"].append(latency)
            except Exception:
                samples["requests"] += 1
                samples["errors"] += 1
            await asyncio.sleep(random.uniform(args["min_pause"], args["max_pause"]))


async def worker_async(args: dict) -> dict:
    samples = {"ttft": [], "itl": [], "latency": [], "chunks": 0, "requests": 0, "errors": 0}
    stop_event = asyncio.Event()
    tasks = []
    n = args["users"]

    async def ramp():
        delay = args["ramp_up"] / max(n, 1) if args["ramp_up"] else 0
        for _ in range(n):
            if stop_event.is_set():
                return
            tasks.append(asyncio.create_task(user_loop(args, samples, stop_event)))
            if delay:
                await asyncio.sleep(delay)

    started = time.time()
    ramp_task = asyncio.create_task(ramp())
    end = started + args["duration"]
    while time.time() < end:
        await asyncio.sleep(1)
    stop_event.set()
    ramp_task.cancel()
    for t in tasks:
        t.cancel()
    await asyncio.gather(ramp_task, *tasks, return_exceptions=True)
    samples["elapsed"] = time.time() - started
    return samples


def worker(args: dict, q):
    q.put(worker_async_run(args))


def worker_async_run(args: dict) -> dict:
    return asyncio.run(worker_async(args))


def main():
    parser = argparse.ArgumentParser(description="Sharded load tester")
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--mode", choices=["text", "long"], default="text")
    parser.add_argument("--users", type=int, default=400, help="TOTAL users across all shards")
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--ramp-up", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--min-pause", type=float, default=0.2)
    parser.add_argument("--max-pause", type=float, default=1.2)
    parser.add_argument("--label", default="sharded")
    parser.add_argument("--total-gpus", type=int, default=1)
    args = parser.parse_args()

    shards = max(1, args.shards)
    base = args.users // shards
    rem = args.users % shards
    per_shard = [base + (1 if i < rem else 0) for i in range(shards)]

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    print(f"Launching {shards} shards driving {args.users} total users "
          f"({per_shard}) against {args.url}")
    t0 = time.time()
    for i in range(shards):
        wargs = {
            "url": args.url,
            "model": args.model,
            "mode": args.mode,
            "users": per_shard[i],
            "duration": args.duration,
            "ramp_up": args.ramp_up,
            "max_tokens": args.max_tokens,
            "request_timeout": args.request_timeout,
            "min_pause": args.min_pause,
            "max_pause": args.max_pause,
        }
        p = ctx.Process(target=worker, args=(wargs, q))
        p.start()
        procs.append(p)

    collected = [q.get() for _ in procs]
    for p in procs:
        p.join()
    wall = time.time() - t0

    ttft = [x for s in collected for x in s["ttft"]]
    itl = [x for s in collected for x in s["itl"]]
    latency = [x for s in collected for x in s["latency"]]
    total_chunks = sum(s["chunks"] for s in collected)
    total_requests = sum(s["requests"] for s in collected)
    total_errors = sum(s["errors"] for s in collected)
    mean_elapsed = sum(s["elapsed"] for s in collected) / len(collected)

    error_rate = total_errors / max(total_requests, 1)
    aggregate_cps = total_chunks / mean_elapsed if mean_elapsed else 0.0

    results = {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": error_rate,
        "total_stream_chunks": total_chunks,
        "ttft_p50_ms": percentile(ttft, 50),
        "ttft_p95_ms": percentile(ttft, 95),
        "ttft_p99_ms": percentile(ttft, 99),
        "itl_p50_ms": percentile(itl, 50),
        "itl_p95_ms": percentile(itl, 95),
        "latency_p50_ms": percentile(latency, 50),
        "latency_p95_ms": percentile(latency, 95),
        "aggregate_stream_chunks_per_s": aggregate_cps,
        "requests_per_s": total_requests / mean_elapsed if mean_elapsed else 0.0,
    }
    config = {
        "label": args.label,
        "model": args.model,
        "mode": args.mode,
        "users": args.users,
        "shards": shards,
        "duration": args.duration,
        "ramp_up": args.ramp_up,
        "max_tokens": args.max_tokens,
        "total_gpus": args.total_gpus,
    }

    out_dir = ROOT / "results_infertutor"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{args.label}_{args.mode}_{args.users}u_{int(time.time())}.json"
    out_file.write_text(json.dumps({"config": config, "results": results}, indent=2))

    print("=" * 60)
    print(f"  shards={shards}  wall={wall:.1f}s  mean_elapsed={mean_elapsed:.1f}s")
    print(f"  requests={total_requests}  errors={total_errors}  err%={100*error_rate:.2f}")
    print(f"  TTFT p95={results['ttft_p95_ms']:.1f} ms   ITL p95={results['itl_p95_ms']:.1f} ms")
    print(f"  throughput={aggregate_cps:.1f} chunks/s   req/s={results['requests_per_s']:.2f}")
    print(f"  Saved {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
