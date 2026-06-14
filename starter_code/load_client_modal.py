"""Modal-native load client for InferTutor Arena.

Runs the sharded load generator from INSIDE Modal's network, co-located with the
vLLM endpoint. This removes the residential-uplink bufferbloat that inflated TTFT
p95 to ~3 s when the client ran on the laptop (which capped the score at ~45M
despite healthy server metrics).

Architecture:
- run_shard  : a CPU-only modal.Function. One container = one shard. Drives N
               async streaming users against the endpoint, returns RAW
               ttft/itl/latency samples.
- main       : local_entrypoint. Fans shards out with .map() so they run in
               parallel Modal containers, pools the raw samples, recomputes true
               percentiles, and writes ONE merged JSON in the scorer schema.

Auth uses the existing .modal.toml token (same one used for every deploy this
session). No new credentials, VM, GPU, or secret needed -- only the server needs
the huggingface secret.

Usage:
    modal run load_client_modal.py --url <ENDPOINT> --label <L> --users 400 \
        --shards 10 --duration 180 --ramp-up 40 --total-gpus 4 --mode text
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("arena-loadgen")
image = modal.Image.debian_slim(python_version="3.11").pip_install("httpx")

ROOT = Path(__file__).parent


def _percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p / 100), len(ordered) - 1)]


def make_png_data_url(width: int = 256, height: int = 192) -> str:
    """Deterministic diagram-like PNG, identical to load_test_infertutor.py so
    the mixed workload sends the same image payload the harness uses."""
    import base64
    import struct
    import zlib

    palette = [(245, 247, 250), (38, 92, 135), (228, 111, 71), (81, 168, 129)]
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            idx = ((x // 24) + (y // 24)) % len(palette)
            if 48 < x < 208 and 70 < y < 92:
                idx = 1
            if 48 < x < 208 and 132 < y < 154:
                idx = 2
            raw.extend(palette[idx])

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


@app.function(image=image, cpu=2.0, timeout=1800)
def run_shard(args: dict) -> dict:
    import asyncio
    import json as _j
    import random
    import time as _t

    import httpx

    system_prompt = args["system_prompt"]
    mode = args.get("mode", "text")
    text_prompts = args["text_prompts"]
    long_prompts = args["long_prompts"]
    image_prompts = args["image_prompts"]
    image_url = args.get("image_url", "")
    url = args["url"].rstrip("/")
    model = args["model"]
    n = args["users"]
    duration = args["duration"]
    ramp_up = args["ramp_up"]
    max_tokens = args["max_tokens"]
    req_timeout = args["request_timeout"]
    min_pause = args["min_pause"]
    max_pause = args["max_pause"]

    samples = {"ttft": [], "itl": [], "latency": [], "chunks": 0, "requests": 0, "errors": 0}

    def choose_user_content(rng):
        """Mirror load_test_infertutor.choose_messages: mixed = 25% image,
        20% long, 55% text. Single-mode runs send only that category."""
        m = mode
        if m == "mixed":
            roll = rng.random()
            if roll < 0.25:
                m = "image"
            elif roll < 0.45:
                m = "long"
            else:
                m = "text"
        if m == "image":
            return [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": rng.choice(image_prompts)},
            ]
        if m == "long":
            return rng.choice(long_prompts)
        return rng.choice(text_prompts)

    async def user_loop(stop):
        async with httpx.AsyncClient(timeout=req_timeout) as client:
            while not stop.is_set():
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": choose_user_content(random)},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "stream": True,
                }
                rs = _t.perf_counter()
                first = None
                cts = []
                ch = 0
                try:
                    async with client.stream(
                        "POST",
                        f"{url}/v1/chat/completions",
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
                                c = _j.loads(line)
                                content = c["choices"][0]["delta"].get("content", "")
                            except Exception:
                                continue
                            if content:
                                now = _t.perf_counter()
                                first = first or now
                                cts.append(now)
                                ch += 1
                        re = _t.perf_counter()
                        if first is None or ch == 0:
                            samples["requests"] += 1
                            samples["errors"] += 1
                            continue
                        gaps = [b - a for a, b in zip(cts, cts[1:])]
                        samples["requests"] += 1
                        samples["chunks"] += ch
                        samples["ttft"].append((first - rs) * 1000)
                        samples["itl"].append((sum(gaps) / len(gaps) * 1000) if gaps else 0.0)
                        samples["latency"].append((re - rs) * 1000)
                except Exception:
                    samples["requests"] += 1
                    samples["errors"] += 1
                await asyncio.sleep(random.uniform(min_pause, max_pause))

    async def driver():
        stop = asyncio.Event()
        tasks = []

        async def ramp():
            delay = ramp_up / max(n, 1) if ramp_up else 0
            for _ in range(n):
                if stop.is_set():
                    return
                tasks.append(asyncio.create_task(user_loop(stop)))
                if delay:
                    await asyncio.sleep(delay)

        start = _t.time()
        rt = asyncio.create_task(ramp())
        end = start + duration
        while _t.time() < end:
            await asyncio.sleep(1)
        stop.set()
        rt.cancel()
        for t in tasks:
            t.cancel()
        await asyncio.gather(rt, *tasks, return_exceptions=True)
        samples["elapsed"] = _t.time() - start
        return samples

    return asyncio.run(driver())


@app.local_entrypoint()
def main(
    url: str,
    label: str = "modal-client",
    mode: str = "text",
    users: int = 400,
    shards: int = 10,
    duration: int = 180,
    ramp_up: int = 40,
    max_tokens: int = 96,
    total_gpus: int = 4,
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    warmup: int = 25,
    min_pause: float = 0.2,
    max_pause: float = 1.2,
):
    prompts = json.loads((ROOT / "prompts.json").read_text())
    system_prompt = prompts["system_prompt"]
    text_prompts = prompts["text"]
    long_prompts = prompts["long"]
    image_prompts = prompts.get("image", text_prompts)
    image_url = make_png_data_url() if mode in ("image", "mixed") else ""

    base = users // shards
    rem = users % shards
    per = [base + (1 if i < rem else 0) for i in range(shards)]

    def mk(u, dur, ru):
        return {
            "system_prompt": system_prompt,
            "mode": mode,
            "text_prompts": text_prompts,
            "long_prompts": long_prompts,
            "image_prompts": image_prompts,
            "image_url": image_url,
            "url": url,
            "model": model,
            "users": u,
            "duration": dur,
            "ramp_up": ru,
            "max_tokens": max_tokens,
            "request_timeout": 180,
            "min_pause": min_pause,
            "max_pause": max_pause,
        }

    if warmup > 0:
        warm_args = [mk(max(per[i] // 2, 1), warmup, min(warmup, 10)) for i in range(shards)]
        print(f"[warmup] {shards} shards x ~{max(per[0] // 2, 1)} users for {warmup}s to warm replicas ...")
        list(run_shard.map(warm_args))

    args_list = [mk(per[i], duration, ramp_up) for i in range(shards)]
    print(f"[measure] {shards} shards driving {users} users ({per}) for {duration}s vs {url}")
    t0 = time.time()
    collected = list(run_shard.map(args_list))
    wall = time.time() - t0

    ttft = [x for s in collected for x in s["ttft"]]
    itl = [x for s in collected for x in s["itl"]]
    latency = [x for s in collected for x in s["latency"]]
    total_chunks = sum(s["chunks"] for s in collected)
    total_requests = sum(s["requests"] for s in collected)
    total_errors = sum(s["errors"] for s in collected)
    mean_elapsed = sum(s["elapsed"] for s in collected) / len(collected)
    err = total_errors / max(total_requests, 1)
    cps = total_chunks / mean_elapsed if mean_elapsed else 0.0

    results = {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": err,
        "total_stream_chunks": total_chunks,
        "ttft_p50_ms": _percentile(ttft, 50),
        "ttft_p95_ms": _percentile(ttft, 95),
        "ttft_p99_ms": _percentile(ttft, 99),
        "itl_p50_ms": _percentile(itl, 50),
        "itl_p95_ms": _percentile(itl, 95),
        "latency_p50_ms": _percentile(latency, 50),
        "latency_p95_ms": _percentile(latency, 95),
        "aggregate_stream_chunks_per_s": cps,
        "requests_per_s": total_requests / mean_elapsed if mean_elapsed else 0.0,
    }
    config = {
        "label": label,
        "model": model,
        "mode": mode,
        "users": users,
        "shards": shards,
        "duration": duration,
        "ramp_up": ramp_up,
        "max_tokens": max_tokens,
        "total_gpus": total_gpus,
    }

    out_dir = ROOT / "results_infertutor"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{label}_{mode}_{users}u_{int(time.time())}.json"
    out.write_text(json.dumps({"config": config, "results": results}, indent=2))

    print("=" * 64)
    print(f"  shards={shards}  wall={wall:.1f}s  mean_elapsed={mean_elapsed:.1f}s")
    print(f"  requests={total_requests}  errors={total_errors}  err%={100 * err:.2f}")
    print(f"  TTFT p50/p95={results['ttft_p50_ms']:.0f}/{results['ttft_p95_ms']:.0f} ms"
          f"   ITL p95={results['itl_p95_ms']:.1f} ms")
    print(f"  throughput={cps:.1f} chunks/s")
    print(f"  Saved {out}")
    print("=" * 64)
