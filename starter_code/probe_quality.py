"""Quality probe for InferTutor Arena.

Sends the OFFICIAL prompts (image + long + text from prompts.json, same 256x192
diagram PNG the harness uses) to a deployed endpoint, NON-streaming, and captures
the full answers. Purpose: verify that answer QUALITY holds under a given server
config (specifically: compiled mode `--no-fast-boot` on MIXED multimodal traffic,
which the capstone spec warns "performed poorly" on multimodal).

The official score multiplies by quality_pass_rate, which the local scorer omits.
This probe is how we check that assumption instead of asserting it.

Usage:
    python probe_quality.py --url <ENDPOINT> --label compiled-mixed --mode mixed
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import time
import zlib
from pathlib import Path

import httpx

ROOT = Path(__file__).parent


def make_png_data_url(width: int = 256, height: int = 192) -> str:
    """Identical to load_test_infertutor.make_png_data_url (the harness image)."""
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


def looks_degenerate(text: str) -> list[str]:
    """Heuristic red flags for broken multimodal output. Returns list of issues."""
    issues = []
    t = (text or "").strip()
    if not t:
        issues.append("EMPTY")
        return issues
    if len(t) < 40:
        issues.append("TOO_SHORT(<40 chars)")
    # excessive single-token / phrase repetition
    words = t.split()
    if len(words) >= 12:
        # longest run of an identical word
        run = best = 1
        for a, b in zip(words, words[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        if best >= 6:
            issues.append(f"WORD_REPEAT(x{best})")
        # unique-word ratio
        uniq = len(set(w.lower() for w in words)) / len(words)
        if uniq < 0.30:
            issues.append(f"LOW_DIVERSITY({uniq:.2f})")
    # non-printable / replacement chars
    if t.count("�") > 2:
        issues.append("REPLACEMENT_CHARS")
    # mojibake-ish: very high ratio of non-ascii
    non_ascii = sum(1 for c in t if ord(c) > 127)
    if non_ascii / max(len(t), 1) > 0.30:
        issues.append("HIGH_NON_ASCII")
    return issues


def ask(client, url, model, system, user_content, max_tokens):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    t0 = time.perf_counter()
    r = client.post(f"{url.rstrip('/')}/v1/chat/completions", json=payload)
    dt = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return {"ok": False, "status": r.status_code, "text": r.text[:300], "ms": dt}
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return {"ok": True, "status": 200, "text": text, "ms": dt, "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", default="probe")
    ap.add_argument("--mode", default="mixed")  # only affects whether image prompts run
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    prompts = json.loads((ROOT / "prompts.json").read_text())
    system = prompts["system_prompt"]
    image_url = make_png_data_url()

    cases = []
    # ALL image prompts (the at-risk path)
    for p in prompts["image"]:
        cases.append(("image", p, [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": p},
        ]))
    # both long prompts
    for p in prompts["long"]:
        cases.append(("long", p, p))
    # first 4 text prompts
    for p in prompts["text"][:4]:
        cases.append(("text", p, p))

    print(f"[probe] {args.label}: {len(cases)} cases vs {args.url}")
    out = {"label": args.label, "url": args.url, "model": args.model, "cases": []}
    n_flags = 0
    with httpx.Client(timeout=180) as client:
        for i, (cat, prompt, content) in enumerate(cases, 1):
            res = ask(client, args.url, args.model, system, content, args.max_tokens)
            flags = looks_degenerate(res.get("text", "")) if res["ok"] else ["HTTP_ERROR"]
            if flags:
                n_flags += 1
            rec = {
                "i": i, "category": cat, "prompt": prompt,
                "ok": res["ok"], "status": res["status"],
                "latency_ms": round(res["ms"], 1),
                "answer": res.get("text", res.get("text", "")),
                "usage": res.get("usage", {}),
                "flags": flags,
            }
            out["cases"].append(rec)
            mark = "  OK " if not flags else "FLAG!"
            ans = (res.get("text") or "").replace("\n", " ")
            print(f"  [{mark}] {i:2d} {cat:5s} | {res['status']} {res['ms']:.0f}ms | {ans[:140]}")
            if flags:
                print(f"         ^ flags: {flags}")

    out["summary"] = {
        "total_cases": len(cases),
        "flagged_cases": n_flags,
        "image_cases": sum(1 for c in out["cases"] if c["category"] == "image"),
        "image_flagged": sum(1 for c in out["cases"] if c["category"] == "image" and c["flags"]),
    }
    out_dir = ROOT / "results_infertutor"
    out_dir.mkdir(exist_ok=True)
    fp = out_dir / f"quality_{args.label}_{int(time.time())}.json"
    fp.write_text(json.dumps(out, indent=2))
    print("=" * 70)
    print(f"  cases={len(cases)}  flagged={n_flags}  "
          f"image_flagged={out['summary']['image_flagged']}/{out['summary']['image_cases']}")
    print(f"  saved {fp}")
    print("=" * 70)


if __name__ == "__main__":
    main()
