#!/usr/bin/env python3
"""What a second user of one model waits for.

A long greedy generation streams from an OpenAI-compatible server
(/v1/chat/completions); once it is halfway, a short question is sent to
the same model. Reported, as medians over the runs:

  - the short question's time to first token and total time;
  - the long stream's tokens per second before, during and after the
    overlap, and its longest gap between two tokens.

Standard library only. Usage:

  bench_shared_model.py URL MODEL [--runs N] [--long-tokens N] [--no-prompt-cache]

`--no-prompt-cache' sends llama-server's `cache_prompt: false', so its
slot cache does not reuse the short question between runs (erllama
keeps only prompts of at least 512 tokens, so its short question is
always prefilled; erllama refuses the unknown field).

for example, against erllama_server and against llama-server on the
same GGUF:

  bench_shared_model.py http://127.0.0.1:18100 qwen
  bench_shared_model.py http://127.0.0.1:18200 qwen
"""
import argparse
import http.client
import json
import statistics
import threading
import time
import urllib.parse

# Runs to the token limit on small instruct models (no early stop).
LONG_PROMPT = (
    "List every number from 1 to 600, each on its own line. "
    "Do not skip any number and do not use ellipses."
)
SHORT_PROMPT = "What is the capital of France? Answer in one word."


EXTRA = {}


def stream(url, model, prompt, max_tokens, on_token):
    """POST a streamed chat completion; call on_token(t) for every content delta."""
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        **EXTRA,
    })
    conn.request("POST", "/v1/chat/completions", body, {"content-type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:200]!r}")
    for raw in resp:
        line = raw.strip()
        if not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        for choice in chunk.get("choices", []):
            if choice.get("delta", {}).get("content"):
                on_token(time.monotonic())
    conn.close()


def one_run(url, model, long_tokens):
    long_times = []
    half = long_tokens // 2
    started = threading.Event()

    def long_job():
        def tok(t):
            long_times.append(t)
            if len(long_times) == half:
                started.set()
        stream(url, model, LONG_PROMPT, long_tokens, tok)
        started.set()

    th = threading.Thread(target=long_job)
    th.start()
    started.wait()
    short_times = []
    t0 = time.monotonic()
    stream(url, model, SHORT_PROMPT, 16, short_times.append)
    t1 = time.monotonic()
    th.join()
    if not short_times or len(long_times) < 10:
        raise RuntimeError("no tokens")

    def rate(a, b):
        ts = [t for t in long_times if a <= t <= b]
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 and ts[-1] > ts[0] else float("nan")

    gaps = [b - a for a, b in zip(long_times, long_times[1:])]
    return {
        "short_ttft_ms": (short_times[0] - t0) * 1000,
        "short_total_ms": (t1 - t0) * 1000,
        "long_tps_before": rate(long_times[0], t0),
        "long_tps_during": rate(t0, t1),
        "long_tps_after": rate(t1, long_times[-1]),
        "long_max_gap_ms": max(gaps) * 1000,
        "long_tokens": len(long_times),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("model")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--long-tokens", type=int, default=1500)
    p.add_argument("--no-prompt-cache", action="store_true")
    a = p.parse_args()
    if a.no_prompt_cache:
        EXTRA["cache_prompt"] = False
    # Warm up: the model loaded and the first compile done.
    stream(a.url, a.model, "Say hi.", 4, lambda t: None)
    runs = [one_run(a.url, a.model, a.long_tokens) for _ in range(a.runs)]
    for k in runs[0]:
        vals = [r[k] for r in runs]
        print(f"{k:18} median {statistics.median(vals):10.1f}   runs {', '.join(f'{v:.1f}' for v in vals)}")


if __name__ == "__main__":
    main()
