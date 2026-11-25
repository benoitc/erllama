#!/usr/bin/env python3
"""What sharing one model and its cache saves, measured over HTTP.

Against one running erllama_server (OpenAI /v1/chat/completions,
streamed, greedy), in order:

  1. first request    the model is not loaded: it loads on demand
  2. loaded           the same model, already loaded
  3. long, cold       a system prompt of about 2700 tokens nobody sent
                      before
  4. long, shared     the same system prompt, another question (as from
                      another client): the cache holds its first 2048
                      tokens (prefixes are kept at 2048-token boundaries)
  5. after eviction   the same, once the cache dropped its rows
  6. turn 2, whole    a second turn whose history a stateless client
                      sends whole (chat completions): only the
                      2048-token prefix is found
  7. turn 2, linked   the same second turn through the Responses API
                      (previous_response_id): the release passes turn
                      1's cache key, so turn 1's whole context is
                      restored

For each: time to first token, total time, prompt tokens, and the cache
counters before and after (GET /status). Standard library only.

  bench_cache.py URL MODEL --unload-cmd CMD --evict-cmd CMD

CMD are shell commands run between steps, for example with the release
in _build/prod/rel/erllama_server:
  --unload-cmd "bin/erllama unload qwen"
  --evict-cmd  "bin/erllama_server eval 'erllama_cache_meta_srv:gc().'"
"""
import argparse
import http.client
import json
import subprocess
import time
import urllib.parse

RULES = " ".join(
    f"Rule {i}: when a request mentions item {i}, answer with the code {1000 + i} "
    f"and explain the reason in one short sentence."
    for i in range(1, 86)
)
SYSTEM = "You are a support assistant for a warehouse. Follow these rules exactly. " + RULES


def post(url, body, stream=True):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    return conn, conn.getresponse()


def ask(url, model, messages, max_tokens=24):
    t0 = time.monotonic()
    conn, resp = post(url, {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
    })
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300]!r}")
    first, text, usage = None, [], {}
    for raw in resp:
        line = raw.strip()
        if not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        usage = chunk.get("usage") or usage
        for c in chunk.get("choices", []):
            piece = c.get("delta", {}).get("content")
            if piece:
                first = first or time.monotonic()
                text.append(piece)
    conn.close()
    t1 = time.monotonic()
    return {
        "ttft_ms": ((first or t1) - t0) * 1000,
        "total_ms": (t1 - t0) * 1000,
        "prompt_tokens": usage.get("prompt_tokens"),
        "text": "".join(text),
    }


def respond(url, model, body):
    """A non-streamed /v1/responses call: total time, id, usage."""
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    t0 = time.monotonic()
    conn.request("POST", "/v1/responses", json.dumps({"model": model, "temperature": 0, **body}),
                 {"content-type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    t1 = time.monotonic()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {data}")
    text = "".join(c.get("text", "") for o in data.get("output", []) for c in o.get("content", []))
    return {"ttft_ms": float("nan"), "total_ms": (t1 - t0) * 1000,
            "prompt_tokens": data.get("usage", {}).get("input_tokens"), "id": data["id"], "text": text}


def cache(url):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    conn.request("GET", "/status")
    c = json.loads(conn.getresponse().read()).get("cache", {})
    conn.close()
    return {k: c.get(k, 0) for k in ("hits_exact", "hits_resume", "hits_longest_prefix", "misses", "saves_cold", "saves_finish", "bytes_ram")}


def delta(before, after):
    return {k: after[k] - before[k] for k in after if after[k] != before[k]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("model")
    p.add_argument("--unload-cmd", required=True)
    p.add_argument("--evict-cmd", required=True)
    a = p.parse_args()

    def sh(cmd):
        subprocess.run(cmd, shell=True, check=True, capture_output=True)

    user = lambda q: {"role": "user", "content": q}
    sysm = {"role": "system", "content": SYSTEM}
    steps = []

    def step(name, messages, before_cmd=None, call=None):
        if before_cmd:
            sh(before_cmd)
        before = cache(a.url)
        r = call() if call else ask(a.url, a.model, messages)
        r["cache"] = delta(before, cache(a.url))
        steps.append((name, r))
        return r

    sh(a.unload_cmd)
    step("1 first request (loads the model)", [user("Say hi.")])
    step("2 loaded", [user("Say hello.")])
    step("3 long, cold", [sysm, user("A customer asks about item 7.")])
    # Let the saves of the long prompt land before reusing it.
    time.sleep(1.0)
    step("4 long, shared", [sysm, user("A customer asks about item 42.")])
    step("5 after eviction", [sysm, user("A customer asks about item 13.")], a.evict_cmd)
    sh(a.evict_cmd)
    q1, q2 = "A customer asks about item 7.", "And item 8?"
    t1 = respond(a.url, a.model, {"instructions": SYSTEM, "input": q1, "max_output_tokens": 24})
    time.sleep(1.0)
    step("6 turn 2, whole (stateless)", [sysm, user(q1), {"role": "assistant", "content": t1["text"]},
                                         user(q2)])
    step("7 turn 2, linked (Responses)", None, call=lambda: respond(
        a.url, a.model, {"previous_response_id": t1["id"], "input": q2, "max_output_tokens": 24}))

    print(f"{'step':32} {'first token':>11} {'total':>8} {'prompt':>7}  cache")
    for name, r in steps:
        ttft = "-" if r["ttft_ms"] != r["ttft_ms"] else f"{r['ttft_ms']:.0f} ms"
        print(f"{name:32} {ttft:>11} {r['total_ms']:5.0f} ms {r['prompt_tokens'] or 0:7}  {r['cache']}")


if __name__ == "__main__":
    main()
