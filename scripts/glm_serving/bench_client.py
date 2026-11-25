# Copyright (c) 2026 Benoit Chesneau. Licensed under the MIT License.
# See the LICENSE file at the project root.
"""One benchmark leg against an OpenAI-compatible chat endpoint.

    bench_client.py URL MODEL STREAMS|probe THINKING_OFF [EXTRA_JSON]

The four prompts of TensorFold's docs/dec.py, greedy, 250 tokens,
streamed. STREAMS=1 sends them one after another; STREAMS=4 sends all
four at once. THINKING_OFF is how the endpoint turns thinking off:
`reasoning_effort' (erllama) or `chat_template_kwargs' (TensorFold).
EXTRA_JSON is merged into each request body (e.g. erllama's
`{"speculative": {"type": "mtp", "n_max": 1}}').

`probe' sends the first prompt alone. Prints one JSON object: each
request's time to the first content token, its completion tokens (the
usage block), the bytes that arrived as reasoning, cached prompt tokens,
finish reason, the server's speculative block when it sends one and a
hash of the text; and the leg's aggregate output tokens/s (completion
tokens over the leg's wall time).
"""
import hashlib
import json
import sys
import threading
import time
import urllib.parse
import http.client

PROMPTS = [
    "Write five detailed paragraphs about the history of ocean navigation.",
    "Explain how a modern operating system schedules threads. Be thorough and use several paragraphs.",
    "Describe the engineering of suspension bridges in five paragraphs.",
    "Write five paragraphs on the development of the printing press and its consequences.",
]
MAXTOK = 250


def body(model, prompt, thinking_off, extra):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
         "max_tokens": MAXTOK, "stream": True, "stream_options": {"include_usage": True}}
    if thinking_off == "reasoning_effort":
        b["reasoning_effort"] = "none"
    else:
        b["chat_template_kwargs"] = {"enable_thinking": False}
    b.update(extra)
    return json.dumps(b).encode()


def one(url, model, prompt, thinking_off, extra, out, i):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
    t0 = time.time()
    conn.request("POST", u.path.rstrip("/") + "/v1/chat/completions", body(model, prompt, thinking_off, extra),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    rec = {"prompt": i, "status": r.status, "start": t0, "ttft": None, "completion_tokens": None,
           "cached_tokens": None, "finish_reason": None, "speculative": None, "chunks": 0,
           "reasoning_bytes": 0}
    text = []
    if r.status != 200:
        rec["error"] = r.read().decode(errors="replace")[:300]
    else:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("error"):
                rec["error"] = json.dumps(ev["error"])[:300]
            for ch in ev.get("choices", []):
                rc = (ch.get("delta") or {}).get("reasoning_content")
                if rc:
                    rec["reasoning_bytes"] += len(rc.encode())
                c = (ch.get("delta") or {}).get("content")
                if c:
                    if rec["ttft"] is None:
                        rec["ttft"] = time.time() - t0
                    text.append(c)
                    rec["chunks"] += 1
                if ch.get("finish_reason"):
                    rec["finish_reason"] = ch["finish_reason"]
            if ev.get("speculative"):
                rec["speculative"] = ev["speculative"]
            if ev.get("usage"):
                rec["completion_tokens"] = ev["usage"].get("completion_tokens")
                rec["cached_tokens"] = (ev["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
    rec["end"] = time.time()
    rec["seconds"] = rec["end"] - t0
    t = "".join(text)
    rec["text_sha"] = hashlib.sha256(t.encode()).hexdigest()[:16]
    rec["text_bytes"] = len(t.encode())
    conn.close()
    out[i] = rec


def main():
    url, model, mode, thinking_off = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    if mode == "probe":
        # One request, the first prompt: where its text arrives.
        out = [None]
        one(url, model, PROMPTS[0], thinking_off, extra, out, 0)
        print(json.dumps(out[0]))
        return
    streams = int(mode)
    out = [None] * len(PROMPTS)
    t0 = time.time()
    if streams == 1:
        for i, p in enumerate(PROMPTS):
            one(url, model, p, thinking_off, extra, out, i)
    else:
        th = [threading.Thread(target=one, args=(url, model, p, thinking_off, extra, out, i))
              for i, p in enumerate(PROMPTS)]
        for x in th:
            x.start()
        for x in th:
            x.join()
    wall = time.time() - t0
    toks = sum(r["completion_tokens"] or 0 for r in out)
    for r in out:
        r["start"] -= t0
        r["end"] -= t0
    print(json.dumps({"streams": streams, "wall": wall, "tokens": toks, "aggregate_tps": toks / wall,
                      "requests": out}))


if __name__ == "__main__":
    main()
