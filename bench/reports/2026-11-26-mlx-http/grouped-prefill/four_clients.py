"""Replay the four simultaneous requests against a fresh erllama HTTP server."""
import argparse
import importlib.util
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("client", help="path to mac_compare.py from commit ddc11057")
    parser.add_argument("port", type=int)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("mac_compare", args.client)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    client.PORT = args.port
    client.request("erllama", "Write a short sentence about rain.", 8)
    prompts = client.PROMPTS + [
        "Explain five ways to test a database transaction under concurrent writes. "
        "Give a detailed example and explain what each test proves."
    ]
    with ThreadPoolExecutor(4) as pool:
        start = time.monotonic()
        jobs = [pool.submit(client.request, "erllama", prompt, 128) for prompt in prompts]
        replies = [job.result() for job in jobs]
        elapsed = time.monotonic() - start
    if any(reply["tokens"] != 128 for reply in replies):
        raise ValueError("expected 128 output tokens per client")
    first = [reply["ttft_ms"] for reply in replies]
    print(json.dumps({
        "first_visible_ms": first,
        "median_first_visible_ms": statistics.median(first),
        "group_wall_s": elapsed,
        "total_tokens_per_s": 512 / elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
