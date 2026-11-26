# MLX HTTP serving on Apple M4 Pro

Qwen3.8-27B on a 48 GiB Mac. The client used the OpenAI-compatible
`/v1/chat/completions` endpoint with greedy sampling, thinking off and
streaming. The model was already loaded. The timings include the HTTP request,
prefill, generation and the final usage event. The first-text measurement uses
the first non-empty content delta.

| erllama MLX | First visible text | Full reply | End-to-end tokens/s |
| --- | ---: | ---: | ---: |
| One client, cold prompt | 727 ms | 10.01 s | 12.79 |
| One client, immediate retry | 186 ms | 9.43 s | 13.57 |
| Four clients, cold prompts | 2764 ms | 14.24 s for all four | 35.95 total |

The single-client rows give medians across six requests each. Every initial
request had zero cached tokens. The retries restored 55, 54 and 53 prompt
tokens for navigation, Erlang and bridges respectively, in both passes. The
four-client figure comes from one group of four simultaneous streams, each
of which returned 128 visible output tokens with thinking off. [Request timings](requests.csv)

On the same Mac and single-client workload, the earlier TensorFold run with
the same weights measured 11.80 tokens/s cold and 12.29 on retry. First text
on retry arrived in 113 ms. The earlier erllama llama.cpp run with the
UD-Q4_K_M GGUF measured 9.07 and 9.64 tokens/s respectively. These runs took
place in different measurement windows. [Original measurements](https://github.com/erllama/erllama/tree/ddc11057a2f2e1f4dfcb0e6d0df0fb162ac1e353/bench/reports/2026-11-26-mac-qwen-27b)

The [grouped prefill follow-up](grouped-prefill/README.md) repeats the four-client workload with six variants and reports where the first-token time went.

## Reproduction

Use the [Qwen checkpoint](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit/tree/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c)
with MLX 0.32.5 and MLX-LM 0.32.2. Register its directory with
`n_ctx => 2048`. Set the RAM cache budget to 1 GiB, both minimum token
thresholds to 32, and the boundary trim and alignment to one token.

Use the three prompts and streaming client from the
[earlier Mac comparison](https://github.com/erllama/erllama/blob/ddc11057a2f2e1f4dfcb0e6d0df0fb162ac1e353/bench/reports/2026-11-26-mac-qwen-27b/mac_compare.py).
For each single-client pass, start a fresh server with one slot, warm it
with the eight-token rain request, then request 128 tokens for each prompt
and immediately repeat it. Reverse the prompt order in the second pass.
Check the server's cache counters for each request.

For the concurrent pass, start a fresh server with four slots and use
the same warm-up. Submit the three prompts and the following fourth prompt
at the same time. Aggregate throughput is the sum of returned tokens divided
by the wall time from submission to the last completed request.

> Explain five ways to test a database transaction under concurrent writes. Give a detailed example and explain what each test proves.
