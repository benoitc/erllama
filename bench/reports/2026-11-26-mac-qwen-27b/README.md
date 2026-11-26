# Qwen3.8-27B on the Apple M4 Pro

26 November 2026. One streaming HTTP client sent three prompts (navigation, Erlang and bridges), each immediately repeated. The 48 GiB Mac ran one server at a time. The second pass reversed both the engine order and the prompt order. Each of the four fresh server processes received a separate warm-up. Timings start when the client sends the request. First text is the first visible content chunk. Throughput divides 128 output tokens by the full request time. Each row gives medians across six requests.

| Engine | Request | First visible text | Full reply | End-to-end tokens/s |
|---|---|---:|---:|---:|
| TensorFold | first | 673 ms | 10.86 s | 11.80 |
| TensorFold | retry | 113 ms | 10.44 s | 12.29 |
| erllama | first | 784 ms | 14.10 s | 9.07 |
| erllama | retry | 271 ms | 13.28 s | 9.64 |

Each engine used a 2,048-token context, one sequence, greedy decoding, thinking off, ordinary decoding and an enabled prompt cache. All 24 timed responses reached 128 tokens with visible content. Both engines reported the same input token counts per prompt. All first requests reported zero reused tokens, including after the engine order was reversed.

- TensorFold reused 49, 48, 47 prompt tokens respectively on the three retries.
- erllama reused 55, 54, 53 prompt tokens respectively on the three retries.

TensorFold 1.3.5 used the official native Metal binary with MLX affine four-bit weights at revision 22d8d538154e0e5b6f8dcb1fcc74a60a0798104c. erllama 48dda6c used llama.cpp Metal with the Unsloth UD-Q4_K_M GGUF at revision 4af51b3542f775b0a8e4b310bfb3f17e01c306c9. Both conversions are based on Qwen3.8-27B. Model conversion, GPU kernels, cache handling and server overhead all contribute to these end-to-end timings. The experiment did not isolate their individual costs.

[All request timings](requests.csv) · [erllama cache counters](erllama-cache-stats.json) · [TensorFold weights](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit/tree/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c) · [erllama weights](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4af51b3542f775b0a8e4b310bfb3f17e01c306c9)

## Reproducing the comparison

Build erllama from the repository root with Erlang/OTP 28 on your PATH and `rebar3 compile`. Use a separate working directory for the model downloads and generated results. Put the pinned MLX checkpoint in `mlx/`, the single UD-Q4_K_M file in `gguf/`, and the extracted TensorFold 1.3.5 Mac release beside them. Leave local port 19083 free. The driver uses the Erlang executable on PATH.

```sh
# Set this in the erllama checkout, then change into the prepared working directory.
driver="$PWD/bench/reports/2026-11-26-mac-qwen-27b/mac_compare.py"
cd "$benchmark_dir"
python3 "$driver" tensorfold
python3 "$driver" erllama
python3 "$driver" erllama --pass-number 2
python3 "$driver" tensorfold --pass-number 2
```

Each run writes its server log and per-request JSON in the working directory. The archived CSV uses the HTTP usage counters for TensorFold and the erllama server's cache counters for restored prompt tokens.
