# How long a KV checkpoint pauses another stream

Two TinyLlama requests shared an Apple M4 Pro. One request finished and saved its
KV state to RAM while the other kept generating. The times below are milliseconds,
reported as median / p95 over 30 runs per cell.

| Prompt tokens on finishing stream | Packed bytes | Time in KV pack | Neighbor token gap, saving off | Neighbor token gap, saving on |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 3,336,480 | 5.49 / 5.97 | 6.00 / 6.18 | 6.55 / 7.09 |
| 512 | 11,991,840 | 6.28 / 7.11 | 6.17 / 6.40 | 7.45 / 8.40 |
| 1,536 | 35,072,800 | 9.09 / 12.28 | 6.73 / 7.78 | 10.65 / 15.32 |

At 1,536 prompt tokens the 35.1 MB checkpoint increased the median gap by
3.92 ms and the p95 gap by 7.54 ms. At 128 prompt tokens the corresponding
increases were 0.55 ms and 0.91 ms. The synchronous pack time alone
overstates the added delay a neighboring client observes on this setup.

## Measurement

erllama `a1cd337`, Erlang/OTP 28.3.1, macOS 27.2, M4 Pro with 48 GiB,
TinyLlama 1.1B Chat GGUF (SHA-256
`9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`). The model used Metal, two sequence slots, unified
KV, a 4,096-token context and a 512-token batch. Each trial loaded a fresh
model, warmed it with four tokens, and used a separate cache namespace. Each
pair used the same prompts and alternated which condition went first.

After one token from the longer request, the client started a second stream.
The requests generated 20 and 48 tokens respectively. Cold and periodic saves
were suppressed. The finish-save threshold allowed exactly one save in the
enabled trial and zero in its control. Every measured prompt was cold.

The driver traces the start of the finish-save decision and reads the packed
payload length without copying it. Pack time comes from the model's
`pack_total_ns` counter. The observer timestamps token messages in its own
Erlang process. A token already sent may arrive as the save begins. We measure
the largest of the three adjacent gaps straddling that event in both conditions.
This is the pause visible to a streaming library client. HTTP buffering,
persistent disk I/O and multi-host checkpoints need separate measurements.
p95 uses the nearest-rank estimator (29th of 30 observations).

## Reproduce

From the repository root, set `LLAMA_TEST_MODEL` to a local copy of TinyLlama.
The CSV records one row for each of the 180 trials.

```sh
rebar3 compile
run_dir=$(mktemp -d)
erlc -Werror -o "$run_dir" bench/reports/2026-11-26-checkpoint-pause/checkpoint_pause.erl
PAUSE_OUTPUT="$run_dir/pause.csv" erl +S 8:8 -noshell -pa "$run_dir" _build/default/lib/*/ebin -eval 'checkpoint_pause:run(),halt().'
```

[Trial data](pause.csv) · [Benchmark driver](checkpoint_pause.erl)
