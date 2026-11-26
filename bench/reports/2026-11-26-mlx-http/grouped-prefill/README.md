# Four concurrent MLX prompts on Apple M4 Pro

Keep per-request prefill and batched decode for Qwen3.8-27B. The pairwise
implementation had the lowest median first-text time among the grouped versions.
It took 3.19 s and delivered 32.37 tokens/s. Repeating the serving baseline in
the same experiment gave 2.84 s and 34.00 tokens/s.

Each run used a fresh server, the same loaded model, four slots, an empty RAM
prompt cache and the eight-token warm-up from the [serving benchmark](../README.md).
Four clients then sent the same four prompts concurrently with thinking off.
All 68 timed responses returned 128 tokens of visible text with zero cached
prompt tokens. Each run's throughput is 512 tokens divided by the time to the
last response. The table gives medians across runs, with first visible text
measured per client and then summarized per run.

| Implementation | Runs | First visible text | Total tokens/s |
| --- | ---: | ---: | ---: |
| Per-request prefill, serving baseline | 6 | 2.84 s | 34.00 |
| MLX-LM PromptProcessingBatch with right padding | 1 | 10.84 s | 7.97 |
| Grouped prefill with direct MLX cache operations | 2 | 6.26 s | 28.74 |
| Group only fresh prompts, then extract each row | 2 | 4.70 s | 31.89 |
| Also keep each row's allocated KV pages | 2 | 5.90 s | 29.11 |
| Keep the group through checkpoints and the suffix | 2 | 6.46 s | 27.71 |
| Group fresh prompts in pairs by length | 2 | 3.19 s | 32.37 |

The scheduler supplied all four prompts in one tick. With the test cache
policy it saved a checkpoint one token before each prompt ended. The next
tick processed the four one-token suffixes. The baseline spent 2.02 s in
the first tick and 0.27 s in the suffix tick. Grouping all fresh prompts
reduced the first tick to 1.73 s, but its suffix ticks took 1.03 s and
3.34 s. The paired version brought its suffix ticks down to 0.33 s and
0.51 s. These are elapsed worker-call times. They do not isolate GPU work,
memory stalls or checkpoint serialization. macOS reported memory pressure
during the single PromptProcessingBatch run. The results cover short cold
prompts on this Mac. [Per-run timings](runs.csv) · [Per-request timings](requests.csv)

After the timed series, a fresh-server check of the replay command with the
pairwise patch returned first text in 2.66 s at 36.57 total tokens/s. All
four prompts were cold. Including this third pairwise run gives medians of
2.98 s to first text and 32.92 total tokens/s.

## Correctness

With the pairwise patch, one HTTP client disconnected after visible text
arrived. The other three returned all 24 requested tokens. A subsequent
retry restored 55 of 56 prompt tokens and returned the same text as the
solo request. A separate small Qwen hybrid test compared each attention and
recurrent layer against independent prefill on Metal, including unequal
lengths, chunked prefill, slot reuse and checkpoint restore. The comparison
checked shapes and token positions and used tolerances of 0.001 relative and
0.0001 absolute for the numerical state.

## Reproduce

Start from the serving implementation at `5833ba6`. Use the pinned MLX
environment, model, context length, cache settings and prompts in the
[serving benchmark](../README.md). Use a fresh server for each run. Give it
four slots and the model name `qwen-27b`. The following commands use the
same HTTP client and first-visible-text calculation as the original run.

```sh
mkdir -p /tmp/erllama-prefill-client/driver
git show ddc11057:bench/reports/2026-11-26-mac-qwen-27b/mac_compare.py > /tmp/erllama-prefill-client/driver/mac_compare.py
python3 bench/reports/2026-11-26-mlx-http/grouped-prefill/four_clients.py /tmp/erllama-prefill-client/driver/mac_compare.py 8080
```

To run the paired version, apply the [experimental patch](pair-prefill.patch)
in a separate worktree at `5833ba6`, build and restart the server in that
worktree, then repeat the same client command. Verify zero cached input
tokens in the server's cache counters for the four measured requests.
