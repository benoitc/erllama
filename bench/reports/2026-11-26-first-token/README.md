# First visible text after prefill

On an Apple M4 Pro, Qwen2.5-1.5B-Instruct Q4_K_M showed its first visible text about 2 ms earlier with `early_stream_token => true` for one stream. The next text chunk took about 1 ms longer to arrive. The four-stream latency difference was small relative to run-to-run variation. The option defaults to `false`.

| Concurrent requests | Early sampling | First visible text (ms) | First to second visible chunk (ms) | Whole batch (ms) | Total output tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | off | 44.80 | 6.73 | 302.31 | 132.31 |
| 1 | on | 42.92 | 8.01 | 303.64 | 131.74 |
| 4 | off | 120.04 | 10.39 | 521.39 | 306.88 |
| 4 | on | 118.66 | 11.67 | 519.44 | 308.02 |

The paired median change in time to first visible text was −1.74 ms for one stream (14 of 20 pairs faster) and −0.82 ms for four streams (13 of 20). Throughput differed by less than 0.5% in either direction. Every request returned 40 tokens. Output matched across settings for each prompt and concurrency.

## Method

Two resident instances used the same GGUF and Metal backend in one erllama server, with four sequence slots, a 4,096-token context, a 512-token batch and greedy sampling. Requests were shorter than the 1,024-token checkpoint threshold. Each client streamed a chat completion over loopback HTTP and timed the first two nonempty content deltas. For the four-client case, clients started together. The batch rate uses the earliest request start and the latest finish.

After two warm-up pairs at each concurrency, each prompt group ran once per setting for 20 pairs. Setting order rotated to reduce clock and temperature effects. The table takes the median across batches and uses the median client latency within each batch. An earlier comparison using separate processes showed noticeable timing drift across process starts.

The four prompts were:

- Count from 1 to 200 in English, one number per line.
- List all the chemical elements with their atomic numbers.
- Give me fifty examples of common English verbs with definitions.
- List the countries of the world alphabetically, one per line.

The run exercised ordinary text streaming on one Mac. The separate correctness tests cover warm checkpoints and cancellation. [Paired HTTP measurements](paired.csv)
