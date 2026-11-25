# TinyLlama on erllama and Ollama

25 November 2026. We sent the same 44-token prompt to erllama and Ollama using TinyLlama
1.1B Chat Q4_K_M. Every reply contained 128 tokens. The figures are
medians of six requests per server on each machine, alternating which
server ran first. Client tokens/s covers the full request, from sending
the prompt on loopback to reading the final stream event.

| Machine | Server | Client tokens/s | First visible text | Full reply | Server-reported decode tokens/s |
|---|---|---:|---:|---:|---:|
| Apple M4 Pro | erllama | 223.3 | 30.3 ms | 573 ms | 225.0 |
| Apple M4 Pro | Ollama | 214.0 | 29.0 ms | 598 ms | 224.8 |
| NVIDIA GB10 | erllama | 259.7 | 12.3 ms | 493 ms | 262.6 |
| NVIDIA GB10 | Ollama | 240.2 | 62.9 ms | 533 ms | 273.3 |

On the **M4 Pro**, each timed request followed an unrelated prompt with
the model already loaded. Ollama reused one token on every run. The
first visible text arrived about 1.3 ms earlier with Ollama, and the
full reply finished about 25 ms earlier with erllama.

On the **GB10**, both servers were warmed once and then received the
measured prompt six times. Ollama reported faster decode, while erllama
delivered the first text about 51 ms earlier and finished about 40 ms
earlier. Compare the servers within each machine because the prompt
history and Ollama versions differ between machines.

## Reproducing the workload

Both servers used one GPU, one client, the same GGUF (SHA-256
`9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`)
and a 2,048-token context. The client streamed `POST /api/generate` with
`raw: true`, temperature 0, seed 42, repeat penalty 1.0 and a 128-token
output cap. Time to first visible text ends at the first nonempty
response chunk. Each server repeated its own output within a machine.
The generated wording differed between servers.

The Mac used erllama `3b7bf3f` and Ollama 0.40.17. Before each timed
request both servers received the raw warm-up text below with a two-token
output cap. Ollama's runner was unloaded before the warm-up so an earlier
copy of the measured prompt could not be reused. Model loading and the
warm-up were outside the timer.

```text
Mountain ranges form where tectonic plates meet. The process
```

The GB10 used the erllama-val integration build and Ollama 0.32.4.
Both servers received one warm-up on different text, followed by the
six timed requests. The measured prompt below was identical on both
machines and ended at the period.

```text
<|user|>
Write a numbered list of 30 practical ways to save water at home. Give one full sentence for each item.</s>
<|assistant|>
1.
```

[Mac request timings and output](mac.json) · [GB10 request timings and output](spark.json)

## Mac first-text trace

In a separate six-pair trace with a 16-token reply cap, the same prompt
after unrelated text reached the client in a median 30.2 ms on erllama
and 29.4 ms on Ollama. On erllama the first decode call took 26.1 ms
and the HTTP adapter delivered the first text in 0.2 ms. The decode
call samples the prompt logits and submits the next decode, so it can
include a wait for earlier Metal work. These measurements time call
boundaries. [Per-request trace after unrelated text](mac-first-text.csv)
