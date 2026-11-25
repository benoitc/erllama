# Roadmap

What erllama does not do yet, with rough scope and the reason for each
item. It is a backlog, with no fixed milestone; what each release
shipped is in the CHANGELOG of each app. Issues and pull requests are
welcome.

## Tensor parallelism

- **More than two hosts.** A split model is verified on two hosts;
  larger groups are experimental and untested. Four-host testing is
  the next step.
- **Metal.** Tensor parallelism needs a CUDA build with NCCL.

## Speculative decoding

- **Ngram drafting under concurrency.** Ngram drafting runs only while
  its request is the only one on the model; with others it falls back
  to ordinary steps. MTP drafting already runs for several requests in
  one step.
- **More draft sources.** llama.cpp's ngram-map and ngram-cache
  drafters.
- **Drafting with logprobs or thinking.** A request asking for
  logprobs, or with thinking on, decodes without drafts.
- **Verifier context isolation.** `verify/4` leaves the context ready
  to decode after it runs, whatever its state before. A caller that
  needs the exact earlier state has to reset its sampler itself.
  Restoring it waits for a caller that needs it.

## Multimodal

Remaining after vision and audio input: caching the KV of requests
with media (they bypass the cache today), sessions and sticky
sequences with media, media prefill batched with other requests,
video and webp (needs an ffmpeg subprocess), and mtmd's experimental
text to speech.

## Transcription API (whisper.cpp)

Apart from the audio input above, which feeds audio to multimodal
models: a speech recognition surface for plain Whisper GGUFs with
timestamps and segments. whisper.cpp is its own runtime with its own
context shape; likely a sister application sharing the cache.

## Non-GGUF model loading

ONNX, safetensors, PyTorch checkpoints. llama.cpp does not load them,
so the path is a conversion step when the server fetches a model, or a
second backend on another runtime.

## Bit-exact multi-turn resume

A warm restore decodes the last cached token again to get its logits,
which can change a near-tied sample. `continue/3` avoids that cost for
a pinned session, but the sampler and its random state are still
rebuilt per request. Saving them with the KV at a turn boundary would
make a multi-turn reply identical to the unbroken stream. The same
goes for sampler state such as the repetition penalty's window.

## Pipeline parallelism

Splitting a model by layer ranges across nodes, hidden states passed
between them. llama.cpp's public API does not run a range of layers
(`llama_decode` always runs the whole model), so this needs a patch to
llama.cpp, upstreamed or carried locally.

## Cache

- **Semantic or approximate matching.** The cache is exact by design:
  a hit is a SHA-256 match on the rendered prompt. A semantic candidate
  proposer is deferred to a later major version.
- **KV state compression.** KV state is bulky (about 1 GB for 30k
  tokens on a 70B model). Generic lz4 or zstd helps a little;
  TurboQuant is unproven for this. Both wait for benchmark data.
- **Sharing across nodes.** The cache is node-local; sharing it through
  a disk tier on a shared filesystem or a small announce protocol is
  open.
- **Tool-id maps in KV files.** Not planned: storing an HTTP-layer tool
  map inside cache files would tie the cache's stable format to it.
  Revisit only if keeping them apart costs measurable I/O.

## Observability

- **Telemetry events.** Counters are plain atomics. A `telemetry`
  event surface would let you wire Prometheus, OpenTelemetry or statsd
  without forking the metrics module.
- **Per-GPU memory pressure.** The NVIDIA pressure source sums all
  GPUs. A multi-GPU host wants pressure and eviction per device;
  `list_devices/0` gives the per-device probe to build it on.

## Prompts and tokens

- **Incremental chat-template render.** `continue/3` takes only the
  new turn's tokens, but you still render the whole conversation to
  get them. A delta render that takes the prior token count and the
  new messages would skip the full template pass on every turn.
- **Streaming tokenize for very large prompts.** `tokenize/2` and the
  chat-template renderers allocate the whole output at once (up to
  about 256 MB at the 64 MiB text cap). A chunked variant would let
  prefill start earlier and cap native allocations.

## Remote devices

The ggml RPC backend is removed from the vendored llama.cpp and
`GGML_RPC` is forced off. Distribution happens at the Erlang level
instead (tensor parallelism, the server).

## Code structure

Refactors deferred until the documented boundaries are clean:

- one shared per-sequence MTP round in C, used by the single and the
  batched round, so a fix to verification or rollback is made once;
- `erllama_nif_decode.c` split by pipeline (step, ngram step, media
  prefill, embeddings, argmax forward), as `erllama_nif_mtp.c` already
  is;
- extractions from `erllama_model`: the prompt-cache lookup chain, the
  stop-sequence buffer, the sequence pool and sessions, batch building,
  and MTP batching.
