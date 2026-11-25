# Opt-in suites that need models or hardware

This page lists every test that skips unless you point it at a model
file or at GPU hosts, and tells you which ones to run after a given
kind of change. You need it before you merge a change the default
`rebar3 eunit`, `rebar3 proper` and `rebar3 ct` cannot check: those
runs use the stub backend and skip everything below when the
environment variables are unset, so a green default run says nothing
about llama.cpp, the GPU or the network.

Run the suites from the repository root. Paths for the tensor-parallel
suites are relative to the hosts' home unless absolute.

## Library suites on one machine

| Suite | What it checks | Environment | Needs |
|---|---|---|---|
| `erllama_real_model_SUITE` | Load and free, tokenize and decode, `kv_pack`/`kv_unpack` round trip, cold then warm completion and the warm speed-up, longest-prefix resume, chat templates, grammars and samplers, logprobs, chunked prefill, `verify/4`, ngram speculation, `continue/3`, device and placement options, shared reads during generation and unload | `LLAMA_TEST_MODEL` | a small GGUF (TinyLlama) |
| `erllama_chat_SUITE` | The chat autoparser NIF: templates, apply then parse, partial parse, tool, JSON-schema and thinking grammars | `LLAMA_TEST_MODEL` | the same GGUF |
| `erllama_stale_logits_SUITE` | A sequence left out of a decode answers `{error, stale_logits}`, and re-decoding its last token gives the token it would have emitted | `LLAMA_TEST_MODEL` | the same GGUF |
| `erllama_nif_tests` (EUnit, gated cases) | `step` without a prefill answers `no_logits`, context and batch overflow guards, bad chat content, split and device options, and the prefill fuzz property | `LLAMA_TEST_MODEL` | the same GGUF |
| `prop_erllama_nif` (PropEr) | `prop_prefill_never_crashes` | `LLAMA_TEST_MODEL` | the same GGUF |
| `erllama_family_SUITE` | Model family probe; warm hits on a recurrent model (rollback or the documented cold fallback, same reply); partial hits stay warm; SWA warm hits | `LLAMA_TEST_RECURRENT_MODEL`, `LLAMA_TEST_SWA_MODEL` (each group skips alone) | a Mamba GGUF, a Gemma 3 GGUF |
| `erllama_vision_SUITE` | mmproj capabilities, chat with an image, raw stream with a media marker, marker count mismatch, media with a session refused, sequential media requests; audio chat | `LLAMA_TEST_VISION_MODEL` + `LLAMA_TEST_MMPROJ`; audio case: `LLAMA_TEST_AUDIO_MODEL` + `LLAMA_TEST_AUDIO_MMPROJ` | a vision GGUF and its projector (SmolVLM-256M); an audio-capable pair |

```bash
LLAMA_TEST_MODEL=/path/to/tinyllama-1.1b-q4_k_m.gguf rebar3 ct \
    --dir apps/erllama/test --suite erllama_real_model_SUITE,erllama_chat_SUITE,erllama_stale_logits_SUITE
LLAMA_TEST_MODEL=/path/to/tinyllama-1.1b-q4_k_m.gguf rebar3 eunit --module=erllama_nif_tests
```

## Server suites

These run with stub models; only the cases below need a real file.

| Suite | Gated case | Environment |
|---|---|---|
| `erllama_server_http_SUITE` | the real-model cases; `real_model_image_chat` | `LLAMA_TEST_MODEL`; `LLAMA_TEST_VISION_MODEL` + `LLAMA_TEST_MMPROJ` |
| `erllama_server_rpc_SUITE` | the real-model cases, on the release node | `LLAMA_TEST_MODEL` |
| `erllama_server_agent_SUITE` | `real_model_reads_a_file` | `LLAMA_TEST_MODEL` (a tool-capable GGUF) |
| `erllama_server_compact_SUITE` | `real_model_summary` | `LLAMA_TEST_MODEL` |
| `erllama_server_convs_SUITE` | `real_model_warm_after_restart` | `LLAMA_TEST_MODEL` |
| `erllama_server_registry_tests` (EUnit) | `gguf_real_model_test` | `LLAMA_TEST_MODEL` |

## Tensor-parallel suites on CUDA hosts

Every suite below skips unless `ERLLAMA_TP_HOSTS` names the hosts (ssh
targets, comma-separated, in rank order), each prepared with
`scripts/tp_hosts.sh setup`, `sync` and `build`. The hosts must reach
each other by those names, and the build must be the test profile.
`erllama_tp_bootstrap_SUITE` is the exception: it needs no GPU and runs
in the default `rebar3 ct`.

Variables every suite reads (through `erllama_tp_hosts`):

| Variable | Meaning |
|---|---|
| `ERLLAMA_TP_HOSTS` | the hosts, in rank order |
| `ERLLAMA_TP_MODEL` | the GGUF the dense cases split (default `erllama-tp/tinyllama-1.1b-chat.gguf`) |
| `ERLLAMA_TP_NCCL_ENV` | extra environment for every rank node, space-separated `K=V` |
| `ERLLAMA_TP_NCCL_ENV_<Rank>` | extra environment for one rank (its `NCCL_SOCKET_IFNAME`, say) |

| Suite | What it checks | Extra environment | Needs |
|---|---|---|---|
| `erllama_tp_nccl_SUITE` | Matrix fixture, decode-sized FP32 and prefill-sized BF16 reductions, a missing rank, repeated open and close, an even and an uneven split against one GPU, serving through erllama, the transport NCCL chose and the bytes on the interface | `ERLLAMA_TP_EXPECT_NET` (`Socket` or `IB`), `ERLLAMA_TP_EXPECT_IF` (interfaces): without them the transport cases only log | two CUDA hosts |
| `erllama_tp_failure_SUITE` | A rank killed mid-stream, stale, duplicate and out-of-order commands, a cancel during prefill, each followed by a request served | none | two CUDA hosts |
| `erllama_tp_checkpoint_SUITE` | A checkpoint restored by a new group; a save missing a rank's share publishes nothing; GLM restores of one and four sequences, byte for byte | `ERLLAMA_TP_GLM` (GLM cases) | two CUDA hosts; a GLM-5.3-Flash GGUF |
| `erllama_tp_glm_SUITE` | GLM-5.3-Flash split against the same file on one GPU, four streams on one GPU and split, the decode sweep, the cache after a reload | `ERLLAMA_TP_GLM_REF` (fits one host), `ERLLAMA_TP_GLM`; `ERLLAMA_TP_GLM_SEQ` (sequences, default 4), `ERLLAMA_TP_GLM_UNIFIED=1` (one KV stream), `ERLLAMA_TP_GLM_RS` (`n_rs_seq`) | two hosts, GPUs free |
| `erllama_tp_rollback_SUITE` | Recurrent-state snapshots rolled back by 1 and 2 against decoding without drafts; rounds of one known token and two drafted ones | `ERLLAMA_TP_GLM_REF` (required), `ERLLAMA_TP_GLM` (defaults to it) | two hosts, GPUs free |
| `erllama_tp_mtp_SUITE` | MTP decode with and without drafts, divergence margin, rollback state against a fresh prefill, batched rounds (accept, reject, cancel, checkpoint, speed), four requests through `erllama_model`, the server bench, a two-turn cached conversation, and two refusal cases on one GPU | `ERLLAMA_TP_GLM`, `ERLLAMA_TP_GLM_MTP` (required); `ERLLAMA_TP_GLM_ONE` (one-GPU cases); `ERLLAMA_TP_MTP_SPLIT` (shares, e.g. `0.46,0.54`); `ERLLAMA_TP_BENCH_CLIENT` (`scripts/glm_serving/bench_client.py`, for `server_bench`); `ERLLAMA_TP_BENCH_OUT` (JSON output dir); `ERLLAMA_TP_BATCH_SPEED=0` (skip the speed part); `ERLLAMA_TP_RESUME_ONE=1` (stop after the one-request checkpoint compare) | two hosts, GPUs free; GLM-5.3-Flash and its MTP GGUF |
| `erllama_tp_capacity_SUITE` | A model larger than one GPU split over the group and served, each rank's peak memory and llama.cpp's placement; whether one host can hold it | `ERLLAMA_TP_BIG_MODEL` (required); `ERLLAMA_TP_CAPACITY_ONE_HOST=1` (the one-host case, which can run the host out of memory) | two hosts, GPUs free |

```bash
ERLLAMA_TP_HOSTS=hostA,hostB rebar3 ct --dir apps/erllama/test \
    --suite erllama_tp_nccl_SUITE,erllama_tp_failure_SUITE,erllama_tp_checkpoint_SUITE
```

## Which suite to run after which change

| You changed | Run |
|---|---|
| the cache, the save path, the publish protocol, the ladder | the default EUnit and PropEr runs, then `erllama_real_model_SUITE` and `erllama_family_SUITE`; `erllama_tp_checkpoint_SUITE` if the checkpoint payload or the KV format changed |
| NIF decode, sampling, KV operations (`apps/erllama/c_src`) | `erllama_nif_tests` and `prop_erllama_nif` with `LLAMA_TEST_MODEL`, `erllama_real_model_SUITE`, `erllama_stale_logits_SUITE`, `erllama_family_SUITE` |
| chat templates or tool-call parsing | `erllama_chat_SUITE`, `erllama_real_model_SUITE`, `erllama_server_agent_SUITE` |
| speculative decoding (ngram or MTP), the tick order | `erllama_speculative_tests` (stub), the ngram cases of `erllama_real_model_SUITE`, `erllama_stale_logits_SUITE`; for MTP, `erllama_tp_mtp_SUITE` (its `ERLLAMA_TP_GLM_ONE` cases run on one GPU) and `erllama_tp_rollback_SUITE` |
| tensor parallelism (`erllama_tp*`, `erllama_model_tp`, `erllama_tp_nif.cpp`) | `erllama_tp_bootstrap_SUITE`, then `erllama_tp_nccl_SUITE`, `erllama_tp_failure_SUITE`, `erllama_tp_checkpoint_SUITE`; the GLM suites for hybrid-model work |
| multimodal | `erllama_vision_SUITE` (with the audio pair if you touched audio), `real_model_image_chat` in `erllama_server_http_SUITE` |
| the vendored llama.cpp | every single-machine suite above, and on the CUDA hosts at least `erllama_tp_nccl_SUITE` (see [UPDATE_LLAMA.md](../apps/erllama/UPDATE_LLAMA.md) and [tensor-parallel.md](tensor-parallel-design.md#when-you-bump-llamacpp)) |
