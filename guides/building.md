# Building erllama

erllama is a single OTP application with a single NIF
(`erllama_nif.so`), in `apps/erllama/` of this repository. The first
compile builds the vendored `c_src/llama.cpp/` (about 3 minutes on a
fast machine), then compiles the NIF sources and a CRC table.
Subsequent builds reuse the cmake cache and finish in seconds.

## Toolchain requirements

| | Required | Notes |
|---|---|---|
| Erlang/OTP | **28** | rebar.config declares `{minimum_otp_vsn, "28"}`. |
| rebar3 | **3.25.0+** | Earlier 3.24.x is fine for compile but the CI pinned version is 3.25.0. |
| C++17 toolchain | clang 14+ or gcc 11+ | Apple clang as shipped on macOS works. |
| cmake | **3.20+** | llama.cpp's own minimum is 3.18; we set 3.20 for the FindErlang module. |
| pthreads | yes | Linked via CMake's `Threads::Threads`. |

Build-time dependencies are platform-specific; the recipes below
match what CI installs.

## Linux (Ubuntu 24.04 amd64 / arm64)

```bash
sudo apt-get install -y build-essential cmake
# Erlang/OTP 28 from erlef setup-beam (manual install also fine).
asdf install erlang 28.0 && asdf local erlang 28.0
asdf install rebar 3.25.0 && asdf local rebar 3.25.0
rebar3 compile
```

OpenMP is intentionally disabled in `c_src/CMakeLists.txt`
(`set(GGML_OPENMP OFF ...)`); the system `libgomp.a` ships without
`-fPIC` on stock Ubuntu, which would break the shared NIF link with
`R_X86_64_TPOFF32 against hidden symbol gomp_tls_data`. Disabling
OpenMP at the ggml level avoids that entirely; the GPU paths
(Metal/CUDA) are unaffected.

CUDA is off by default. Enable with:

```bash
ERLLAMA_OPTS=-DGGML_CUDA=ON rebar3 compile
```

`ERLLAMA_OPTS` goes to the CMake configure step and
`ERLLAMA_BUILDOPTS` to the llama.cpp build step, so
`ERLLAMA_BUILDOPTS=-j4` caps the parallel jobs.

### CUDA with NCCL (tensor parallelism)

To split a model over several machines you need CUDA and NCCL in the
build. Set the CUDA architecture of your GPU (121 for GB10) and point
`NCCL_ROOT` at a directory holding NCCL's `include/` and `lib/`:

```bash
ERLLAMA_OPTS="-DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DNCCL_ROOT=$HOME/nccl" \
ERLLAMA_BUILDOPTS=-j6 \
rebar3 compile
```

`scripts/tp_hosts.sh` prepares Linux CUDA hosts this way without root
(NCCL from the pip wheel, OTP 28.3.1, rebar3) and builds the test
profile on each host. See [tensor parallelism](tensor-parallel.md)
for forming the ranks and loading a split model.

## macOS (Apple Silicon and Intel)

```bash
brew install erlang@28 rebar3 cmake
echo 'export PATH="$(brew --prefix erlang@28)/bin:$PATH"' >> ~/.zshrc
rebar3 compile
```

Metal and Apple BLAS (Accelerate) are auto-detected and on by
default. Compile is ~30 s after the first ggml build is cached.

## FreeBSD (14.2 / 14.4)

```sh
# The cached FreeBSD VM image (or a freshly-installed system) ships
# an older libpcre2 than the git package in the latest pkg repo
# expects (PCRE2_10.47 not defined). Refresh first so git can load.
pkg install -y pcre2

# erllama needs OTP 28+; the base `erlang` package is 26.x.
# erlang-runtime28 installs OTP 28 under /usr/local/lib/erlang28.
pkg install -y erlang-runtime28 cmake bash gmake git

export PATH="/usr/local/lib/erlang28/bin:/usr/local/bin:$PATH"

# llama.cpp's build-info cmake script invokes `git rev-parse`. When
# the build directory's owner differs from the user (typical inside
# CI VMs), git refuses with "dubious ownership"; allow the path.
git config --global --add safe.directory "$PWD"

# rebar3 isn't always available as a pkg; fetch it once.
fetch https://github.com/erlang/rebar3/releases/download/3.25.0/rebar3 -o rebar3
chmod +x rebar3

./rebar3 compile
```

## Erlang ERTS detection

The build needs `erl_nif.h` from the Erlang installation. erllama
uses `c_src/CMake/FindErlang.cmake` (adopted from erlang-rocksdb)
which runs `erl -noshell -eval` to read `code:lib_dir/0` /
`code:root_dir/0` and exports `ERLANG_ERTS_INCLUDE_PATH`. If the
caller pre-sets the `ERTS_INCLUDE_DIR` environment variable, that
takes precedence (useful for cross-compilation or pinned headers).

## What the build produces

- `priv/erllama_nif.so`: the single NIF, statically linked against
  the vendored `c_src/llama.cpp` (libllama, libggml, ggml-cpu, plus
  the platform GPU/BLAS backends) and `c_src/crc32c.c`.
- `_build/default/lib/erllama/ebin/*.beam`: Erlang modules.
- `_build/cmake/` (under `apps/erllama/` in this repository): the
  CMake build dir, cached for incremental builds.

Besides the C sources, the NIF compiles six C++ shims:
`erllama_safe.cpp` (catches C++ exceptions before they reach
the BEAM), `erllama_chat_nif.cpp` (chat templates and output parsing),
`erllama_mtmd_nif.cpp` (the multimodal projector), `erllama_spec_nif.cpp`
(ngram speculation), `erllama_mtp.cpp` (the MTP drafter) and
`erllama_tp_nif.cpp` (tensor parallelism).

## Common build issues

- **`'erl_nif.h' file not found`**: `ERTS_INCLUDE_DIR` is wrong.
  `FindErlang.cmake` should resolve it automatically; if it fails,
  set the env var explicitly:
  `ERTS_INCLUDE_DIR=$(erl -noshell -eval 'io:format("~s",[filename:join([code:root_dir(),"erts-"++erlang:system_info(version),"include"])]),halt().') rebar3 compile`.
- **`R_X86_64_TPOFF32 against hidden symbol gomp_tls_data`**: your
  `libgomp.a` is non-PIC. erllama's CMakeLists already sets
  `GGML_OPENMP OFF` to avoid this. If you re-enabled OpenMP, build
  a PIC `libgomp` or leave it off.
- **`PCRE2_10.47 not defined`** when running git on FreeBSD: refresh
  `pcre2` first: `pkg install -y pcre2`. The cached VM image lags
  the latest repo.
- **macOS metal init slow on first model load**: the lazy
  `llama_backend_init` runs on the first `erllama:load_model/1` call
  and discovers Metal devices. eunit cases that load a model need
  a generator timeout >5 s; see
  `apps/erllama/test/erllama_nif_tests.erl:load_model_rejects_non_existent_path_test_/0`
  for the pattern.

## Verifying the build

```bash
rebar3 fmt --check
rebar3 compile
rebar3 xref
rebar3 dialyzer
rebar3 lint
rebar3 eunit
rebar3 ct          # real-model and hardware suites skip without their env vars
```

End-to-end against a real GGUF:

```bash
LLAMA_TEST_MODEL=/path/to/tinyllama-1.1b-chat.gguf \
    rebar3 ct --dir apps/erllama/test --suite erllama_real_model_SUITE
```

Without the env var the suite skips, so default `rebar3 ct` stays
green on machines without a model file.

## Bumping the vendored llama.cpp

See [UPDATE_LLAMA.md](../apps/erllama/UPDATE_LLAMA.md).

## Optional MLX-LM backend on Apple Silicon

The experimental backend loads a local Qwen3.5-family MLX text checkpoint.
The tested model is [Qwen3.8-27B, MLX four-bit](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit/tree/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c).
A Python helper holds the model on the GPU. Install the optional dependencies
in a virtual environment on the Mac (tested with arm64 Python 3.13).

```sh
python3.13 -m venv "$HOME/.venvs/erllama-mlx"
export ERLLAMA_MLX_PYTHON="$HOME/.venvs/erllama-mlx/bin/python"
"$ERLLAMA_MLX_PYTHON" -m pip install -r apps/erllama/priv/mlx/requirements.txt
```

```erlang
{ok, M} = erllama:load_model(#{backend => erllama_model_mlx,
                               model_path => "/models/Qwen3.8-27B-MLX-4bit"}).
{ok, #{reply := Text}} = erllama:complete(M, <<"Explain Erlang supervision.">>,
                                         #{temperature => 0.0, response_tokens => 32}).
ok = erllama:unload(M).
```

Set `context_opts => #{n_seq_max => 4}` for four active sequences. Prefill runs
per sequence. Ready decode rows share one MLX forward pass per scheduler tick.
The checkpoint holds each layer's attention or recurrent state and the last
token needed to resume. The normal erllama cache policy selects which prefixes
to save. Model files are hashed at load so a changed checkpoint gets a fresh
cache namespace. Packing and restoring use a temporary safetensors file in a
private directory on the Mac and the configured erllama RAM or disk tier.

See [Serving MLX over HTTP](https://github.com/benoitc/erllama/blob/v0.13.0/guides/server/http.md#serving-mlx-on-a-mac) for a streaming example.
On a 48 GiB M4 Pro serving four short cold prompts, per-sequence prefill delivered
first text in 2.84 s. Grouping fresh prompts in pairs took 3.19 s.

The current backend supports greedy text generation. The following checks define the next milestones.

| Feature | Acceptance check |
| --- | --- |
| Temperature, top-p and seed | Match direct MLX-LM with a fixed seed. Cancelling another stream must preserve the surviving request's sampler state. |
| Long-context serving | Measure first visible text, throughput and peak memory at one and four clients as prefix length grows. Measure the cost of changing batch membership. |
| Additional model families and embeddings | Check each family's tokenizer, cache layout and output against its MLX reference before enabling it through HTTP. |
| Vision and speculative decoding | Add a model-specific reference for image inputs or a drafter, then exercise cancellation and warm resume through the server. |

Set `ERLLAMA_MLX_TEST_MODEL` to the pinned checkpoint directory to run the
hardware cases from the repository root.

```sh
rebar3 ct --dir apps/erllama/test --suite erllama_mlx_SUITE
rebar3 ct --dir apps/erllama_server/test --suite erllama_server_http_SUITE --case mlx_chat_stream
```
