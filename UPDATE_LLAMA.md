# Updating the vendored llama.cpp

Barrel Inference vendors a pinned copy of llama.cpp under `c_src/llama.cpp/`.
The current pin is **b9585**.

This file documents the bump procedure.

## Why pin

- Reproducible builds: every developer and CI run compiles the same
  source.
- Hex.pm-friendly: published packages contain the full source so no
  network access is needed at install time.
- Backend stability: llama.cpp moves fast, especially in the model
  zoo. We control when we adopt new architectures.

## What we ship

We vendor only the parts we need. Currently:

```
c_src/llama.cpp/
  CMakeLists.txt            llama.cpp's top-level CMake
  LICENSE                   MIT (dual MIT/Apache, llama.cpp pick MIT)
  cmake/                    CMake helpers (toolchain files, etc)
  include/                  public headers (llama.h, etc)
  src/                      llama core (model.cpp, context.cpp, etc)
  common/                   common_chat_* + jinja (autoparser path)
  vendor/                   header-only deps common/ links against
    cpp-httplib             pulled in by common's hf-cache / download
    nlohmann                nlohmann::json used by common chat code
    miniaudio, sheredom, stb other small header-only deps
  ggml/
    CMakeLists.txt
    cmake/                  ggml CMake helpers (common.cmake, GitVars.cmake)
    include/                public ggml headers
    src/
      CMakeLists.txt
      ggml*.c, ggml*.cpp, ggml*.h  core ggml + frontends
      gguf.cpp                     GGUF file format
      ggml-cpu/                    CPU SIMD kernels (mandatory)
      ggml-metal/                  Apple GPU backend (Apple Silicon)
      ggml-cuda/                   NVIDIA GPU backend (Linux x86-64)
      ggml-blas/                   BLAS backend (OpenBLAS / Accelerate)
```

Excluded (unused or out-of-scope for v1):

- `tools/`, `examples/`, `tests/`, `docs/`, `models/`, `gguf-py/`,
  `benches/`, `ci/`, `scripts/`, `grammars/`, `.git/`,
  `.github/`, `AUTHORS`, `.devops/`, `app/`
- ggml backends we do not link: Vulkan, SYCL, OpenCL, CANN, Hexagon,
  HIP, MUSA, RPC, ZDNN, ZenDNN, Virtgpu, Webgpu, OpenVINO

If a user needs one of the excluded backends they can build Barrel Inference
against an unvendored llama.cpp via `git+` rebar dep instead of the
hex package; that path is supported but unsupported in this scaffold.

## Bumping

Pick a tag from <https://github.com/ggml-org/llama.cpp/tags>. Newer
tags are usually fine; check the changelog for breaking C-API changes
to `llama_state_seq_*` (the cache layer depends on those).

```sh
# 1. Clone the new tag into a scratch directory.
cd /tmp
git clone --depth=1 --branch=<TAG> https://github.com/ggml-org/llama.cpp llama.cpp.new

# 2. Sync the parts we vendor.
cd /Users/benoitc/Projects/barrel_inference
rm -rf c_src/llama.cpp
mkdir -p c_src/llama.cpp/ggml/src

cp -r /tmp/llama.cpp.new/{src,include,cmake,common,vendor,CMakeLists.txt,LICENSE} \
      c_src/llama.cpp/
cp -r /tmp/llama.cpp.new/ggml/{include,cmake,CMakeLists.txt} \
      c_src/llama.cpp/ggml/
cp /tmp/llama.cpp.new/ggml/src/CMakeLists.txt \
   c_src/llama.cpp/ggml/src/
cp /tmp/llama.cpp.new/ggml/src/ggml*.c \
   /tmp/llama.cpp.new/ggml/src/ggml*.cpp \
   /tmp/llama.cpp.new/ggml/src/ggml*.h \
   /tmp/llama.cpp.new/ggml/src/gguf.cpp \
   c_src/llama.cpp/ggml/src/
cp -r /tmp/llama.cpp.new/ggml/src/{ggml-cpu,ggml-metal,ggml-cuda,ggml-blas} \
      c_src/llama.cpp/ggml/src/

# 3. Rebuild and run the full test gauntlet.
rm -rf _build
rebar3 compile
rebar3 fmt --check && rebar3 lint && rebar3 xref \
    && rebar3 eunit && rebar3 proper && rebar3 ct

# 4. Update the pin reference in this file and in
#    c_src/llama.cpp/.version (if present).

# 5. Commit with a message naming the new tag.
```

## Known local patches

The vendored tree carries a handful of additive patches the bumper has
to re-apply when upstream churns the affected lines. Grep for
`barrel_inference local addition` to find them.

| File | What it adds | Why |
|---|---|---|
| `include/llama.h` :: `llama_model_params` | `bool prefetch` | Threads `POSIX_MADV_WILLNEED` vs `POSIX_MADV_RANDOM` through to the mmap, used by the server's `weight_residency = lazy / lazy_then_pin_resident` modes. |
| `src/llama-model.cpp` :: `llama_model_default_params`, `init_mappings` call | initialises `prefetch = true` (back-compat); plumbs it through to `init_mappings`. | Same. |
| `src/llama-model.h` :: `llama_model` | `n_mappings()` + `get_mapping()` member fns. | So the NIF can `mincore` / `mlock` the model's mmap regions without breaking pimpl encapsulation. |
| `src/llama-model.cpp` :: same impls + `llama_model_n_mappings` / `llama_model_get_mapping` C API. | Same. |
| `common/chat.h` :: `common_chat_templates_inputs` | `bool skip_parser_synthesis` | Lets the NIF render the prompt without paying the per-request PEG synthesis cost; pairs with the cached params arena. |
| `common/chat.cpp` :: `common_chat_templates_apply_jinja` | Branches to `common_chat_template_direct_apply_impl` when `skip_parser_synthesis` is true. | Same. |

When upstream rewrites any of these chunks, the bump procedure has to
re-apply the diff. None are intrusive — each is a few lines added next
to existing functionality.

## Configuration knobs

The CMake configure step honours these env vars (passed via
`BARREL_INFERENCE_OPTS` to `do_cmake.sh`):

```
BARREL_INFERENCE_OPTS="-DGGML_CUDA=ON"           # enable CUDA on Linux x86-64
BARREL_INFERENCE_OPTS="-DGGML_METAL=OFF"         # disable Metal on Darwin
BARREL_INFERENCE_OPTS="-DGGML_BLAS=OFF"          # disable BLAS
BARREL_INFERENCE_OPTS="-DCMAKE_BUILD_TYPE=Debug" # debug build
```

The build step honours `BARREL_INFERENCE_BUILDOPTS` (passed to `cmake --build`).

## Why we ship `common/`

llama.cpp's `common/` carries the chat-template pipeline (`common_chat_*`,
the PEG autoparser, jinja runtime) that the autoparser path in barrel
depends on (`apps/barrel_inference/c_src/barrel_inference_chat_nif.cpp`).
It also pulls in HTTP / Hugging Face download helpers we do not use,
which is why `common/` indirectly depends on `vendor/cpp-httplib` and
`vendor/nlohmann` — those have to ship too even though we never call
the HTTP code path. We could prune `hf-cache.cpp` + `download.cpp` and
drop the http vendor cost, but the diff drifts on each bump; carrying
~5 MB extra source is cheaper than maintaining the patch.
