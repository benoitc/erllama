# Updating the vendored llama.cpp

erllama vendors a pinned copy of llama.cpp under
`c_src/llama.cpp/`. The pinned tag is in `c_src/llama.cpp/.version`
(currently **v1.0.0**). You need this page when you bump that pin.
Paths on this page are relative to `apps/erllama/`; run the commands
from the repository root.

Why pin: reproducible builds, no network access at install time (the hex
package ships the source), and control over when new model
architectures are adopted.

## What is vendored

`scripts/vendor_llama.sh` copies these upstream paths wholesale:

```
cmake/ CMakeLists.txt common/ ggml/ include/ src/ LICENSE vendor/ tools/mtmd/
```

and then removes whole directories we never build:

- ggml backends other than CPU, Metal, CUDA and BLAS
  (`ggml/src/ggml-{cann,et,hexagon,hip,musa,opencl,openvino,rpc,sycl,virtgpu,vulkan,webgpu,zdnn,zendnn}`)
- `.gitignore` files

`vendor/` is kept whole: upstream builds it as CMake targets that
`common/` links against. `tools/mtmd/` is the multimodal library
(libmtmd: mmproj projectors, image/audio input); it is built through
the upstream `LLAMA_BUILD_MTMD` standalone hook with `MTMD_VIDEO`
forced off, and it links `vendor/{stb,miniaudio,hash,sheredom}`.

The vendor script applies every patch in `scripts/patches/`, in name
order, to the extracted tarball before copying it:

- [cuda-dgx-spark-q4k-moe-mmq.patch](scripts/patches/cuda-dgx-spark-q4k-moe-mmq.patch)
  sends an eight-row Q4_K `MUL_MAT_ID` (a mixture-of-experts layer with
  eight rows, as an MTP verify of four requests gives) through MMQ
  instead of MMVQ on DGX Spark (GB10), where it is faster; other GPUs,
  types and row counts are unchanged.
- The [FP32 logits patch](scripts/patches/cuda-fp32-logits.patch) exposes
  an explicit FP32 NCCL sum for the final vocabulary scores. The graph's
  normal reduction still uses the upstream precision heuristic.
- [hybrid-k-graph-reuse.patch](scripts/patches/hybrid-k-graph-reuse.patch)
  checks the dense attention mask built for the hybrid K-only input, so
  GLM can reuse its decode graph with unified KV and fused K/V rows
  enabled.
- [meta-recurrent-snapshot-view.patch](scripts/patches/meta-recurrent-snapshot-view.patch)
  lets the split backend (`ggml-backend-meta.cpp`) split a view that
  walks the split axis one window at a time, which GLM's recurrent-state
  snapshots take; without it a split GLM with snapshots aborts.
- [meta-split-mirrored-read.patch](scripts/patches/meta-split-mirrored-read.patch)
  makes the split backend read a mirrored tensor from the first device
  that holds data (its reg's `ggml_backend_dev_holds_data`), not always
  device 0. erllama's tensor-parallel ranks hold one real device; the
  others are placeholders that read as zeros, so a checkpoint of a
  mirrored tensor on any rank but the first held zeros.

When updating the pin, check whether upstream has fixed each one; once
it has, remove that patch. A patch failure leaves the current vendored
tree intact.

`common/` is needed for the chat template pipeline
(`common_chat_*`, the PEG autoparser, the jinja runtime). It links
`vendor/cpp-httplib` for its Hugging Face download helpers, which the
NIF never calls; `c_src/CMakeLists.txt` sets `LLAMA_OPENSSL=OFF` so that
code is compiled without TLS and no OpenSSL dependency is pulled in.

## Bump the pin

Pick a tag from <https://github.com/ggml-org/llama.cpp/tags>. Both the
release tags (`v1.0.0`) and the build tags (`b12359`) work. Check the
release notes for changes to the C API the NIF wraps
(`c_src/erllama_safe.cpp` lists every `llama_*` / `ggml_*` symbol) and to
`common/chat.h`.

```sh
apps/erllama/scripts/vendor_llama.sh v1.0.0   # replace with the new tag
rm -rf _build apps/erllama/_build
rebar3 compile
rebar3 xref && rebar3 dialyzer && rebar3 fmt --check && rebar3 lint
rebar3 eunit && rebar3 proper
LLAMA_TEST_MODEL=/path/to/small.gguf rebar3 ct
```

The script checks the retained files against the patched tarball.
Update the tag in this page and
add a CHANGELOG line, then commit with a message naming the new tag.

### Check the C++ shims

Every file in `c_src/*.cpp` builds against llama.cpp's C++ headers,
not only its C API, so a bump can change them without touching
`llama.h`. Read the upstream diff of the headers each one includes
(upstream paths, under `c_src/llama.cpp/`):

| Shim | What it wraps |
|---|---|
| `c_src/erllama_safe.cpp` | the `noexcept` wrappers around every `llama_*` / `ggml_*` call that can throw |
| `c_src/erllama_chat_nif.cpp` | `common_chat_*`, the autoparser (`common/chat.h`) |
| `c_src/erllama_spec_nif.cpp` | ngram speculation (`common/speculative.h`) |
| `c_src/erllama_mtp.cpp` | the MTP drafter (`common_speculative`, a draft context) |
| `c_src/erllama_mtmd_nif.cpp` | libmtmd (`tools/mtmd/`) |
| `c_src/erllama_tp_nif.cpp` | ggml's internal backend interfaces (`ggml/src/ggml-backend-impl.h`) and the split backend (`ggml/src/ggml-backend-meta.cpp`) |

A changed signature fails the build; a changed behaviour shows only in
the real-model suites (see
[internals/hardware-suites.md](../../internals/hardware-suites.md)).

### Check tensor parallelism on every bump

The tensor-parallel devices rely on ggml internals that change between
pins. On the CUDA hosts, build with NCCL and run at least
`erllama_tp_nccl_SUITE`, then `erllama_tp_failure_SUITE` and
`erllama_tp_checkpoint_SUITE` (see "When you bump llama.cpp" in
[internals/tensor-parallel-design.md](../../internals/tensor-parallel-design.md#when-you-bump-llamacpp)):

```sh
scripts/tp_hosts.sh sync  hostA hostB
scripts/tp_hosts.sh build hostA hostB
ERLLAMA_TP_HOSTS=hostA,hostB rebar3 ct --dir apps/erllama/test \
    --suite erllama_tp_nccl_SUITE,erllama_tp_failure_SUITE,erllama_tp_checkpoint_SUITE
```

For hybrid-model work (GLM-5.3-Flash), also run the GLM suites, and
check whether the local `meta-*` patches still apply and are still
needed.

## Build knobs

`do_cmake.sh` passes `ERLLAMA_OPTS` to the CMake configure step and
`do_llama.sh` passes `ERLLAMA_BUILDOPTS` to `cmake --build`:

```sh
ERLLAMA_OPTS="-DGGML_CUDA=ON"           # enable CUDA on Linux x86-64
ERLLAMA_OPTS="-DGGML_CUDA=ON -DGGML_CUDA_NCCL=ON -DNCCL_ROOT=$HOME/nccl"
                                        # CUDA with NCCL, for tensor parallelism
ERLLAMA_OPTS="-DGGML_METAL=OFF"         # disable Metal on Darwin
ERLLAMA_OPTS="-DGGML_BLAS=OFF"          # disable BLAS
ERLLAMA_OPTS="-DCMAKE_BUILD_TYPE=Debug" # debug build
ERLLAMA_BUILDOPTS="-j 4"                # limit build parallelism
```

## Notes

- If you need a backend that is pruned (Vulkan, SYCL, ...), drop it from
  `PRUNE_GGML` in `scripts/vendor_llama.sh` and remove the matching
  `GGML_*=OFF` line in `c_src/CMakeLists.txt`.
- The cache layer depends on `llama_state_seq_*`; bumps that change those
  signatures show up as NIF compile errors in `erllama_safe.cpp`.
- The tag in `.version` is part of the KV cache key (`nif_kv_format` in
  `c_src/erllama_nif_kv.c`), so every bump starts the cache cold: rows
  saved by the previous build are never restored.
