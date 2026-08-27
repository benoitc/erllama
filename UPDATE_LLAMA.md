# Updating the vendored llama.cpp

erllama vendors a pinned, unmodified copy of llama.cpp under
`c_src/llama.cpp/`. The pinned tag is in `c_src/llama.cpp/.version`
(currently **b10593**). You need this page when you bump that pin.

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

No file is edited. `common/` is needed for the chat template pipeline
(`common_chat_*`, the PEG autoparser, the jinja runtime). It links
`vendor/cpp-httplib` for its Hugging Face download helpers, which the
NIF never calls; `c_src/CMakeLists.txt` sets `LLAMA_OPENSSL=OFF` so that
code is compiled without TLS and no OpenSSL dependency is pulled in.

## Bump the pin

Pick a tag from <https://github.com/ggml-org/llama.cpp/tags>. Check the
release notes for changes to the C API the NIF wraps
(`c_src/erllama_safe.cpp` lists every `llama_*` / `ggml_*` symbol) and to
`common/chat.h`.

```sh
scripts/vendor_llama.sh b10068          # replace with the new tag
rm -rf _build
rebar3 compile
rebar3 xref && rebar3 dialyzer && rebar3 fmt --check && rebar3 lint
rebar3 eunit && rebar3 proper
LLAMA_TEST_MODEL=/path/to/small.gguf rebar3 ct
```

The script fails if any vendored file differs from the tarball, so a
clean run proves the tree is upstream. Update the tag in this page and
add a CHANGELOG line, then commit with a message naming the new tag.

## Build knobs

`do_cmake.sh` passes `ERLLAMA_OPTS` to the CMake configure step and
`do_llama.sh` passes `ERLLAMA_BUILDOPTS` to `cmake --build`:

```sh
ERLLAMA_OPTS="-DGGML_CUDA=ON"           # enable CUDA on Linux x86-64
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
