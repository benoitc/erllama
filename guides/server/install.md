# Install the server and CLI

The release archive includes the Erlang runtime, the inference library and both commands.
The native archives run on these platforms:

| Archive | Build and runtime |
| --- | --- |
| macos-arm64 | Apple Silicon, built on macOS 15, Metal. Install Homebrew OpenSSL 3 if the runtime needs it. |
| linux-amd64 | Ubuntu 24.04, x86-64 CPU with AVX2, BMI2, FMA and F16C |
| linux-arm64 | Ubuntu 24.04, AArch64 CPU |
| freebsd-amd64 | FreeBSD 14.4, amd64 CPU with AVX2, BMI2, FMA and F16C |

## Install from GitHub

```sh
curl -fsSL https://github.com/erllama/erllama/releases/download/v0.14.0/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
erllama_server foreground
```

In another terminal, use `erllama status` or `curl http://127.0.0.1:8080/ready`.
To load a local model and send it a prompt:

```sh
erllama load /path/to/model.gguf --name demo
erllama run demo "Say hello."
```

The installer chooses the archive for the current machine, verifies its SHA-256, and installs it under
`~/.local/lib/erllama` with launchers in `~/.local/bin`. Use `sh -s -- --prefix /path/to/prefix`
after the pipe for a different installation directory. Existing model data stays in the server's data directory.

## Install from local files

Download `install.sh`, `SHA256SUMS` and the matching `.tar.gz` from the
[same release](https://github.com/erllama/erllama/releases/tag/v0.14.0) to one directory, then run:

```sh
sh ./install.sh --from .
```

## GX10 / DGX Spark

The ARM64 container builds llama.cpp for the GB10 CUDA GPU (SM 121).
Run it on Linux with Docker and the NVIDIA Container Toolkit:

```sh
docker run -d --name erllama --gpus all --network host \
  -v erllama-data:/data -v /path/to/models:/models:ro \
  ghcr.io/erllama/erllama:0.14.0-gx10
docker exec erllama erllama load /models/model.gguf --name demo --gpu-layers 99
docker exec erllama erllama run demo "Say hello."
curl http://127.0.0.1:8080/ready
```

For an offline GX10, download `erllama-0.14.0-gx10.docker.tar.gz` and its
`.sha256` file from the release. Run `sha256sum -c erllama-0.14.0-gx10.docker.tar.gz.sha256`
and `docker load -i erllama-0.14.0-gx10.docker.tar.gz`, then use the same run command.

Host networking keeps the default HTTP listener on the host loopback interface.
Use `-e ERLLAMA_HTTP_PORT=8081` if port 8080 is occupied. The default Erlang node name
also needs to be free on the host. The named volume holds downloaded models and server state.
To stop the container, use `docker stop erllama`.
