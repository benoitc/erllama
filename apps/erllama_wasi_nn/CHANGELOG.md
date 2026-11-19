# Changelog

All notable changes to erllama_wasi_nn are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- First version. A WebAssembly component that imports `wasi:nn` and
  runs under erlang_wasm can now use models loaded in erllama. You map
  each name the component may load to a loaded model and a task:
  a text completion, or an embedding. The component sends text and
  reads the reply, or the vector as 32-bit floats. Replies come once,
  without streaming.
- Not on Hex yet: it needs a version of erlang_wasm that is not
  released, so it is used from this repository.
