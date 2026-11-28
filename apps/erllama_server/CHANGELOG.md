# Changelog

All notable changes to erllama_server are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.2.0] - 2026-11-28

### Added

- The release includes a mutual TLS distribution profile for groups across hosts.
- HTTP status API 2 reports grouped models, rank readiness and loss reasons. Prometheus exposes group state and lost slots.
- Start serving and rank roles from the same release with separate config files.
- Register a tensor-parallel group to serve a model by name and see when a rank is lost.
- Tested with two GB10 hosts on RoCE and TCP. Fleet served chat, withdrew the
  replica after follower loss, and served again with a new generation. See the
  [two-host guide](../../guides/server/tensor-parallel.md#fleet-acceptance-and-matched-transport-comparison).

### Changed

- Model pulls and local registration accept numbered GGUF shards as one
  model. The registry verifies the complete set before serving it.

- An embedding request (OpenAI `/v1/embeddings`, Ollama `/api/embed`)
  now counts against `max_active_runs`, like a generation: when every
  slot is in use it answers 429 instead of running on top of them.
  Model listings and token counts take no slot and still answer.
- A remote caller can no longer choose the module an agent session
  generates with (the undocumented `generator` key of `agent_start`);
  it was a test hook.
- A streamed Responses request that stops at `max_output_tokens` now
  ends with `response.incomplete` instead of `response.completed`, as
  OpenAI does; the response it carries already had status `incomplete`.
  A client that waits only for `response.completed` must also accept
  `response.incomplete`.

### Fixed

- A cancelled Responses request was reported and stored as completed.
  It now ends with status `cancelled` and the text made so far, and the
  stream ends with `response.incomplete` carrying that response. Stored,
  it reads back the same way, also when the client went away before the
  run had stopped; with `store: false` nothing is kept. Continuing it
  starts from the last turn that completed.
