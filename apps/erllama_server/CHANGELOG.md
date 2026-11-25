# Changelog

All notable changes to erllama_server are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Changed

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
