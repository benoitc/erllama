# DSpark hardware probe

This experiment runs inside isolated Erlang rank processes using the native library from 8a0ac02. It uses that build's private context resource layout. Load the probe only alongside the same erllama NIF used to compile it.

- `weights.py HEAD TARGET OUTPUT` writes a standalone draft GGUF. It copies shared token embeddings and the output projection from the target if the draft lacks them.
- `ctx.c`, `probe.cpp` and `dsprobe.erl` let the rank test read the selected target layers, feed accepted positions to the head and request greedy guesses. The head context reserves five sampled output rows for a block of up to five positions.
- `trial.erl` drives ordinary decode, real target verification, per-sequence rollback and the transition between solo and batched decode. The two-host hardware setup uses erllama's existing `erllama_tp_hosts` test helper. The hostnames, temporary paths and rank build in these scripts match the recorded experiment.
- `compare.py` compares the captured FP32 rows with a whole-model reference. `replay.cpp` reproduces the sampled-output limit and replays the captured rows through the standalone head on CPU.

The raw host logs and captured features are in `~/erllama-tp/results/2026-11-25-deepseek-dspark-features/` on gx10-fa8e. The parent report lists which cases ran.
