# DSpark on split DeepSeek V4 Flash 0731

Two GB10s ran erllama's FP32 logits build (8a0ac02) with the same target GGUF and 70-token Erlang coding prompt as the prior decode comparison. We compared the features that the 0731 DSpark head reads at layers 41, 42 and 43 against a whole-model run on one GB10. All three used the same token IDs, 2,048 context tokens, four sequence slots, unified KV and two rollback snapshots per sequence.

| Features | Relative RMS difference from one GPU | Agreement between split ranks |
|---|---:|---|
| 70-token prefill | 0.106–0.130% | Byte-identical at all three layers |
| Next token, teacher-forced | 13.93–15.10% | Byte-identical at all three layers |

Rank 0 already holds a complete feature vector for this head. Adding the two rank vectors roughly doubles its magnitude. The continuation missed our predeclared comparison threshold of 10% relative RMS and 0.995 cosine. This experiment did not identify the source of that difference. [Per-layer comparisons](features.csv)

## Head replay

The ggml-org BF16 head (revision `64da7c18350c54db6406cd085731f2a34363bf66`) declares a five-position block and borrows the target token embeddings and output projection. We copied those two tensors into a temporary standalone draft GGUF from the original target, retaining their F16 and Q8_0 representations.

The first live attempt made 64 teacher-forced draft calls and produced zero guesses. Its context allowed one sampled output per sequence while the head requested four. A CPU replay with the backend sampler enabled reproduced the native error. Raising the limit to five produced guesses with both captured feature sources.

| Anchor position | Single-GPU features | Split features | Next two reference tokens |
|---|---|---|---|
| 70 | 442, 85 | 442, 85 | 344, 260 |
| 71 | 260, 5553 | 260, 5553 | 260, 5553 |

These two anchors establish that the split features can drive the actual head. They are too few to estimate an acceptance rate or serving speed. [Replay output](replay.csv)
