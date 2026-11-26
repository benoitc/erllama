"""Rebuild context-speed.png from the erllama timings and published ds4 CSV (requires matplotlib)."""
import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True, layout="constrained")
for column, (name, title, color) in enumerate([
    ("qwen", "Qwen2.5 1.5B · M4 Pro", "#176a88"),
    ("glm", "GLM-5.3-Flash · two GB10s", "#9a4a24"),
    ("deepseek-erllama", "DeepSeek V4 Flash", "#386d51"),
]):
    with (ROOT / f"{name}.csv").open() as src:
        rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(src)]
    depths = sorted({r["context_tokens"] for r in rows})
    assert depths == [2048, 4096, 6144, 8192]
    assert all(r["extension_tokens"] == 2048 and r["sampled_tokens"] == 128
               and r["eog_steps"] == 0 for r in rows)
    for panel, metric in enumerate(["extension", "decode"]):
        ax = axes[panel, column]
        values = []
        for depth in depths:
            trials = [r for r in rows if r["context_tokens"] == depth]
            assert {r["repeat"] for r in trials} == {1, 2} and len(trials) == 2
            rates = [(r["extension_tokens"] / r["first_token_us"] if metric == "extension"
                      else (r["sampled_tokens"] - 1) / r["remaining_us"]) * 1e6 for r in trials]
            values.append(mean(rates))
            ax.vlines(depth, min(rates), max(rates), color=color, linewidth=2)
            ax.scatter([depth] * len(rates), rates, edgecolors=color, facecolors="white", zorder=3)
        ax.plot(depths, values, color=color, linewidth=1.6, label="erllama · two GB10s" if name == "deepseek-erllama" else None)
        ax.set_xticks(depths, labels=[f"{d:,}" for d in depths])
        ax.set_ylim(0, ax.dataLim.ymax * 1.13)
        ax.grid(axis="y", alpha=.24)
        ax.spines[["top", "right"]].set_visible(False)
        if panel == 0:
            ax.set_title(title, fontsize=14, pad=10)
            ax.set_ylabel("New prompt tokens/s")
        else:
            ax.set_ylabel("Generated tokens/s")
            ax.set_xlabel("Context tokens before generation")
with (ROOT / "ds4-gb10.csv").open() as src:
    reference = list(csv.DictReader(src))
depths = [int(r["ctx_tokens"]) for r in reference]
assert depths == [2048, 4096, 6144, 8192]
assert all(int(r["prefill_tokens"]) == 2048 and int(r["gen_steady_tokens"]) == 127 for r in reference)
for panel, field in enumerate(["prefill_tps", "gen_steady_tps"]):
    ax = axes[panel, 2]
    ax.plot(depths, [float(r[field]) for r in reference], marker="s", linestyle="--",
            color="#565295", label="ds4 · one GB10 (published)")
    ax.set_ylim(0, ax.dataLim.ymax * 1.13)
axes[0, 2].legend(loc="lower left", fontsize=9, frameon=False)
fig.suptitle("Prompt extension and generation as the context grows", fontsize=16)
fig.savefig(ROOT / "context-speed.png", dpi=165, facecolor="white")
