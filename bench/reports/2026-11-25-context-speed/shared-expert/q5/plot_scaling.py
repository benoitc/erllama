"""Plot the original-weight, single-stream erllama trials. Requires Matplotlib."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parent
DEPTHS = [2048, 4096, 6144, 8192]
SERIES = [("single.csv", "One GX10", "#596478", "s"), ("baseline.csv", "Two GX10s", "#08766e", "o")]


def read_trials(name):
    with (ROOT / name).open() as stream:
        rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    expected = {(depth, rep) for depth in DEPTHS for rep in (1, 2)}
    assert len(rows) == len(expected) and {(r["context_tokens"], r["repeat"]) for r in rows} == expected
    assert all(r["extension_tokens"] == 2048 and r["sampled_tokens"] == 128 and r["eog_steps"] == 0 for r in rows)
    return rows


def rate(rows, phase):
    if phase == "decode":
        tokens = sum(r["sampled_tokens"] - 1 for r in rows)
        elapsed = sum(r["remaining_us"] for r in rows)
    else:
        tokens = sum(r["extension_tokens"] for r in rows)
        elapsed = sum(r["first_token_us"] for r in rows)
    return tokens * 1e6 / elapsed


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
    fig.subplots_adjust(left=.09, right=.98, top=.69, bottom=.24, wspace=.25)
    fig.text(.07, .93, "DeepSeek Flash 0731 on one and two GX10s", fontsize=19, weight="bold")
    data = [(read_trials(name), label, color, marker) for name, label, color, marker in SERIES]
    for ax, phase, title in zip(axes, ("decode", "prompt"), ("Generation", "Prompt extension")):
        a, b = (rate(rows, phase) for rows, *_ in data)
        ax.set_title(f"{title}   +{100*(b/a-1):.0f}%", loc="left", fontsize=14, weight="bold")
        for rows, label, color, marker in data:
            grouped = [[r for r in rows if r["context_tokens"] == depth] for depth in DEPTHS]
            pooled = [rate(group, phase) for group in grouped]
            ax.plot(DEPTHS, pooled, color=color, marker=marker, label=label, lw=2.3, ms=6)
            for depth, group in zip(DEPTHS, grouped):
                ends = [rate([r], phase) for r in group]
                ax.vlines(depth, min(ends), max(ends), color=color, linewidth=1)
                ax.scatter([depth, depth], ends, color=color, marker="_", s=40)
        ax.set_xticks(DEPTHS)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("Context tokens before generation")
        ax.set_ylabel("Tokens/s")
        ax.set_ylim(0, 26 if phase == "decode" else 800)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["right", "top"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.061,.89),
               ncol=2, frameon=False)
    fig.text(.07,.085,"Original ds4 weights • one greedy stream • two runs per context",fontsize=10)
    fig.text(.07,.04,"Markers pool tokens and elapsed time. Whiskers span both runs. Gains pool all eight trials.",fontsize=9,color="#4b5563")
    fig.savefig(ROOT / "gx10-scaling.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
