"""Summarize the completed factorial experiment and plot fixed-setting costs/errors."""
import argparse
import csv
from pathlib import Path
from statistics import median
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    root = parser.parse_args().directory
    with (root / "metadata.toml").open("rb") as stream:
        repeats = tomllib.load(stream)["repeats"]
    rows = read_rows(root / "trials.csv")
    methods = ("augmented", "parallel")
    variants = ("baseline", "weighted", "grouped", "both")
    summary = []
    for method in methods:
        baseline = median(float(r["seconds"]) for r in rows
                          if r["method"] == method and r["variant"] == "baseline")
        for variant in variants:
            trials = [r for r in rows if (r["method"], r["variant"]) == (method, variant)]
            assert len(trials) == repeats and {int(r["repeat"]) for r in trials} == set(range(1, repeats + 1))
            for r in trials:
                steps = read_rows(root / f"{method}_{variant}_{r['repeat']}_steps.csv")
                assert len(steps) == 10 and float(steps[-1]["time_fs"]) == 1.0
            times = [float(r["seconds"]) for r in trials]
            summary.append(dict(method=method, variant=variant, median_seconds=median(times),
                                min_seconds=min(times), max_seconds=max(times),
                                time_ratio_baseline=baseline / median(times),
                                median_allocated_GiB=median(float(r["allocated_bytes"]) / 1024**3 for r in trials),
                                peak_rank=max(int(r["peak_rank"]) for r in trials),
                                max_augmented_rank=max(int(r["max_augmented_rank"]) for r in trials),
                                max_local_dimension=max(int(r["max_local_dimension"]) for r in trials),
                                truncation_misses=max(int(r["truncation_misses"]) for r in trials),
                                state_error=max(float(r["state_error"]) for r in trials),
                                density_error=max(float(r["density_error"]) for r in trials),
                                trace_error=max(float(r["trace_error"]) for r in trials)))
    with (root / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    panels = (("median_seconds", "Propagation time [s]", True),
              ("median_allocated_GiB", "Cumulative allocations [GiB]", True),
              ("peak_rank", "Peak bond rank", False),
              ("state_error", "Final full-state relative difference", True))
    colors = ("#89939f", "#009e73", "#e69f00", "#0072b2")
    for ax, (key, title, logarithmic) in zip(axes.flat, panels):
        for j, (variant, color) in enumerate(zip(variants, colors)):
            selected = [next(r for r in summary if r["method"] == m and r["variant"] == variant) for m in methods]
            values = [r[key] for r in selected]
            locations = np.arange(2) + (j - 1.5) * 0.2
            ax.bar(locations, values, width=0.18, color=color, label=variant.capitalize())
            if key == "median_seconds":
                error = [[r["median_seconds"] - r["min_seconds"] for r in selected],
                         [r["max_seconds"] - r["median_seconds"] for r in selected]]
                ax.errorbar(locations, values, yerr=error, fmt="none", ecolor="#333333", capsize=2)
            for x, y in zip(locations, values):
                ax.annotate(f"{y:.3g}", (x, y), xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=8)
        ax.set_xticks([0, 1], ["Augmented BUG", "Parallel BUG"])
        ax.set_title(title)
        if logarithmic:
            ax.set_yscale("log")
        ax.margins(y=0.24)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"Holstein HEOM · 10 sites · split · 0–1 fs\n"
                 f"h=0.1 fs · rank cap 64 · {repeats} runs per case · fixed settings")
    fig.savefig(root / "comparison.svg")
    fig.savefig(root / "comparison.png", dpi=180)


if __name__ == "__main__":
    main()
