"""Summarize independently completed common-time checkpoints, not extrapolations."""
import csv
import sys
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(label):
    if not label or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in label):
        raise ValueError("invalid label")
    root = Path(__file__).parent / "output" / label
    rows = read(root / "measurements.csv")
    summary = []
    names = sorted({r["name"] for r in rows})
    for name in names:
        for t in sorted({int(r["time_fs"]) for r in rows if r["name"] == name}):
            selected = [r for r in rows if r["name"] == name and int(r["time_fs"]) == t]
            summary.append(dict(name=name, time_fs=t, repetitions=len(selected),
                dt_fs=selected[0]["dt_fs"],
                seconds=median(float(r["seconds"]) for r in selected),
                min_seconds=min(float(r["seconds"]) for r in selected),
                max_seconds=max(float(r["seconds"]) for r in selected),
                allocated_GiB=median(float(r["allocated_GiB"]) for r in selected),
                peak_rank=max(int(r["peak_rank"]) for r in selected),
                truncation_misses=max(int(r["truncation_misses"]) for r in selected),
                max_state_difference=max(float(r["max_state_difference"]) for r in selected),
                max_density_difference=max(float(r["max_density_difference"]) for r in selected),
                max_trace_error=max(float(r["max_trace_error"]) for r in selected),
                reference_checked=all(r["reference_checked"] == "true" for r in selected),
                accuracy_qualified=all(r["accuracy_qualified"] == "true" for r in selected)))
    for row in summary:
        tt = next((r for r in summary if r["name"] == "tamen" and r["time_fs"] == row["time_fs"]), None)
        row["time_ratio_to_tamen"] = row["seconds"] / tt["seconds"] if tt else ""
        row["qualified_time_ratio_to_tamen"] = row["time_ratio_to_tamen"] if tt and row["accuracy_qualified"] and tt["accuracy_qualified"] else ""
    with (root / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    styles = {"tamen": ("tAMEn", "#222222", "-", "o"),
              "augmented": ("Augmented · order 1", "#0072b2", "-", "o"),
              "parallel": ("Parallel · order 1", "#d55e00", "-", "o"),
              "augmented2": ("Augmented · order 2", "#0072b2", "--", "D"),
              "parallel2": ("Parallel · order 2", "#d55e00", "--", "D")}
    specs = [("seconds", "Cumulative propagation time [s]", True),
             ("max_state_difference", "Maximum full-state relative difference", True),
             ("peak_rank", "Maximum saved bond rank", False),
             ("max_density_difference", "Maximum density-matrix difference", True)]
    for name in names:
        selected = [r for r in summary if r["name"] == name]
        title, color, line, marker = styles.get(name, (name, "#009e73", ":", "s"))
        for ax, (key, _, _) in zip(axes.flat, specs):
            ax.plot([r["time_fs"] for r in selected], [r[key] for r in selected],
                    label=title, color=color, linestyle=line, marker=marker, markersize=5)
    for ax, (key, title, log) in zip(axes.flat, specs):
        ax.set_title(title)
        ax.set_xlabel("Simulated time [fs]")
        if log:
            ax.set_yscale("log")
        ax.set_xticks(sorted({r["time_fs"] for r in summary}))
        ax.grid(alpha=.2)
        if key in ("max_state_difference", "max_density_difference"):
            ax.axhline(1e-3, color="#009e73", linewidth=1, label="Target 1e-3")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Holstein HEOM · 10 sites · split · weighted truncation · current implementation")
    fig.savefig(root / "comparison.svg")
    fig.savefig(root / "comparison.png", dpi=180)


if __name__ == "__main__":
    main(sys.argv[1])
