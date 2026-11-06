"""Plot measured before/after performance, keeping steps and trajectories separate."""
import csv
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
steps = []
for method in ("augmented", "parallel"):
    with (ROOT / f"data/optimized_steps_final_{method}/timings.csv").open() as f:
        steps.extend(csv.DictReader(f))
with (ROOT / "data/optimized_trajectories/summary.csv").open() as f:
    trajectories = list(csv.DictReader(f))

methods = ("augmented", "parallel")
implementations = ("baseline", "optimized")
for method in methods:
    for implementation in implementations:
        selected = [r for r in steps if (r["method"], r["implementation"]) == (method, implementation)]
        assert {int(r["trial"]) for r in selected} == {1, 2, 3} and len(selected) == 3
        selected = [r for r in trajectories if (r["method"], r["implementation"]) == (method, implementation)]
        assert len(selected) == 1
        with (ROOT / "data/optimized_trajectories" / f"{method}_{implementation}_steps.csv").open() as f:
            records = list(csv.DictReader(f))
        assert len(records) == (20 if method == "augmented" else 50)
        assert float(records[-1]["time_fs"]) == 1.0

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "figure.facecolor": "white"})
fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
colors = {"baseline": "#a7afb8", "optimized": "#177e89"}
for ax, rows, column, scale, title, xlabel in (
    (axes[0, 0], steps, "seconds", 1, "One step at t = 1 fs (median of 3)", "Time [s]"),
    (axes[0, 1], steps, "allocated_bytes", 1e-9, "One-step allocations (median of 3)", "Cumulative allocation [GB]"),
    (axes[1, 0], trajectories, "seconds", 1, "Localized initial state: 0–1 fs", "Propagation time [s]"),
    (axes[1, 1], trajectories, "allocated_bytes", 1e-9, "Trajectory allocations: 0–1 fs", "Cumulative allocation [GB]"),
):
    largest = 0
    for offset, implementation in zip((-0.18, 0.18), implementations):
        values = [median(float(r[column]) for r in rows
                         if r["method"] == method and r["implementation"] == implementation) * scale
                  for method in methods]
        positions = [i + offset for i in range(2)]
        ax.barh(positions, values, height=0.3, color=colors[implementation], label=implementation.capitalize())
        largest = max(largest, *values)
        for y, value in zip(positions, values):
            ax.annotate(f"{value:.3g}", (value, y), xytext=(4, 0), textcoords="offset points", va="center")
    ax.set(yticks=[0, 1], yticklabels=["Augmented BUG", "Parallel BUG"], xlabel=xlabel, title=title)
    ax.set_xlim(0, largest * 1.26)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.2)
axes[0, 0].legend(frameon=False, loc="lower right")
fig.suptitle("Holstein HEOM · 10 sites · split ket/bra\n8 Julia workers · 1 BLAS thread · rank cap 32", fontsize=14)
directory = ROOT / "figures"
directory.mkdir(exist_ok=True)
fig.savefig(directory / "optimization.svg")
fig.savefig(directory / "optimization.png", dpi=180)
