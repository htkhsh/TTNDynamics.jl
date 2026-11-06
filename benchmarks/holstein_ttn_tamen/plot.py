"""Plot recorded common-time results; incomplete runs stop at their last point."""
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent / "output/five_fs"
rows = list(csv.DictReader((root / "comparison.csv").open()))
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
labels = {"tt": "TT tAMEn (BLAS 8)", "tt_serial": "TT tAMEn (BLAS 1)",
          "tree": "TTN, 0.5 fs / 8 points (BLAS 1)",
          "tree_interval1": "TTN, 1 fs / 6 points (BLAS 1)",
          "tree_blas8": "TTN, 0.5 fs / 8 points (BLAS 8)",
          "tree_tight": "TTN, tighter", "tree_loose": "TTN, looser"}
for name in dict.fromkeys(r["name"] for r in rows):
    group = [r for r in rows if r["name"] == name]
    times = sorted({float(r["time_fs"]) for r in group})
    for ax, key in zip(axes, ("seconds", "max_state_error", "maxrank")):
        values = [statistics.median(float(r[key]) for r in group if float(r["time_fs"]) == t) for t in times]
        ax.plot(times, values, "o-", ms=4, label=labels.get(name, name))
for ax in axes:
    ax.set_xlabel("Time (fs)")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Cumulative propagation time (s)")
axes[0].set_yscale("log")
axes[1].set_ylabel("Maximum state error so far")
axes[1].set_yscale("log")
axes[1].axhline(1e-3, color="0.4", ls="--", lw=1, label="Accuracy target")
axes[1].text(0.98, 1e-3, "Target: 1e-3", transform=axes[1].get_yaxis_transform(),
             ha="right", va="bottom", fontsize=8, color="0.4")
axes[2].set_ylabel("Maximum bond rank at sample")
axes[0].legend(frameon=False, fontsize=8)
for suffix in ("png", "svg"):
    fig.savefig(root / f"comparison.{suffix}", dpi=180)
