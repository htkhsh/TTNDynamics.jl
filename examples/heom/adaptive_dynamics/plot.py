"""Plot adaptive HEOM sample/attempt CSVs; requires NumPy and Matplotlib."""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ttn-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
samples = {label: np.genfromtxt(HERE / "data" / f"heom_adaptive_{label}.csv", delimiter=",", names=True)
           for label in ("loose", "tight")}
steps = {label: np.genfromtxt(HERE / "data" / f"heom_adaptive_{label}_steps.csv", delimiter=",",
                            names=True, dtype=None, encoding="utf-8")
         for label in ("loose", "tight")}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, layout="constrained")
fig.suptitle("HEOM with adaptive TTN augmented BUG", fontsize=18)
fig.supxlabel("Time (hbar = 1)")
fine = samples["tight"]
t = fine["time"]
colors = {"loose": "#2864b7", "tight": "#df7625"}
labels = {"loose": "rtol = 1e-3", "tight": "rtol = 1e-5"}

ax = axes[0, 0]
for level, color in ((1, "#2864b7"), (2, "#df7625")):
    ax.plot(t, fine[f"p{level}_ttn"], color=color, lw=2, label=f"TTN P({level})")
    ax.plot(t, fine[f"p{level}_dense"], "o", color=color, ms=4,
            fillstyle="none", label=f"Dense P({level})")
ax.set(title=f"Populations at {len(t)} saved times", ylabel="Population")
ax.legend(ncol=2, fontsize=8)

for label in ("loose", "tight"):
    data, attempts, color = samples[label], steps[label], colors[label]
    kept = attempts[attempts["accepted"] == 1]
    rejected = attempts[attempts["accepted"] == 0]
    axes[0, 1].semilogy(data["time"][1:], np.maximum(data["density_error"][1:], 1e-18),
                        "o-", ms=3, color=color, label=labels[label])
    axes[1, 0].step(np.r_[kept["time"], kept["t_end"][-1]],
                    np.r_[kept["dt"], kept["dt"][-1]], where="post", color=color,
                    label=f"{labels[label]}: {len(kept)} accepted")
    if len(rejected):
        axes[1, 0].scatter(rejected["time"], rejected["dt"], marker="x", color=color,
                            label=f"{len(rejected)} rejected")
    axes[1, 1].semilogy(kept["time"], np.maximum(kept["error_ratio"], 1e-12), ".",
                        color=color, label=labels[label])
    finite_rejections = rejected[np.isfinite(rejected["error_ratio"])]
    if len(finite_rejections):
        axes[1, 1].semilogy(finite_rejections["time"], finite_rejections["error_ratio"],
                            "x", color=color)
    axes[2, 0].semilogy(data["time"], np.maximum(data["trace_drift"], 1e-18),
                        color=color, label=labels[label])
    axes[2, 1].step(data["time"], data["max_rank"], where="post", color=color,
                    label=labels[label])

axes[0, 1].set(title="Density error against dense ADO exponential",
               ylabel=r"$\|\rho_0^{TTN}-\rho_0^{dense}\|_F$")
axes[1, 0].set(title="Internal trial steps (crosses are rejections)", ylabel="dt")
axes[1, 0].set_yscale("log")
axes[1, 1].axhline(1, color="#333333", ls="--", lw=1, label="Acceptance threshold")
axes[1, 1].set(title="Whole-TTN local error estimate / tolerance", ylabel="Error ratio")
axes[2, 0].set(title="Trace drift; root correction disabled",
               ylabel=r"$|\mathrm{Tr}\,\rho_0(t)-\mathrm{Tr}\,\rho_0(0)|$")
axes[2, 1].set(title="Maximum TTN bond rank at saved times", ylabel="Rank")
for i, ax in enumerate(axes.flat):
    ax.grid(alpha=0.22)
    ax.set_xlim(t[0], t[-1])
    if i:
        ax.legend(fontsize=8)
for suffix in ("svg", "png"):
    destination = FIGURES / f"heom_adaptive_dynamics.{suffix}"
    fig.savefig(destination, dpi=150)
    print(destination)
