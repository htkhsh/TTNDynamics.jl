"""Plot the Julia validation output; NumPy/Matplotlib are optional example deps."""
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ttn-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"svg.fonttype": "none", "font.size": 10})
root = Path(__file__).resolve().parent
figures = root / "figures"
figures.mkdir(parents=True, exist_ok=True)
with (root / "data" / "runs.csv").open() as stream:
    runs = list(csv.DictReader(stream))
method = runs[0]["method"]
labels = [run["label"] for run in runs]
descriptions = {run["label"]: (f"h = {float(run['value']):g} fs" if method == "fixed"
                              else f"rtol = {float(run['value']):g}") for run in runs}
data = {label: np.atleast_1d(np.genfromtxt(root / "data" / f"dynamics_{label}.csv", delimiter=",", names=True))
        for label in labels}
steps = {label: np.atleast_1d(np.genfromtxt(root / "data" / f"steps_{label}.csv", delimiter=",", names=True))
         for label in data}
fine = data[labels[-1]]
site_count = sum(name.startswith("p") and name.endswith("_ttn") for name in fine.dtype.names)
fig, axes = plt.subplots(4, 2, figsize=(12, 13), constrained_layout=True)
fig.suptitle(f"Holstein–TFD: {method} TTN vs dense evolution", fontsize=16)
colors = plt.get_cmap("tab10").colors
for j in range(1, site_count + 1):
    color = colors[(j - 1) % len(colors)]
    for ax, key in ((axes[0, 0], "p"), (axes[0, 1], "n")):
        ax.plot(fine["time_fs"], fine[f"{key}{j}_dense"], color=color, lw=1.8, label=f"site {j} dense")
        ax.plot(fine["time_fs"][::3], fine[f"{key}{j}_ttn"][::3], "o", color=color, ms=4,
                mfc="none", label=f"site {j} TTN ({descriptions[labels[-1]]})")
axes[0, 0].set(title="Electronic populations", ylabel="Population")
axes[0, 1].set(title="Physical oscillator occupations", ylabel="⟨a†a⟩")
for label, color in zip(labels, ("#dd7136", "#2766ad")):
    description = descriptions[label]
    d = data[label]
    axes[1, 0].semilogy(d["time_fs"], np.maximum(d["density_error"], 1e-16), color=color, label=f"density / {description}")
    axes[1, 0].semilogy(d["time_fs"], np.maximum(d["occupation_error"], 1e-16), "--", color=color, label=f"occupation / {description}")
    axes[1, 1].semilogy(d["time_fs"], np.maximum(d["norm_drift"], 1e-16), color=color, label=description)
    axes[3, 0].semilogy(d["time_fs"], np.maximum(d["energy_drift_ifs"], 1e-18), color=color, label=description)
    axes[2, 0].step(d["time_fs"], d["max_rank"], where="post", color=color, label=description)
    s = steps[label]
    accepted = s["accepted"] == 1
    axes[2, 1].plot(s["time_fs"][accepted], s["dt_fs"][accepted], ".-", color=color, label=description)
    if np.any(~accepted):
        axes[2, 1].plot(s["time_fs"][~accepted], s["dt_fs"][~accepted], "x", color=color,
                        label=f"rejected / {description}")
axes[1, 0].set(title="Errors against the same finite-basis dense model", ylabel="Absolute error")
axes[1, 1].set(title="Norm drift (no renormalization)", ylabel="|‖ψ(t)‖ − ‖ψ(0)‖|")
axes[2, 0].set(title="Maximum bond rank", ylabel="Rank")
axes[2, 1].set(title=f"Internal {method} steps", ylabel="Step size / fs")
axes[3, 0].set(title="Energy drift", ylabel="|E(t) − E(0)| / fs⁻¹")
for j in range(1, 2 * site_count + 1):
    site, sign = (j + 1) // 2, "+" if j % 2 else "−"
    axes[3, 1].plot(fine["time_fs"], fine[f"thermal_n{j}"],
                    "-" if j % 2 else "--", color=colors[(site - 1) % len(colors)],
                    label=f"site {site}, {sign}ω")
axes[3, 1].set(title="Transformed-mode occupations (finer TTN run)", ylabel="⟨N±⟩")
for ax in axes.flat:
    ax.set_xlabel("Time / fs")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
for extension in ("svg", "png"):
    path = figures / f"dynamics.{extension}"
    fig.savefig(path, dpi=160)
    print(path)
