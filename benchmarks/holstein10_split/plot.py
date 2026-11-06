"""Plot recorded trajectories, including partial runs with their actual end time."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    files = sorted((BASE / "data").glob("*/steps.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for path in files:
        rows = read(path)
        if not rows:
            continue
        values = lambda key: np.array([float(row[key]) for row in rows])
        time = values("time_fs")
        label = f"{path.parent.name} ({time[-1]:g} fs)"
        axes[0, 0].plot(time, values("rho_1_1_re"), label=label)
        axes[0, 1].plot(time, values("maxrank"), label=label)
        axes[1, 0].plot(time, values("cumulative_seconds"), label=label)
        axes[1, 1].plot(time[1:], values("seconds")[1:], label=label)
    for ax, title, ylabel in zip(axes.flat,
            ["Initially occupied site", "Maximum bond rank", "Propagation cost", "Cost per recorded step"],
            ["Population", "Rank", "Cumulative seconds", "Seconds"]):
        ax.set(title=title, xlabel="Time (fs)", ylabel=ylabel)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("10-site Holstein HEOM · split ket/bra · localized initial state")
    fig.supxlabel("Exploratory runs: tolerances and thread settings differ; legend shows each end time.", fontsize=9)
    for suffix in ("svg", "png"):
        fig.savefig(OUT / f"trajectories.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
