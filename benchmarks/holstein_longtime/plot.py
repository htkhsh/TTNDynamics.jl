"""Plot only the common time range established by compare.jl."""
import argparse
import csv
from pathlib import Path
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def bounded_rows(path, end):
    result = []
    for row in rows(path):
        try:
            time = float(row["time_fs"])
        except (KeyError, ValueError, TypeError):
            continue  # An uncommitted trailing row after a killed writer.
        if time <= end + 1e-10:
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    folder = args.output / ("comparison_partial" if args.partial else "comparison")
    with (folder / "comparison.toml").open("rb") as stream:
        meta = tomllib.load(stream)
    end = meta["compared_until_fs"]
    labels = list(dict.fromkeys([meta["reference"], *meta["runs"]]))
    model = meta["model"]
    n, initial = model["sites"], model["initial_site"]
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    colors = dict(zip(labels, ["#333333", "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00"]))
    for label in labels:
        color = colors.get(label)
        obs = bounded_rows(args.output / label / "observables.csv", end)
        steps = bounded_rows(args.output / label / "steps.csv", end)
        time = [float(r["time_fs"]) for r in obs]
        axes[0, 0].plot(time, [float(r[f"rho_{initial}_{initial}_re"]) for r in obs], label=label, color=color)
        axes[0, 1].plot(range(1, n + 1), [float(obs[-1][f"rho_{i}_{i}_re"]) for i in range(1, n + 1)],
                        marker="o", ms=3, color=color)
        axes[0, 2].plot([float(r["time_fs"]) for r in steps], [int(r["maxrank"]) for r in steps], color=color)
        axes[1, 0].plot([float(r["time_fs"]) for r in steps], [float(r["cumulative_seconds"]) for r in steps], color=color)
    for file, key, ax in (("density_differences.csv", "density_absolute_difference", axes[1, 1]),
                          ("state_differences.csv", "state_relative_difference", axes[1, 2])):
        data = rows(folder / file)
        for label in meta["runs"]:
            selected = [r for r in data if r["run"] == label]
            values = np.array([float(r[key]) for r in selected])
            values[values <= 0] = np.nan
            ax.semilogy([float(r["time_fs"]) for r in selected], values, color=colors.get(label))
    titles = [f"Population at initial site {initial}", f"Populations at {end:g} fs", "Maximum bond rank",
              "Cumulative propagation time [s]", "Density difference from reference", "Full-state relative difference"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Time [fs]")
        ax.grid(alpha=0.2)
    axes[0, 1].set_xlabel("Site")
    axes[0, 0].legend(frameon=False)
    scope = "Complete" if meta["complete"] else "PARTIAL"
    fig.suptitle(f"Holstein HEOM · {n} sites · split · J={model['hopping_cm']:g}, ω₀={model['frequency_cm']:g} cm⁻¹\n"
                 f"{scope}: 0–{end:g} fs · reference convergence: {meta['reference_status']}")
    fig.savefig(folder / "comparison.svg")
    fig.savefig(folder / "comparison.png", dpi=180)


if __name__ == "__main__":
    main()
