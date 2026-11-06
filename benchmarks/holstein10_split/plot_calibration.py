"""Plot only completed 0–1 fs measurements; do not extrapolate to 100 fs."""
import csv
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent


def rows(name):
    with (BASE / "data" / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    train = rows("short_tamen.csv")
    bug = rows("calibration.csv")
    assert len(train) == 6 and len(bug) == 2, "Calibration is incomplete"
    samples = []
    for tolerance in (.005, .001):
        group = [r for r in train if float(r["tolerance"]) == tolerance]
        assert len(group) == 3
        samples.append((f"tAMEn, tol={tolerance:g}", median(float(r["seconds"]) for r in group),
                        float(group[0]["state_error"]), float(group[0]["density_error"]), "#3475b7", "o"))
    for row in bug:
        samples.append((f"Parallel BUG, h={row['h']} fs", float(row["seconds"]),
                        float(row["state_error"]), float(row["density_error"]), "#c97632", "s"))
    for path in sorted((BASE / "data").glob("augmented_*/summary.csv")):
        with (path.parent / "steps.csv").open() as stream:
            trajectory = list(csv.DictReader(stream))
        if not trajectory or abs(float(trajectory[-1]["time_fs"]) - 1.0) > 1e-12:
            raise ValueError(f"Expected a completed 0–1 fs trajectory: {path.parent.name}")
        with path.open() as stream:
            records = list(csv.DictReader(stream))
        assert len(records) == 1
        row = records[0]
        samples.append((f"Augmented BUG, h={row['h']} fs", float(row["seconds"]),
                        float(row["state_error"]), float(row["density_error"]), "#27866d", "D"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for label, seconds, state_error, density_error, color, marker in samples:
        for ax, error in zip(axes, (state_error, density_error)):
            ax.scatter(seconds, error, s=65, color=color, marker=marker, label=label)
            right = marker != "o"
            ax.annotate(label, (seconds, error), xytext=(-6 if right else 6, 7),
                        textcoords="offset points", ha="right" if right else "left", fontsize=8)
    for ax, title in zip(axes, ("Full hierarchy: relative difference", "Density matrix: absolute difference")):
        ax.set(xscale="log", yscale="log", xlabel="Propagation time for 0–1 fs (seconds)",
               ylabel="Difference from tighter TT reference", title=title)
        ax.grid(alpha=.2, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=.12,y=.2)
    fig.suptitle("10-site Holstein HEOM · split · localized initial state · 1 fs pilot")
    fig.supxlabel("tAMEn: median of 3 runs. BUG: one run per h, maxrank=32; see report for truncation-budget failures.", fontsize=9)
    out = BASE / "figures"; out.mkdir(exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(out / f"calibration.{suffix}", dpi=160)


if __name__ == "__main__":
    main()
