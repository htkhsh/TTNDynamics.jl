"""Summarize completed allocation and temporal-order trials."""
import csv
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).parent / "output"


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    allocation = []
    for phase in ("before", "after"):
        rows = read(ROOT / phase / "timings.csv")
        for method in ("augmented", "parallel"):
            selected = [r for r in rows if r["method"] == method]
            assert len(selected) == 3
            allocation.append(dict(phase=phase, method=method,
                seconds=median(float(r["seconds"]) for r in selected),
                min_seconds=min(float(r["seconds"]) for r in selected),
                max_seconds=max(float(r["seconds"]) for r in selected),
                allocated_GiB=median(int(r["bytes"]) / 1024**3 for r in selected),
                peak_rank=max(int(r["peak_rank"]) for r in selected)))
    write(ROOT / "allocation_summary.csv", allocation)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, key, label in zip(axes, ("seconds", "allocated_GiB"),
                              ("Propagation time [s]", "Cumulative allocations [GiB]")):
        for j, (phase, color) in enumerate((("before", "#89939f"), ("after", "#009e73"))):
            selected = [r for r in allocation if r["phase"] == phase]
            loc = [i + (j - .5) * .32 for i in range(2)]
            values = [r[key] for r in selected]
            ax.bar(loc, values, .3, color=color, label=phase.capitalize())
            if key == "seconds":
                ax.errorbar(loc, values, yerr=[[r["seconds"]-r["min_seconds"] for r in selected],
                    [r["max_seconds"]-r["seconds"] for r in selected]], fmt="none", ecolor="#444", capsize=3)
            for x, y in zip(loc, values):
                ax.annotate(f"{y:.3g}", (x, y), xytext=(0, 5), textcoords="offset points", ha="center")
        ax.set_xticks([0, 1], ["Augmented BUG", "Parallel BUG"])
        ax.set_title(label)
        ax.margins(y=.22)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Buffered orthogonalization · weighted truncation · 0–1 fs · 3 runs")
    fig.savefig(ROOT / "allocations.svg")
    fig.savefig(ROOT / "allocations.png", dpi=180)

    order_root = ROOT / "order_clean"
    rows = []
    for directory in ("order_clean", "order_refined"):
        rows.extend(dict(r, directory=directory) for r in read(ROOT / directory / "trials.csv"))
    assert len(rows) == 26
    summary = []
    for order in (1, 2):
        for method in ("augmented", "parallel"):
            stepsizes=sorted({float(r["h"]) for r in rows if int(r["order"])==order and r["method"]==method})
            for h in stepsizes:
                selected = [r for r in rows if int(r["order"]) == order and
                            r["method"] == method and float(r["h"]) == h]
                assert len(selected) == 2
                for rep in (1, 2):
                    steps = read(ROOT / selected[0]["directory"] / f"{method}_order{order}_h{h}_rep{rep}_steps.csv")
                    assert len(steps) == round(1 / h) and float(steps[-1]["time_fs"]) == 1
                summary.append(dict(order=order, method=method, h=h,
                    seconds=median(float(r["seconds"]) for r in selected),
                    min_seconds=min(float(r["seconds"]) for r in selected),
                    max_seconds=max(float(r["seconds"]) for r in selected),
                    allocated_GiB=median(int(r["bytes"]) / 1024**3 for r in selected),
                    state_error=max(float(r["state_error"]) for r in selected),
                    density_error=max(float(r["density_error"]) for r in selected),
                    peak_rank=max(int(r["peak_rank"]) for r in selected),
                    max_augmented_rank=max(int(r["max_augmented_rank"]) for r in selected),
                    truncation_misses=max(int(r["truncation_misses"]) for r in selected)))
    write(order_root / "summary.csv", summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for method, color in (("augmented", "#0072b2"), ("parallel", "#d55e00")):
        for order, marker in ((1, "o"), (2, "D")):
            selected = sorted([r for r in summary if r["method"] == method and r["order"] == order], key=lambda r:r["h"])
            label = f"{method.capitalize()} · order {order}"
            for ax, key in zip(axes, ("h", "seconds")):
                ax.plot([r[key] for r in selected], [r["state_error"] for r in selected],
                        marker=marker, color=color, linestyle="-" if order == 1 else "--", label=label)
                if key == "seconds":
                    for r in selected:
                        # Label the accuracy-matched candidates; every step size
                        # remains available on the convergence panel and in CSV.
                        offsets = {
                            ("augmented", 1, .05): (-67, 7),
                            ("augmented", 2, .1): (-42, 10),
                            ("parallel", 1, .025): (7, 7),
                            ("parallel", 2, .125): (-60, 8),
                            ("parallel", 2, .1): (7, -14),
                        }
                        offset = offsets.get((method, order, r["h"]))
                        if offset is None:
                            continue
                        ax.annotate(f"h={r['h']}", (r[key], r["state_error"]),
                            xytext=offset, textcoords="offset points", fontsize=8)
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axhline(1e-3, color="#009e73", lw=1, alpha=.7, label="Full-state target 1e-3")
        ax.grid(alpha=.2)
        ax.set_ylabel("Final full-state relative difference")
    axes[0].set_xlabel("Step size [fs]")
    ticks = [.0125, .025, .05, .1, .25, .5]
    axes[0].set_xticks(ticks, [str(h) for h in ticks])
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("Propagation time [s]")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Holstein HEOM · 10 sites · split · 0–1 fs · 2 runs per setting")
    fig.savefig(order_root / "comparison.svg")
    fig.savefig(order_root / "comparison.png", dpi=180)


if __name__ == "__main__":
    main()
