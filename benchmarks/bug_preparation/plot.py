"""Plot the measured old/new comparison and generate its report."""
import csv
import os
from pathlib import Path
from statistics import median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ttn-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / "data" / "runs.csv").open() as stream:
    rows = [{k: v if k in ("case", "version") else float(v) for k, v in row.items()}
            for row in csv.DictReader(stream)]
names = list(dict.fromkeys(r["case"] for r in rows))
groups = {(name, version): [r for r in rows if r["case"] == name and r["version"] == version]
          for name in names for version in ("before", "after")}

def med(name, version, field):
    return median(r[field] for r in groups[name, version])

plt.rcParams.update({"svg.fonttype": "none", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
fig.suptitle("BUG preparation and workspaces: old/new implementation", fontsize=15)
for ax, field, scale, ylabel in zip(axes, ("wall_seconds", "bytes"), (1, 1e-9),
                                  ("Median wall time / s", "Allocated memory per run / GB")):
    for version, offset, color in (("before", -0.19, "#a6afb7"), ("after", 0.19, "#297e75")):
        values = [med(n, version, field) * scale for n in names]
        ranges = [[r[field] * scale for r in groups[n, version]] for n in names]
        error = [[v-min(rs) for v, rs in zip(values, ranges)],
                 [max(rs)-v for v, rs in zip(values, ranges)]]
        ax.bar([i+offset for i in range(len(names))], values, width=0.36,
               color=color, label=version, yerr=error, capsize=3)
    ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names])
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
figures = root / "figures"
figures.mkdir(exist_ok=True)
for suffix in ("svg", "png"):
    path = figures / f"comparison.{suffix}"
    fig.savefig(path, dpi=160)
    print(path)

lines = ["# BUG preparation and local workspaces", "",
    "The full step and first half step start from the same state. They now share "
    "its orthogonalization, old operator contractions, recursive complement QR "
    "factorizations and reduced initial data. Updated bases are owned by each solve. "
    "The second half step prepares its changed input independently. Preparation "
    "lives for one trial; no persistent cache or public API is added. An unnecessary "
    "deep copy of old operator contractions was also removed from every BUG step.", "",
    "Local Krylov actions now reuse private scratch arrays for mode permutations "
    "and matrix products. Each operator application still returns a fresh result "
    "because Arnoldi retains earlier outputs. Scratch storage is scoped to a "
    "single sequential local solve and is never shared between branches. This "
    "also benefits fixed stepping. Profiling identified repeated core contractions "
    "and array allocation as the next target after preparation reuse alone gave "
    "small timing gains.", "",
    "The integration formula, truncation budgets, local solves and acceptance test "
    "are unchanged. Fixed stepping remains the standard workflow.", "",
    "## Measurements", "",
    "The old source snapshot and current solver run in the same Julia process, "
    "using identical model wrappers. Each case has two full warmups per version; "
    "measured pairs alternate order. GC remains enabled, with explicit collection "
    "outside each measured call. Times include wrapper setup and observations, "
    "excluding model construction and numerical comparisons. Adaptive cases enable "
    "stage timing in both versions; fixed cases have no stage timers.", "",
    "```text", (root / "data" / "environment.txt").read_text().strip(), "```", "",
    "HEOM uses three hierarchy modes of size 3, t=0–10, no trace correction. "
    "Adaptive: initial h=0.2, output interval 0.2, atol=1e-8, rtol=1e-5; fixed: "
    "h=0.1, truncation atol=1e-10. TFD uses the two-site 300 K Holstein model, "
    "four transformed modes, t=0–50 fs, Fock sizes 3 and 5. Adaptive: initial "
    "h=2 fs, output interval 1 fs, atol=1e-9, rtol=1e-5; fixed: h=0.5 fs, "
    "truncation atol=1e-10. Steps land on observation times.", "",
    "Wall times and allocated bytes below are medians; allocation is cumulative, "
    "not peak resident memory. Error bars in the figure show minimum–maximum.", "",
    "| Case | Before, s | After, s | Speedup | Allocation reduction | Accepted / rejected | Max rank |",
    "|---|---:|---:|---:|---:|---:|---:|"]
for name in names:
    before, after = med(name, "before", "wall_seconds"), med(name, "after", "wall_seconds")
    reduction = 100 * (1-med(name, "after", "bytes")/med(name, "before", "bytes"))
    row = groups[name, "after"][0]
    lines.append(f"| {name} | {before:.4f} | {after:.4f} | {before/after:.3f}× | {reduction:.1f}% | "
                 f"{int(row['accepted'])} / {int(row['rejected'])} | {int(row['max_rank'])} |")
lines += ["", "![Comparison](../benchmarks/bug_preparation/figures/comparison.svg)", "",
    "Shared preparation is charged once to `coarse_bug`. The time saved by reusing "
    "it appears in `accepted_update` / `rejected_update`, since the first half step "
    "no longer repeats setup. Phase medians below are independent and need not "
    "sum to a particular run's wall time.", "",
    "| Case | Error-estimation BUG, before → after (s) | Accepted updates, before → after (s) | Difference + QR, before → after (s) |",
    "|---|---:|---:|---:|"]
for name in names:
    if name.endswith("adaptive"):
        fields = ("coarse_bug", "accepted_update", "difference")
        cells = [f"{med(name, 'before', f):.4f} → {med(name, 'after', f):.4f}" for f in fields]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
lines += ["", "## Numerical checks and limits", "",
    "Every warmup and measured pair checks saved times, ranks, and (for adaptive "
    "runs) identical step sizes and acceptance/rejection histories. Densities and "
    "other saved observables agree within 1e-11; the final dense state comparison "
    "is performed outside timing. This compares implementations, not convergence "
    "to an infinite Fock space or an independent dense evolution.", "",
    "| Case | Maximum saved density difference | Maximum relative final-state difference |",
    "|---|---:|---:|"]
for name in names:
    group = groups[name, "after"]
    lines.append(f"| {name} | {max(r['density_difference'] for r in group):.3e} | "
                 f"{max(r['final_state_difference'] for r in group):.3e} |")
lines += ["", "The result applies to these small-rank models and this machine. "
    "The full error-estimation solve and both half-step solves still run. "
    "Further speedups require profiling their local contractions, Krylov actions "
    "and truncations. Replacing step doubling would change the error estimator "
    "and needs separate numerical validation.", "",
    "## Reproduce", "", "From the package root, with a pre-change solver `src` snapshot:", "",
    "```sh",
    "OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=examples benchmarks/bug_preparation/run.jl /path/to/baseline/src 5",
    "python3 benchmarks/bug_preparation/plot.py", "```", "",
    "The baseline snapshot used for this run was `/tmp/ttn-bug-before-20261105/src`. "
    "It is a local temporary artifact; the source hashes above identify the measured "
    "versions. [Raw results](../benchmarks/bug_preparation/data/runs.csv) are retained "
    "with this report. Re-running replaces the data, figures and report.", ""]
report = root.parents[1] / "docs" / "bug_preparation.md"
report.write_text("\n".join(lines))
print(report)
