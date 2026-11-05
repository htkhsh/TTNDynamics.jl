"""Summarize repeated measurements and render standalone timing figures."""
import csv
import os
from pathlib import Path
from statistics import median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ttn-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
repo = root.parents[1]
with (root / "data" / "runs.csv").open() as stream:
    rows = [{k: v if k == "case" else float(v) for k, v in row.items()}
            for row in csv.DictReader(stream)]
groups = {name: [r for r in rows if r["case"] == name]
          for name in dict.fromkeys(r["case"] for r in rows)}
# Use an actual run nearest the median, so its phase times add up to its total.
representatives = [min(group, key=lambda r: abs(r["total"] - median(g["total"] for g in group)))
                   for group in groups.values()]
phases = [
    ("accepted_update", "Accepted updates", "#297e75"),
    ("coarse_bug", "Error-estimation BUG", "#d58b39"),
    ("difference", "Difference + QR", "#805ba5"),
    ("rejected_update", "Rejected half steps", "#bd5355"),
    ("candidate_norm", "Candidate norm", "#6688bc"),
    ("observations", "Observations", "#8c999e"),
    ("other", "Other driver work", "#c8cdd1"),
]
plt.rcParams.update({"svg.fonttype": "none", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
fig.suptitle("Adaptive augmented BUG: measured cost by stage", fontsize=15)
labels = [f"{r['case']}\nrank {int(r['max_rank'])}" for r in representatives]
for ax, fraction in zip(axes, (False, True)):
    bottom = [0.0] * len(representatives)
    for key, label, color in phases:
        height = [r[key] / r["total"] * 100 if fraction else r[key] for r in representatives]
        ax.bar(labels, height, bottom=bottom, label=label, color=color, width=0.65)
        bottom = [b + h for b, h in zip(bottom, height)]
    ax.set_ylabel("Share of driver time / %" if fraction else "Driver wall time / s")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
axes[1].set_ylim(0, 100)
axes[0].legend(fontsize=8)
figures = root / "figures"
figures.mkdir(exist_ok=True)
for suffix in ("svg", "png"):
    destination = figures / f"adaptive_timing.{suffix}"
    fig.savefig(destination, dpi=160)
    print(destination)

environment = (root / "data" / "environment.txt").read_text().strip()
lines = [
    "# Adaptive integration timing", "",
    "The standard workflow uses a prescribed time step and checks convergence with half that step. "
    "Adaptive integration remains available for choosing time steps and for changing time scales.", "",
    "These measurements use the actual adaptive HEOM/TFD APIs with `record_timing=true`. "
    "Both paths are warmed up by complete runs, then timed and untimed runs alternate. "
    "GC is enabled; an explicit collection precedes each measured call. "
    "No tests, dense reference work or plotting run concurrently with the measurements.", "",
    "```text", environment, "```", "",
    "HEOM: three hierarchy modes of size 3, t=0–10, save interval 0.2, "
    "initial dt=0.2, atol=1e-8, rtol=1e-5, no trace correction. "
    "TFD: the two-site 300 K Holstein example, t=0–50 fs, save interval 1 fs, "
    "initial dt=2 fs, atol=1e-9, rtol=1e-5; Fock cutoffs 3 and 5. "
    "Steps land on each output time. Fock size 5 has no dense accuracy check in this benchmark.", "",
    "## Measurements", "",
    "Each breakdown is the actual repetition nearest the median driver time. "
    "Ranks are the maximum saved rank and maximum temporary augmented rank. "
    "The driver total excludes generator construction and TFD initial enrichment; "
    "the outer wall times saved in the CSV include these wrapper operations. "
    "Model construction is outside both measured intervals.", "",
    "| Case | Saved / augmented rank | Accepted / rejected | Driver median (range), s | Timing on/off wall ratio |",
    "|---|---:|---:|---:|---:|",
]
for r in representatives:
    group = groups[r["case"]]
    ratio = median(g["wall_seconds"] for g in group) / median(g["untimed_wall_seconds"] for g in group)
    lines.append(f"| {r['case']} | {int(r['max_rank'])} / {int(r['max_augmented_rank'])} | "
                 f"{int(r['accepted'])} / {int(r['rejected'])} | {median(g['total'] for g in group):.4f} "
                 f"({min(g['total'] for g in group):.4f}–{max(g['total'] for g in group):.4f}) | {ratio:.3f} |")
lines += ["", "| Case | Error-estimation BUG | Difference + QR | Accepted updates | Rejected half steps | Other |",
          "|---|---:|---:|---:|---:|---:|"]
for r in representatives:
    keys = ("coarse_bug", "difference", "accepted_update", "rejected_update")
    shares = [100*r[k]/r["total"] for k in keys]
    other = 100-sum(shares)
    lines.append(f"| {r['case']} | " + " | ".join(f"{v:.1f}%" for v in [*shares, other]) + " |")
lines += ["", "Other combines candidate-norm QR, observation snapshots/callbacks and driver bookkeeping; "
          "the CSV and figure keep these separate. Difference + QR measures the complete `ttn_distance` "
          "call, including construction of the temporary difference tensor. "
          "The two half steps are classified as accepted updates only after acceptance; rejected work "
          "and failed local solves are included in the accounting.", "",
          "![Timing breakdown](../benchmarks/adaptive_timing/figures/adaptive_timing.svg)", "",
          "## Optimization targets", ""]
for r in representatives:
    relevant = {label: r[key] for key, label, _ in phases if key in
                ("coarse_bug", "difference", "rejected_update", "candidate_norm")}
    largest = max(relevant, key=relevant.get)
    lines.append(f"- {r['case']}: the largest measured error-control cost is **{largest}** "
                 f"({100*relevant[largest]/r['total']:.1f}% of driver time).")
lines += ["", "Shared BUG preparation and local contraction workspaces are evaluated in the "
          "[before/after optimization report](bug_preparation.md). Removing the extra full step "
          "would require a validated alternative time-error estimator; the stage breakdown alone "
          "does not establish the speedup of a proposed change. Larger ranks, different operators or output intervals can "
          "change the relative costs. Timing on/off ratios include run-to-run noise.", "",
          "## Numerical checks", "",
          "All timed/untimed pairs have identical saved times and acceptance/rejection histories, "
          "with final-state TTN distance below 1e-11 times max(1, norm(state)).", "",
          "| Case | Maximum timing-on/off state difference | Final dense relative error |",
          "|---|---:|---:|"]
for r in representatives:
    difference = max(g["timing_state_difference"] for g in groups[r["case"]])
    error = r["dense_relative_error"]
    lines.append(f"| {r['case']} | {difference:.3e} | " + (f"{error:.3e}" if error == error else "not measured") + " |")
lines += ["", "## Reproduce", "", "From the TTNDynamics package root:", "", "```sh",
          "OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=examples benchmarks/adaptive_timing/run.jl 5",
          "python3 benchmarks/adaptive_timing/plot.py", "```", "",
          "Raw data: [runs.csv](../benchmarks/adaptive_timing/data/runs.csv), "
          "[attempts.csv](../benchmarks/adaptive_timing/data/attempts.csv). "
          "Repeating the benchmark replaces these files, the figures and this report.", ""]
report = repo / "docs" / "adaptive_timing.md"
report.write_text("\n".join(lines))
print(report)
