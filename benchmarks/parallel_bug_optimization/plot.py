"""Generate the paired optimization and common-error-target report."""
import csv
import os
from pathlib import Path
from statistics import median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ttn-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
def read(name):
    with (root / "data" / name).open() as stream:
        return list(csv.DictReader(stream))
rows = read("runs_threads4.csv")
stages = read("stages_threads4.csv")
accuracy = read("accuracy_runs.csv")
cases = list(dict.fromkeys(r["case"] for r in rows))
def med(case, threaded, version, field, source=rows):
    return median(float(r[field]) for r in source if r["case"] == case and
                  r["threaded"] == threaded and r["version"] == version)
def acc(case, variant, field):
    return median(float(r[field]) for r in accuracy if r["case"] == case and r["variant"] == variant)
labels = [c.replace("heom", "HEOM M=").replace("_6_", "\n6 modes\n")
          .replace("_2_", "\n2 modes\n").replace("tfd3", "TFD\n3 sites")
          .replace("tfd4_large", "TFD\n4 sites, n=3") for c in cases]
plt.rcParams.update({"svg.fonttype": "none", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
fig.suptitle("Parallel BUG: compact assembly optimization", fontsize=15)
colors = {"false": "#4865a9", "true": "#238b81"}
for threaded, offset, label in (("false", -0.18, "Serial execution"), ("true", 0.18, "4 workers")):
    speed = [med(c, threaded, "before", "seconds") / med(c, threaded, "after", "seconds") for c in cases]
    axes[0, 0].bar([i+offset for i in range(len(cases))], speed, width=0.34, color=colors[threaded], label=label)
    reduction = [100*(1-med(c, threaded, "after", "bytes")/med(c, threaded, "before", "bytes")) for c in cases]
    axes[0, 1].bar([i+offset for i in range(len(cases))], reduction, width=0.34, color=colors[threaded])
axes[0, 0].axhline(1, color="#444444", ls="--", lw=1)
axes[0, 0].set_ylabel("Before / after median time")
axes[0, 1].set_ylabel("Cumulative allocation reduction / %")
for ax in axes[0]:
    ax.set_xticks(range(len(cases)), labels, fontsize=8)
axes[0, 0].set_ylim(0,1.45)
axes[0, 0].legend(fontsize=9,loc="upper center",ncol=2)
phase_names = ("prepare", "solve", "assemble", "finish", "other")
phase_colors = ("#8aa1c8", "#4865a9", "#238b81", "#d17d35", "#bababa")
bottom = [0.0, 0.0]
for phase, color in zip(phase_names, phase_colors):
    values = [1e3*med("heom4_6_split", "true", v, phase, stages) for v in ("before", "after")]
    axes[1, 0].bar([0, 1], values, bottom=bottom, color=color, label=phase)
    bottom = [a+b for a, b in zip(bottom, values)]
axes[1, 0].set(xticks=[0, 1], xticklabels=["Before", "After"],
               ylabel="Sum of phase medians / ms", title="HEOM M=4, 6 modes, split / 4 workers")
axes[1, 0].set_ylim(0,1.25*max(bottom))
axes[1, 0].legend(fontsize=8,loc="upper center",ncol=3)
acases = list(dict.fromkeys(r["case"] for r in accuracy))
for j, (v, label, color) in enumerate(zip(("augmented", "parallel_serial", "parallel_threads"),
        ("Augmented", "Parallel, serial", "Parallel, 4 workers"), ("#4865a9", "#238b81", "#d17d35"))):
    axes[1, 1].bar([i+(j-1)*0.26 for i in range(len(acases))], [1e3*acc(c, v, "seconds") for c in acases],
                   width=0.25, color=color, label=label)
axes[1, 1].set(yscale="log", ylabel="Median time to target error / ms",
               title="Final full-state relative error ≤ 10⁻⁴", xticks=range(len(acases)),
               xticklabels=["HEOM M=4, 6 modes\nfused", "TFD / 3 sites"])
axes[1, 1].legend(fontsize=8)
for ax in axes.flat:
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
figures = root / "figures"; figures.mkdir(exist_ok=True)
for extension in ("svg", "png"):
    fig.savefig(figures / f"comparison.{extension}", dpi=170)
plt.close(fig)

max_distance = max(float(r["state_distance"]) for r in rows)
max_density = max(float(r["density_difference"]) for r in rows)
max_omitted = max(float(r["omitted_difference"]) for r in rows)
same_ranks = all(r["ranks_equal"] == "true" for r in rows)
rank_mismatch_cases = ", ".join(sorted({r["case"] for r in rows if r["ranks_equal"] != "true"}))
lines = ["# Parallel BUG optimization", "",
    "The update formula and public options are unchanged. The implementation "
    "now contracts only the newly added basis directions, skips empty blocks, "
    "reuses temporary storage across product terms, and reuses complete leaf "
    "bases and their reduced operators. No tolerances are relaxed and no "
    "rank-expansion retry is enabled.", "",
    "The initial sampling profile (M=4, six-mode split HEOM, serial execution) "
    "put about 51% of active evolution samples inside local Arnoldi solves, "
    "19% in preparation, and 17% in the general RHS contractions for block "
    "assembly. These are inclusive sampling counts, not disjoint wall times. "
    "The new rectangular contractions remove multiplication by explicitly "
    "zero old-basis rows and avoid per-term permutation allocations.", "",
    "![Optimization and target accuracy](../benchmarks/parallel_bug_optimization/figures/comparison.svg)", "",
    "## Paired timing", "",
    "Five warmed runs per version, rotating before/after order in the same "
    "Julia process, with four worker threads and BLAS=1. Both variants use "
    "the same trajectory/observation loop, t=0…0.5, h=0.05, truncation atol=1e-10 "
    "and maxrank=64. Model setup, initial TFD enrichment, all solves, truncation "
    "and observations are timed. GC is collected before each call; any GC "
    "during the call remains included. This was not an isolated host; report "
    "ranges as well as medians and avoid interpreting small timing differences.", "",
    "| Case | Execution | Before ms (min–max) | After ms (min–max) | Speedup | Allocation reduction |",
    "|---|---|---:|---:|---:|---:|"]
for c in cases:
    for t in ("false", "true"):
        def times(v):
            return [1e3*float(r["seconds"]) for r in rows if r["case"] == c and r["threaded"] == t and r["version"] == v]
        b, a = times("before"), times("after")
        reduction = 100*(1-med(c, t, "after", "bytes")/med(c, t, "before", "bytes"))
        lines.append(f"| {c} | {'4 workers' if t == 'true' else 'serial'} | "
                     f"{median(b):.2f} ({min(b):.2f}–{max(b):.2f}) | "
                     f"{median(a):.2f} ({min(a):.2f}–{max(a):.2f}) | {median(b)/median(a):.2f}× | {reduction:.1f}% |")
lines += ["", "Allocation reduction is cumulative allocated bytes, not peak memory. "
    "The larger TFD fixture has four sites, eight thermofield modes and three "
    "Fock states per mode (104,976 dense coefficients); it is compared in TTN "
    "form without a dense reference. The smaller fixtures match the "
    "[initial implementation comparison](parallel_bug.md).", "",
    "## Numerical equivalence", "",
    f"Across these trajectories the largest before/after TTN distance is {max_distance:.3e}, "
    f"the largest observed density difference is {max_density:.3e}, and the "
    f"largest change in a reported h*eta diagnostic is {max_omitted:.3e}. "
    f"Saved rank histories are {'identical in all cases' if same_ranks else 'different in '+rank_mismatch_cases+'; identical in the remaining cases'}. "
    "Before/after truncation-budget pass/fail histories also agree. HEOM root trace correction "
    "is disabled; TFD has no norm correction. State differences at rounding "
    "or truncation scale do not establish physical cutoff convergence.", "",
    "| Case | Maximum rank | Steps exceeding truncation budget (of 10) | Maximum trace / norm drift |",
    "|---|---:|---:|---:|"]
for c in cases:
    data = [r for r in rows if r["case"] == c]
    lines.append(f"| {c} | {max(int(r['max_rank']) for r in data)} | "
                 f"{max(int(r['truncation_failures']) for r in data)} | "
                 f"{max(float(r['trace_or_norm_drift']) for r in data):.3e} |")
lines += ["", "The large TFD case hits maxrank=64 and therefore does not meet "
    "the requested truncation budget on every step. It measures performance "
    "and equivalence of the same capped calculation; it is not an accuracy "
    "or rank-convergence claim. Increasing maxrank would require a separate "
    "convergence and cost study.", "",
    "## Where time remains", "",
    "The following separately instrumented runs split preparation, joined "
    "node solves, block assembly, and truncation/invariant finalization. "
    "Production code reads no clocks for this benchmark. The benchmark loads "
    "isolated instrumented copies with checked insertion points. Phase medians "
    "need not sum to the median uninstrumented trajectory time.", "",
    "| Case / 4 workers | Version | Prepare ms | Local ODE ms | Assemble ms | Truncate/correct ms | Other ms |",
    "|---|---|---:|---:|---:|---:|---:|"]
for c in cases:
    for v in ("before", "after"):
        values = [1e3*med(c, "true", v, p, stages) for p in phase_names]
        lines.append(f"| {c} | {v} | " + " | ".join(f"{x:.2f}" for x in values) + " |")
lines += ["", "## Common accuracy target", "",
    "For two small models, search h=0.05/2^j (j=0…8) from coarse to fine "
    "and take the first grid whose final full-state relative error is ≤1e-4 "
    "against an independent dense matrix exponential. Warm that selected "
    "configuration and measure five times. This is a comparison at a common "
    "target on a discrete set of grids, not an optimized work/error curve or "
    "a guarantee on errors at all intermediate times. Parallel timings here "
    "use the optimized implementation.", "",
    "| Case | Method | Selected h | Final relative error | Median ms (min–max) |",
    "|---|---|---:|---:|---:|"]
for c in acases:
    for v in ("augmented", "parallel_serial", "parallel_threads"):
        times = [1e3*float(r['seconds']) for r in accuracy if r['case'] == c and r['variant'] == v]
        lines.append(f"| {c} | {v} | {acc(c,v,'h'):.7g} | {acc(c,v,'relative_error'):.3e} | "
                     f"{median(times):.2f} ({min(times):.2f}–{max(times):.2f}) |")
lines += ["", "The common-h optimization saves work, but does not reduce the time "
    "discretization error of parallel BUG. Use the target-error comparison "
    "when choosing an integrator; augmented BUG remains the default. The "
    "optional basis-expansion retry of the parallel paper is still unimplemented; "
    "the existing omitted-block diagnostics and step doubling do not guarantee "
    "detection of directions invisible to the reduced equations.", "",
    "## Reproduce", "", "From the package root:", "", "```sh",
    "OPENBLAS_NUM_THREADS=1 julia --threads=4 --startup-file=no --project=. benchmarks/parallel_bug_optimization/run.jl 5",
    "OPENBLAS_NUM_THREADS=1 julia --threads=4 --startup-file=no --project=. benchmarks/parallel_bug_optimization/accuracy.jl 5",
    "python3 benchmarks/parallel_bug_optimization/plot.py", "```", "",
    "The [benchmark folder](../benchmarks/parallel_bug_optimization/) contains "
    "CSV data, the baseline sampling profile, instrumentation, and a frozen "
    "pre-optimization parallel step. The frozen step uses unchanged shared "
    "contraction/preparation/truncation primitives, whose hashes are recorded "
    "below. Ordinary before/after timings do not include phase instrumentation. "
    "The recorded measurements were collected in two batches: the large case "
    "was resumed after identifying its rank-cap limitation. The workspace-file "
    "hash changed only because an allocation comment was clarified; its "
    "numerical function body did not change between batches.", "",
    "```text", (root / "data" / "environment_threads4.txt").read_text().strip(), "```", "",
    "The underlying first-order method is unchanged from "
    "G. Ceruti, J. Kusch, C. Lubich and D. Sulz, “A Parallel Basis Update and "
    "Galerkin Integrator for Tree Tensor Networks,” SIAM J. Sci. Comput. 48(1), "
    "A27–A48 (2026), [doi:10.1137/24M1714502](https://doi.org/10.1137/24M1714502).", ""]
report = root.parents[1] / "docs" / "parallel_bug_optimization.md"
report.write_text("\n".join(lines))
print(report)
