"""Plot paired timings and convergence; write a reproducible comparison report."""
import csv
import os
from pathlib import Path
from statistics import median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ttn-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
def read_csv(name):
    with (root / "data" / name).open() as f:
        return [{k: v if k in ("case", "variant") else float(v)
                 for k, v in row.items()} for row in csv.DictReader(f)]

rows = read_csv("runs_threads4.csv")
checks = read_csv("convergence_threads4.csv")
single = read_csv("runs_threads1.csv")
cases = list(dict.fromkeys(r["case"] for r in rows))
variants = ["augmented", "parallel_serial", "parallel_threads"]
names = ["Augmented", "Parallel, serial", "Parallel, 4 threads"]
colors = ["#4865a9", "#238b81", "#d17d35"]
def med(case, variant, field, source=rows):
    return median(r[field] for r in source if r["case"] == case and r["variant"] == variant)
def check(case, variant, h):
    return next(r for r in checks if r["case"] == case and r["variant"] == variant and r["h"] == h)
labels = [c.replace("heom", "HEOM M=").replace("_6_", "\n6 modes\n")
          .replace("_2_", "\n2 modes\n").replace("tfd3", "TFD\n3 sites") for c in cases]
plt.rcParams.update({"svg.fonttype": "none", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
fig.suptitle("Parallel BUG: time, allocations and accuracy", fontsize=15)
for ax, field, scale, ylabel in zip(axes[0], ("seconds", "bytes"), (1e3, 1e-6),
                                  ("Wall time / ms (median, min–max)", "Cumulative allocations / MB")):
    for j, (v, color, name) in enumerate(zip(variants, colors, names)):
        values = [scale*med(c, v, field) for c in cases]
        groups = [[scale*r[field] for r in rows if r["case"] == c and r["variant"] == v] for c in cases]
        err = [[a-min(g) for a, g in zip(values, groups)], [max(g)-a for a, g in zip(values, groups)]]
        ax.bar([i+(j-1)*0.26 for i in range(len(cases))], values, width=0.25,
               color=color, label=name, yerr=err, capsize=2)
    ax.set(ylabel=ylabel, xticks=range(len(cases)), xticklabels=labels)
for v, color, name in zip(variants[:2], colors[:2], names[:2]):
    for h, style in ((0.05, "o-"), (0.025, "s--")):
        axes[1, 0].semilogy(range(len(cases)), [check(c, v, h)["relative_error"] for c in cases],
                            style, color=color, label=f"{name}, h={h}")
axes[1, 0].set(ylabel="Final full-state relative error vs dense", xticks=range(len(cases)), xticklabels=labels)
speedups = [med(c, "parallel_serial", "seconds")/med(c, "parallel_threads", "seconds") for c in cases]
axes[1, 1].bar(range(len(cases)), speedups, color=colors[2])
axes[1, 1].axhline(1, color="#444444", lw=1, ls="--")
axes[1, 1].set(ylabel="Threading speedup (same parallel BUG formula)", xticks=range(len(cases)), xticklabels=labels)
axes[0, 0].legend(fontsize=9)
axes[1, 0].legend(fontsize=8)
for ax in axes.flat:
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
figures = root / "figures"
figures.mkdir(exist_ok=True)
for extension in ("svg", "png"):
    fig.savefig(figures / f"comparison.{extension}", dpi=170)
plt.close(fig)

lines = ["# Parallel BUG for HEOM and TFD", "",
    "See the [optimization report](parallel_bug_optimization.md) for a frozen-baseline "
    "comparison of compact assembly and a common-error-target comparison. "
    "The source hashes below identify the implementation used for this measurement.", "",
    "Both model APIs accept `method=:parallel`; augmented BUG remains the default. "
    "Use a fixed grid for the standard workflow, or the keyword form for adaptive time steps.", "",
    "```julia",
    "fixed = heom_evolve(system, initial, 0:0.01:1; method=:parallel, threaded=true)",
    "adaptive = heom_evolve(system, initial; tspan=(0.,1.), dt=0.01,",
    "                       method=:parallel, threaded=true, atol=1e-8, rtol=1e-5)",
    "# tfd_evolve accepts the same method/threaded options.",
    "```", "",
    "Parallel BUG [1] solves every node equation in the old bases, then assembles "
    "the evolved old/old block and explicit single-new-direction blocks. The "
    "double-new block is discarded. Only the local ODE phase is threaded; "
    "preparation, assembly and truncation still traverse the tree serially. "
    "`threaded=false` runs the identical formula serially. A one-worker Julia "
    "process falls back to serial execution.", "",
    "HEOM inserts the trace functional's subtree spaces into the old basis "
    "and protects them during truncation. The measurements disable trace "
    "correction. TFD retains its initial `H*initial` enrichment and does not "
    "correct norm or energy. Parallel BUG does not generally conserve those "
    "quadratic quantities exactly.", "",
    "## Measurements", "",
    "![Timing and accuracy](../benchmarks/parallel_bug/figures/comparison.svg)", "",
    "Measured with BLAS=1, Julia worker threads=4, five warmed runs per variant, "
    "rotating execution order and collecting garbage before each timed run. "
    "The timed call includes model generator construction, enrichment, all "
    "steps and observations. Compilation warm-up and dense reference work are "
    "outside the timed region. Fixed grids: t=0…0.5, h=0.05; truncation atol=1e-10, "
    "maxrank=64. All measured steps met their truncation budgets. No TTN is "
    "expanded by the solver; dense arrays are used only in the benchmark reference.", "",
    "| Case | Augmented ms | Parallel serial ms | Parallel 4 threads ms | Threading speedup |",
    "|---|---:|---:|---:|---:|"]
for c in cases:
    a, s, t = [1e3*med(c, v, "seconds") for v in variants]
    lines.append(f"| {c} | {a:.2f} | {s:.2f} | {t:.2f} | {s/t:.2f}× |")
lines += ["", "Speedups compare the same mathematical parallel method. "
    "The augmented/parallel comparison at a common h includes a change of "
    "algorithm and accuracy; it is not a speedup at matched error. These "
    "small trees can be limited by scheduling, allocation and serial work. "
    "The results do not establish a universal benefit from more threads.", "",
    "| Case | Augmented MB | Parallel serial MB | Parallel threaded MB | Max rank (aug. / par.) |",
    "|---|---:|---:|---:|---:|"]
for c in cases:
    alloc = [med(c, v, "bytes")/1e6 for v in variants]
    ranks = [int(med(c, v, "max_rank")) for v in variants[:2]]
    lines.append(f"| {c} | {alloc[0]:.2f} | {alloc[1]:.2f} | {alloc[2]:.2f} | {ranks[0]} / {ranks[1]} |")
lines += ["", "MB denotes cumulative allocated bytes, not peak resident memory.", "",
    "## Accuracy and time-step refinement", "",
    "The reference is a matrix exponential of independent direct ADO or "
    "thermofield equations at the same finite hierarchy/Fock cutoff. This "
    "checks numerical evolution, not physical cutoff convergence. Errors "
    "below are final full-state relative 2-norm errors.", "",
    "| Case | Augmented error h → h/2 | Parallel error h → h/2 | Trace / norm drift (parallel, h) |",
    "|---|---:|---:|---:|"]
for c in cases:
    a, b = [check(c, "augmented", h)["relative_error"] for h in (0.05, 0.025)]
    p, q = [check(c, "parallel_serial", h)["relative_error"] for h in (0.05, 0.025)]
    d = med(c, "parallel_serial", "invariant_drift")
    lines.append(f"| {c} | {a:.3e} → {b:.3e} | {p:.3e} → {q:.3e} | {d:.2e} |")
agreement = max(r["serial_thread_distance"] for r in rows + single)
lines += ["", f"Serial and threaded parallel BUG final TTNs agree within {agreement:.2e} "
    "in absolute TTN distance over all measured cases. The one-worker fallback "
    "was measured separately:", "",
    "| Case | One worker, serial ms | One worker, threaded=true ms |",
    "|---|---:|---:|"]
for c in cases:
    lines.append(f"| {c} | {1e3*med(c, 'parallel_serial', 'seconds', single):.2f} | "
                 f"{1e3*med(c, 'parallel_threads', 'seconds', single):.2f} |")
lines += ["", "## Limits", "",
    "This implements the base first-order algorithm (Algorithms 1–4 of [1]). "
    "`omitted_block_norms[node]` reports h times the norm of the discarded "
    "new/new derivative block. It is a local diagnostic, not a global error "
    "bound. The optional rank-expansion retry strategy in section 4.6 is not "
    "implemented. Adaptive mode uses step doubling; neither that estimate "
    "nor the omitted-block diagnostic can detect directions invisible to "
    "the reduced equations. Check time-step/rank convergence and initial "
    "basis enrichment. The split HEOM trace protection still requires maxrank ≥ M.", "",
    "## Reproduce", "", "From the package root:", "", "```sh",
    "OPENBLAS_NUM_THREADS=1 julia --threads=1 --startup-file=no --project=. benchmarks/parallel_bug/run.jl 5",
    "OPENBLAS_NUM_THREADS=1 julia --threads=4 --startup-file=no --project=. benchmarks/parallel_bug/run.jl 5",
    "python3 benchmarks/parallel_bug/plot.py", "```", "",
    "[Benchmark sources and raw data](../benchmarks/parallel_bug/) include all "
    "runs, convergence data and environment/source hashes. Re-running replaces "
    "the corresponding CSV files, figures and this report.", "", "```text",
    (root / "data" / "environment_threads4.txt").read_text().strip(), "```", "",
    "## References", "",
    "1. G. Ceruti, J. Kusch, C. Lubich and D. Sulz, “A Parallel Basis Update and "
    "Galerkin Integrator for Tree Tensor Networks,” SIAM J. Sci. Comput. 48(1), "
    "A27–A48 (2026), [doi:10.1137/24M1714502](https://doi.org/10.1137/24M1714502); "
    "[open preprint](https://arxiv.org/abs/2412.00858).",
    "2. The authors’ [reproduction code](https://github.com/JonasKu/Publication-Parallel-BUG-for-TTNs), "
    "Section5.1/parallel_TTN_integrator, was consulted for the block assembly. "
    "The local Julia implementation uses SoP contractions and Arnoldi exponential actions.", ""]
report = root.parents[1] / "docs" / "parallel_bug.md"
report.write_text("\n".join(lines))
print(report)
