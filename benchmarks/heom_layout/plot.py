"""Render the measured fused/split comparison and its standalone report."""
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
    rows = [{k: v if k in ("method", "layout") else float(v) for k, v in row.items()}
            for row in csv.DictReader(stream)]
with (root / "data" / "convergence.csv").open() as stream:
    checks = [{k: v if k == "layout" else float(v) for k, v in row.items()}
              for row in csv.DictReader(stream)]
cases = list(dict.fromkeys((int(r["dimension"]), r["method"]) for r in rows))
groups = {(d, method, layout): [r for r in rows if r["dimension"] == d and
          r["method"] == method and r["layout"] == layout]
          for d, method in cases for layout in ("fused", "split")}

def med(d, method, layout, field):
    return median(r[field] for r in groups[d, method, layout])

colors = {"fused": "#4263aa", "split": "#25887e"}
plt.rcParams.update({"svg.fonttype": "none", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
fig.suptitle("HEOM: fused system leaf vs separate ket/bra leaves", fontsize=15)
for ax, field, scale, ylabel in zip(axes[0], ("wall_seconds", "bytes"), (1e3, 1e-6),
                                  ("Median wall time / ms", "Allocated memory per run / MB")):
    for layout, offset in (("fused", -0.19), ("split", 0.19)):
        values = [med(d, method, layout, field) * scale for d, method in cases]
        ranges = [[r[field]*scale for r in groups[d, method, layout]] for d, method in cases]
        error = [[v-min(a) for v, a in zip(values, ranges)],
                 [max(a)-v for v, a in zip(values, ranges)]]
        ax.bar([i+offset for i in range(len(cases))], values, width=0.36,
               color=colors[layout], label=layout, yerr=error, capsize=3)
    ax.set_xticks(range(len(cases)), [f"M={d}\n{method}" for d, method in cases])
    ax.set_ylabel(ylabel)
dimensions = sorted({d for d, method in cases if method == "fixed"})
for layout in colors:
    axes[1, 0].semilogy(dimensions, [med(d, "fixed", layout, "operator_factor_bytes")/1024
                                    for d in dimensions], "o-", color=colors[layout], label=layout)
    data = sorted([r for r in checks if r["layout"] == layout], key=lambda r: r["dimension"])
    for key, style, label in (("coarse_density_error", "o-", "h=0.05"),
                              ("fine_density_error", "s--", "h=0.025")):
        axes[1, 1].semilogy([r["dimension"] for r in data], [r[key] for r in data],
                            style, color=colors[layout], label=f"{layout}, {label}",
                            mfc="none" if layout == "split" else colors[layout],
                            ms=8 if layout == "split" else 5)
axes[1, 0].set(xlabel="System dimension M", ylabel="Local operator matrix payload / KiB")
axes[1, 1].set(xlabel="System dimension M", ylabel="Maximum density error vs dense HEOM")
for ax in axes.flat:
    ax.grid(alpha=0.2, axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
figures = root / "figures"
figures.mkdir(exist_ok=True)
for suffix in ("svg", "png"):
    path = figures / f"comparison.{suffix}"
    fig.savefig(path, dpi=160)
    print(path)

lines = ["# Fused and split HEOM system layouts", "",
    "Select the representation on the model; all constructors, observables and "
    "fixed/adaptive evolution then use that layout:", "", "```julia",
    "system = HEOMTTNSystem(H, modes, hierarchy_sizes; system_layout=:split)",
    "initial = heom_initial_state(system, rho)",
    "result = heom_evolve(system, initial, 0:0.05:1; atol=1e-10)", "```", "",
    "`:fused` is the compatible default, with dimensions `[M^2, sizes...]`. "
    "`:split` uses `[M, M, sizes...]`, ket first. Both flatten to the same "
    "column-major HEOM vector. In the default split tree, ket and bra are siblings; "
    "their parent connects to a balanced hierarchy subtree. Custom trees are supported too.", "",
    "The split generator uses H/V on ket and transpose(H/V) on bra, avoiding "
    "M²-by-M² local factors. The number of product terms grows from 1+3K to "
    "2+5K for K bath modes. The trace functional has rank M on both system "
    "leaves, so the current trace-protected algorithm keeps those ranks at M "
    "and requires `maxrank >= M`, even with `correct_trace=false`.", "",
    "## Timing and storage", "",
    f"{len(next(iter(groups.values())))} pairs run in alternating order after two complete warmups per layout. "
    "Julia and BLAS each use one thread. GC is enabled and explicitly collected "
    "before each timed call. This measures end-to-end `heom_evolve`, including "
    "generator and trace-functional construction, preparation, "
    "propagation and observations; model/initial-state construction, dense reference "
    "work and plotting are outside timing. No other test or numerical job ran "
    "concurrently. Reported times are medians; plot error bars show min–max.", "",
    "The fixture uses a complex Hermitian tridiagonal H, two Hermitian coupling "
    "operators and two exponential modes with box sizes `[2,2]`. H has diagonal "
    "−0.2…0.2 and upper off-diagonal 0.25+0.07i. V has diagonal 0.6…1.2 and "
    "upper off-diagonal 0.1i; W has diagonal −0.3…0.7 and off-diagonal 0.07. "
    "Rates are 0.8 and 1.2, forward coefficients 0.05+0.01i and 0.025+0.005i, "
    "with conjugate backward coefficients and default scaling. The initial "
    "density is 0.95|1><1| + 0.05I/M.", "",
    "All runs cover t=0–1 with `maxrank=64` and `correct_trace=false`. "
    "Fixed stepping uses h=0.05 and truncation atol=1e-10; h=0.025 is checked "
    "separately outside timing. The M=4 adaptive check uses initial dt=0.1, "
    "saveat=0:0.1:1, atol=1e-8, rtol=1e-4 and the default truncation fraction 0.1.", "",
    "| M | Method | Fused, ms | Split, ms | Fused / split time | Fused / split allocated MB |",
    "|---:|---|---:|---:|---:|---:|"]
for d, method in cases:
    f, s = med(d, method, "fused", "wall_seconds"), med(d, method, "split", "wall_seconds")
    fb, sb = med(d, method, "fused", "bytes"), med(d, method, "split", "bytes")
    lines.append(f"| {d} | {method} | {f*1e3:.2f} | {s*1e3:.2f} | {f/s:.2f}× | {fb/1e6:.2f} / {sb/1e6:.2f} |")
lines += ["", "Ratios greater than one mean split is faster. Allocations are cumulative "
    "bytes, not peak resident memory. Local-operator storage below counts matrix "
    "payload only, excluding dictionary/object overhead and transient construction arrays. "
    "Ranks are maxima over saved states; augmented ranks and unsaved adaptive "
    "internal/rejected states are excluded.", "",
    "| M | Fused / split operator KiB | Fused / split max saved rank | Fused / split max system–bath rank |",
    "|---:|---:|---:|---:|"]
for d in dimensions:
    fields = ("operator_factor_bytes", "max_rank", "system_bath_rank")
    pairs = [(med(d, "fixed", "fused", k), med(d, "fixed", "split", k)) for k in fields]
    storage, ranks, cut = pairs
    lines.append(f"| {d} | {storage[0]/1024:.2f} / {storage[1]/1024:.2f} | "
                 f"{int(ranks[0])} / {int(ranks[1])} | {int(cut[0])} / {int(cut[1])} |")
lines += ["", "![Measured comparison](../benchmarks/heom_layout/figures/comparison.svg)", "",
    "For this fixture, the tradeoff is between extra split nodes/terms and "
    "smaller local matrix products. Split allocates more cumulative memory in "
    "every measured case, despite storing smaller operator matrices; this "
    "benchmark does not establish lower peak memory use. The crossover is specific to these models "
    "and this machine; it is not a general M threshold. Larger hierarchy ranks "
    "or different topologies can change the result. Fused remains the default.", "",
    "## Accuracy", "",
    "The dense reference applies the direct ADO equations without calling the "
    "product-sum generator. Matrix exponentiation at dt=0.025 supplies the "
    "reference trajectory, with global dimensions 16, 64, 256 and 1024. "
    "This validates the same finite box cutoff, not physical hierarchy convergence.", "",
    "| M | Layout | Generator relative error | Max density error, h → h/2 | Final full-state relative error, h → h/2 |",
    "|---:|---|---:|---:|---:|"]
for r in checks:
    lines.append(f"| {int(r['dimension'])} | {r['layout']} | {r['generator_relative_error']:.2e} | "
                 f"{r['coarse_density_error']:.3e} → {r['fine_density_error']:.3e} | "
                 f"{r['coarse_final_error']:.3e} → {r['fine_final_error']:.3e} |")
lines += ["", "The layouts can have different truncation budgets per edge and different "
    "adaptive meshes. Equal local tolerances do not imply identical trajectories. "
    "The following comparisons use their common observation times; trace drift "
    "is measured with root correction disabled.", "",
    "| M | Method | Max density difference between layouts | Fused / split accepted (rejected) | Max trace drift |",
    "|---:|---|---:|---:|---:|"]
for d, method in cases:
    f, s = groups[d, method, "fused"][0], groups[d, method, "split"][0]
    drift = max(r["trace_drift"] for layout in colors for r in groups[d, method, layout])
    lines.append(f"| {d} | {method} | {f['layout_density_difference']:.3e} | "
                 f"{int(f['accepted'])} ({int(f['rejected'])}) / {int(s['accepted'])} ({int(s['rejected'])}) | {drift:.2e} |")
for d, method in cases:
    if method == "adaptive":
        errors = [med(d, method, layout, "density_error") for layout in colors]
        full = [med(d, method, layout, "final_relative_error") for layout in colors]
        lines += ["", f"M={d} adaptive maximum density errors are {errors[0]:.3e} (fused) "
                  f"and {errors[1]:.3e} (split); final full-state relative errors are "
                  f"{full[0]:.3e} and {full[1]:.3e}, respectively."]
lines += ["", "## Tree diagrams", "",
    "The M=4 final trees show actual saved ranks:", "",
    "![Fused tree](../benchmarks/heom_layout/trees/fused_final.svg)", "",
    "![Split tree](../benchmarks/heom_layout/trees/split_final.svg)", "",
    "## Reproduce", "", "From the package root:", "", "```sh",
    "OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=examples benchmarks/heom_layout/run.jl 5",
    "python3 benchmarks/heom_layout/plot.py", "```", "",
    "[Raw timings](../benchmarks/heom_layout/data/runs.csv) and "
    "[convergence data](../benchmarks/heom_layout/data/convergence.csv) are saved "
    "alongside the scripts. Re-running replaces these outputs and this report.", "",
    "```text", (root / "data" / "environment.txt").read_text().strip(), "```", "",
    "Related density-vectorization work: R. Borrelli, “Density matrix dynamics in "
    "twin-formulation,” J. Chem. Phys. 150, 234102 (2019), "
    "[doi:10.1063/1.5099416](https://doi.org/10.1063/1.5099416).", ""]
report = root.parents[1] / "docs" / "heom_layout.md"
report.write_text("\n".join(lines))
print(report)
