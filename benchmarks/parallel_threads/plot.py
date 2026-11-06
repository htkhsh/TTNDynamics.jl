"""Plot parallel BUG scaling and regenerate its report from retained CSVs."""
import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import median
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ttn-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {"heom4_8_fused": "HEOM fused", "heom4_8_split": "HEOM split", "tfd4_3": "TFD 4 sites"}
WORKERS = (1, 2, 4, 8)


def load(path):
    with path.open() as f:
        return list(csv.DictReader(f))


groups, augmented = defaultdict(list), defaultdict(list)
for p in WORKERS:
    for target, directory in ((groups, HERE), (augmented, HERE.parent / "augmented_threads")):
        for row in load(directory / "data" / f"runs_threads{p}.csv"):
            target[row["case"], p, row["threaded"] == "true"].append(row)


def med(name, p, threaded, field="seconds", source=groups):
    return median(float(r[field]) for r in source[name, p, threaded])


stages = defaultdict(list)
for row in load(HERE / "data" / "stages_threads4.csv"):
    stages[row["case"], row["threaded"] == "true"].append(row)
fields = ("prepare", "solve", "assemble", "finish", "other")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "axes.titleweight": "bold"})
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
colors = ("#187e9c", "#df8548", "#7360a7")
for (name, label), color in zip(LABELS.items(), colors):
    times = np.array([med(name, p, True) for p in WORKERS])
    speeds = np.array([med(name, p, False) for p in WORKERS]) / times
    axes[0, 0].plot(WORKERS, speeds, "o-", color=color, label=label)
    old_speed = [med(name,p,False,source=augmented)/med(name,p,True,source=augmented) for p in WORKERS]
    axes[0, 0].plot(WORKERS, old_speed, "--", color=color, alpha=0.5)
    axes[0, 1].plot(WORKERS, 100 * speeds / WORKERS, "o-", color=color)
    lows = [min(float(r["seconds"]) for r in groups[name,p,True]) for p in WORKERS]
    highs = [max(float(r["seconds"]) for r in groups[name,p,True]) for p in WORKERS]
    axes[1, 0].errorbar(WORKERS, times, yerr=[times-lows, highs-times], fmt="o-", capsize=3, color=color)
axes[0, 0].axhline(1, color="0.6", ls=":", lw=1)
axes[0, 0].set(title="Parallel BUG: trajectory speedup", ylabel="T(serial) / T(threaded), same process")
axes[0, 0].legend(frameon=False, fontsize=9,
    title="Solid: parallel · dashed: augmented", title_fontsize=8)
axes[0, 1].set(title="Parallel efficiency", ylabel="Speedup / workers (%)")
axes[1, 0].set(title="Five steps, t = 0 to 0.25", ylabel="Seconds (median, min–max)")
for ax in axes.flat[:3]:
    ax.set(xlabel="Julia worker threads", xticks=WORKERS)
    ax.grid(axis="y", color="0.9")
keys = [(name, threaded) for name in LABELS for threaded in (False, True)]
bottom = np.zeros(len(keys))
for field, label, color in zip(fields,
        ("Preparation", "Local ODEs", "Assembly", "Truncation", "Observations / other"),
        ("#b9c8d1", "#187e9c", "#62a48b", "#df8548", "#dedede")):
    totals = np.array([sum(median(float(r[f]) for r in stages[k]) for f in fields) for k in keys])
    vals = 100 * np.array([median(float(r[field]) for r in stages[k]) for k in keys]) / totals
    axes[1, 1].bar(range(len(keys)), vals, bottom=bottom, color=color, label=label)
    bottom += vals
axes[1, 1].set(title="Four-worker process: wall-time phases", ylabel="Share (%)", ylim=(0,100),
    xticks=range(len(keys)), xticklabels=[f"{LABELS[n]}\n{'threads' if t else 'serial'}" for n,t in keys])
axes[1, 1].tick_params(axis="x", labelsize=8)
axes[1, 1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5,-0.2), ncol=2)
(HERE / "figures").mkdir(exist_ok=True)
for ext in ("svg", "png"):
    fig.savefig(HERE / "figures" / f"scaling.{ext}", dpi=180)

lines = ["# Parallel BUG thread scaling", "",
    "For up to 65 leaves and 16 workers, see the [larger-tree study](parallel_large_trees.md).", "",
    "This measures `parallel_bug_step` with `threaded=false` versus `threaded=true`. All node equations use old bases and may run concurrently, including nodes at different tree levels. Preparation, cross-block assembly and final truncation remain sequential. The solver implementation is unchanged by this benchmark.", "",
    "The fixtures and trajectory loop are reused directly from the [augmented BUG study](augmented_threads.md): HEOM M=4 with eight size-3 hierarchy modes (fused/split), and four-site TFD with eight size-3 thermal modes. Each full state space contains 104,976 complex coefficients; no global dense state is formed.", "",
    "![Scaling and phase timings](../benchmarks/parallel_threads/figures/scaling.svg)", "",
    "## Conditions", "",
    "Julia 1.12.6, AMD Ryzen Threadripper PRO 9985WX, KrylovKit 0.10.4. Julia workers = 1/2/4/8, BLAS = 1, GC workers = 1. Each case uses two warmups per execution mode followed by five alternating serial/threaded pairs in the same process. Explicit GC is outside timing; GC during evolution is included. The host is shared, so small differences should be interpreted with the observed ranges.", "",
    "Five fixed steps cover t=0…0.25 with h=0.05, truncation atol=1e-10, maxrank=64 and default Krylov options. Model construction and initial TFD enrichment are excluded; per-step preparation, solves, assembly, truncation and observations are included. HEOM root trace correction is off; TFD has no norm correction. This is an early-rank-growth workload, not a long-time scaling or convergence claim.", "",
    "## Whole-trajectory speedup", "",
    "For HEOM, two workers give the best observed whole-trajectory speedups: 1.30× (fused) and 1.39× (split), or about 65–70% efficiency. Four workers give 1.18–1.25×, and eight only 1.07–1.09×. This is a clearer benefit than the earlier augmented measurements, but it saturates quickly. TFD is variable: 0.89× / 1.15× / 1.01× at 2/4/8 workers, so this dataset does not establish a consistent gain there.", "",
    "Speedup is the ratio of serial and threaded medians from the same process. Efficiency is speedup divided by worker count. The last column separately reports the one-worker serial baseline ratio; it is more exposed to between-process load changes.", "",
    "| Case | Workers | Serial s | Threaded s (min–max) | Speedup | Efficiency | 1-worker baseline ratio |",
    "|---|---:|---:|---:|---:|---:|---:|"]
for name, label in LABELS.items():
    for p in WORKERS:
        values = [float(r["seconds"]) for r in groups[name,p,True]]
        tm, serial = median(values), med(name,p,False)
        speed = serial/tm
        lines.append(f"| {label} | {p} | {serial:.3f} | {tm:.3f} ({min(values):.3f}–{max(values):.3f}) | {speed:.2f}× | {100*speed/p:.1f}% | {med(name,1,False)/tm:.2f}× |")
lines += ["", "Dashed curves in the speedup plot reproduce the earlier augmented measurements. They compare scaling of each method against its own sequential execution. Absolute runtimes at equal h do not establish which method reaches a common physical error faster: the two integration formulas and their ranks/time errors differ. See the [common-error-target comparison](parallel_bug_optimization.md) for smaller cases.", "",
    "## Numerical equivalence", "",
    "Each serial/threaded pair checks final TTN distance, sampled densities, full rank histories, all truncation pass/fail flags, and the discarded-block diagnostic h*eta. Allocations below are cumulative, not peak memory.", "",
    "| Case | Max kept / augmented rank | Largest local dimension | Truncation misses / 5 | Max trace / norm drift | Serial/thread distance | Allocated GB (4 workers) |",
    "|---|---:|---:|---:|---:|---:|---:|"]
for name, label in LABELS.items():
    rows = sum((groups[name,p,t] for p in WORKERS for t in (False,True)), [])
    mx = lambda k: max(float(r[k]) for r in rows)
    lines.append(f"| {label} | {mx('max_rank'):.0f} / {mx('max_augmented_rank'):.0f} | {mx('max_local_dimension'):.0f} | {mx('truncation_failures'):.0f} | {mx('invariant_drift'):.2e} | {mx('state_distance'):.2e} | {med(name,4,True,'bytes')/1e9:.2f} |")
lines += ["", "These checks establish equivalence of execution modes, not accuracy against the exact dynamics. Norm drift and discarded-block norms can reflect the parallel BUG formula; they are not concurrency errors. Any rank-cap truncation misses remain visible in the raw data.", "",
    "## Four-worker phase timings", "",
    "Three repeated trajectories per mode use an isolated, checked source copy for instrumentation. Results match the ordinary solver. The ordinary scaling trials above contain no additional timers.", "",
    "| Case | Execution | Preparation s | Local ODEs s | Assembly s | Truncation s | Observations / other s |",
    "|---|---|---:|---:|---:|---:|---:|"]
for (name, threaded), rows in stages.items():
    values = [median(float(r[f]) for r in rows) for f in fields]
    lines.append(f"| {LABELS[name]} | {'threaded' if threaded else 'serial'} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
node_times = defaultdict(list)
for row in load(HERE / "data" / "nodes_threads4.csv"):
    if row["threaded"] == "false":
        node_times[row["case"],int(row["node"])].append(float(row["solve_seconds"]))
lines += ["", "| Case | Local ODE phase speedup | Most expensive node | Share of serial local solve routine time |",
    "|---|---:|---:|---:|"]
for name, label in LABELS.items():
    times = {n:median(v) for (c,n),v in node_times.items() if c==name}
    node = max(times,key=times.get)
    ratio = median(float(r['solve']) for r in stages[name,False])/median(float(r['solve']) for r in stages[name,True])
    lines.append(f"| {label} | {ratio:.2f}× | {node} | {100*times[node]/sum(times.values()):.1f}% |")
lines += ["", "HEOM's heaviest node accounts for about 39% of local solve routine time, versus roughly 72% in the earlier augmented study; its ODE phase benefits more from concurrency. TFD still spends about 76% in one node. Only the local ODE phase is threaded. Changes in other phase timings are not parallel execution of those phases; allocation/GC placement and host variability affect them, particularly the TFD observation/other time. These separate instrumented runs should not be used to reconstruct the ordinary-run speedup exactly.", "",
    "Node timers include local workspace setup and solve bookkeeping. Their sum is not concurrent wall time. All-node concurrency removes the child-to-parent ODE dependency, but does not split a single expensive node solve among workers. Uneven node cost, sequential preparation/assembly/truncation, allocations and task overhead can therefore still limit scaling.", "",
    "## Reproduce", "", "From the TTNDynamics package root:", "", "```sh",
    "for n in 1 2 4 8; do", "  OPENBLAS_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 julia --threads=$n --project=. benchmarks/parallel_threads/run.jl 5", "done",
    "OPENBLAS_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 julia --threads=4 --project=. -e 'include(\"benchmarks/parallel_threads/run.jl\"); include(\"benchmarks/parallel_threads/stages.jl\"); benchmark_parallel_stages()'",
    "python3 benchmarks/parallel_threads/plot.py", "```", "",
    "[Raw CSVs and environment/source hashes](../benchmarks/parallel_threads/data/), [runner](../benchmarks/parallel_threads/run.jl), [phase instrumentation](../benchmarks/parallel_threads/stages.jl). Plotting also reads the retained augmented dataset for dashed reference curves.", "",
    "The implemented parallel BUG formula follows G. Ceruti, J. Kusch, C. Lubich and D. Sulz, *A Parallel Basis Update and Galerkin Integrator for Tree Tensor Networks*, SIAM J. Sci. Comput. 48(1), A27–A48 (2026), [doi:10.1137/24M1714502](https://doi.org/10.1137/24M1714502). The optional rank-expansion/rejection strategy is not enabled; see the [algorithm notes](parallel_bug.md).", ""]
(HERE.parents[1] / "docs" / "parallel_threads.md").write_text("\n".join(lines))
