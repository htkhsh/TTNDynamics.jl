"""Regenerate scaling figures and the measurement report from retained CSVs."""
from pathlib import Path
from collections import defaultdict
from statistics import median
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {"heom4_8_fused": "HEOM fused", "heom4_8_split": "HEOM split", "tfd4_3": "TFD 4 sites"}
WORKERS = (1, 2, 4, 8)
groups = defaultdict(list)
for p in WORKERS:
    with (HERE / "data" / f"runs_threads{p}.csv").open() as f:
        for row in csv.DictReader(f):
            groups[row["case"], p, row["threaded"] == "true"].append(row)


def med(name, p, threaded, field="seconds"):
    return median(float(r[field]) for r in groups[name, p, threaded])


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "axes.titleweight": "bold"})
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
colors = ("#187e9c", "#df8548", "#7360a7")
for (name, label), color in zip(LABELS.items(), colors):
    times = np.array([med(name, p, True) for p in WORKERS])
    speeds = np.array([med(name, p, False) for p in WORKERS]) / times
    axes[0, 0].plot(WORKERS, speeds, "o-", color=color, label=label)
    axes[0, 1].plot(WORKERS, 100 * speeds / WORKERS, "o-", color=color)
    lows = [min(float(r["seconds"]) for r in groups[name, p, True]) for p in WORKERS]
    highs = [max(float(r["seconds"]) for r in groups[name, p, True]) for p in WORKERS]
    axes[1, 0].errorbar(WORKERS, times, yerr=[times-lows, highs-times], fmt="o-",
                      capsize=3, color=color, label=label)
axes[0, 0].axhline(1, color="0.6", ls="--", lw=1)
axes[0, 0].set(title="Augmented BUG: trajectory speedup", ylabel="T(serial) / T(threaded), same process")
axes[0, 0].legend(frameon=False)
axes[0, 1].set(title="Parallel efficiency", ylabel="Speedup / workers (%)")
axes[1, 0].set(title="Five steps, t = 0 to 0.25", ylabel="Seconds (median, min–max)")
for ax in axes.flat[:3]:
    ax.set(xlabel="Julia worker threads", xticks=WORKERS)
    ax.grid(axis="y", color="0.9")

stages = defaultdict(list)
with (HERE / "data" / "stages_threads4.csv").open() as f:
    for row in csv.DictReader(f):
        stages[row["case"], row["threaded"] == "true"].append(row)
keys = [(name, threaded) for name in LABELS for threaded in (False, True)]
bottom = np.zeros(len(keys))
for field, label, color in (("prepare", "Preparation", "#b9c8d1"),
                            ("update", "Subtree updates", "#187e9c"),
                            ("finish", "Truncation", "#df8548"),
                            ("other", "Observations / other", "#d9d9d9")):
    vals = np.array([median(float(r[field]) for r in stages[k]) for k in keys])
    totals = np.array([sum(median(float(r[f]) for r in stages[k])
                           for f in ("prepare", "update", "finish", "other")) for k in keys])
    vals = 100 * vals / totals
    axes[1, 1].bar(range(len(keys)), vals, bottom=bottom, color=color, label=label)
    bottom += vals
axes[1, 1].set(title="Four-worker process: wall-time phases", ylabel="Share (%)", ylim=(0, 100),
               xticks=range(len(keys)), xticklabels=[f"{LABELS[n]}\n{'threads' if t else 'serial'}" for n, t in keys])
axes[1, 1].tick_params(axis="x", labelsize=8)
axes[1, 1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
(HERE / "figures").mkdir(exist_ok=True)
for ext in ("svg", "png"):
    fig.savefig(HERE / "figures" / f"scaling.{ext}", dpi=180)

lines = ["# Augmented BUG thread scaling", "",
    "Augmented BUG now updates independent sibling subtrees concurrently. A parent waits for both updated child bases before its Galerkin solve. This changes scheduling, not the integration formula. Both fixed and adaptive HEOM/TFD APIs honor `threaded=true` (default); `threaded=false` runs sequentially. Start Julia with `--threads=N`. The one-worker path falls back to sequential execution.", "",
    "```julia", "result = heom_evolve(system, initial, 0:0.05:0.25;",
    "                     method=:augmented, threaded=true)",
    "# The same options apply to tfd_evolve and adaptive keyword time spans.", "```", "",
    "![Scaling and phase timings](../benchmarks/augmented_threads/figures/scaling.svg)", "",
    "## Measurement", "",
    "Julia 1.12.6, AMD Ryzen Threadripper PRO 9985WX, KrylovKit 0.10.4; BLAS and GC each use one thread. Separate processes use 1, 2, 4, and 8 Julia workers. Each case has two warmup trajectories per mode, followed by five alternating serial/threaded pairs; explicit GC is outside timing, while GC during evolution is included. The host is shared: ranges show observed variability, not confidence intervals.", "",
    "Each trajectory has five fixed steps (`h=0.05`, `t=0…0.25`), absolute truncation budget `1e-10`, `maxrank=64`, and default Krylov settings. Timings include every step's preparation, updates, truncation and observable contractions, but exclude model construction and TFD's initial `H*initial` basis enrichment. No dense hierarchy/state is constructed. Allocations are cumulative, not peak memory. This measures early rank growth; longer trajectories can have different ranks and scaling.", "",
    "HEOM has system dimension M=4 and eight hierarchy modes, each of size 3 (fused and split layouts). TFD has four sites, eight thermal modes and Fock size 3. All three global state spaces have 104,976 complex entries; tree leaves/terms and actual rank sizes differ. The previous small cases were M=4/six size-2 HEOM modes (1,024 entries) and three-site/size-2 TFD (576 entries). These cutoff choices define performance workloads, not converged bath/Fock models.", "",
    "## End-to-end scaling", "",
    "The same-process median speedups with 2–8 workers are only about 1.00–1.07×, with overlapping timing ranges. These workloads show no substantial scalable speedup. Four workers deliver roughly 25–27% efficiency and eight workers about 13%. Between-process load variation matters: for example, the serial TFD median differs between the one- and four-worker processes even though both execute sequentially.", "",
    "Speedup is the serial median divided by the threaded median from the same process; efficiency is that speedup divided by the worker count. The last column also gives the conventional one-worker serial baseline ratio for comparison. GC and BLAS thread counts are fixed in both modes.", "",
    "| Case | Workers | Serial s | Threaded s (min–max) | Same-process speedup | Efficiency | 1-worker baseline ratio |",
    "|---|---:|---:|---:|---:|---:|---:|"]
for name, label in LABELS.items():
    for p in WORKERS:
        values = [float(r["seconds"]) for r in groups[name, p, True]]
        tm, serial = median(values), med(name, p, False)
        speed = serial / tm
        lines.append(f"| {label} | {p} | {serial:.3f} | {tm:.3f} ({min(values):.3f}–{max(values):.3f}) | {speed:.2f}× | {100*speed/p:.1f}% | {med(name,1,False)/tm:.2f}× |")
lines += ["", "## Numerical checks", "",
    "Every pair checks the final full-TTN distance with local contractions, sampled density matrices, complete saved rank history, and every truncation pass/fail flag. Root trace correction is disabled; TFD has no norm correction.", "",
    "| Case | Max kept / augmented rank | Largest local dimension | Truncation misses / 5 | Max invariant drift | Serial/thread distance | Allocated GB (4 workers) |",
    "|---|---:|---:|---:|---:|---:|---:|"]
for name, label in LABELS.items():
    rows = sum((groups[name, p, t] for p in WORKERS for t in (False, True)), [])
    mx = lambda k: max(float(r[k]) for r in rows)
    lines.append(f"| {label} | {mx('max_rank'):.0f} / {mx('max_augmented_rank'):.0f} | {mx('max_local_dimension'):.0f} | {mx('truncation_failures'):.0f} | {mx('invariant_drift'):.2e} | {mx('state_distance'):.2e} | {med(name,4,True,'bytes')/1e9:.2f} |")
lines += ["", "A truncation miss means the chosen rank cap did not meet the per-step budget. These runs establish concurrency equivalence and performance; they do not establish time/rank convergence or global physical accuracy.", "",
    "## Wall-time breakdown", "",
    "Separate instrumented runs in a four-worker process report three repetitions per mode. Instrumentation is inserted into an isolated source copy, checked against the ordinary solver, and is absent from the main scaling runs. Subtree updates include basis construction, local ODEs, upward contractions and result publication. Root ODE time is part of updates and cannot overlap with its descendants; it is shown separately, not added again.", "",
    "| Case | Execution | Preparation s | Updates s | Truncation s | Observations / other s | Root ODE s |",
    "|---|---|---:|---:|---:|---:|---:|"]
for (name, threaded), rows in stages.items():
    values = [median(float(r[f]) for r in rows) for f in ("prepare", "update", "finish", "other", "root_solve")]
    lines.append(f"| {LABELS[name]} | {'threaded' if threaded else 'serial'} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
node_times = defaultdict(list)
with (HERE / "data" / "nodes_threads4.csv").open() as f:
    for row in csv.DictReader(f):
        if row["threaded"] == "false":
            node_times[row["case"], int(row["node"])].append(float(row["solve_seconds"]))
lines += ["", "The following serial measurements show how unevenly local ODE work is distributed. Shares use medians of accumulated per-node solve times across the five steps; workspace construction and basis algebra are outside these solve timers.", "",
    "| Case | Most expensive node | Its local ODE time s | Share of all local ODE time |",
    "|---|---:|---:|---:|"]
for name, label in LABELS.items():
    times = {node: median(values) for (case, node), values in node_times.items() if case == name}
    node = max(times, key=times.get)
    lines.append(f"| {label} | {node} | {times[node]:.3f} | {100*times[node]/sum(times.values()):.1f}% |")
lines += ["", "In these measurements one internal core accounts for about 72% of local ODE time in each HEOM layout and 92% in TFD. The primary limitation is concentrated work: that single solve is sequential, and subtree scheduling cannot divide it among workers. Preparation and truncation also remain serial, but do not explain most of the measured time. Optimizing that core's contraction/Arnoldi action is a more useful next target than increasing the subtree worker count for these cases.", "",
    "Dependencies from leaves to root, uneven subtree costs, memory allocation and task overhead limit scaling; increasing the thread count does not remove these costs. Parallel BUG is a distinct algorithm that solves all node equations in old bases, so its timing/accuracy tradeoff should be evaluated separately. Small timing differences here are comparable to run-to-run variation; use one worker or `threaded=false` for these workloads unless a measurement of the actual model shows a benefit.", "",
    "The [parallel BUG scaling study](parallel_threads.md) now measures the same fixtures, time interval and worker counts with the parallel BUG formula.", "",
    "## Reproduce", "", "From the TTNDynamics package root:", "", "```sh",
    "for n in 1 2 4 8; do", "  OPENBLAS_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 julia --threads=$n --project=. benchmarks/augmented_threads/run.jl 5", "done",
    "OPENBLAS_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 julia --threads=4 --project=. -e 'include(\"benchmarks/augmented_threads/run.jl\"); include(\"benchmarks/augmented_threads/stages.jl\"); benchmark_stages()'",
    "python3 benchmarks/augmented_threads/plot.py", "```", "",
    "Raw [timings, diagnostics and environment/source hashes](../benchmarks/augmented_threads/data/), [runner](../benchmarks/augmented_threads/run.jl), and [phase instrumentation](../benchmarks/augmented_threads/stages.jl) are retained. Node solve times are elapsed durations; their sum is not parallel wall time. GC worker count is fixed explicitly because Julia otherwise changes it with the worker count.", "",
    "## References", "",
    "1. G. Ceruti, C. Lubich and D. Sulz, *Rank-adaptive time integration of tree tensor networks*, SIAM J. Numer. Anal. 61(1), 194–222 (2023), [doi:10.1137/22M1473790](https://doi.org/10.1137/22M1473790).",
    "2. G. Ceruti, J. Kusch, C. Lubich and D. Sulz, *A Parallel Basis Update and Galerkin Integrator for Tree Tensor Networks*, SIAM J. Sci. Comput. 48(1), A27–A48 (2026), [doi:10.1137/24M1714502](https://doi.org/10.1137/24M1714502).", ""]
(HERE.parents[1] / "docs" / "augmented_threads.md").write_text("\n".join(lines))
