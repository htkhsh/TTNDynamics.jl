"""Regenerate the larger-tree scaling figures and report from retained CSVs."""
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
LABELS = {"heom16": "HEOM / 17 leaves", "heom32": "HEOM / 33 leaves",
          "heom64": "HEOM / 65 leaves", "tfd16": "TFD / 17 leaves"}
WORKERS = (1,2,4,8,16)
ENVIRONMENT = dict(line.split(": ",1) for line in
    (HERE/"data"/"environment_threads1.txt").read_text().splitlines() if ": " in line)


def load(path):
    with path.open() as f:
        return list(csv.DictReader(f))


groups = defaultdict(list)
for p in WORKERS:
    for row in load(HERE / "data" / f"runs_threads{p}.csv"):
        groups[row["case"],p,row["threaded"]=="true"].append(row)


def med(name,p,threaded,field="seconds"):
    return median(float(r[field]) for r in groups[name,p,threaded])


stages, node_times = defaultdict(list), defaultdict(list)
for row in load(HERE / "data" / "stages_threads8.csv"):
    stages[row["case"],row["threaded"]=="true"].append(row)
for row in load(HERE / "data" / "nodes_threads8.csv"):
    if row["threaded"]=="false" and row["skipped"]=="false":
        node_times[row["case"],int(row["node"])].append(float(row["solve_seconds"]))

plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
    "axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none",
    "axes.titleweight":"bold"})
fig,axes = plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
colors = ("#64a791","#16859b","#315681","#ae6944")
for (name,label),color in zip(LABELS.items(),colors):
    times = np.array([med(name,p,True) for p in WORKERS])
    speed = np.array([med(name,p,False) for p in WORKERS])/times
    axes[0,0].plot(WORKERS,speed,"o-",color=color,label=label)
    axes[0,1].plot(WORKERS,100*speed/WORKERS,"o-",color=color)
    lows = [min(float(r["seconds"]) for r in groups[name,p,True]) for p in WORKERS]
    highs = [max(float(r["seconds"]) for r in groups[name,p,True]) for p in WORKERS]
    axes[1,0].errorbar(WORKERS,times,yerr=[times-lows,highs-times],fmt="o-",color=color,capsize=3)
axes[0,0].axhline(1,color="0.6",ls="--",lw=1)
axes[0,0].set(title="Parallel BUG: larger trees",ylabel="T(serial) / T(threaded), same process")
axes[0,0].legend(frameon=False,fontsize=9)
axes[0,1].set(title="Parallel efficiency",ylabel="Speedup / workers (%)")
axes[1,0].set(title="One step: t = 0.20 to 0.25",ylabel="Seconds (median, min–max)",yscale="log")
for ax in axes.flat[:3]:
    ax.set(xlabel="Julia worker threads",xticks=WORKERS)
    ax.grid(axis="y",color="0.9")
fields = ("prepare","solve","assemble","finish","other")
keys = [(name,t) for name in ("heom64","tfd16") for t in (False,True)]
totals = np.array([sum(median(float(r[f]) for r in stages[k]) for f in fields) for k in keys])
bottom = np.zeros(len(keys))
for f,label,color in zip(fields,("Preparation","Local ODEs","Assembly","Truncation","Other"),
        ("#b9c8d1","#16859b","#64a791","#ae6944","#dedede")):
    values = 100*np.array([median(float(r[f]) for r in stages[k]) for k in keys])/totals
    axes[1,1].bar(range(len(keys)),values,bottom=bottom,color=color,label=label)
    bottom += values
axes[1,1].set(title="Eight-worker process: wall-time phases",ylabel="Share (%)",ylim=(0,100),
    xticks=range(len(keys)),xticklabels=[f"{LABELS[n]}\n{'threads' if t else 'serial'}" for n,t in keys])
axes[1,1].tick_params(axis="x",labelsize=8)
axes[1,1].legend(frameon=False,fontsize=8,loc="upper center",bbox_to_anchor=(0.5,-0.18),ncol=3)
figures=HERE/"figures";figures.mkdir(exist_ok=True)
for ext in ("svg","png"):
    fig.savefig(figures/f"scaling.{ext}",dpi=180)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,3.6),constrained_layout=True)
for ax,name in zip(axes,("heom64","tfd16")):
    times={n:median(v) for (c,n),v in node_times.items() if c==name}
    order=sorted(times,key=times.get,reverse=True)
    shares=[100*times[n]/sum(times.values()) for n in order]
    ax.bar(range(len(order)),shares,color="#16859b")
    ax.set(title=LABELS[name],xlabel="Active node solves, sorted by cost",ylabel="Serial local solve time (%)")
    ax.grid(axis="y",color="0.9")
for ext in ("svg","png"):
    fig.savefig(figures/f"node_work.{ext}",dpi=180)
plt.close(fig)

lines=["# Parallel BUG on larger trees","",
    "This study increases the number of physical leaves and independent node solves. It uses the existing parallel BUG implementation without solver changes, comparing `threaded=false` and `true` in processes with 1, 2, 4, 8 and 16 Julia workers. One-worker requested threading falls back to the identical serial path; differences between its paired timings are variability, not parallel acceleration.","",
    "![Larger-tree scaling](../benchmarks/parallel_large_trees/figures/scaling.svg)","",
    "## Controlled workloads","",
    "All systems have electronic dimension M=4 and local bath dimension 3. HEOM uses 16, 32 or 64 modes, giving 17, 33 or 65 leaves (33, 65 or 129 total nodes). TFD uses 16 signed thermal modes, giving 17 leaves and 33 nodes. A system leaf and a balanced bath subtree meet at the root. Coupling types/sites are interleaved in bath order.","",
    "HEOM keeps the sum of forward coefficients at 0.64+0.08im and uses conjugate backward coefficients. Four diagonal coupling types repeat; decay rates sample 0.8…1.5. TFD has four sites with two physical oscillators per site, transformed at beta=2 into signed pairs. Frequencies sample 0.6…1.0 and each physical amplitude is 0.3/sqrt(oscillators per site), keeping the sum of squared amplitudes at 0.36. These are controlled performance fixtures, not fitted/converged bath models. Increasing mode count also increases the SoP term count; this is not a pure topology change at fixed operator size.","",
    "The common step size is h=0.05. Four serial steps first construct a snapshot at t=0.20. Every timed call starts from that same read-only snapshot and advances one step to t=0.25, with truncation atol=1e-10, maxrank=64 and default Krylov settings. Root invariant correction is off. Each call includes QR/environment preparation, local solves, block assembly and truncation. Snapshot/model construction, observations and numerical comparisons are outside timing. This differs from the [earlier five-step trajectory study](parallel_threads.md); compare scaling ratios with care, not absolute times.","",
    "TFD's initial basis seed includes H times the vacuum state. Number terms annihilating the vacuum are omitted only during that one-time seed construction; the full Hamiltonian is used for evolution. A small fixture check confirms equality of the full and reduced seed directions. Redundant zero-weight bases can differ from earlier bootstrap constructions.","",
    f"Julia {ENVIRONMENT['Julia']}, {ENVIRONMENT['CPU']}, KrylovKit {ENVIRONMENT['KrylovKit']}. BLAS threads = {ENVIRONMENT['BLAS threads']}; GC threads = {ENVIRONMENT['GC threads']}. After {ENVIRONMENT['warmups per mode']} warmup per execution mode, {ENVIRONMENT['repetitions']} serial/threaded pairs alternate call order. Explicit GC runs outside timing; collections during a step are included. This is a shared host: ranges are observed variability, not confidence intervals.","",
    "| Case | Leaves / nodes | Tree height | SoP terms | Input → output max rank | Local solves / skipped full leaves | Largest local dimension |",
    "|---|---:|---:|---:|---:|---:|---:|"]
for name,label in LABELS.items():
    r=groups[name,1,False][0]
    lines.append(f"| {label} | {r['leaves']} / {r['nodes']} | {r['height']} | {r['terms']} | {r['input_rank']} → {r['output_rank']} | {r['local_solves']} / {r['skipped_leaf_solves']} | {r['max_local_dimension']} |")
lines += ["","Input trees with bond ranks: "+", ".join(f"[{LABELS[n]}](../benchmarks/parallel_large_trees/figures/trees/{n}.svg)" for n in LABELS)+". They are rendered with TTNPlots from the saved topology CSVs.","",
    "## Measured speedup","",
    "For the 65-leaf HEOM tree, the same-process speedup is 1.36× / 1.76× / 1.87× / 1.86× at 2/4/8/16 workers. Its best threaded median is 4.30 s at eight workers; sixteen workers take 4.50 s. Thus this workload benefits more than the earlier small trees, but remains far from linear scaling. Efficiency is about 23% at eight workers and 12% at sixteen.","",
    "The best measured threaded medians for the 17- and 33-leaf HEOM trees are 0.269 s and 1.187 s, both at four workers. Same-process ratios can rise when a particular process's serial baseline is slower: for example, the 17-leaf ratio is highest at sixteen workers even though its threaded median is slower than at four. Prefer the absolute timings and ranges as well as ratios when choosing a worker count.","",
    "Speedup is serial median / threaded median within the same process. Efficiency is that ratio / worker count. The last column separately reports the one-worker serial baseline ratio; between-process host variation can change it.","",
    "| Case | Workers | Serial s | Threaded s (min–max) | Speedup | Efficiency | 1-worker baseline ratio |",
    "|---|---:|---:|---:|---:|---:|---:|"]
for name,label in LABELS.items():
    for p in WORKERS:
        values=[float(r['seconds']) for r in groups[name,p,True]]
        tm,serial=median(values),med(name,p,False)
        speed=serial/tm
        lines.append(f"| {label} | {p} | {serial:.3f} | {tm:.3f} ({min(values):.3f}–{max(values):.3f}) | {speed:.2f}× | {100*speed/p:.1f}% | {med(name,1,False)/tm:.2f}× |")
lines += ["","## Numerical checks and rank limits","",
    "Each pair compares final TTN distance, density, retained ranks, truncation status, local-solve/skip counts and omitted-block h*eta diagnostics. The original snapshot is checked for mutation after all trials. No dense global tensor is built. Allocations are cumulative, not peak memory.","",
    "The [fixture verification](../benchmarks/parallel_large_trees/data/verification.txt) passed 31 checks of tree sizes, coupling normalization, initial states and seed-direction equality. All 120 repeated timing rows have matching serial/threaded ranks and pass the equivalence assertions; source hashes and group counts were checked separately.","",
    "| Case | Max serial/thread distance | Density difference | Pre-step truncation misses / 4 | Measured step meets tolerance | Largest discarded bound | Invariant drift in measured step | Allocated GB (8 workers) |",
    "|---|---:|---:|---:|---|---:|---:|---:|"]
for name,label in LABELS.items():
    rows=sum((groups[name,p,t] for p in WORKERS for t in (False,True)),[])
    mx=lambda k:max(float(r[k]) for r in rows)
    met=all(r['truncation_met']=='true' for r in rows)
    lines.append(f"| {label} | {mx('state_distance'):.2e} | {mx('density_difference'):.2e} | {mx('pretruncation_misses'):.0f} | {'yes' if met else 'no'} | {mx('discarded_bound'):.2e} | {mx('invariant_step_drift'):.2e} | {med(name,8,True,'bytes')/1e9:.2f} |")
lines += ["","The TFD fixture reaches maxrank=64 and misses the requested truncation budget. Its measurements compare identical capped calculations; they are not an accuracy or rank-convergence claim. Norm drift also contains the parallel BUG time-discretization contribution. The HEOM results must still be checked for time and bath-cutoff convergence before physical use.","",
    "## Why speedup saturates","",
    "A separate eight-worker process measures three instrumented steps per mode on an isolated source copy, checked against the ordinary solver. The ordinary timing trials above contain no extra timers. Only local ODEs run concurrently; preparation, assembly and truncation are sequential. Timing changes in those sequential phases reflect variation/cache/GC effects, not new parallelism.","",
    "| Case | Execution | Preparation s | Local ODEs s | Assembly s | Truncation s | Other s |",
    "|---|---|---:|---:|---:|---:|---:|"]
for (name,t),rows in stages.items():
    values=[median(float(r[f]) for r in rows) for f in fields]
    lines.append(f"| {LABELS[name]} | {'threads' if t else 'serial'} | "+" | ".join(f"{v:.3f}" for v in values)+" |")
hs={t:{f:median(float(r[f]) for r in stages['heom64',t]) for f in fields} for t in (False,True)}
remaining=sum(hs[True][f] for f in ('prepare','assemble','finish'))
phase_speed=hs[False]['solve']/hs[True]['solve']
share=100*remaining/sum(hs[True].values())
lines += ["",f"For the 65-leaf HEOM case, the local ODE phase falls from {hs[False]['solve']:.2f} s to {hs[True]['solve']:.2f} s ({phase_speed:.2f}×), while preparation, assembly and truncation retain about {remaining:.2f} s. Those sequential phases occupy about {share:.0f}% of the sum of phase medians with eight workers. Further whole-step acceleration therefore needs work on preparation and assembly as well as the local solves.",""]
lines += ["","![Node cost distribution](../benchmarks/parallel_large_trees/figures/node_work.svg)","",
    "| Case | Local ODE phase speedup | Heaviest active node | Its share of serial local solve time | Sum / heaviest node time |",
    "|---|---:|---:|---:|---:|"]
for name in ("heom64","tfd16"):
    times={n:median(v) for (c,n),v in node_times.items() if c==name}
    node=max(times,key=times.get)
    ratio=median(float(r['solve']) for r in stages[name,False])/median(float(r['solve']) for r in stages[name,True])
    lines.append(f"| {LABELS[name]} | {ratio:.2f}× | {node} | {100*times[node]/sum(times.values()):.1f}% | {sum(times.values())/times[node]:.2f} |")
lines += ["","Node timers include workspace setup and local solve bookkeeping. Their sums are not parallel wall time. Full-leaf skips are excluded from the cost-distribution figure. Sum/heaviest is an optimistic work-imbalance ceiling for the local routine alone, not a whole-step speedup prediction. More leaves supply more independent work, but imbalance, serial phases, allocation/GC and task scheduling remain.","",
    "## Larger TFD pilot","",
    "An additional 33-leaf / 65-node TFD fixture was probed once at 8 workers: approximately 100.1 s serial versus 54.0 s threaded (1.85×), with about 50.6 GB cumulative allocations per call. It uses input rank 64, reaches augmented rank 128, and misses the truncation budget. This single pilot pair has no separate per-mode warmup and is excluded from the repeated scaling table. Its state difference was 5.60e-15. The [raw pilot log](../benchmarks/parallel_large_trees/data/pilot.txt) also records setup costs and the smaller pilot cases.","",
    "## Reproduce","","From the TTNDynamics package root:","","```sh",
    "OPENBLAS_NUM_THREADS=1 julia --project=. benchmarks/parallel_large_trees/verify_fixtures.jl",
    "for n in 1 2 4 8 16; do",
    f"  OPENBLAS_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 julia --threads=$n --project=. benchmarks/parallel_large_trees/run.jl {ENVIRONMENT['repetitions']}",
    "done",
    "OPENBLAS_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1 julia --threads=8 --project=. benchmarks/parallel_large_trees/stages.jl",
    "julia --project=examples benchmarks/parallel_large_trees/trees.jl",
    "python3 benchmarks/parallel_large_trees/plot.py","```","",
    "The [raw data and source/environment hashes](../benchmarks/parallel_large_trees/data/), [fixtures](../benchmarks/parallel_large_trees/fixtures.jl), [runner](../benchmarks/parallel_large_trees/run.jl), and [phase instrumentation](../benchmarks/parallel_large_trees/stages.jl) are retained. To repeat the extra TFD pilot, run `pilot.jl tfd32` with eight workers; it is substantially more expensive.","",
    "For the algorithm and its limits, see [parallel BUG](parallel_bug.md) and G. Ceruti, J. Kusch, C. Lubich and D. Sulz, *A Parallel Basis Update and Galerkin Integrator for Tree Tensor Networks*, [doi:10.1137/24M1714502](https://doi.org/10.1137/24M1714502).",""]
(HERE.parents[1]/"docs"/"parallel_large_trees.md").write_text("\n".join(lines))
