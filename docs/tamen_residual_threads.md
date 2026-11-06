# Reusing and threading global tAMEn residual construction

These measurements precede the later
[weighted-cut buffer optimization and phase timings](tamen_residual_compression.md).

Global residual additions now reuse node-owned projection and QR buffers.
With `threaded=true`, independent subtrees of each addition also execute
concurrently. The global acceptance bound, term order, weighted truncation
budget and rank cap are unchanged. This continues the
[canonical-addition optimization](tamen_certification.md).

## Usage and ownership

```julia
workspace = TAMEnWorkspace()
result = heom_tamen_evolve(system, initial, times;
    workspace, als_sweep=:etter, threaded=true)
```

The solver reuses scratch across residual terms/checks and sequential intervals.
Passing a workspace also retains it across separate evolution calls. Each node
owns eight growing buffers for projection intermediates, joined QR input,
block reflectors, Q and R. `mul!`, `permutedims!`, LAPACK `geqrt!` and `gemqrt!`
fill active views; small LAPACK work arrays still allocate. Returned tensors
own their arrays and never reference retained scratch.

`workspace_stats.residual_buffer_grows` and `residual_buffer_reuses` count buffer
requests; `residual_tasks` counts spawned addition tasks. They are not counts of
all allocations or ALS tasks. Workspaces cannot be shared by concurrent calls.
Both child tasks join on errors before a lease is released, allowing subsequent
reuse. Replacing a workspace releases its retained high-water buffer storage.

The task budget follows estimated QR work in each subtree. A child receiving
zero after rounding runs on the caller, leaving the budget for its heavier
sibling. This is a relative balance heuristic, without an absolute task-size
cutoff. Operator terms, weighted cuts and the final residual QR are still
sequential; auxiliary residual ALS also remains sequential.

## Measurements, 2026-11-06

Julia 1.12.6, AMD Ryzen Threadripper PRO 9985WX, BLAS 1. Five warmed trials per
configuration; medians below. Kernel order alternates each trial; Julia
processes run sequentially. Compilation, model setup and explicit GC before
each trial are excluded; GC during each call is included. Allocations are
**cumulative allocated bytes, not peak memory**.

The HEOM fixture uses eight bath modes, split two-state system, bath dimension
3, five Chebyshev time unknowns, interval 0.2 and linear tolerance `1e-8`.
Certification uses the identical serialized accepted iterate and residual
rank cap 64. The controlled case has 32 balanced leaves, physical dimension 4,
input rank 8, twelve two-mode operator terms and residual rank cap 24. It uses
seeded random tensors to test contraction scaling, not physical dynamics.
Neither fixture is the ten-site Brownian Holstein production benchmark.

Four-thread process, certificate only:

| Implementation | HEOM (s) | HEOM allocated (MB) | 32 leaves (s) | 32 leaves allocated (MB) |
|---|---:|---:|---:|---:|
| Previous canonical addition | 0.2162 | 722.78 | 0.2438 | 860.24 |
| New kernels; fresh scratch per term, serial | 0.1986 | 544.57 | 0.2253 | 699.25 |
| Reuse, serial | 0.1911 | 395.76 | 0.2064 | 530.24 |
| Fresh scratch per term, threaded | 0.2009 | 544.61 | 0.2133 | 699.26 |
| Reuse and threaded | 0.1892 | 395.81 | 0.1737 | 530.25 |

Combined gains over the previous canonical implementation are **1.14×** for
this HEOM certificate and **1.40×** for the 32-leaf certificate. In this process,
threading on top of reuse gives only 1.01× for the small HEOM case and 1.19× for
32 leaves. Thus the small HEOM case mainly benefits from fewer allocations;
these figures do not imply fourfold thread scaling.

Thread scaling with buffer reuse (one-thread **serial reuse** baseline;
four/eight-thread rows use **reuse and threaded**, BLAS 1 throughout):

| Threads | HEOM (s) | Speedup | Efficiency | 32 leaves (s) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1883 | 1.00 | 100% | 0.2059 | 1.00 | 100% |
| 4 | 0.1892 | 1.00 | 24.9% | 0.1737 | 1.19 | 29.6% |
| 8 | 0.2305 | 0.82 | 10.2% | 0.1918 | 1.07 | 13.4% |

Efficiency is speedup divided by thread count. Eight threads are slower than
four for both fixtures. Within the eight-thread process, serial reuse takes
0.1982 s for HEOM and 0.2169 s for 32 leaves; threading ratios are 0.86× and
1.13× respectively. Threading adds net overhead in the small HEOM case;
scheduling and GC are plausible contributors, but their costs were not isolated.
Julia's default GC thread
configuration is retained across processes; hardware/cache/GC variation also
affects cross-process comparisons. Four threads gives the lowest measured
time for the 32-leaf case; serial reuse is competitive for HEOM. Neither
result establishes a universal optimum.

The retained numeric buffer payload is 4.83 MB for the HEOM certificate and
18.31 MB for 32 leaves (excluding object headers and vector capacity overhead).
All variants agree with the previous residual within `4.7e-24` of RHS norm
for HEOM and `5.2e-15` for the controlled case. Compression/error-bound checks
are performed in every timed trial, outside the timed region.

## Whole HEOM trajectory

Five consecutive intervals to dimensionless time 1, accepted warm starts,
`maxrank=32`, Etter ALS and four threads in both implementations:

| Implementation | Total (s) | Global certificate time (s) | Allocated (GB) |
|---|---:|---:|---:|
| Previous canonical addition | 4.454 | 3.904 | 11.877 |
| Reuse and threaded addition | 3.981 | 3.473 | 6.398 |

The whole trajectory improves by **1.12×**, with **46.1%** fewer allocated
bytes. Trial ranges are 4.436–4.769 s before and 3.941–4.048 s after. Both use
10 solution sweeps and 6 global checks, finish at rank 26, have maximum relative
certified residual `7.17e-9`, and trace error `9.27e-13`. Endpoint relative
distance is `1.12e-15`. The persistent residual scratch payload is 15.69 MB.
The initial state, tolerances, invariant correction and acceptance checks
were not relaxed to obtain the improvement.

Global certification still dominates total cost. Its term loop and weighted
truncation remain serial, limiting gains even when subtree additions scale.

## Validation

Four-thread full suites: TTNSolver **1973/1973**, TTNDynamics **595/595**.
Targeted tests cover shrinking/growing ranks and trees, wide QR, scratch
poisoning/output ownership, serial/threaded agreement, failure joining and
lease reuse, concurrent independent workspaces, restrictive rank caps and
public solver statistics. Independent review probes additionally cover QR
block boundaries, zero/rank-deficient matrices and both-child failures.

## Reproduction

From the shared package directory, run sequentially (using your normal Julia
depot, or the scratch depot used for these measurements):

```bash
export JULIA_DEPOT_PATH=/tmp/ttn-tamen-depot:/home/takahashi/.julia
export OPENBLAS_NUM_THREADS=1
previous_pkg=$(python3 TTNDynamics/benchmarks/tamen_residual_threads/reference_package.py)
TAMEN_THREADS_LABEL=previous julia --threads=4 --project="$previous_pkg" \
  TTNDynamics/benchmarks/tamen_residual_threads/run.jl
julia --threads=4 --project=TTNSolver TTNDynamics/benchmarks/tamen_residual_threads/run.jl
julia --threads=1 --project=TTNSolver TTNDynamics/benchmarks/tamen_residual_threads/run.jl
julia --threads=8 --project=TTNSolver TTNDynamics/benchmarks/tamen_residual_threads/run.jl
python3 TTNDynamics/benchmarks/tamen_residual_threads/summarize.py
```

The isolated previous package changes only the global certificate to the
[pre-buffer canonical fixture](../../TTNSolver/test/fixtures/tamen_canonical_reference.jl).
It keeps current ALS and other solver behavior. `TAMEN_THREADS_SNAPSHOTS`
overrides `/tmp/ttn-residual-thread-states`; `TAMEN_THREADS_REPEATS` overrides
five trials. [Benchmark code](../benchmarks/tamen_residual_threads/run.jl),
[raw results and summary](../benchmarks/tamen_residual_threads/data/summary.csv).
