# Historical environment-only tAMEn threading

These measurements precede `als_sweep=:etter`; see [the new parallel sweep](tamen_etter.md)
for its implementation and timings. The path measured here parallelizes only
**upward environment contractions**, without the parallel ALS sweep of Etter, *Parallel ALS algorithm
for solving linear systems in the hierarchical Tucker representation* (2016),
[Section 4.2 and Algorithm 4](https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2015/2015-25_rev2.pdf),
[published article](https://doi.org/10.1137/15M1038852).

Etter precomputes the environments for all children, allows these cached values
to remain unchanged while siblings are updated, and uses child orthogonalization
to make concurrent local solves possible. The sweep descends from root to leaves
and returns from leaves to root. This changes the update ordering of serial ALS.
Its ideal balanced-tree scaling is limited by tree depth and workload/rank balance;
the paper's O(d/log d) speedup is not a guarantee for this implementation or HEOM.

Our `_tamen_upward` reads fixed tensors and contracts independent subtrees using
Julia tasks. `_tamen_sweep!` still updates the gauge and solves local GMRES problems
sequentially, using updated environments on traversed edges. Residual assembly,
residual enrichment and truncation are also sequential. Implementing Etter's
sweep requires changes to the algorithm and ownership of basis updates, rather
than adding `@spawn` to the current child loop. Non-Hermitian HEOM requires its
own convergence and invariant checks.

## Measurements, 2026-11-06

These measurements use the original `residual_method=:global`, now explicitly
pinned in the benchmark. The later default-local residual update changes total
solve cost; see the [local/global residual comparison](tamen_local_residual.md).
The scaling table below is a historical baseline, not a measurement of that
new default. The scope of threaded environment construction is unchanged.

AMD Ryzen Threadripper PRO 9985WX, Julia 1.12.6, OpenBLAS with one thread.
Separate Julia processes with 1, 2, 4 and 8 worker threads; default Julia GC
thread configuration retained. Both paths warmed, then five repetitions each
with serial/threaded order alternating. GC is enabled; an explicit collection
before each trial is outside the timer. Timing includes allocation and any GC
during the call. Processes were run sequentially.

The full-solve fixture is a two-state split HEOM with eight bath modes, hierarchy
dimension 3 per mode, localized initial state, dimensionless interval 0.2,
Chebyshev unknown count 5, linear residual tolerance 1e-8 and maxrank 24.
It converges in four sweeps to spatial rank 14 with relative linear residual
2.79e-9. These are controlled test coefficients, not a fitted Brownian bath or
the ten-site Holstein benchmark.

The environment-only fixture has 64 spatial leaves plus one time leaf, physical
dimension 4, maximum spatial rank 32, five time unknowns and 129 space-time SoP
terms. It uses deterministic random orthonormal spatial bases. It measures a
single environment construction, **not an entire integration step**.

Medians; speedup S = one-thread serial time / p-thread time, efficiency E = S/p:

| Workers | HEOM solve (s) | Solve S | Solve E | 64-leaf environment (s) | Environment S | Environment E |
|---:|---:|---:|---:|---:|---:|---:|
| 1, serial baseline | 0.7567 | 1.00 | 100% | 0.2067 | 1.00 | 100% |
| 2 | 0.7494 | 1.01 | 50.5% | 0.2034 | 1.02 | 50.8% |
| 4 | 0.7677 | 0.99 | 24.6% | 0.1056 | 1.96 | 48.9% |
| 8 | 0.7532 | 1.00 | 12.6% | 0.06965 | 2.97 | 37.1% |

The HEOM timings show no convincing whole-solve speedup. In the same eight-thread
process, toggling `threaded` gives 0.7714 s versus 0.7532 s, a 1.024 ratio, but
the trial ranges overlap (0.7367–0.7813 s and 0.7352–0.7750 s). This small change
should not be interpreted as a demonstrated gain. The smaller two-mode fixture
slows from about 11.5 ms to 13.6 ms when task parallelism is enabled on eight threads.

For the eight-mode solve, the serial residual construction takes 92–93% of total
time, and the currently parallelized environments only 0.65–0.69%. Eliminating
that environment cost entirely would therefore improve this case by less than
1%. The solve allocates about 2.36 GB cumulatively per call (not peak storage).
Replacing repeated global residual sums/compressions with local residual updates
is the main candidate to reduce total work before scaling the ALS solves.

There is also an avoidable scheduling limitation: the new root has a large spatial
child and a very cheap time-leaf child. The current budget assigns at least one
worker to each. With two workers the spatial subtree receives only one worker;
with four/eight it receives three/seven. The measured two-worker contraction
ratio is therefore mostly timing variation, not useful parallelism in the spatial
tree. A work-aware budget is a separate improvement from implementing Etter ALS.

Both full-solve paths have identical sweep/matvec counts and endpoint differences
below 1e-14. The environment outputs agree exactly for the synthetic fixtures.
No solver code was changed for these measurements.

## Reproduction

From `TTNDynamics`, run each command sequentially:

```bash
OPENBLAS_NUM_THREADS=1 julia --threads=1 --project=. benchmarks/tamen_threads/run.jl
OPENBLAS_NUM_THREADS=1 julia --threads=2 --project=. benchmarks/tamen_threads/run.jl
OPENBLAS_NUM_THREADS=1 julia --threads=4 --project=. benchmarks/tamen_threads/run.jl
OPENBLAS_NUM_THREADS=1 julia --threads=8 --project=. benchmarks/tamen_threads/run.jl
python3 benchmarks/tamen_threads/summarize.py
```

[Benchmark code](../benchmarks/tamen_threads/run.jl),
[raw data and summaries](../benchmarks/tamen_threads/data/summary.csv).
The summary retains both the common one-thread baseline and paired serial/threaded
ratios within each process, along with trial ranges and GC times. The script also
measures a 16-leaf contraction fixture and the smaller two-mode HEOM solve.
