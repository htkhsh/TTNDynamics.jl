# Etter-style parallel ALS for TTN tAMEn

`als_sweep=:etter, threaded=true` runs sibling solution subtrees concurrently.
The option works with fixed/adaptive HEOM intervals, fused/split system leaves,
trace protection and accepted-interval warm starts.

```julia
result = heom_tamen_evolve(system,initial,0:0.2:1.0;
    als_sweep=:etter, threaded=true,
    time_points=8, rtol=1e-8, maxrank=32)
```

The default remains `als_sweep=:serial`, the existing sequential Gauss–Seidel
sweep. `als_sweep=:etter, threaded=false` runs the same new sweep serially and
is the baseline for thread speedup. Changing the sweep algorithm can change
iteration counts; compare accepted residuals and states, not just timings.
This implementation uses shared-memory Julia tasks, not MPI/distributed memory.

## Algorithm and implementation choices

The sweep follows Etter's Section 4.2, Definition 33 and Algorithm 4:

1. Solve the parent local system.
2. SVD the parent toward each child and capture its orthonormal complement.
3. Freeze those operator/RHS messages and solve the siblings independently.
4. QR each updated child toward its parent and rebuild upward messages.
5. Solve the parent again in the updated child bases.

When preparing a sibling's frame, cached upward contractions use only the
unitary `Vt` rotation; the actual child receives `S*Vt`. Mixing these two frames
would change the sibling's local equation. Partial RHS messages retain source
bond indices and use mixed overlaps, including when source and solution ranks differ.

The parent starting iterate is reconstructed by projection of its pre-branch
tensor into the updated child bases. This avoids the inverse-singular-value
connector in the paper, retains null directions without division, and leaves
the final Galerkin equation unchanged. Finite-tolerance GMRES paths can differ
from the published algorithm. The paper's SPD convergence analysis is not a
guarantee for the non-Hermitian HEOM space-time system.

Each task owns disjoint tensor, scratch and counter slots. Parents join both
children even on exceptions; a failed sweep publishes no partially updated
tree. Workspace dictionaries and counters are merged by the caller after joining.
The time leaf stays fixed and does not consume a solution-task budget; all
available workers pass into the spatial tree. The current split uses subtree
node counts rather than a measured per-node cost model.

Residual ALS, rank enrichment, protected truncation and final root invariant
repair remain sequential. The measurements below predate the later
[buffer reuse and threading of global residual additions](tamen_residual_threads.md).
`timings.sweep_seconds`
measures the main solution sweeps and is included in `als_seconds`.
`environment_seconds` overlaps these counters; `als_parallel_tasks` counts
only spawned solution-subtree tasks. Never accept an interval from a local
residual estimate alone; the existing global bound remains mandatory.

## Measurements (2026-11-06)

Julia 1.12.6, AMD Ryzen Threadripper PRO 9985WX (64 cores), BLAS 1, three warmed
trials per configuration. Speedup uses the median serialized Etter time in the
**same process** divided by the threaded median. Efficiency is speedup divided
by Julia thread count. These are shared-memory measurements on this machine.

The synthetic problem has 16 or 64 spatial leaves of dimension 4, maximum rank 24,
five time unknowns, and 33 or 129 space-time product terms. It measures one ALS
sweep including preparation and upward environments, without residual
certification or rank enrichment. It is not a converged propagation benchmark.

| Spatial leaves | Threads | Serialized sweep (s) | Threaded sweep (s) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
|16|1|0.1160|0.1146|1.01×|101%|
|16|2|0.1163|0.0718|1.62×|81%|
|16|4|0.1187|0.0519|2.29×|57%|
|16|8|0.1179|0.0514|2.29×|29%|
|64|1|1.3975|1.4019|1.00×|100%|
|64|2|1.4369|0.8238|1.74×|87%|
|64|4|1.4394|0.4637|3.10×|78%|
|64|8|1.4026|0.3363|4.17×|52%|

The 1-thread entries are identical algorithms with timing variation, not
superlinear speedup. Serial/threaded matvec counts are identical (168 for 16
leaves, 595 for 64); relative state differences are at most 8.13e-15. The larger
tree exposes more independent work. Upper-level solves, frame preparation,
memory allocation/GC and workload imbalance still limit scaling.

The whole HEOM check uses a two-state system with split ket/bra leaves and
eight conjugate-paired bath modes of hierarchy dimension 3, a dimensionless
interval 0.2 and five Chebyshev unknowns. Linear `rtol=1e-8`, `maxrank=32`,
default local residual updates, one cold interval. This is not the physical
10-site Holstein comparison.

| Julia threads | Existing sequential ALS total (s) | Serialized Etter total (s) | Threaded Etter total (s) | Etter thread speedup |
|---:|---:|---:|---:|---:|
|1|0.3363|0.3369|0.3394|0.99×|
|2|0.4357|0.3379|0.4236|0.80×|
|4|0.3696|0.3698|0.3685|1.00×|
|8|0.3684|0.3767|0.3652|1.03×|

There is **no consistent whole-solve speedup in this small HEOM case**. All
paths require 4 sweeps and reach spatial rank 14. Etter uses 1144 matvecs versus 1017
for the existing sequential sweep. The 4-thread Etter solution-sweep time falls
from 0.0472 s to 0.0402 s, but residual work alone takes about 0.318 s of its 0.369 s
total (86%). These small-run timings also vary between trials; see raw data.

Both algorithms pass the same global residual tolerance (maximum reported
relative bound 2.82e-9), with full-state endpoint difference 1.91e-9 and maximum
trace error 4.47e-13. Threading the same Etter algorithm changes states by less
than 1e-14. Matching the linear collocation tolerance is not a separate temporal
error guarantee. This study does not establish an advantage over TT/tAMEn or BUG.
Global residual formation/certification is the next whole-solve optimization
target; local solves now have a measured parallel path.

The subsequent [residual certification optimization](tamen_certification.md)
reduces assembly/copy/QR work and measures about 1.30× on a five-interval run.
The table above records the original Etter measurement before that change.

## Reproduction

From `TTNDynamics`, run separate processes sequentially:

```sh
for threads in 1 2 4 8; do
    OPENBLAS_NUM_THREADS=1 julia --threads="$threads" --startup-file=no --project=. benchmarks/tamen_etter/run.jl
done
python3 benchmarks/tamen_etter/summarize.py
```

[`run.jl`](../benchmarks/tamen_etter/run.jl) warms all code paths, alternates
trial order, leaves GC enabled, and takes three timed trials per setting.
Model setup and explicit collection occur outside the timer. Set
`TAMEN_ETTER_REPEATS` to change the trial count or `TAMEN_ETTER_CASES` to a
comma-separated subset of `heom8,sweep16,sweep64`.

Raw timings, allocation counts, GMRES work, task counts and state comparisons
are in [`data/`](../benchmarks/tamen_etter/data/). The benchmark asserts
threaded/serialized output agreement and equal work for synthetic sweeps;
HEOM assertions include convergence, global residual tolerance and trace error.

## Validation

The tests include complex non-Hermitian fixed points, explicit dense child
Galerkin contractions, unequal source ranks, shrinking bonds, zero/tiny SVD
data, arbitrary root/leaf order, pinned time bases, scratch reuse, child-failure
join/rollback, warm starts, Chebyshev/CN adaptive propagation and independent
invariants. Both fused and split HEOM results are checked against dense exponentials.
Package suites passed 1724/1724 (TTNSolver) and 595/595 (TTNDynamics) with 4 threads.
An independent review also passed 117 assertions, including ill-conditioned
spectra, alternate pinned-leaf positions and dense serial/Etter comparisons.

## Reference

S. Etter, *Parallel ALS Algorithm for Solving Linear Systems in the Hierarchical
Tucker Representation*, SIAM Journal on Scientific Computing 38(4), A2585–A2609
(2016), [doi:10.1137/15M1038852](https://doi.org/10.1137/15M1038852),
[author manuscript](https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2015/2015-25_rev2.pdf).
The parent initialization adaptation and remaining sequential stages are
described above; this is not the paper's distributed-memory implementation or
its full parallel ALS(SD) solver.
