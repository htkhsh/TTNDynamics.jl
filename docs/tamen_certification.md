# Optimizing global tAMEn residual verification

These measurements cover the first canonical-addition optimization. The later
[buffer reuse and subtree threading](tamen_residual_threads.md) comparison uses
this implementation as its baseline.

Global residual verification now uses exact canonical term addition and avoids
repeated QR/copies. This is the default for both sequential and Etter ALS;
no new solver option is needed. The stopping test remains
`residual_bound <= max(atol,rtol*norm(b))`.

## What changed

Profiling the eight-mode split HEOM case found that a final global check
allocated about 1.08 GB and spent about 95% of its time in compression.
The old path built a full acted copy of the solution, added it using zero-filled
block cores, copied and orthogonalized that sum in `truncate!`, then
orthogonalized again after cutting ranks. This repeated for every product term.

The new path applies local operators at the leaves while forming the sum.
At an internal node, the two input cores are projected separately using their
child QR transfers. Their parent columns are concatenated and QR-factorized;
no zero-filled three-index direct-sum core is built. The result is an exact
root-orthogonal sum, within roundoff, with independently owned storage.

Weighted truncation then consumes that canonical tensor directly, eliminating
its initial copy/QR. The next canonical addition accepts noncanonical inputs,
so post-cut QR can be deferred. One final QR before taking the root norm
restores orthogonality. Public `truncate!` still validates on a private copy
and returns a root-orthogonal state; the deferred mode is private to verification.

The operator term order, per-term budget, rank cap and weighted cut algorithm
are unchanged. If a compressed partial residual is `r̂` and accumulated discarded
bound is `δ`, the final bound is `norm(r̂)+δ`. Exact addition/QR does not add
approximation error; each new cut adds its bound by the triangle inequality.
As before, this excludes floating-point roundoff. The implementation does not
subtract squared norms such as `||b||²+||Ax||²−2real⟨b,Ax⟩`, which can lose
a small residual through cancellation. A restrictive rank cap still prevents
acceptance when the discarded bound is too large.

## What was consulted in TTSolver

[`TTSolver/src/amen_solve.jl`](../../TTSolver/src/amen_solve.jl), especially
`amenany_sweep`, constructs residuals from projected RHS/action contractions
and reuses local matvec/environment storage. [`TTSolver/src/tamen.jl`](../../TTSolver/src/tamen.jl)
retains `ZAX`/`ZY` across sweeps. Its stopping rule uses the maximum reported
local residual or update size, depending on `trunc_norm`.
These supplied the useful principle of keeping calculations in local compressed
coordinates and avoiding expanded intermediates.

That local stopping measure was **not substituted for the TTN global bound**:
the existing auxiliary residual remains an estimate, and every accepted
interval still requires a global certificate. No TTSolver code or behavior was
changed. This optimization follows the TTN's existing weighted truncation and
does not introduce a new time integrator or change temporal error control.

## Measurements (2026-11-06)

Julia 1.12.6 on AMD Ryzen Threadripper PRO 9985WX, four Julia threads,
one BLAS thread; medians of five warmed trials. The model is the same
eight-mode, two-state split HEOM case as the [Etter study](tamen_etter.md),
with hierarchy dimension 3, interval length 0.2, five Chebyshev unknowns,
linear `rtol=1e-8`, `maxrank=32`, and threaded Etter ALS. Time is dimensionless.

| Measurement | Previous (s) | Optimized (s) | Speedup | Previous allocations (GB) | Optimized allocations (GB) |
|---|---:|---:|---:|---:|---:|
| One global check, identical accepted iterate | 0.3177 | 0.2451 | 1.30× | 1.081 | 0.723 |
| Full first interval, cold start | 0.3808 | 0.3039 | 1.25× | 1.152 | 0.793 |
| Five intervals to time 1, accepted warm starts | 6.8009 | 5.2468 | 1.30× | 17.264 | 11.877 |

Allocations are cumulative bytes allocated during each timed call, **not peak
resident memory**. The standalone check reduces allocations by 33%; the
five-interval run reduces them by 31%. GC time in the five-interval run falls
from 1.516 s to 1.041 s. The reference and optimized paths run in separate
processes on the same machine; ordinary timing variation remains, so raw trials
are retained rather than treating the last decimal as significant.

The standalone check uses the exact same serialized space-time iterate and
`residual_maxrank=64`; both returned residuals have maximum rank 29. Their
TTN difference, divided by the RHS norm, is 2.76e-16. The full first interval
uses four sweeps and one global check in both versions. Across five intervals,
both use ten sweeps and six global checks, reaching spatial rank 26. Endpoint
relative difference is 2.23e-15; maximum trace error is 9.27e-13. Every interval
passes the same global residual tolerance. No check was removed to obtain these
timings, and time discretization was identical.

Residual work still takes about 4.67 s of the optimized 5.25 s trajectory (89%).
This removes avoidable assembly work but does not eliminate the bottleneck.
Further targets are the remaining weighted QR/SVD cuts and allocation reuse;
any alternative norm evaluation must still resolve small residuals and account
for truncation error.

## Reproduction

From `/home/takahashi/.julia/dev` (or the corresponding checkout root):

```sh
reference_project=$(python3 TTNDynamics/benchmarks/tamen_certification/reference_package.py)
OPENBLAS_NUM_THREADS=1 TAMEN_CERT_LABEL=reference julia --threads=4 --startup-file=no --project="$reference_project" TTNDynamics/benchmarks/tamen_certification/run.jl
OPENBLAS_NUM_THREADS=1 TAMEN_CERT_LABEL=optimized julia --threads=4 --startup-file=no --project=TTNSolver TTNDynamics/benchmarks/tamen_certification/run.jl
python3 TTNDynamics/benchmarks/tamen_certification/summarize.py
```

The helper creates an isolated temporary package with the old residual routine
from the [reference fixture](../../TTNSolver/test/fixtures/tamen_residual_reference.jl).
It does not modify either checkout or monkey-patch a running solver. Package
dependencies must already be instantiated. `run.jl` loads that solver before the
HEOM wrapper, and records its source path to identify the backend actually used.

The reference run stores a space-time fixture and endpoint tensors under
`/tmp/ttn-residual-comparison-states`; the optimized run reads these. Override
`TAMEN_CERT_SNAPSHOTS` for concurrent independent benchmark runs. Both processes
use identical input arrays for the standalone check. Set `TAMEN_CERT_REPEATS`
to change the default five trials. Compilation, warm-up, setup and explicit GC
are outside the timer; GC remains enabled. Raw rows and medians are in
[`benchmarks/tamen_certification/data/`](../benchmarks/tamen_certification/data/).

## Validation and limits

Tests cover dense complex residuals, arbitrary topology/leaf order, unequal and
overcomplete ranks, tiny residuals, zero terms, single-leaf trees, independent
storage, deferred-gauge addition and conservative bounds when rank caps force
truncation. Existing tests also exercise protected/fixed-leaf compression,
nonconvergence, fused/split HEOM, warm starts, adaptive intervals and both ALS
sweeps. Package suites passed 1828/1828 (TTNSolver) and 595/595 (TTNDynamics).
Independent review passed 5184 assertions, including exhaustive output/input
alias checks and overcomplete complex gauges with scale ratios up to 1e24.
Those checks do not turn the roundoff-excluded bound into a rigorous floating-
point certificate for arbitrarily ill-conditioned inputs.

This remains a sequential, termwise global check. Product-term count and
verification ranks still affect cost. The measurements concern a small
dimensionless HEOM problem, not the physical 10-site Holstein calculation or
a new comparison with TT/tAMEn.
