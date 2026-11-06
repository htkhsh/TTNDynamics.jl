# Cross-interval tAMEn warm starts

Fixed and adaptive `heom_tamen_evolve` now default to `warm_start=true`.
The most recent accepted space-time solution and auxiliary residual supply
the next interval's initial guess. Disable with `warm_start=false` to compare
against cold interval initialization. Residual reuse within a single interval
is independent of this option.

```julia
fixed = heom_tamen_evolve(system,initial,0:0.2:1;
    warm_start=true,time_points=5,rtol=1e-8,maxrank=32)
adaptive = heom_tamen_evolve(system,initial;tspan=(0.,1.),dt=0.2,
    warm_start=true,time_points=5,solve_rtol=1e-8,maxrank=32)
```

The predictor retains the previous spatial bases. It interpolates the old curve
inside its normalized interval, replays its increments scaled by the new/old
duration ratio, and anchors them to the new initial-state projection. No
polynomial is evaluated beyond the old interval. Different time-point counts
or Chebyshev/CN schemes can reuse solution bases, while the auxiliary residual
starts fresh. Every interval rebuilds its operator/RHS environments and uses
the supplied initial state for its RHS.

An owned `TAMEnWarmStart` hint copies all tensor arrays and time nodes. A hint
is a guess, never an acceptance certificate. Tree/dimension mismatch or initial
state/previous-endpoint disagreement triggers cold fallback. The endpoint test
uses 128 machine epsilons relative to the larger state norm. Nonfinite temporal
scaling/prediction also falls back. Final global residual verification and HEOM
trace protection remain mandatory.

In the adaptive driver, a hint becomes the next initializer only after both
linear and temporal acceptance. Rejected trials retain the previous accepted
state and hint. The driver keeps only one hint locally; it is not cached in the
workspace or retained in diagnostic history, even with `store_solutions=true`.
One call cannot silently initialize a different trajectory through a reused
workspace. Separate workspaces permit concurrent readers of the same hint.

Direct `tamen_step` callers can explicitly retain and pass hints:

```julia
y, first_info = tamen_step(x,L,0.2;store_warm_start=true)
z, next_info = tamen_step(y,L,0.1;
    warm_start=first_info.warm_start,store_warm_start=true)
```

Direct storage requires linear convergence; callers implementing their own
temporal controller must additionally require temporal acceptance. Diagnostics
report `warm_start_used`, `residual_warm_start_used` and `warm_start_status`.
`timings.warm_start_seconds` includes hint transfer and storage. Treat hint
fields and shared tree topology as read-only.

## Reproduction

The benchmark uses the same two-state split HEOM with eight bath modes and
hierarchy dimension three as the [local residual comparison](tamen_local_residual.md),
now for five fixed intervals of length 0.2 to dimensionless time 1.0. Both modes
use local residual updates, Chebyshev time_points=5, linear rtol=1e-8,
maxrank=32, kickrank=3 and threaded=false. These are controlled test coefficients,
not the ten-site Brownian Holstein model.

From `TTNDynamics`:

```bash
OPENBLAS_NUM_THREADS=1 julia --threads=4 --project=. benchmarks/tamen_warm_start/run.jl
```

The script warms both paths, alternates five measured trials of each, leaves GC
enabled and collects before each timer. Model construction and compilation are
excluded; the entire fixed-interval wrapper call is timed. See
[code](../benchmarks/tamen_warm_start/run.jl),
[raw runs](../benchmarks/tamen_warm_start/data/runs.csv),
[summary](../benchmarks/tamen_warm_start/data/summary.csv), and
[per-interval diagnostics](../benchmarks/tamen_warm_start/data/intervals_warm.csv).

## Measurements, 2026-11-06

Ryzen Threadripper PRO 9985WX, Julia 1.12.6, four Julia threads and BLAS=1.
Medians from five warmed trials:

| Five-interval total | Cold starts | Warm starts |
|---|---:|---:|
| Wall time | 9.236 s | 5.506 s |
| Cumulative allocation | 25.84 GB | 17.34 GB |
| Solution sweeps | 18 | 10 |
| Global residual checks | 7 | 6 |
| ALS time | 1.571 s | 0.445 s |
| Residual processing | 7.634 s | 4.998 s |
| Warm transfer/storage | 0 | 0.0239 s |
| Maximum endpoint rank | 32 | 26 |

Warm starts are **1.68× faster** in this fixture, with **32.9% less cumulative
allocation** (not peak memory). Relative endpoint difference is 7.52e-9.
Every interval in both runs meets the same 1e-8 linear residual tolerance and
the benchmark checks trace errors below 1e-10. This is a warm/cold initialization
comparison, not parallel speedup or a TT-versus-TTN performance comparison.

Per-interval sweep counts are cold `4,3,3,4,4` versus warm `4,2,1,2,1`.
Only the first interval initializes cold; all subsequent warm intervals reuse
both the space-time solution and auxiliary residual. Global verification remains
the dominant cost. Warm starts reduce repeated basis construction and ALS work,
but do not remove the required global certificate.

## Validation and scope

Tests cover duration/order/scheme changes, changed operators, incompatible
initial states/trees, exact linear-in-time prediction, ownership and concurrent
reads, failed solves, rejection rollback, compact history, dense fused/split
HEOM evolution, trace preservation and opt-out equivalence. Iteration/rank counts
and the adaptive mesh may change with the guess; accepted tolerances do not.

The existing TT implementation `TTSolver/src/tamen.jl` retains solution and
auxiliary residual cores across its intervals. This implementation follows that
reuse concept with explicit ownership and acceptance-controlled storage. The
predictor is a TTN implementation choice, not an assertion of identical TT
iteration ordering. The subsequent [Etter-style sweep](tamen_etter.md) supports
these same accepted-interval hints; the timings above use `als_sweep=:serial`.
