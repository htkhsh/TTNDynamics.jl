# HEOM with space-time TTN AMEn (experimental)

`heom_tamen_evolve` provides a tAMEn-type alternative to stepwise BUG: all
Chebyshev collocation samples within an interval are solved together in a TTN.
The construction follows the space-time approach of
[Dolgov, arXiv:1403.8085](https://arxiv.org/abs/1403.8085), with recursive tree
Galerkin ALS and residual enrichment between sweeps. Fixed and adaptive intervals
are available, with optional Etter-style parallel ALS. A speed advantage over
TT/tAMEn or BUG has not been demonstrated.

```julia
result = heom_tamen_evolve(system, initial, 0:0.2:1.0;
    time_points=8, rtol=1e-8, maxrank=32, maxsweeps=12)
```

Set `als_sweep=:etter, threaded=true` to solve sibling subtrees concurrently,
following [Etter's parallel ALS](https://doi.org/10.1137/15M1038852).
`threaded=false` serializes the same sweep for comparison. The default
`als_sweep=:serial` retains sequential local solves; its `threaded=true` only
parallelizes upward contractions. This selection also works with adaptive
intervals and warm starts. Global residual acceptance and trace protection
apply in either mode. See [implementation details and scaling](tamen_etter.md).

Times must use the inverse units of the supplied HEOM generator, exactly as
for the other HEOM solvers. Each adjacent pair defines one fixed interval.
`rtol` and `atol` control the collocation linear-system residual; they do not
control temporal discretization error. Increase `time_points` from 8 to 16,
or halve interval lengths, to check time convergence separately.

Both `system_layout=:fused` and `:split` work. The trace functional is inserted
and protected, and a final root solve enforces its collocation equations at
all time nodes. No trace correction or Euclidean normalization is applied.
Split leaves require enough rank for the trace directions (`maxrank ≥ M`).
The time leaf retains its full basis and is exempt from the spatial rank cap.

The return value includes endpoint `densities`, `traces`, `ranks`,
`final_state`, and interval `diagnostics`. Each diagnostic contains
`converged`, `relative_residual_bound`, `residual_compression_bound`, sweep
`history`, GMRES work counts, `invariant_values` at every time node, and
`timings` split into ALS, residual formation and enrichment. Set
`store_solutions=true` to retain each space-time TTN and access its samples:

```julia
result = heom_tamen_evolve(system, initial, [0.0,0.2];store_solutions=true)
sol = first(result.diagnostics).solution
rho_at_node = root_density_matrix(time_slice(sol,4),system)
time_at_node = sol.nodes[4]  # relative to the interval start
```

Use `using TTNSolver` when calling `time_slice` directly. Without
`store_solutions`, history retains small diagnostics rather than full TTNs.
A failed GMRES solve or an interval failing the global residual tolerance
throws `TAMEnConvergenceError`; propagation does not advance with that iterate.
Raise rank/sweep limits or shorten intervals after examining the failure.

## Small validation example

From the package root:

```sh
julia --startup-file=no --project=. examples/heom/tamen_dynamics/run.jl
```

The example uses a two-state system, split ket/bra leaves, and a conjugate
pair of underdamped modes with explicit test coefficients. Units are
dimensionless; this is not the 10-site physical Holstein benchmark.
It compares 4 and 8 time points per interval against an independently
assembled dense HEOM exponential. CSVs in `examples/heom/tamen_dynamics/data/`
record populations, full-state errors, trace drift, residual bounds, sweeps
and ranks at every collocation point. `summary.csv` includes propagation times
after warm-up. Dense expansion is confined to this small reference example.

Baseline validation before cross-interval warm starts, on 2026-11-06
(five intervals of length 0.2, linear residual
`rtol=1e-10`, no trace correction):

| Time points per interval | Maximum full-state relative error | Maximum trace error | Maximum spatial rank |
|---|---:|---:|---:|
| 4 | 2.63e-7 | 1.78e-15 | 9 |
| 8 | 8.07e-12 | 3.11e-15 | 9 |

Errors include every collocation point. The equal linear residual tolerances
and different state errors illustrate the separate time-discretization error.
This tiny example is a correctness check, not a speed comparison with TT/tAMEn.

## Adaptive intervals and independent observation times

```julia
result = heom_tamen_evolve(system, initial;
    tspan=(0.0,1.0), dt=0.5, saveat=0:0.05:1,
    time_scheme=:cheb, time_points=8,
    atol=1e-8, rtol=1e-6, solve_rtol=1e-10,
    time_error_damp=100, maxrank=32, threaded=true)
```

Here `atol/rtol` control the temporal estimate. `solve_atol/solve_rtol` separately
control the linear-system residual. The reduced-space time estimate evaluates
an integrated defect on a finer temporal grid and includes initial projection
error; it does not run a second spatial ALS propagation. It is a local estimate,
so tolerance refinement is still needed for global accuracy.

Intervals are accepted only after both linear convergence and temporal checks.
Rejected attempts retry from the last accepted state. `dtmin`, `dtmax`,
`maxsteps` and `maxreject` limit retries; `TAMEnIntegrationError` retains the
accepted state and attempt history if a limit is hit. `time_error_damp=100`
follows the existing TT tAMEn default; zero disables only temporal rejection.

`time_scheme=:cn` or `"CN"` selects Crank–Nicolson. In that scheme,
`time_points` is odd and at least 3, including the initial sample. Chebyshev
counts unknown samples, excluding the initial point. Both retain the full time
basis. The positional `heom_tamen_evolve(system,initial,times;...)` API keeps its
fixed-interval behavior and its linear-residual meaning for `atol/rtol`.

`saveat` does not change the internal interval grid. Chebyshev output uses
barycentric interpolation, CN output uses piecewise linear interpolation, and
the initial output is exactly the supplied initial state. Return fields include
`accepted_times`, `accepted` and `rejected`. Every attempt diagnostic has
`.solve` for its solver report; local-GMRES failures instead record the failing
node/residual. With `store_solutions=true`, accepted and globally unconverged
space-time iterates are accessible in `.solve.solution`.

```sh
julia --threads=4 --startup-file=no --project=. examples/heom/tamen_dynamics/adaptive.jl
```

This split HEOM example compares Chebyshev and CN with a dense exponential and
writes `adaptive_*_samples.csv` and `adaptive_*_attempts.csv`. The attempt files
include acceptance, interval size, temporal error ratio, linear residual and
environment contraction time. Its test bath and time units remain dimensionless.

For that example on 2026-11-06 (`rtol=1e-5`, damping 10, five time points),
Chebyshev used 5 accepted / 1 rejected intervals with final full-state error
3.81e-9; CN used 110 / 2 with error 2.63e-7. Maximum saved trace errors were
6.66e-15 and 7.51e-14. These are convergence checks on a tiny test model, not
timing evidence for a large Holstein system.

## Cross-interval warm starts

Both fixed and adaptive `heom_tamen_evolve` default to `warm_start=true`.
The previous accepted space-time solution and auxiliary residual initialize
the next interval. Set `warm_start=false` for comparison. The initial RHS,
trace protection, all operator environments and global acceptance check are
rebuilt for each solve. A changed time grid/scheme reuses solution bases but
resets the residual seed. Rejected adaptive trials cannot replace the accepted
hint; compact diagnostics never retain hints. Only one accepted hint is held
locally, independently of the reusable workspace.

See [warm-start usage and multi-interval timings](tamen_warm_start.md).

## Parallel sweeps and remaining work

`als_sweep=:etter, threaded=true` parallelizes independent sibling local solves
as well as upward environments. Children receive frozen orthonormal complements,
own their tensor/scratch slots, and join before the parent solves again. A fixed
time leaf consumes no solution task budget. The default `als_sweep=:serial`
retains sequential Gauss–Seidel local updates. The library leaves BLAS thread
settings to the caller; use BLAS=1 when measuring task parallelism.

`workspace=TAMEnWorkspace()` reuses local buffers and temporal grids across
intervals. Use separate workspaces for concurrent runs. Reusing a workspace does
not reuse old environments after a basis/operator change. The adaptive run holds
an exclusive lease, released on failure. `workspace_stats` reports reuse, and
`timings.environment_seconds` is a subset of `timings.als_seconds`.

See [Etter sweep measurements](tamen_etter.md) for current whole-solve and
single-sweep scaling, and [the earlier environment-only measurements](tamen_parallelism.md).
`timings.sweep_seconds` is a subset of `als_seconds`; `als_parallel_tasks`
counts spawned solution tasks. Auxiliary residual updates remain sequential.
Global certification now combines term addition and QR, avoids zero-filled block
cores, and defers redundant post-cut orthogonalization. See the
[residual verification optimization](tamen_certification.md) for matched checks
and whole-trajectory measurements; the acceptance bound is unchanged.
Global residual additions now also reuse node-owned projection/QR buffers and
thread independent subtrees with `threaded=true`; weighted cuts and the term
loop stay sequential. `workspace_stats` adds `residual_buffer_grows`,
`residual_buffer_reuses` and `residual_tasks`. See the
[reuse/threading comparison](tamen_residual_threads.md).
The subsequent [weighted residual compression optimization](tamen_residual_compression.md)
adds reusable cut buffers and per-phase global-verification timings while
preserving the same acceptance bound.
The [ten-site Holstein comparison](holstein_ttn_tamen.md) uses physical fs units,
refined TT references and a shared state/density accuracy target; incomplete
TTN trials are compared only over their accepted common-time prefix.

General multiple-invariant support is available in the tensor solver; the HEOM
wrapper always owns its trace functional and density observer, rejecting keyword
overrides. There is still no Euclidean norm correction, which would be unsuitable
as a default for HEOM decay. Auxiliary local residual sweeps are now available;
see [local residual updates and timings](tamen_local_residual.md). Time-dependent
operators/RHS, residual updates interleaved with individual solution-node updates,
preconditioning and node-interleaved enrichment remain gaps from
full TT tAMEn functionality. Large-system speed comparisons should use matched
full-state accuracy after these improvements.
