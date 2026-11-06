# Local residual updates for TTN tAMEn

`heom_tamen_evolve` now defaults to `residual_method=:local`. The auxiliary TTN
`z ≈ b-Ax` is updated by fixed-rank ALS using mixed environment contractions.
It supplies basis-enrichment directions without assembling and compressing a
global residual on every solution sweep. The auxiliary TTN is reused between
sweeps within an interval. [Cross-interval warm starts](tamen_warm_start.md) were
added subsequently; the single-interval comparison below is unaffected.

```julia
result = heom_tamen_evolve(system,initial;tspan=(0.,1.),dt=0.2,
    time_points=5,solve_rtol=1e-8,maxrank=24,
    residual_method=:local,residual_sweeps=1,residual_check_every=4)
```

The old path remains available as `residual_method=:global`. Both modes preserve
the same global stopping criterion, and the HEOM wrapper still protects the trace.
Temporal tolerance/interval control is independent of the linear residual test.

The local projected norm is **not** a certified residual. A small estimate asks
for a global check; checks also run every fourth sweep by default, and on the
last allowed sweep. An interval is accepted only after the existing global
residual-plus-compression bound meets its tolerance. Failed checks supply a
compressed residual to refresh the auxiliary basis. No squared-norm cancellation
formula or trace renormalization is used.

`history` entries distinguish `residual_estimate` from the verified
`residual_bound`. Unchecked entries set bound fields to `Inf` and
`global_residual_checked=false`. Final bounds always describe the returned
iterate. Diagnostics include `global_checks` and `residual_updates`.
`timings.residual_seconds` equals `residual_local_seconds + residual_global_seconds`.

## Eight-mode comparison, 2026-11-06

The fixture matches the eight-mode example in the [threading measurement](tamen_parallelism.md):
two system states, split ket/bra, eight bath modes of dimension three, localized
initial state, dimensionless duration 0.2, five Chebyshev unknowns, linear
relative tolerance 1e-8, maxrank 24 and kickrank 3. These are controlled test
coefficients, not the ten-site Brownian Holstein benchmark.

Julia 1.12.6 on Ryzen Threadripper PRO 9985WX; four Julia threads, BLAS=1,
`threaded=false` to isolate the residual algorithm. Both modes are warmed;
five trials alternate execution order, with GC enabled and explicit collection
before timing. Setup and compilation are excluded. Table entries are medians.

| Measurement | Global residual each sweep | Local updates + global verification |
|---|---:|---:|
| Whole interval | 0.7824 s | 0.3978 s |
| Residual processing total | 0.7265 s | 0.3471 s |
| Local residual updates | 0 | 0.0100 s |
| Global verification | 0.7265 s | 0.3371 s |
| Cumulative allocated bytes | 2.363 GB | 1.160 GB |
| Solution sweeps | 4 | 4 |
| Global checks | 4 | 1 |
| Maximum endpoint rank | 14 | 14 |
| Verified relative residual | 2.790e-9 | 2.820e-9 |

The whole solve is **1.97× faster**, with **50.9% fewer allocated bytes**
(allocation totals, not peak resident memory). Relative endpoint difference is
2.32e-9. Maximum trace drift across time nodes is 1.78e-15 in local mode.
The local relative estimates are 3.23e-2, 2.32e-5, 2.31e-7 and 1.22e-9;
only the fourth iterate is globally checked, and its bound is 2.82e-9.

The remaining global check takes approximately 85% of the new total. One check
at the final higher ranks is more expensive than the earlier low-rank checks,
so reducing four checks to one does not produce a fourfold whole-solve speedup.
This result measures the residual implementation, not parallel efficiency or
performance relative to TT tAMEn. Subsequent changes added
[Etter parallel ALS](tamen_etter.md) and [cross-interval warm starts](tamen_warm_start.md);
the measurements here use sequential ALS and one cold interval.

## Reproduction and validation

From `TTNDynamics`:

```bash
OPENBLAS_NUM_THREADS=1 julia --threads=4 --project=. benchmarks/tamen_local_residual/run.jl
```

[Benchmark code](../benchmarks/tamen_local_residual/run.jl),
[raw runs](../benchmarks/tamen_local_residual/data/runs.csv),
[summary](../benchmarks/tamen_local_residual/data/summary.csv),
[local sweep history](../benchmarks/tamen_local_residual/data/history_local.csv).

Tests cover complex mixed contractions against dense residuals, arbitrary tree
ordering, auxiliary rank limits, independent RNG/state ownership, zero data,
an auxiliary basis that misses a nonzero residual, mandatory failed/final
checks, Chebyshev/CN agreement with the global path, dense HEOM fused/split
propagation, multiple invariants, adaptive timing and workspace concurrency.

## References

- Dolgov, [A tensor decomposition algorithm for large ODEs with conservation laws](https://arxiv.org/abs/1403.8085): time-discretized systems and residual enrichment.
- Etter, [Parallel ALS algorithm for solving linear systems in the hierarchical Tucker representation, Section 4.4](https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2015/2015-25_rev2.pdf): low-rank residual approximation by solving `I z = b-Ax` with ALS. Our auxiliary sweeps are sequential, and enrichment is a basis union between solution sweeps. This is not the full published parallel ALS(SD) or node-interleaved TT AMEn algorithm.
