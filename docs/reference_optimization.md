# Reference-guided TTN optimizations

The TENSO and PyTreeNet snapshots under `TTNSolver/ref` were inspected before
these changes. Benchmarks compare our old and new Julia implementations;
they do not measure Python-versus-Julia performance.

## BUG: unchanged local tolerances

Both augmented and parallel BUG now request `eager=true` from KrylovKit, so
Arnoldi checks convergence before reaching the maximum Krylov dimension.
The requested tolerance remains `1e-12`, with dimension 30 and 100 iterations.

Omitted operators in provably orthonormal subtree/QR-complement frames now
remain exact identities. Buffered leaf/core actions skip those matrix products.
No approximate identity test is used. General public environments and mixed
old/new basis overlaps retain their Gram/overlap matrices. The same conserved
directions, weighted truncation and `system_layout=:split` are retained.

10-site Holstein, Brownian frequency 800 cm⁻¹, hopping 400 cm⁻¹, 20 bath
exponents, local hierarchy size 5. Julia 1.12.6, AMD Threadripper PRO 9985WX,
4 Julia workers, BLAS 1 thread, warm-up then
three samples; times below are medians. The mature-state test uses the **same
saved PS2 state at 3 fs**, maximum rank 64, as input to one 0.05 fs BUG step.
Truncation tolerance is `5e-7`, cap 64, threading enabled, invariant correction
disabled. This is a complete-step implementation comparison, not a new BUG
trajectory from 0 to 3 fs or a fresh TT+tAMEn comparison.

| Method | Before | Eager only | Eager + identities | Overall speedup |
|---|---:|---:|---:|---:|
| Augmented BUG | 31.2517 s | 11.5979 s | 8.3954 s | 3.72× |
| Parallel BUG | 7.6662 s | 4.7970 s | 4.2992 s | 1.78× |

Relative state differences from the baseline are `1.84e-14` and `3.42e-13`;
uncorrected trace drift is below `2.3e-15`. Allocation totals fall from
3.046 to 2.134 GB for augmented BUG and 2.108 to 1.871 GB for parallel BUG.
These are cumulative allocated bytes, not peak memory. Identity skipping
primarily removes arithmetic rather than scratch allocation.

From the localized initial state, four steps to 0.2 fs reach only ranks 12/10.
Augmented BUG improves 0.0463→0.0374 s (1.24×); parallel BUG is essentially
unchanged at 0.0355→0.0351 s. State differences are `1.20e-10`/`2.67e-11`,
well below the step truncation budget; trace errors remain below `8e-15`.
The benefit is therefore strongly rank dependent.

## PS2: remove cancelling root updates

At the symmetric sweep turnaround the root has three consecutive solves with
the same frozen generator, for `-h/2`, `h`, and `-h/2`. They cancel and are now
omitted. The 43-node tree uses 166 local solves instead of 169. A singleton
tree retains its sole full-step solve. The dense and split-HEOM checks verify
the result and invariant protection. This small change does not address all
of PS2's large local problems or absorb full physical leaves into parents.

## TTNO: experimental residual backend

`TreeTensorOperator` merges equal subtree products and optionally compresses
the operator with an explicit Frobenius error bound. `tamen_step(...;
residual_backend=:ttno)` uses a compressed TTNO for global certification while
local/Etter ALS still uses SoP. Its certificate includes
`operator_discarded_bound * norm(current_iterate)` at every check. The default
backend remains `:sop`. See [API and error contract](../../TTNSolver/docs/ttno.md).

On the existing eight-mode certification fixture, the largest operator rank
falls from 43 to 4 after compression. Three-run medians:

| Global certificate | Time | Allocated |
|---|---:|---:|
| Streamed SoP | 0.2348 s | 401.1 MB |
| Structural TTNO without numerical compression | 0.7927 s | 790.8 MB |
| Compressed TTNO | 0.0294 s | 102.0 MB |

The warmed compressed-operator construction costs 0.00177 s separately.
The relative residual bound is `2.80e-9` (SoP) versus `2.87e-9` (TTNO), both
below `1e-8`; the compressed residuals differ by `4.37e-11` relative to the RHS.

For a complete interval of length 0.2 in that eight-mode model, five time
points, `rtol=1e-8`, rank cap 32 and threaded Etter ALS, total time including
operator construction improves **0.2372→0.0886 s (2.68×)**. Both use four sweeps
and one global check. Endpoint difference is below `1e-14`; trace error is
`3.25e-13`. This is not the 10-site physical-fs benchmark above.

The large Holstein case remains a limitation: the 3 fs rank-64 state needs a
predicted QR input of **182 GB** for the structural physical TTNO. For a
0.5 fs/eight-point space-time operator compressed with the `rtol=1e-4` policy,
the maximum operator rank is still 39 and the predicted input is **109 GB**.
These are single-array size estimates, not peak-memory measurements; these
large applications were not executed. Application now rejects predicted QR
inputs above 1 GiB before allocating them. Other projection intermediates and
outputs also consume memory; the guard is not a bound on total memory.

Making TTNO useful for this large state requires compression during application
or a different residual contraction, avoiding an exact operator-rank × state-
rank product before QR. PyTreeNet's symbolic Gaussian elimination and minimum
vertex cover construction are also not implemented here. Small-case speedups
do not establish large-case speedups.

## Verification and reproduction

The benchmark scripts, parameters, all samples and baseline reconstruction
patch are in [benchmarks/reference_optimization](../benchmarks/reference_optimization/README.md).
The baseline uses an isolated package snapshot, without changing live methods.
Existing uncommitted work and the ignored reference sources are preserved.

Numerical review covered complex transpose conventions, nonorthogonal/mixed
frames, ownership, failure paths, compression bounds and operator reuse across
growing iterates. No blocking issues were found. Final checks passed:
TTNSolver 2644/2644, TTNDynamics 595/595, PS2 numerical 43/43, split-HEOM
physical 75/75 and comparison runner 8/8. The physical PS2 error decreases
5.58e-5→1.40e-5→3.49e-6 as the time step halves .2→.1→.05 fs.

Consulted local sources:

- [TENSO sparse propagation](../../TTNSolver/ref/pytenso-main/src/tenso/operator/sparse.py):
  identity mean fields and omission of turnaround updates.
- [TENSO tree layouts](../../TTNSolver/ref/pytenso-main/src/tenso/heom/eom.py):
  physical indices can live directly on numerical cores; its default ket/bra
  arrangement differs from our separate split leaves.
- [PyTreeNet TTNO](../../TTNSolver/ref/PyTreeNet-main/pytreenet/ttno/ttno_class.py)
  and [state diagrams](../../TTNSolver/ref/PyTreeNet-main/pytreenet/ttno/state_diagram.py):
  shared operator channels and compressed operator representation.

The Julia code uses the project's own algebra and QR/SVD routines; source code
from those packages was not copied. Directed message caching and scratch reuse
already existed and are not counted as new optimizations.
