# Rank-adaptive PS2 trial

Experimental, benchmark-local implementation of the symmetric two-site
projector-splitting sweeps in Chen and Franco, JCP163,104109(2025),
https://doi.org/10.1063/5.0278591, Algorithms4–6. Local autonomous linear
equations use non-Hermitian Arnoldi exponential actions. This trial is not a
new default or a public solver API.

Approved scope: verify dense small systems, rank growth, and trace protection;
compare the existing ten-site split Brownian Holstein fixture at1fs, then5fs
within roughly one hour of benchmark computation. Preserve earlier outputs.

Implementation choices: moving orthogonality centre with oriented cached
operator environments; protected source-side spaces in both gauge moves and
SVD cuts; full physical leaf bases; optional extra retained singular vectors
to prevent premature rank locking. These choices need accuracy checks, not
just timing. A sum of local cut tails is not a propagated global error bound.
Half the cut budget is reserved for a final protected rank-cap enforcement
when gauge moves temporarily exceed the cap. There is no trace renormalization.
Physical leaf bases remain full; the internal bond ranks adapt.

Run from the shared package directory (set `HOLSTEIN_DEPOT_PATH` if necessary):

```bash
OPENBLAS_NUM_THREADS=1 julia --threads=4 --project=TTNSolver TTNDynamics/benchmarks/holstein_ps2/verify.jl
HOLSTEIN_JULIA_THREADS=4 bash TTNDynamics/benchmarks/holstein_longtime/run.sh ../holstein_ps2/verify_physical.jl
HOLSTEIN_JULIA_THREADS=4 bash TTNDynamics/benchmarks/holstein_longtime/run.sh ../holstein_ps2/verify_runner.jl
PS2_OUTPUT=my_trial HOLSTEIN_JULIA_THREADS=4 timeout --kill-after=15s 3500s bash TTNDynamics/benchmarks/holstein_longtime/run.sh ../holstein_ps2/batch.jl
PS2_OUTPUT=my_trial python3 TTNDynamics/benchmarks/holstein_ps2/summarize.py
PS2_OUTPUT=my_trial MPLCONFIGDIR=/tmp/ttn-mpl python3 TTNDynamics/benchmarks/holstein_ps2/plot.py
```

Existing case directories are never overwritten. The saved TT references in
`../holstein_ttn_tamen/output/five_fs` must be present with matching model,
source/runtime metadata, wrapper hash and qualified convergence checks. Their
state-file hashes and the trial script hashes are stored immutably. If a process
is interrupted, run `analyze.jl` with the same output label after it has stopped;
it uses only atomic accepted checkpoints and marks interrupted cases explicitly.

The measured calibration uses `krylov_eager=true`: KrylovKit checks convergence
after each basis expansion while keeping the same tolerance and maximum size.
This avoids always constructing30vectors on short local intervals. Both modes
remain available in `ps2_step`; `verify.jl` checks agreement and fewer actions.
The initial non-eager attempt and its source snapshot remain in
`output/holstein5`; the subsequent trial is in `output/holstein5_eager`.

Timing excludes setup, warmup, observations, serialization and reference
comparisons, but includes GC during propagation. TT-only absent diagnostics use
zero/NaN placeholders. Full-state and density errors at integer-fs checkpoints
must both be below1e-3; local cut-budget flags are recorded separately. A
completed1fs calibration is never presented as a completed5fs trajectory.

Progress:
- [x] Implement local trial and numerical tests (40 assertions).
- [x] Check the physical fixture against dense/TT references (75 +8 runner assertions).
- [x] Measure calibrated1fs runs and attempt5fs within the time budget;
  report qualified common-time results (PS2 stopped at3.8fs, verified through3fs).

The Japanese report is in [docs/holstein_ps2.md](../../docs/holstein_ps2.md).
`summary.csv` contains common-time comparisons; `phase_summary.csv` separates
local exponential solves, environment construction and QR/SVD splitting. GC is
included in propagation and phase times, so its column must not be added again.

Measured result: at3fs PS2(h=0.1fs,padding1) took1069.198s versus TT tAMEn's
median1.288s. Both met the1e-3 state/density target; PS2 was about830times slower
in this implementation and fixture. Local exponential actions accounted for
91.4% of PS2 propagation time. TT completed5fs; PS2 did not complete5fs.
