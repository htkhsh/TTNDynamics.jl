# Holstein: TT / TTN tAMEn

Ten-site split HEOM, localized initial state, Brownian bath frequency800cm⁻¹,
hopping400cm⁻¹,5fs. Uses the same bath expansion and ADO scaling in both formats.

From the shared package directory:

```bash
HOLSTEIN_JULIA_THREADS=4 bash TTNDynamics/benchmarks/holstein_longtime/run.sh ../holstein_ttn_tamen/verify.jl
HOLSTEIN_JULIA_THREADS=4 timeout --kill-after=15s 3600s bash TTNDynamics/benchmarks/holstein_longtime/run.sh ../holstein_ttn_tamen/batch.jl
```

`batch.jl` runs references, TT with BLAS8/1, then TTN with BLAS1 sequentially.
Existing result directories are never overwritten. `common.jl` exposes explicit
`root`, `cfg` and `settings` keywords for subsequent calibration runs.
Override `HOLSTEIN_DEPOT_PATH` if the existing benchmark depot is unavailable.

Two additional calibration entrypoints preserve the linear tolerance:
`calibrate.jl` tests a1fs interval with6 temporal points, while
`calibrate_blas8.jl` keeps the original0.5fs/8-point setting and uses BLAS8.
Their results are separate from the original trial. For a deliberately stopped
trial with an accepted checkpoint, run `recover.jl NAME [NAME...]` before analysis. It
retains the original step log and marks the trial `calibration_stopped`; it does
not count an unfinished interval as accepted propagation time. Its recorded
`wall_seconds` is the last status update, not the total cost of interrupted work;
`wall_seconds_is_complete=false` makes that limitation explicit.

```bash
HOLSTEIN_JULIA_THREADS=4 bash TTNDynamics/benchmarks/holstein_longtime/run.sh ../holstein_ttn_tamen/run.jl analyze
python3 TTNDynamics/benchmarks/holstein_ttn_tamen/summarize.py
MPLCONFIGDIR=/tmp/ttn-mpl python3 TTNDynamics/benchmarks/holstein_ttn_tamen/plot.py
```

Every accepted interval is checkpointed; integer-fs states support independent
accuracy checks. `status.toml` records initialization failures and time limits;
`run_status.csv` lists all attempted runs, including those with no accepted
interval. `comparison.csv` contains only common-time comparisons and marks the
1e-3 state/density target explicitly. Full5fs speed comparisons require completed
runs and a refined-reference difference below1e-4 throughout the trajectory.
TT-specific diagnostics unavailable through the existing wrapper use zero/NaN
placeholders; these do not mean the TT algorithm performed no internal checks.
`summary.csv` reports common-time medians and accuracy-qualified timing ratios;
`phase_summary.csv` records the TTN verification breakdown. GC time overlaps the
measured phases and must not be added to their sum.

See [results and limitations](../../docs/holstein_ttn_tamen.md).

`batch_tt_acceptance.jl` re-runs references, TT and the TT-style TTN cases
(`tree_tt*`: `truncation_rtol`, `certify=false`, `local_tol_factor`, and
`local_operator=:channels` for `tree_tt_channels*`) into `output/tt_acceptance`.
Set `HOLSTEIN_TTN_TAMEN_OUTPUT` to choose another output root; `summarize.py`
and `plot.py` take the same root name as their first argument. Do not edit the
hashed sources while a batch runs: every case records source hashes, and
`analyze()` rejects a root whose cases disagree. Uncertified TTN cases are
validated only against the refined reference. Results:
[holstein_tt_acceptance.md](../../docs/holstein_tt_acceptance.md).
