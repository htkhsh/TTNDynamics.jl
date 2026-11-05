# Adaptive timing benchmark

This benchmark separates the full BUG step for error estimation, accepted
and rejected half-step updates, TTN difference construction plus QR, candidate
norm, observations and remaining driver work. Timing is enabled explicitly
with `record_timing=true`; the normal solver path reads no clocks.

From the TTNDynamics package root:

```sh
OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=examples benchmarks/adaptive_timing/run.jl 5
python3 benchmarks/adaptive_timing/plot.py
```

Each HEOM/TFD case is warmed up with full timed and untimed runs. Five repeated
pairs alternate their execution order. Numerical trajectories and acceptance
histories must match before results are recorded. GC remains enabled, and
reference construction, plotting and other tests are outside timed calls.

`data/runs.csv` contains whole-run totals and `data/attempts.csv` every trial.
`data/environment.txt` records Julia, CPU and BLAS settings. The plot uses the
actual repetition at the median driver duration so its parts sum to its total.
See [the report](../../docs/adaptive_timing.md) for results and limitations.
