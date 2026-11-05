# BUG preparation and workspace comparison

Compare the current solver with a pre-change `TTNSolver/src` snapshot. The
baseline is loaded as a separate module; both solvers use the same model layer.
Two warmups per version precede five alternating pairs per case. Final states,
saved observables, ranks and adaptive acceptance histories must agree.

From the TTNDynamics package root:

```sh
OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=examples benchmarks/bug_preparation/run.jl /path/to/baseline/src 5
python3 benchmarks/bug_preparation/plot.py
```

The baseline must expose the same solver API, including optional stage timing.
Keep its complete `src` directory before changing the solver; it is not bundled
with this benchmark. This run used `/tmp/ttn-bug-before-20261105/src`.

The benchmark records wall time, allocated bytes, GC time, adaptive phase times
and numerical differences in `data/runs.csv`. Environment metadata and source
hashes are in `data/environment.txt`. No tests or other numerical jobs should
run concurrently. See [the report](../../docs/bug_preparation.md).
