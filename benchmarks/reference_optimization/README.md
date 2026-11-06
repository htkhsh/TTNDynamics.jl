# Reference-guided optimization checks

`bug.jl` compares complete BUG updates on the 10-site split Holstein/Brownian
problem (400 cm⁻¹ hopping, 800 cm⁻¹ bath frequency). It warms each method, then
records three runs with Julia 4 threads / BLAS 1 thread. Default: four 0.05 fs
steps from the localized state. `REFERENCE_OPT_STATE=/path/to/time_3.jls`
instead measures one step from that saved state; use a separate output folder.

Load the selected TTNSolver package **first**, using a package source snapshot
for the baseline rather than redefining methods in a running process. The
initial run builds `data/problem.jls` with the existing Holstein helper. Later
runs read this identical model fixture and need only TTNSolver dependencies.

```sh
export JULIA_DEPOT_PATH=/tmp/ttn-tamen-depot:/tmp/ttndynamics-fixed-depot:/tmp/ttdynamics-julia-depot:$HOME/.julia
export JULIA_LOAD_PATH=$PWD/TTNSolver:$PWD/TTNDynamics:$PWD/TTDynamics/examples/holstein:$PWD/TTSolver:@stdlib
OPENBLAS_NUM_THREADS=1 julia --startup-file=no --compiled-modules=existing --threads=4 \
    TTNDynamics/benchmarks/reference_optimization/bug.jl
```

For baseline measurements select the saved pre-change package first in
`JULIA_LOAD_PATH`, set `REFERENCE_OPT_LABEL=baseline` and run it before the
optimized package. `REFERENCE_OPT_OUTPUT` selects the result directory.
`python3 TTNDynamics/benchmarks/reference_optimization/baseline_package.py`
creates that isolated package from the current sources and the retained reverse
patch; it prints its path. Patch mismatches fail rather than modifying the
working checkout. `data/source_hashes.json` records the compared sources.
Pass `--eager` to reconstruct the intermediate implementation with early
Arnoldi checks only, and measure it with `REFERENCE_OPT_LABEL=eager`.
The default optimized label compares endpoints against `baseline_*.jls` there.
These are same-input implementation comparisons, not new time-step convergence
studies. Do not compare against a baseline built from different model settings.

`ttno.jl` compares global certificates on the existing eight-mode fixture from
`tamen_certification`. `REFERENCE_OPT_RESIDUAL` selects that serialized
`(A,x,b,budget,maxrank)` tuple. Run with `--project=TTNSolver`, 4 threads and
BLAS 1 thread. It records exact structural TTNO and optionally numerically
compressed TTNO alongside SoP; compilation is recorded separately. When the
10-site fixture and PS2 3 fs snapshot exist, it also records operator ranks and
the largest predicted application QR input. That size is not peak memory.

`interval.jl` compares complete tAMEn intervals on the same eight-mode model,
including TTNO construction, convergence checks, endpoint differences and trace.
Run it separately from other benchmarks to avoid CPU contention.

Results: [report](../../docs/reference_optimization.md). Serialized local
fixtures are ignored by Git; tables and parameters are retained under `data/`.
