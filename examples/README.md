# Examples

Examples are grouped by model and task. Generated data and figures stay next
to the script that creates them, in `data/`, `figures/` and `trees/` folders.

From the TTNDynamics package root, prepare the shared example environment once:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

Run a Julia example with `--project=examples`, then run its `plot.py` if it has
one. NumPy and Matplotlib are only needed for the optional Python plots.

| Example | Julia entry point | Plot |
|---|---|---|
| HEOM operator and dense check | `heom/operator_check/run.jl` | Writes two SVGs directly |
| Fixed-step augmented BUG | `heom/bug_dynamics/run.jl` | `heom/bug_dynamics/plot.py` |
| Adaptive augmented BUG | `heom/adaptive_dynamics/run.jl` | `heom/adaptive_dynamics/plot.py` |
| Two-site Holstein–TFD (fixed by default) | `holstein_tfd/dynamics.jl` | `holstein_tfd/plot_dynamics.py` |

For example:

```sh
julia --project=examples examples/heom/bug_dynamics/run.jl
python3 examples/heom/bug_dynamics/plot.py
```

The standard workflow uses fixed time steps and compares `h` with `h/2`.
The Holstein configuration selects `method = :fixed` by default, using
`fixed_dt_fs` and `fixed_atol` (the truncation budget). Select `:adaptive` to
use `initial_dt_fs`, `save_interval_fs`, `adaptive_atol` and `tolerances`.
The method can also be overridden for one run:

```sh
julia --project=examples examples/holstein_tfd/dynamics.jl adaptive
python3 examples/holstein_tfd/plot_dynamics.py
```

Fixed runs save observables at every integration step; a final partial step
lands on `tmax_fs` when needed. The plotting script reads `data/runs.csv` to
select the latest method and its labels. HEOM keeps separate fixed and adaptive
demonstrations in the directories listed above.

For repeated timing measurements and stage-by-stage figures, see
[the adaptive timing benchmark](../benchmarks/adaptive_timing/README.md).

The `common/` directory contains model-tree labels used by the examples.
`heom/common/` holds the small dense HEOM reference implementation shared by
the tests and HEOM demos. `holstein_tfd/dense_reference.jl` holds its separate
finite-basis TFD reference. Dense code is only for small validation systems.

Each dynamics example keeps its configuration and entry point with the model;
generated CSVs, figures and tree snapshots are reproducible output. Rerunning
an example replaces the corresponding outputs.
