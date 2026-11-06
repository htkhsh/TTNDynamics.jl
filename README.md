# TTNDynamics.jl

TTNDynamics implements tree tensor dynamics for scaled HEOM and finite-
temperature thermofield (TFD) models.

HEOM supports a fused system leaf (`system_layout=:fused`, default) or separate
ket/bra leaves (`:split`). See the [layout comparison](docs/heom_layout.md).

Requires Julia 1.11 or later. From the package root, prepare the example
environment and run an example:

```sh
julia --startup-file=no --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=examples examples/heom/bug_dynamics/run.jl
julia --startup-file=no --project=examples examples/holstein_tfd/dynamics.jl
```

## Time-evolution algorithms

HEOM also supports experimental **space-time TTN AMEn** via `heom_tamen_evolve`,
following [Dolgov's space-time formulation](https://arxiv.org/abs/1403.8085).
It supports Chebyshev/CN, adaptive interval lengths, independent output times
and trace protection for fused and split layouts. `als_sweep=:etter, threaded=true`
enables sibling ALS solves following [Etter](https://doi.org/10.1137/15M1038852).
See the [parallel ALS measurements](docs/tamen_etter.md) and
[usage, validation and remaining gaps](docs/tamen.md).

HEOM and TFD support first-order **augmented BUG** [1] and **parallel BUG** [3],
both with adaptive tensor ranks. The solver keyword `method=:parallel` selects
parallel BUG; `:augmented` is the default. `threaded=true` (default) runs sibling
subtrees concurrently for augmented BUG and all-node solves for parallel BUG.
See the scaling measurements for [augmented BUG](docs/augmented_threads.md)
and [parallel BUG](docs/parallel_threads.md), and the
[algorithm comparison](docs/parallel_bug.md).
The [larger-tree study](docs/parallel_large_trees.md) extends parallel BUG
measurements to 65 leaves and 16 workers.
The [TT tAMEn comparison](docs/tamen_comparison.md) measures propagation time
at a common full-state error target.
The [rank study](docs/bug_rank_regimes.md) tests when BUG methods become faster.
The [long-time Holstein comparison](docs/holstein_longtime.md) provides a common
Brownian-bath setup, checkpoints and accuracy checks for TT/tAMEn and both BUG methods.
The [optimization report](docs/parallel_bug_optimization.md) compares compact
assembly with the initial implementation and measures a common error target.

The standard examples use fixed time steps and compare `h` with
`h/2` to check temporal convergence. The Holstein example's configuration
`method = :fixed` / `:adaptive` selects the time-grid workflow; it is separate
from the solver's BUG selection.

Adaptive integration compares one full step with two half steps and adjusts the
step size using a whole-TTN local-error estimate, including truncation and
invariant-correction contributions.

The augmented BUG step follows the recursive rank-adaptive TTN method of
Ceruti, Lubich and Sulz [1]. Adaptive step-doubling is the error-control layer
used here, with general ODE background in [2].

Example visualizations use the separate TTNPlots package; plotting details
and all example commands are in the [example index](examples/README.md).

See the [implementation guide](docs/guide.md) for API conventions, numerical
limits, detailed validation results, related code and literature references.

## References

1. G. Ceruti, C. Lubich and D. Sulz, “Rank-adaptive time integration of tree
   tensor networks,” *SIAM Journal on Numerical Analysis* **61**(1), 194–222
   (2023), [doi:10.1137/22M1473790](https://doi.org/10.1137/22M1473790).
2. Step-doubling is a general local-error estimation technique for ODE solvers;
   see E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
   Equations I: Nonstiff Problems*, 2nd revised ed. (1993),
   [doi:10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1).
3. G. Ceruti, J. Kusch, C. Lubich and D. Sulz, “A Parallel Basis Update and
   Galerkin Integrator for Tree Tensor Networks,” *SIAM Journal on Scientific
   Computing* **48**(1), A27–A48 (2026),
   [doi:10.1137/24M1714502](https://doi.org/10.1137/24M1714502).
