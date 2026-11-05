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

HEOM and TFD use recursive, first-order augmented BUG integration with adaptive
tensor ranks. The standard examples use fixed time steps and compare `h` with
`h/2` to check temporal convergence. The Holstein example selects this workflow
with `method = :fixed`; `method = :adaptive` enables step-doubling control.

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
