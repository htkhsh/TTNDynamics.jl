# TTNDynamics.jl implementation guide

This guide contains the API conventions, numerical details and validation
notes behind the package overview. Run commands from the `TTNDynamics` package
root. The [example index](../examples/README.md) lists runnable demonstrations
and their generated outputs.

This package provides a scaled HEOM generator, initial states, trace/root-ADO
observables, and a thermofield (TFD) model layer with augmented or parallel BUG evolution.
Fixed time steps are the standard workflow; adaptive step-size control is
optional. Tensor ranks remain adaptive in both. The examples include
independent dense comparisons.

Requires Julia 1.11 or later. Run commands from the TTNDynamics package root:

See [the example index](../examples/README.md) for the folder layout, run commands
and locations of generated data, figures and tree snapshots.

```sh
julia --startup-file=no --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=examples examples/heom/operator_check/run.jl
julia --startup-file=no --project=examples examples/heom/bug_dynamics/run.jl
python3 examples/heom/bug_dynamics/plot.py
julia --startup-file=no --project=examples examples/heom/adaptive_dynamics/run.jl
python3 examples/heom/adaptive_dynamics/plot.py
julia --startup-file=no --project=examples examples/holstein_tfd/dynamics.jl
python3 examples/holstein_tfd/plot_dynamics.py
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

The package and example manifests are intentionally untracked. Neither package
needs the existing TTSolver, TTDynamics or HEOMKit as a runtime dependency.

The adaptive HEOM and Holstein examples use the separate `examples` environment
to load the sibling **TTNPlots** package. Its core writes SVG/text with no
runtime dependencies; its optional extension reads topology and ranks.
Visualization adds no runtime dependency to TTNDynamics.
The examples write initial and final tree diagrams, with physical leaf labels,
local dimensions and actual bond ranks, as `.svg` and `.txt`:
`examples/heom/adaptive_dynamics/trees/tree_{initial,loose,tight}.*` and
`examples/holstein_tfd/trees/tree_{initial,coarse,fine}.*` for fixed steps
(`tree_{initial,loose,tight}.*` in adaptive mode).
The TFD initial tree is the supplied product state before basis enrichment.

```julia
using TTNDynamics, LinearAlgebra

H = ComplexF64[0 0.3; 0.3 0]
V = ComplexF64[1 0; 0 -1]
mode = HEOMMode(V, 0.8, 0.7+0.2im, 0.7-0.2im)
system = HEOMTTNSystem(H, [mode], [3])   # occupations 0, 1, 2
state = heom_initial_state(system, ComplexF64[1 0; 0 0])
L = heom_generator(system)
derivative = L * state
@assert abs(dot(trace_observable(system), L, state)) < 1e-12
rho0 = root_density_matrix(state, system)
trajectory = heom_evolve(system, state, 0:0.01:1.0;
                         atol=1e-10, maxrank=32)
rho_final = last(trajectory.densities)
```

## Conventions

- Use consistent units and hbar=1. H and each coupling V must be Hermitian.
- Physical dimensions are `[M^2, hierarchy_sizes...]` for `system_layout=:fused`
  (default), or `[M, M, hierarchy_sizes...]` for `:split`. The system indices
  follow Julia column-major order (ket fastest). Mode 1 of the hierarchy
  is the fastest auxiliary index. This differs from the existing TTSolver
  ket/bra/bath Kronecker order; an explicit permutation is covered by tests.
- Hierarchy sizes count occupations `0:size-1`, with independent box cutoffs.
  There is no total-tier cutoff, terminator or residual correction.
- `rate` describes `exp(-rate*t)`, with nonnegative real part. `forward` and
  `backward` are explicit independent coefficients. For conjugate pole pairs,
  the backward coefficient typically comes from the conjugate partner; it
  must not be inferred by conjugating each forward coefficient separately.
- The stored ADO is the unscaled ADO divided by
  `prod(scale[k]^n[k] * sqrt(n[k]!))`. The default scale is
  `sqrt(max(abs(forward), abs(backward), eps(Float64)))`. Pass an explicit
  positive scale to match another representation (e.g. sqrt(abs_coeff) from
  NoiseExp). No physical state is automatically normalized.
- The raising/lowering connections are
  `-im*scale*sqrt(n+1)*(V*rho_plus-rho_plus*V)` and
  `-im*sqrt(n)/scale*(forward*V*rho_minus-backward*rho_minus*V)`.

`heom_tree(system; bath_grouping=:coupling)` first groups bath modes with exactly
equal coupling matrices, then balances those groups. In Holstein this keeps
each site's Brownian components together. Physical axis ordering is preserved,
even for interleaved modes. See the [weighted truncation / tree comparison](weighted_tree.md).

`heom_tree(system)` puts the system branch under the root and balances the bath
subtree. The branch is one fused leaf or two sibling ket/bra leaves. Pass
`tree=...` to initial/trace constructors for another tree; physical positions
follow that tree's `leaf_order`. Without bath modes the tree is the system
branch alone.

```julia
system = HEOMTTNSystem(H, modes, hierarchy_sizes; system_layout=:split)
state = heom_initial_state(system, rho)
result = heom_evolve(system, state, 0:0.01:1; atol=1e-10)
```

The layout is a model property, used consistently by generators, initial
states, trace functionals and both evolution APIs. Split generators act with
H/V on ket and transpose(H/V) on bra, avoiding M²-by-M² local factors. Initial
ket/bra bases have size M and their correlation is routed through their least
common ancestor, supporting arbitrary trees and finite non-Hermitian matrices.
No full hierarchy expansion is needed.

Trace protection retains the entire subtree space of `vec(I)`. In split form
the ket and bra leaf ranks therefore remain M, requiring `maxrank >= M`, even
with `correct_trace=false`. This limits system-leaf compression; splitting
does not guarantee a speedup. The number of product terms grows from 1+3K to
2+5K for K modes. See the [measured comparison](heom_layout.md) for timing,
operator storage, rank and accuracy results.

`heom_initial_state` also accepts finite non-Hermitian or unnormalized
matrices, for later response calculations. `root_density_matrix` contracts
the auxiliary vacuum directly, without expanding the hierarchy.

## Validation and current limits

The examples compare a three-mode model with direct dense ADO equations,
including nonzero higher ADOs and complex conjugate rates. The reference code
in `examples/heom/common/dense_heom_reference.jl` is for small validation problems only.
Tests also cover trace annihilation, scaling similarity, custom trees, the
existing ket/bra convention, size-one cutoffs and a bath-free system.

Running `examples/heom/operator_check/run.jl` also writes
`examples/heom/operator_check/figures/heom_operator_check.svg`, a vector diagram
of the TTN, local HEOM modes, edge ranks and dense-reference errors, and
`examples/heom/operator_check/figures/heom_dynamics.svg`,
which plots the two system populations and zero-tier trace drift over time.
The trajectory uses direct dense ADO equations with classical RK4 as a small
reference; it is not TTN time propagation. Both plots use only Julia's standard
library, so plotting dependencies stay out of both packages.

`L * state` uses exact block sums and can multiply TTN ranks by the number of
terms. `dot(x, L, y)` contracts products directly and avoids that expansion.
`heom_evolve(system, state, times; ...)` uses local contractions throughout.
Each interval between adjacent times is one augmented BUG step; the first
time labels the supplied initial state. The result contains zero-tier
`densities`, `traces`, node-by-sample `ranks`, per-step `diagnostics`, and
`final_state`. Only observables and the final TTN are retained.
This is a first-order recursive augmented BUG step [1].

Both BUG methods use amplitude-weighted truncation by default. Pass
`truncation=:local` to restore conservative local-core compression. This option
works with fixed and adaptive `heom_evolve` and `tfd_evolve`. The default HEOM
tree remains `bath_grouping=:balanced`; weighting changes the compression,
not the topology. See the [comparison](weighted_tree.md).
The [allocation optimization and second-order trials](bug_improvements.md)
report measured speed, memory allocation and accuracy on the same Holstein
model. The second-order methods are available in the benchmark module.
The subsequent [5 fs comparison against tAMEn](tamen_recomparison.md) includes
full-state errors and measured rank growth over a longer interval.

The trace functional is included in the bases and protected during SVD
truncation. Set `correct_trace=false` to check conservation without the
optional root correction. Tests cover conservation with active rank truncation
and no correction, unnormalized initial states, and bath-free dynamics.
`atol` controls the absolute truncation budget per step, not time error;
reduce the time step to assess the latter. Check
`info.truncation.tolerance_met` for every step when imposing `maxrank`.

## TTN dynamics comparison

`examples/heom/bug_dynamics/run.jl` evolves a 3-mode model with box sizes `[3,3,3]`
as a TTN and writes data under `examples/heom/bug_dynamics/data/`. The dense
reference matrix is assembled independently from the direct ADO equations
and exponentiated only inside this validation example. The plotting script
uses NumPy/Matplotlib (optional, no Julia runtime plotting dependency) and
writes figures under `examples/heom/bug_dynamics/figures/` with populations,
density error, trace drift before/after correction, and maximum bond rank.

The current example on Julia 1.12.6 gives maximum density errors of about
`1.23e-3` at `h=0.02` and `6.09e-4` at `h=0.01`, consistent with first-order
time convergence; the fine run's final full-hierarchy relative error is
`7.11e-4`. Maximum bond rank is 9 and trace drift is below `3e-15`.
These are small-system checks, not a performance benchmark or a guarantee of
physical positivity. The implementation uses constant model coefficients
and does not yet compile TTNOs.

## Adaptive time steps and independent output times

Both fixed and adaptive APIs accept `method=:augmented` (default) or
`method=:parallel`. Parallel BUG solves all old-basis node equations before
block assembly; `threaded=true` enables concurrent node solves when Julia has
multiple worker threads. `threaded=false` runs the same formula serially.
Augmented BUG also honors `threaded`: sibling subtrees update concurrently,
while a parent waits for both children. Preparation and final truncation stay
serial. See the [larger-system scaling measurements](augmented_threads.md).
The [parallel BUG scaling study](parallel_threads.md) measures the same larger
fixtures at 1/2/4/8 workers and separates local ODE time from sequential phases.
The [larger-tree study](parallel_large_trees.md) adds up to 65 physical leaves
and 16 workers, using a prepared state for each measured step.
The [TT tAMEn comparison](tamen_comparison.md) instead measures full trajectories
at matched accuracy against an independently converged reference.
The [rank study](bug_rank_regimes.md) adds developed HEOM states and compares
both BUG methods with tAMEn at a common error target.
The [long-time Holstein driver](holstein_longtime.md) uses a configurable common
model and observation grid, with validated resume and reference-refinement checks.
See the [parallel BUG comparison](parallel_bug.md) for usage, measured timing,
accuracy, and the distinction between step doubling and rank-expansion retries.

```julia
trajectory = heom_evolve(system, state;
    tspan=(0.0, 1.0), dt=0.05, saveat=0:0.1:1,
    atol=1e-8, rtol=1e-5, maxrank=32)
println(trajectory.stats)  # accepted / rejected / total attempts
```

This keyword form opts into step-doubling control. `dt` is an initial proposal;
`saveat` specifies observation times, with endpoints included. The integrator
takes additional internal steps as needed, landing on saved times without
interpolation. The existing positional-times API remains fixed-step.
The controller compares one full BUG step with two half steps and accepts the
two-half-step result. This package-specific controller builds on the first-
order augmented BUG step [1] and the general ODE step-doubling approach [2].

In this adaptive API, atol/rtol control the estimated **whole-TTN local error**.
The truncation budget is derived from these with `truncation_fraction=0.1`;
this differs deliberately from fixed-step atol, which is a truncation budget.
All three trial substeps must meet their truncation budgets. Failed local
Arnoldi solves, rank-cap tolerance misses, and excessive estimated errors cause
rejection/retry. No rejected trial changes the accepted state. A cap that is
too small can require increasing maxrank, rather than ever smaller time steps.

`diagnostics` includes rejected attempts, while `densities`, `traces`, and
`ranks` contain only requested samples. `dtmin`, `dtmax`, `maxsteps`, and
`maxreject` bound the work; `AdaptiveIntegrationError` exposes the stopping
reason, last accepted time/state, and attempt history. `correct_trace=false`
disables the optional root correction while retaining protected trace bases.

`examples/heom/adaptive_dynamics/run.jl` compares rtol=1e-3 and 1e-5 with the
independent dense ADO equations. It writes sample and attempt CSVs; the plotting
script produces figures in `examples/heom/adaptive_dynamics/figures/`. The
current 3-mode example runs to t=10 with saveat=0:0.2:10; the dense reference uses the same
sampling interval. Maximum density error decreases from about `3.88e-4` to
`4.37e-6`. There are 51 saved samples in each run, versus 71 and 87 accepted
internal steps (4 and 12 rejections). Trace drift without root correction stays
below `2.1e-14`. These are measured small-system checks; local error tolerances
do not guarantee the same global error for other models.

## Holstein–TFD dynamics

```julia
system = holstein_tfd_system([0.0,0.0], 0.3, [0.7,0.7], [0.2,0.2];
                             beta=2.0, boson_size=3)
initial = tfd_initial_state(system, ComplexF64[1 0;0 0])
result = tfd_evolve(system, initial, 0:0.01:1;
                    atol=1e-10, maxrank=32)
```

Each adjacent pair of times is one augmented BUG step and one output interval;
the first time labels the supplied initial state. Equal spacing gives a fixed
step size. This API accepts strictly increasing finite times, including
nonuniform grids, and performs no rejection or time-step adaptation. Here
`atol` is the absolute truncation budget per step; check time convergence by
reducing the spacing separately. Inspect
`diagnostics[k].truncation.tolerance_met` when imposing `maxrank`. The result
has the same observables, ranks and final state as the adaptive API, with one
diagnostic per interval and no adaptive `stats`. With one time, it returns an
independent initial snapshot without basis enrichment or propagation.

To enable time-step adaptation, use the keyword form:

```julia
result = tfd_evolve(system, initial; tspan=(0,1), dt=0.1,
                    saveat=0:0.05:1, atol=1e-9, rtol=1e-5)
```

The Holstein example defaults to `method = :fixed` and compares `fixed_dt_fs`
with half that step, using `fixed_atol` as the truncation budget. Its fixed
runs save every step. Set `method = :adaptive` (or run `dynamics.jl adaptive`)
to compare the configured relative tolerances using `adaptive_atol`. In that
mode, `initial_dt_fs` is a proposal and `save_interval_fs` sets the output grid.

For either adaptive model API, `record_timing=true` adds stage timings in
`result.timings` and per-attempt `diagnostics[k].timings`. These measure the
error-estimation BUG step, accepted/rejected half-step updates, TTN difference
construction plus QR, candidate norm and observations. See the
[measurement report](adaptive_timing.md) for reproducible measurements.
Shared preparation for the full step and first half step reduces repeated QR
and operator contractions. Local contraction workspaces reduce temporary
allocations in both fixed and adaptive stepping; see the
[before/after comparison](bug_preparation.md).

The builder has one physical oscillator per site, nearest-neighbor hopping
`-hopping`, and optional periodic closure (the two-site bond is counted once).
Use consistent energy/time units with hbar=1. Each positive physical frequency
becomes two adjacent transformed modes, `+omega` and `-omega`, with amplitudes
`g*sqrt(nbar+1)` and `g*sqrt(nbar)`, where `nbar=1/expm1(beta*omega)`.
The coupling convention is `g*V*(a+a†)/sqrt(2)`. `beta=Inf` is zero temperature.
The thermal mapping follows [de Vega and Bañuls (2015)](https://doi.org/10.1103/PhysRevA.92.052116).

For modes that are already thermally transformed (including signed ID data),
use `TFDMode(V, signed_frequency, amplitude)` and
`TFDTTNSystem(H, modes, boson_sizes)` directly. Amplitudes are not squared
weights; this constructor neither doubles the modes nor applies another
temperature factor. Physical-pair occupations require explicit `thermal_pairs`
metadata; generic unpaired models still return transformed-mode occupations.

The electronic leaf stores `vec(sqrt(rho))` with the physical electron fastest
and an inactive electronic ancilla. H and V act as `kron(I_ancilla,H/V)`.
The other leaves start in vacuum. `tfd_density_matrix` traces the baths and
electronic ancilla by contraction. Positive semidefinite unnormalized input
is accepted; no state normalization or norm correction occurs during evolution.
The initial state is factorized; a bare electronic Gibbs state does not imply
an interacting Gibbs state of the full model.

`tfd_observables` and `tfd_evolve` return the electronic density, wavefunction
norm, energy and occupations. Expectations are raw `<psi|O|psi>`, so divide by
the squared norm yourself only when normalized expectations are intended.
Physical oscillator occupation uses the inverse Bogoliubov transform:
`(nbar+1)Nplus + nbar*(Nminus+I) + sqrt(nbar*(nbar+1))*(a†b†+ab)`.
Normal ordering precedes Fock truncation; multiplying already truncated
creation/annihilation matrices would incorrectly remove the upper boundary term.

`tfd_evolve` initially enriches all subtree bases with `H*initial`, keeping the
state unchanged. This bootstraps coupling directions that can be invisible to
rank-one BUG equations. Set `enrich_initial=false` for a prepared initial basis.
The one-time exact SoP application can temporarily multiply bond ranks by the
number of operator terms. Initial enrichment does not detect every subsequent
missing direction; local step-doubling estimates still require independent
time/rank convergence checks. The initial basis enrichment is enabled by
default and can be disabled with `enrich_initial=false`.

Edit [examples/holstein_tfd/config.jl](../examples/holstein_tfd/config.jl) to set
energies in cm^-1, time in fs and temperature in K, using the same unit and
coupling conventions as the older TTDynamics example. The defaults are two
sites at 300 K, two physical oscillators (four transformed modes), Fock size 3,
and 0–50 fs. Initial electronics may be localized or bare Gibbs. The script
writes observation/step CSVs for two step sizes (or adaptive tolerances), and
the Python script writes
`dynamics.svg` / `.png`: populations, physical/transformed occupations, errors,
norm/energy drift, bond ranks and internal steps.

This increment uses discrete undamped oscillators, without Brownian ID fitting
or counterterms. Dense comparisons use the same finite transformed basis
(dimension 324 by default), not the infinite oscillator model. Increase the
Fock cutoff separately to check physical convergence. Dense reference work
is restricted to the example/tests and guarded at dimension 2048; package
evolution and measurements use TTN contractions.

With fixed steps on Julia 1.12.6, reducing the step from 1 fs to 0.5 fs lowers
the maximum electronic density error from `4.28e-6` to `1.06e-6`. The fine
run's maximum physical occupation error is `1.48e-6`, final full-state error
is `1.67e-5`, and norm drift is below `7.9e-14`; maximum rank is 9.

In the adaptive validation runs, maximum electronic density
error is `4.41e-6` at rtol=1e-3 and `1.07e-6` at rtol=1e-5. The tight run's
maximum physical occupation error is `1.48e-6`, final full-state error is
`1.67e-5`, and norm drift is below `7.5e-14`. Both runs take 50 accepted steps
at the 1 fs output interval and reach maximum rank 9. These figures validate
the finite-basis implementation, not Fock-cutoff convergence.

## Related code and literature

- [TTDynamics.jl](https://github.com/takahashi/TTDynamics.jl) is the related
  tensor-train dynamics code; its HEOM examples provide historical context for
  the model and unit conventions.
- [HEOMKit.jl](https://github.com/htkhsh/HEOMKit.jl) provides related HEOM
  operators and hierarchy utilities. TTNDynamics accepts explicit mode data
  and does not depend on HEOMKit at runtime.
- [QFiND.jl](https://github.com/DOC-Package/QFiND.jl) provides bath
  discretization methods used by related workflows. TTNDynamics does not run
  Brownian ID fitting; its TFD builder constructs discrete undamped modes.
- [1] G. Ceruti, C. Lubich and D. Sulz, “Rank-adaptive time integration of
  tree tensor networks,” *SIAM Journal on Numerical Analysis* **61**(1),
  194–222 (2023), [doi:10.1137/22M1473790](https://doi.org/10.1137/22M1473790).
- [2] E. Hairer, S. P. Nørsett and G. Wanner, *Solving Ordinary Differential
  Equations I: Nonstiff Problems*, 2nd revised ed. (1993),
  [doi:10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1),
  for the general local-error and adaptive step-size context.
- The finite-temperature thermofield mapping follows de Vega and Bañuls,
  [Phys. Rev. A 92, 052116 (2015)](https://doi.org/10.1103/PhysRevA.92.052116).
