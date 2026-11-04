# TTNDynamics.jl

Physical models for tree tensor dynamics, built on TTNSolver.jl. The first
increment provides a scaled HEOM generator, initial states and trace/root-ADO
observables. TFD is planned as another model layer using the same operator
interface. Time propagation and BUG are not implemented yet.

Requires Julia 1.11 or later. With TTNSolver and TTNDynamics in sibling
directories, run from TTNDynamics:

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.develop(path="../TTNSolver")'
julia --startup-file=no --project=. examples/heom_operator_check.jl
julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'
```

The local Manifest is intentionally untracked. Neither package needs the
existing TTSolver, TTDynamics or HEOMKit as a runtime dependency.

```julia
using TTNDynamics, TTNSolver, LinearAlgebra

H = ComplexF64[0 0.3; 0.3 0]
V = ComplexF64[1 0; 0 -1]
mode = HEOMMode(V, 0.8, 0.7+0.2im, 0.7-0.2im)
system = HEOMTTNSystem(H, [mode], [3])   # occupations 0, 1, 2
state = heom_initial_state(system, ComplexF64[1 0; 0 0])
L = heom_generator(system)
derivative = L * state
@assert abs(dot(trace_observable(system), L, state)) < 1e-12
rho0 = root_density_matrix(state, system)
```

## Conventions

- Use consistent units and hbar=1. H and each coupling V must be Hermitian.
- Physical dimensions are `[M^2, hierarchy_sizes...]`. The system leaf is
  `vec(rho)` in Julia column-major order (ket fastest). Mode 1 of the hierarchy
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

`heom_tree(system)` puts the fused system leaf under the root and balances
the bath subtree. Pass `tree=...` to initial/trace constructors for another
tree; physical positions follow that tree's `leaf_order`. A system without
bath modes is represented by a single leaf.

`heom_initial_state` also accepts finite non-Hermitian or unnormalized
matrices, for later response calculations. `root_density_matrix` contracts
the auxiliary vacuum directly, without expanding the hierarchy.

## Validation and current limits

The example compares a three-mode model with direct dense ADO equations,
including nonzero higher ADOs and complex conjugate rates. The reference code
in `examples/dense_heom_reference.jl` is for small validation problems only.
Tests also cover trace annihilation, scaling similarity, custom trees, the
existing ket/bra convention, size-one cutoffs and a bath-free system.

`L * state` uses exact block sums and can multiply TTN ranks by the number of
terms. `dot(x, L, y)` contracts products directly and avoids that expansion.
The next numerical step is augmented BUG with local contractions and verified
trace preservation; these operator checks do not establish conservation after
time integration or rank truncation.
