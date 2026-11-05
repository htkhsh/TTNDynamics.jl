# Holstein–TFD increment

The approved next step is a small Holstein–TFD model using TTNSolver's existing
adaptive augmented BUG solver. Keep TTSolver/TTDynamics independent and unchanged.

1. Add explicit signed-frequency `TFDMode`/`TFDTTNSystem`, plus a Holstein
   builder (one physical oscillator per site). Use hbar=1 and `x=(a+a†)/sqrt(2)`.
   The builder maps `(omega,g,beta)` to `(omega,g*sqrt(n+1))` and
   `(-omega,g*sqrt(n))`, where `n=1/expm1(beta*omega)`. Explicit signed input
   is already transformed and must not be doubled or temperature weighted again.
2. Fuse physical electron and inactive electronic ancilla in the first leaf:
   `vec(sqrt(rho))`, physical electron fastest. Transform the baths to vacuum.
   SumOfProducts represents `I_anc ⊗ H_e + sum(omega*N + g*V*x)`.
3. Contract reduced electronic density and occupation/energy observables locally.
   Physical oscillator number uses the inverse Bogoliubov transform, including
   normal ordering before projection into the finite basis (`Nminus+I`,
   not multiplication of already truncated ladders). Do not normalize states.
4. Wrap adaptive_evolve, reusing all existing time/rank controls without a norm
   correction. Test against an independently assembled dense Hamiltonian,
   including complex data, arbitrary tree order, zero temperature and no bath.
   Review found a centered-coupling product state with invisible BUG directions;
   add TTNSolver.enrich_basis and seed the initial basis with H*initial without
   changing its amplitudes. Document its one-time temporary rank cost and the
   absence of a general subsequent basis-defect estimator.
5. Add a two-site example, configuration, dense validation data, and plots of
   populations, occupations, errors, norm/energy drift, ranks and time steps.
   Run the package tests and example, then review the change.

Limits: this increment uses discrete undamped oscillators, not the previous
Brownian ID fit. Dense comparisons test the same finite transformed basis;
cutoff convergence is separate from time/rank convergence. Initial electronic
thermal states, if supplied, are bare electronic states, not interacting Gibbs
states. The positive/negative bath representation follows the thermofield
construction described in de Vega & Bañuls, Phys. Rev. A 92, 052116 (2015),
https://doi.org/10.1103/PhysRevA.92.052116 .
