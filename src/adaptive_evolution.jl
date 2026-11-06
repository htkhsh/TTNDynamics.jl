"""
    heom_evolve(system, initial; tspan, dt, saveat=nothing,
                atol=1e-6, rtol=1e-4, correct_trace=true, kwargs...)

Adaptively evolve an HEOM TTN. `dt` is the initial internal step; `saveat` is
an increasing collection of observation times, with both tspan endpoints
included. The internal mesh may contain many more steps than saved samples.
The positional form `heom_evolve(system, initial, times; ...)` retains its
fixed-step behavior and its original meaning of atol as a truncation budget.
Select `method=:augmented` (default) or `:parallel`; `threaded` controls
concurrent sibling subtrees for augmented BUG and all-node solves for parallel BUG.

Here atol/rtol control a whole-TTN local step-doubling estimate. Other controls
(truncation=:weighted, truncation_fraction, maxrank, dtmin/dtmax, maxsteps/maxreject, Krylov options)
are passed to TTNSolver.adaptive_evolve. No global error guarantee is implied.
Use `truncation=:local` for the conservative local-core SVD policy.
The trace functional is protected at every step; correct_trace disables/enables
the optional correction, not the protected basis. Limits raise
AdaptiveIntegrationError containing the last accepted TTN and attempt history.

Return times, zero-tier densities, traces, node-by-sample ranks, final_state,
stats, and diagnostics for all attempts (including rejected ones). Saves only
observables and the final TTN. No full hierarchy or matrix is expanded.
`record_timing=true` adds driver totals in `timings` and per-attempt timings
in `diagnostics`; otherwise these fields are `nothing`.
"""
function heom_evolve(system::HEOMTTNSystem, initial::TTNTensor;
                     tspan, dt, saveat=nothing, atol=1e-6, rtol=1e-4,
                     correct_trace::Bool=true, kwargs...)
    leafdims(initial) == heom_dimensions(system) ||
        throw(DimensionMismatch("initial TTN dimensions differ from HEOM model"))
    generator = heom_generator(system)
    conserved = trace_observable(system; tree=initial.tree)
    result = adaptive_evolve(initial, generator, tspan; dt, saveat, atol, rtol,
        conserved, correct_invariant=correct_trace,
        observe=state -> root_density_matrix(state, system), kwargs...)
    return (; result.times, densities=result.values, traces=tr.(result.values),
            result.ranks, result.diagnostics, result.stats, result.final_state, result.timings)
end
