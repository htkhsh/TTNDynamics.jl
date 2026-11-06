"""
    tfd_evolve(system, initial; tspan, dt, saveat=nothing,
               atol=1e-6, rtol=1e-4, enrich_initial=true, kwargs...)

Adaptive BUG for the thermofield Schrodinger equation. Select `method=:augmented`
(default) or `:parallel`; `threaded` enables concurrency for either method. `dt` is the
initial step proposal; saveat specifies observation times. atol/rtol control
the whole-TTN local step-doubling estimate, not a global error bound. Additional
rank, truncation and Krylov options are passed to TTNSolver.adaptive_evolve.
Compression defaults to `truncation=:weighted`; `:local` selects the
conservative local-core SVD policy.
No linear invariant or norm correction is applied. Monitor norm/energy drift
and check time/rank and Fock-cutoff convergence independently.

By default, enrich the initial subtree bases with `H*initial` while preserving
the initial state. This seeds coupling directions invisible to a product
state's reduced equations. The one-time exact SoP application can temporarily
multiply ranks by the number of terms. Set enrich_initial=false for an already
prepared basis. This bootstrap does not detect all later invisible directions;
step doubling alone is not a guarantee against basis stagnation.

Return times, densities, norms, energies, mode_occupations and
physical_occupations (mode by sample matrices), ranks, diagnostics, stats,
and final_state. Evolution and observation do not expand a global dense state.
`record_timing=true` adds `timings` for the adaptive driver and each attempt;
these exclude initial Hamiltonian construction and basis enrichment here.
With timing disabled these fields are `nothing`.
"""
function tfd_evolve(system::TFDTTNSystem,initial::TTNTensor;
                    tspan,dt,saveat=nothing,atol=1e-6,rtol=1e-4,
                    enrich_initial::Bool=true,kwargs...)
    _check_tfd_state(initial,system)
    H = tfd_hamiltonian(system)
    prepared = enrich_initial ? enrich_basis(initial,H*initial) : initial
    result = adaptive_evolve(prepared,_tfd_generator(H),tspan; dt,saveat,atol,rtol,
        conserved=nothing,correct_invariant=false,observe=_tfd_observer(system,H),kwargs...)
    return (; result.times, densities=[v.density for v in result.values],
        norms=[v.norm for v in result.values], energies=[v.energy for v in result.values],
        mode_occupations=reduce(hcat,[v.mode_occupations for v in result.values]),
        physical_occupations=reduce(hcat,[v.physical_occupations for v in result.values]),
        result.ranks,result.diagnostics,result.stats,result.final_state,result.timings)
end

"""
    tfd_evolve(system, initial, times; atol=1e-10, maxrank=typemax(Int),
               krylov_tol=1e-12, krylovdim=30, krylov_maxiter=100,
               enrich_initial=true, method=:augmented, threaded=true, truncation=:weighted)

Propagate the thermofield Schrodinger equation with one BUG step
between each pair of strictly increasing finite `times`. The first time
labels `initial`; an equally spaced grid gives a fixed step size. No time-step
adaptation, normalization or norm correction is applied.
Select `method=:augmented` (default) or `:parallel`. `threaded` controls
concurrent sibling subtrees for augmented BUG and all-node solves for parallel BUG.
Compression defaults to `truncation=:weighted`; select `:local` for the
conservative local-core SVD policy.

`atol` controls the absolute rank-truncation budget per step, not time error.
Reduce the step size to check temporal convergence. A binding `maxrank` can
miss that budget; inspect each diagnostic's `truncation.tolerance_met`.
By default, enrich the initial bases with `H*initial` without changing its
amplitudes, as in the adaptive API. Set `enrich_initial=false` for a prepared
basis. A single time point returns an independent initial snapshot without
enrichment or evaluating truncation/Krylov options. The method name is always
validated. Inputs are not mutated.

Return `times`, `densities`, `norms`, `energies`, `mode_occupations` and
`physical_occupations` (mode by sample matrices), node-by-sample `ranks`,
per-step `diagnostics`, and `final_state`. Only observables and the final TTN
are retained. Evolution and observations use local TTN contractions.
"""
function tfd_evolve(system::TFDTTNSystem, initial::TTNTensor, times;
                    atol=1e-10, maxrank=typemax(Int), krylov_tol=1e-12,
                    krylovdim=30, krylov_maxiter=100, enrich_initial::Bool=true,
                    method=:augmented, threaded::Bool=true,truncation=:weighted)
    method === :augmented || method === :parallel ||
        throw(ArgumentError("method must be :augmented or :parallel"))
    step = method == :parallel ? parallel_bug_step : augmented_bug_step
    execution = (;threaded,truncation)
    _check_tfd_state(initial,system)
    grid = collect(times)
    !isempty(grid) && all(t -> t isa Real && isfinite(t),grid) ||
        throw(ArgumentError("times must be a nonempty collection of finite real numbers"))
    grid = Float64.(grid)
    all(isfinite,grid) && all(h -> isfinite(h) && h > 0,diff(grid)) ||
        throw(ArgumentError("times must be finite and strictly increasing in Float64"))
    H = tfd_hamiltonian(system)
    state = length(grid) > 1 && enrich_initial ? enrich_basis(initial,H*initial) : copy(initial)
    generator = _tfd_generator(H)
    observe = _tfd_observer(system,H)
    densities = Vector{Matrix{ComplexF64}}(undef,length(grid))
    norms, energies = Vector{Float64}(undef,length(grid)),Vector{Float64}(undef,length(grid))
    mode_occupations = Matrix{Float64}(undef,length(system.modes),length(grid))
    physical_occupations = Matrix{Float64}(undef,length(system.thermal_pairs),length(grid))
    ranks = Matrix{Int}(undef,length(initial.tree.children),length(grid))
    diagnostics = NamedTuple[]
    for sample in eachindex(grid)
        if sample > 1
            state,info = step(state,generator,grid[sample]-grid[sample-1];
                atol,maxrank,krylov_tol,krylovdim,krylov_maxiter,
                conserved=nothing,correct_invariant=false,execution...)
            push!(diagnostics,info)
        end
        value = observe(state)
        densities[sample],norms[sample],energies[sample] = value.density,value.norm,value.energy
        mode_occupations[:,sample] = value.mode_occupations
        physical_occupations[:,sample] = value.physical_occupations
        ranks[:,sample] = tree_ranks(state)
    end
    return (; times=grid,densities,norms,energies,mode_occupations,physical_occupations,
            ranks,diagnostics,final_state=state)
end
