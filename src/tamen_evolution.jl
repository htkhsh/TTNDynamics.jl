"""
    heom_tamen_evolve(system, initial, times; store_solutions=false, warm_start=true, kwargs...)

Experimental fixed-interval, space-time TTN AMEn-type HEOM solver.
Each adjacent pair in `times` is one collocation interval; the first value
labels the initial state. `time_points`, `rtol`, `atol`, `maxrank`, `maxsweeps`,
`kickrank`, `time_scheme`, `als_sweep`, `threaded`, workspace and GMRES options are forwarded
to `tamen_step`. A workspace is reused across the fixed intervals by default.
`warm_start=true` also reuses the preceding accepted space-time solution and
auxiliary residual. Set false for cold initialization. One owned hint is kept
locally, outside the workspace and returned history; it does not accumulate.

The trace functional is protected at all collocation points without trace
correction or normalization. A failed interval throws; `strict=false` is not
allowed here. The residual tolerance controls the discretized linear solve,
not time error: check convergence by increasing `time_points` or subdividing
intervals. With split system leaves, `maxrank` must hold the trace directions.

Return times, endpoint densities/traces/ranks, diagnostics, and final_state.
Diagnostics include all collocation trace values, residual bounds, sweep ranks
and timings. Space-time TTNs are kept only with `store_solutions=true`; otherwise
history stores small diagnostics, so memory does not grow by a TTN per interval.
"""
function heom_tamen_evolve(system::HEOMTTNSystem,initial::TTNTensor,times;
                           store_solutions::Bool=false,strict::Bool=true,
                           workspace=TAMEnWorkspace(),warm_start::Bool=true,kwargs...)
    strict || throw(ArgumentError("HEOM propagation requires converged intervals; use tamen_step for diagnostics"))
    _heom_tamen_owned_options(kwargs)
    leafdims(initial) == heom_dimensions(system) ||
        throw(DimensionMismatch("initial TTN dimensions differ from HEOM model"))
    grid = collect(times)
    !isempty(grid) && all(t->t isa Real && isfinite(t),grid) ||
        throw(ArgumentError("times must be a nonempty collection of finite real numbers"))
    grid = Float64.(grid)
    all(isfinite,grid) && all(h->isfinite(h) && h>0,diff(grid)) ||
        throw(ArgumentError("times must be finite and strictly increasing in Float64"))
    generator = heom_generator(system)
    conserved = trace_observable(system;tree=initial.tree)
    state = copy(initial)
    densities = Matrix{ComplexF64}[]
    traces = ComplexF64[]
    ranks = Matrix{Int}(undef,length(initial.tree.children),length(grid))
    diagnostics = NamedTuple[]
    accepted_hint = nothing
    for j in eachindex(grid)
        if j > 1
            state,info = tamen_step(state,generator,grid[j]-grid[j-1];conserved,workspace,
                warm_start=accepted_hint,store_warm_start=warm_start,kwargs...)
            accepted_hint = warm_start ? info.warm_start : nothing
            record = Base.structdiff(info,(;warm_start=info.warm_start))
            push!(diagnostics,store_solutions ? record : Base.structdiff(record,(;solution=record.solution)))
        end
        rho = root_density_matrix(state,system)
        push!(densities,rho)
        push!(traces,tr(rho))
        ranks[:,j] = tree_ranks(state)
    end
    return (;times=grid,densities,traces,ranks,diagnostics,final_state=state)
end

function _heom_tamen_owned_options(kwargs)
    for key in (:conserved,:conserved_observables,:observe,:store_warm_start)
        haskey(kwargs,key) && throw(ArgumentError("$key is controlled internally by heom_tamen_evolve"))
    end
    return nothing
end

"""
    heom_tamen_evolve(system, initial; tspan, dt, saveat=nothing,
                     atol=1e-8, rtol=1e-6, kwargs...)

Adaptive tAMEn intervals with independent output times. `atol/rtol` control
the temporal estimate; `solve_atol/solve_rtol` control linear convergence.
Supports `time_scheme=:cheb` or `:cn`, `time_error_damp`, `als_sweep`, `threaded`, workspace,
rank/sweep limits, and interval limits from `tamen_evolve`. Trace protection
is mandatory; no 2-norm or trace correction is applied. Diagnostics include
accepted and rejected attempts, with interval solver diagnostics in `.solve`.
`saveat` does not change the internal interval grid.
`warm_start=true` reuses only accepted interval hints; temporal rejections
preserve the previous hint. Set false to disable cross-interval reuse.
"""
function heom_tamen_evolve(system::HEOMTTNSystem,initial::TTNTensor;
                           tspan,dt,saveat=nothing,atol=1e-8,rtol=1e-6,kwargs...)
    _heom_tamen_owned_options(kwargs)
    leafdims(initial)==heom_dimensions(system) ||
        throw(DimensionMismatch("initial TTN dimensions differ from HEOM model"))
    generator = heom_generator(system)
    conserved = trace_observable(system;tree=initial.tree)
    result = tamen_evolve(initial,generator,tspan;dt,saveat,atol,rtol,conserved,
        observe=state->root_density_matrix(state,system),kwargs...)
    return (;result.times,densities=result.values,traces=tr.(result.values),
        result.ranks,result.final_state,result.accepted_times,result.diagnostics,
        result.accepted,result.rejected)
end
