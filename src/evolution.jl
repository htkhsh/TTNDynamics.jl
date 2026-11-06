"""
    heom_evolve(system, initial, times; atol=1e-10, maxrank=typemax(Int),
                krylov_tol=1e-12, krylovdim=30, krylov_maxiter=100,
                correct_trace=true, method=:augmented, threaded=true, truncation=:weighted)

Propagate a TTN with `method=:augmented` (default) or `:parallel` BUG over
strictly increasing finite `times`. `threaded` controls concurrent sibling
subtrees for augmented BUG and all-node solves for parallel BUG.
Compression uses amplitude weights by default; `truncation=:local` selects
the conservative local-core SVD policy.
Each adjacent pair is one integration step (the first time labels `initial`).
The trace functional is included in every augmented and truncated space.
No normalization, dense hierarchy, or global Liouvillian is constructed.

Return `times`, zero-tier `densities`, complex `traces`, node-by-sample `ranks`,
per-step `diagnostics`, and `final_state`. History stores observables, not all
TTNs. `atol` is an absolute truncation budget per step; time error must be
checked by reducing the time step. A binding rank cap is reported in each
step's `truncation.tolerance_met`. `correct_trace=false` exposes uncorrected
trace preservation for validation. Caller inputs are never mutated.
With one time point, return an independent initial snapshot without evaluating
truncation or Krylov options. The method name is always validated.
"""
function heom_evolve(system::HEOMTTNSystem, initial::TTNTensor, times;
                     atol=1e-10, maxrank=typemax(Int), krylov_tol=1e-12,
                     krylovdim=30, krylov_maxiter=100, correct_trace::Bool=true,
                     method=:augmented, threaded::Bool=true,truncation=:weighted)
    method === :augmented || method === :parallel ||
        throw(ArgumentError("method must be :augmented or :parallel"))
    step = method == :parallel ? parallel_bug_step : augmented_bug_step
    execution = (;threaded,truncation)
    leafdims(initial) == heom_dimensions(system) ||
        throw(DimensionMismatch("initial TTN dimensions differ from HEOM model"))
    grid = collect(times)
    !isempty(grid) && all(t -> t isa Real && isfinite(t), grid) ||
        throw(ArgumentError("times must be a nonempty collection of finite real numbers"))
    grid = Float64.(grid)
    all(isfinite, grid) && all(h -> isfinite(h) && h > 0, diff(grid)) ||
        throw(ArgumentError("times must be finite and strictly increasing in Float64"))
    generator = heom_generator(system)
    conserved = trace_observable(system; tree=initial.tree)
    state = copy(initial)
    densities = Vector{Matrix{ComplexF64}}(undef, length(grid))
    traces = Vector{ComplexF64}(undef, length(grid))
    ranks = Matrix{Int}(undef, length(initial.tree.children), length(grid))
    diagnostics = NamedTuple[]
    for sample in eachindex(grid)
        if sample > 1
            state, info = step(state, generator, grid[sample]-grid[sample-1];
                atol, maxrank, conserved, krylov_tol, krylovdim, krylov_maxiter,
                correct_invariant=correct_trace, execution...)
            push!(diagnostics, info)
        end
        densities[sample] = root_density_matrix(state, system)
        traces[sample] = tr(densities[sample])
        ranks[:, sample] = tree_ranks(state)
    end
    return (; times=grid, densities, traces, ranks, diagnostics, final_state=state)
end
