# From TTNDynamics: julia --project=. examples/heom/tamen_dynamics/run.jl
using LinearAlgebra, TTNSolver, TTNDynamics
include("../common/dense_heom_reference.jl")

function run_tamen_check()
    BLAS.set_num_threads(1)
    # Dimensionless small-system validation, with an underdamped conjugate
    # pair. These explicit amplitudes are a test fixture, not a fitted bath.
    H = ComplexF64[0 0.4;0.4 0.1]
    V = ComplexF64[1 0;0 0]
    modes = [HEOMMode(V,0.3+0.8im,0.12-0.02im,0.04-0.01im;scale=0.4),
             HEOMMode(V,0.3-0.8im,0.04+0.01im,0.12+0.02im;scale=0.4)]
    system = HEOMTTNSystem(H,modes,[3,3];system_layout=:split)
    initial = heom_initial_state(system,ComplexF64[1 0;0 0])
    L = dense_heom_reference_matrix(system) # reference only
    initial_dense = vec(dense(initial))
    times = collect(0:0.2:1.0)
    output = joinpath(@__DIR__,"data")
    mkpath(output)
    summaries = NamedTuple[]
    for nt in (4,8)
        # Compilation is excluded from reported propagation time.
        heom_tamen_evolve(system,initial,[0.,0.02];time_points=nt,rtol=1e-10,
                          maxrank=24,maxsweeps=16)
        started = time_ns()
        result = heom_tamen_evolve(system,initial,times;time_points=nt,rtol=1e-10,
                                  maxrank=24,maxsweeps=16,store_solutions=true)
        elapsed = (time_ns()-started)/1e9
        max_state_error = max_trace_error = 0.0
        open(joinpath(output,"collocation_nt$nt.csv"),"w") do io
            println(io,"time,p1_ttn,p1_dense,full_relative_error,trace_error,relative_residual_bound,sweeps,max_spatial_rank")
            for (interval,d) in enumerate(result.diagnostics)
                sol = d.solution
                for j in 2:length(sol.nodes)
                    t = times[interval]+sol.nodes[j]
                    state = time_slice(sol,j)
                    rho = root_density_matrix(state,system)
                    reference = exp(t*L)*initial_dense
                    relative_error = norm(vec(dense(state))-reference)/norm(reference)
                    drift = abs(tr(rho)-1)
                    max_state_error = max(max_state_error,relative_error)
                    max_trace_error = max(max_trace_error,drift)
                    println(io,join((t,real(rho[1,1]),real(reference[1]),relative_error,
                        drift,d.relative_residual_bound,d.sweeps,maximum(tree_ranks(state))),','))
                end
            end
        end
        push!(summaries,(;time_points=nt,seconds=elapsed,max_state_error,max_trace_error,
                         max_rank=maximum(result.ranks)))
        println(last(summaries))
    end
    open(joinpath(output,"summary.csv"),"w") do io
        println(io,join(keys(first(summaries)),','))
        for row in summaries
            println(io,join(values(row),','))
        end
    end
    println("CSV output: ",output)
end

run_tamen_check()
