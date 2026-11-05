# julia --project=examples examples/heom/adaptive_dynamics/run.jl
# python3 examples/heom/adaptive_dynamics/plot.py
using TTNDynamics, TTNSolver, LinearAlgebra
include("../common/dense_heom_reference.jl")
include("../../common/model_tree_output.jl")

function run_adaptive_comparison()
    data_dir = joinpath(@__DIR__, "data")
    tree_dir = joinpath(@__DIR__, "trees")
    mkpath(data_dir)
    mkpath(tree_dir)
    H = ComplexF64[0.2 0.3+0.1im; 0.3-0.1im -0.4]
    V = ComplexF64[1 0.2im; -0.2im -0.5]
    modes = [HEOMMode(V, 0.8, 0.7+0.2im, 0.7-0.2im),
             HEOMMode(V, 0.4+1.2im, 0.3+0.1im, 0.2+0.05im; scale=0.7),
             HEOMMode(V, 0.4-1.2im, 0.2-0.05im, 0.3-0.1im; scale=0.7)]
    system = HEOMTTNSystem(H, modes, [3, 3, 3])
    initial = heom_initial_state(system, ComplexF64[0.7 0.1im; -0.1im 0.3])
    write_model_tree(joinpath(tree_dir,"tree_initial"),system,initial;
        title="HEOM · initial tree",subtitle="Scaled hierarchy · box sizes $(system.hierarchy_sizes)",print_tree=true)
    saveat = 0:0.2:10.0
    # Independent dense ADO work is confined to this validation example.
    reference_matrix = dense_heom_reference_matrix(system)
    propagator = exp(step(saveat) * reference_matrix)
    for (label, rtol) in (("loose", 1e-3), ("tight", 1e-5))
        result = heom_evolve(system, initial; tspan=(first(saveat), last(saveat)), dt=0.2,
            saveat, atol=1e-8, rtol, correct_trace=false)
        reference = vec(dense(initial))
        max_error = 0.0
        path = joinpath(data_dir, "heom_adaptive_$label.csv")
        open(path, "w") do io
            println(io, "time,p1_ttn,p2_ttn,p1_dense,p2_dense,density_error,trace_drift,max_rank")
            for k in eachindex(result.times)
                k > 1 && (reference = propagator * reference)
                rho_ref = reshape(view(reference, 1:4), 2, 2)
                rho = result.densities[k]
                error = norm(rho-rho_ref)
                max_error = max(max_error, error)
                row = (result.times[k], real(rho[1,1]), real(rho[2,2]),
                       real(rho_ref[1,1]), real(rho_ref[2,2]), error,
                       abs(result.traces[k]-result.traces[1]), maximum(result.ranks[:,k]))
                println(io, join(row, ','))
            end
        end
        step_path = joinpath(data_dir, "heom_adaptive_$(label)_steps.csv")
        open(step_path, "w") do io
            println(io, "time,t_end,dt,accepted,reason,error_estimate,tolerance,error_ratio,truncation_budget,truncation_error")
            for d in result.diagnostics
                println(io, join((d.time, d.t_end, d.dt, Int(d.accepted), d.reason,
                    d.error_estimate, d.tolerance, d.error_ratio,
                    d.truncation_budget, d.truncation_error), ','))
            end
        end
        println("$label: rtol=$rtol, ", result.stats, ", saved samples=", length(result.times))
        println("  maximum density error=$max_error, final full relative error=",
                norm(vec(dense(result.final_state))-reference)/norm(reference))
        println("  maximum trace drift (no correction)=", maximum(abs.(result.traces .- result.traces[1])))
        println("  samples: $path\n  attempts: $step_path")
        write_model_tree(joinpath(tree_dir,"tree_$label"),system,result.final_state;
            title="HEOM · $label",subtitle="t = $(last(result.times)) · rtol = $rtol · scaled hierarchy")
    end
end

run_adaptive_comparison()
