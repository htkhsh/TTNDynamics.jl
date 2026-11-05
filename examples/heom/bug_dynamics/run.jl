# Run from TTNDynamics: julia --project=examples examples/heom/bug_dynamics/run.jl
# Plot the saved data: python3 examples/heom/bug_dynamics/plot.py
using TTNDynamics, TTNSolver, LinearAlgebra

include("../common/dense_heom_reference.jl")

function run_bug_comparison()
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    H = ComplexF64[0.2 0.3+0.1im; 0.3-0.1im -0.4]
    V = ComplexF64[1 0.2im; -0.2im -0.5]
    modes = [HEOMMode(V, 0.8, 0.7+0.2im, 0.7-0.2im),
             HEOMMode(V, 0.4+1.2im, 0.3+0.1im, 0.2+0.05im; scale=0.7),
             HEOMMode(V, 0.4-1.2im, 0.2-0.05im, 0.3-0.1im; scale=0.7)]
    system = HEOMTTNSystem(H, modes, [3, 3, 3])
    rho = ComplexF64[0.7 0.1im; -0.1im 0.3]
    initial = heom_initial_state(system, rho)
    # Dense work is confined to this comparison, never the TTN solver.
    reference_matrix = dense_heom_reference_matrix(system)
    for (label, h) in (("coarse", 0.02), ("fine", 0.01))
        times = 0:h:1.0
        result = heom_evolve(system, initial, times; atol=1e-10, krylov_tol=1e-12)
        propagator = exp(h * reference_matrix)
        reference = vec(dense(initial))
        density_errors = Float64[]
        path = joinpath(data_dir, "heom_bug_$label.csv")
        open(path, "w") do io
            println(io, "time,p1_ttn,p2_ttn,p1_dense,p2_dense,density_error,trace_drift,uncorrected_trace_drift,max_rank,discarded_bound,correction_norm,tolerance_met")
            for k in eachindex(result.times)
                k > 1 && (reference = propagator * reference)
                reference_density = reshape(view(reference, 1:4), 2, 2)
                density = result.densities[k]
                error = norm(density - reference_density)
                push!(density_errors, error)
                info = k == 1 ? nothing : result.diagnostics[k-1]
                raw_drift = isnothing(info) ? 0.0 : abs(info.invariant_uncorrected - result.traces[1])
                dropped = isnothing(info) ? 0.0 : info.truncation.discarded_bound
                correction = isnothing(info) ? 0.0 : info.correction_norm
                met = isnothing(info) || info.truncation.tolerance_met
                values = (result.times[k], real(density[1,1]), real(density[2,2]),
                          real(reference_density[1,1]), real(reference_density[2,2]),
                          error, abs(result.traces[k]-result.traces[1]), raw_drift,
                          maximum(result.ranks[:,k]), dropped, correction, Int(met))
                println(io, join(values, ','))
            end
        end
        full_error = norm(vec(dense(result.final_state)) - reference) / norm(reference)
        println("$label: h=$h, maximum density error=$(maximum(density_errors)), final full relative error=$full_error")
        println("  maximum trace drift=$(maximum(abs.(result.traces .- result.traces[1]))), maximum rank=$(maximum(result.ranks))")
        println("  all truncation budgets met: ", all(d -> d.truncation.tolerance_met, result.diagnostics))
        println("  data: $path")
    end
end

run_bug_comparison()
