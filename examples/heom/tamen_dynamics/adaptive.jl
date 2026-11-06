# From TTNDynamics: julia --threads=4 --project=. examples/heom/tamen_dynamics/adaptive.jl
using TTNDynamics, TTNSolver, LinearAlgebra
include("../common/dense_heom_reference.jl")

function run_adaptive_tamen_check()
    BLAS.set_num_threads(1)
    H = ComplexF64[0 0.4;0.4 0.1]
    V = ComplexF64[1 0;0 0]
    modes = [HEOMMode(V,0.3+0.8im,0.12-0.02im,0.04-0.01im;scale=0.4),
             HEOMMode(V,0.3-0.8im,0.04+0.01im,0.12+0.02im;scale=0.4)]
    system = HEOMTTNSystem(H,modes,[2,2];system_layout=:split)
    initial = heom_initial_state(system,ComplexF64[1 0;0 0])
    L = dense_heom_reference_matrix(system)
    output = joinpath(@__DIR__,"data")
    mkpath(output)
    for scheme in (:cheb,:cn)
        result = heom_tamen_evolve(system,initial;tspan=(0.,1.),dt=1.,
            saveat=0:0.05:1,time_scheme=scheme,time_points=5,atol=1e-8,rtol=1e-5,
            time_error_damp=10,solve_rtol=1e-10,maxrank=16,threaded=true)
        open(joinpath(output,"adaptive_$(scheme)_samples.csv"),"w") do io
            println(io,"time,p1_ttn,p1_dense,density_error,trace_error")
            for (t,rho) in zip(result.times,result.densities)
                exact = exp(t*L)*vec(dense(initial))
                exact_rho = reshape(exact[1:4],2,2)
                println(io,join((t,real(rho[1,1]),real(exact_rho[1,1]),
                    norm(rho-exact_rho),abs(tr(rho)-1)),','))
            end
        end
        open(joinpath(output,"adaptive_$(scheme)_attempts.csv"),"w") do io
            println(io,"start,end,dt,accepted,reason,time_error_ratio,linear_residual,sweeps,seconds,environment_seconds")
            for d in result.diagnostics
                s = d.solve
                residual = hasproperty(s,:relative_residual_bound) ? s.relative_residual_bound : Inf
                sweeps = hasproperty(s,:sweeps) ? s.sweeps : 0
                env = hasproperty(s,:timings) ? s.timings.environment_seconds : 0.0
                println(io,join((d.time,d.t_end,d.dt,d.accepted,d.reason,d.error_ratio,
                    residual,sweeps,d.seconds,env),','))
            end
        end
        error = norm(vec(dense(result.final_state))-exp(L)*vec(dense(initial)))
        println((;scheme,result.accepted,result.rejected,full_error=error,
                  trace_error=maximum(abs.(result.traces.-1))))
    end
    println("CSV output: ",output)
end
run_adaptive_tamen_check()
