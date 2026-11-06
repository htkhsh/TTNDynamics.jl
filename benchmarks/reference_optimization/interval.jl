using TTNSolver, LinearAlgebra, TOML
push!(LOAD_PATH,normpath(joinpath(@__DIR__,"../..")))
using TTNDynamics
BLAS.set_num_threads(1)

function main()
    # Same eight-mode model as benchmarks/tamen_certification/run.jl.
    k=8;V=ComplexF64[1 0;0 0];modes=HEOMMode[]
    for j in 1:k÷2
        omega=0.6+0.4j/(k÷2)
        push!(modes,HEOMMode(V,0.3+omega*im,(0.12-0.02im)*2/k,
            (0.04-0.01im)*2/k;scale=0.4/sqrt(k/2)),
            HEOMMode(V,0.3-omega*im,(0.04+0.01im)*2/k,
            (0.12+0.02im)*2/k;scale=0.4/sqrt(k/2)))
    end
    system=HEOMTTNSystem(ComplexF64[0 0.4;0.4 0.1],modes,fill(3,k);system_layout=:split)
    x=heom_initial_state(system,ComplexF64[1 0;0 0]);L=heom_generator(system)
    c=trace_observable(system;tree=x.tree)
    rows=Dict[];reference=nothing
    for backend in (:sop,:ttno)
        solve()=tamen_step(x,L,0.2;time_points=5,rtol=1e-8,maxrank=32,maxsweeps=20,
            als_sweep=:etter,threaded=true,conserved=c,residual_backend=backend)
        solve()
        for rep in 1:3
            GC.gc();m=@timed solve();y,d=m.value
            @assert d.converged && d.relative_residual_bound<=1e-8
            reference=isnothing(reference) ? y : reference
            difference=ttn_distance(y,reference)/norm(reference)
            @assert difference<1e-8
            @assert abs(dot(c,y)-1)<1e-10
            row=Dict("backend"=>string(backend),"repeat"=>rep,"seconds"=>m.time,"bytes"=>m.bytes,
                "global_residual_seconds"=>d.timings.residual_global_seconds,
                "sweeps"=>d.sweeps,"checks"=>d.global_checks,
                "relative_residual_bound"=>d.relative_residual_bound,
                "relative_difference"=>difference,"trace_error"=>abs(dot(c,y)-1),
                "operator_ranks"=>d.residual_operator_ranks,
                "operator_discarded_bound"=>d.residual_operator_discarded_bound)
            push!(rows,row);println(row);flush(stdout)
        end
    end
    open(joinpath(@__DIR__,"data","interval.toml"),"w") do io
        TOML.print(io,Dict("duration"=>0.2,"time_points"=>5,"rtol"=>1e-8,
            "maxrank"=>32,"threads"=>Threads.nthreads(:default),"blas_threads"=>1,"runs"=>rows))
    end
end
main()
