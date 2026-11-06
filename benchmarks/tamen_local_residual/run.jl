# From TTNDynamics: OPENBLAS_NUM_THREADS=1 julia --threads=4 --project=. benchmarks/tamen_local_residual/run.jl
# Same eight-bath-mode split fixture as benchmarks/tamen_threads/run.jl.
using TTNDynamics, TTNSolver, LinearAlgebra, Statistics
BLAS.set_num_threads(1)

function main()
    k = 8
    H = ComplexF64[0 0.4;0.4 0.1]
    V = ComplexF64[1 0;0 0]
    modes = HEOMMode[]
    for j in 1:k÷2
        omega = 0.6+0.4j/(k÷2)
        push!(modes,HEOMMode(V,0.3+omega*im,(0.12-0.02im)*2/k,
                           (0.04-0.01im)*2/k;scale=0.4/sqrt(k/2)),
                    HEOMMode(V,0.3-omega*im,(0.04+0.01im)*2/k,
                           (0.12+0.02im)*2/k;scale=0.4/sqrt(k/2)))
    end
    system = HEOMTTNSystem(H,modes,fill(3,k);system_layout=:split)
    initial = heom_initial_state(system,ComplexF64[1 0;0 0])
    L = heom_generator(system)
    c = trace_observable(system;tree=initial.tree)
    work = Dict(method=>TAMEnWorkspace() for method in (:local,:global))
    solve(method) = tamen_step(initial,L,0.2;time_points=5,rtol=1e-8,maxrank=24,
        maxsweeps=16,conserved=c,residual_method=method,workspace=work[method],threaded=false)
    old,oldinfo = solve(:global)
    new,newinfo = solve(:local)
    difference = ttn_distance(old,new)/norm(old)
    @assert oldinfo.converged && newinfo.converged
    @assert difference < 1e-7
    println((;relative_endpoint_difference=difference,global_sweeps=oldinfo.sweeps,
             local_sweeps=newinfo.sweeps,local_checks=newinfo.global_checks)); flush(stdout)
    out = joinpath(@__DIR__,"data")
    mkpath(out)
    rows = NamedTuple[]
    open(joinpath(out,"runs.csv"),"w") do io
        for rep in 1:5, method in (isodd(rep) ? (:global,:local) : (:local,:global))
            GC.gc()
            timed = @timed solve(method)
            endpoint,info = timed.value
            @assert info.converged && info.history[end].global_residual_checked
            @assert info.relative_residual_bound <= 1e-8
            trace_error = maximum(abs.(info.invariant_values.-1))
            @assert trace_error < 1e-10
            t = info.timings
            row = (;method,repeat=rep,seconds=timed.time,bytes=timed.bytes,gc_seconds=timed.gctime,
                als_seconds=t.als_seconds,residual_seconds=t.residual_seconds,
                residual_local_seconds=t.residual_local_seconds,residual_global_seconds=t.residual_global_seconds,
                enrichment_seconds=t.enrichment_seconds,sweeps=info.sweeps,global_checks=info.global_checks,
                rank=maximum(tree_ranks(endpoint)),relative_residual=info.relative_residual_bound,
                trace_error,relative_endpoint_difference=ttn_distance(endpoint,old)/norm(old))
            isempty(rows) && println(io,join(keys(row),','))
            push!(rows,row)
            println(io,join(values(row),',')); flush(io)
        end
    end
    open(joinpath(out,"summary.csv"),"w") do io
        println(io,"method,seconds,bytes,residual_seconds,residual_local_seconds,residual_global_seconds,sweeps,global_checks,rank")
        for method in (:global,:local)
            group = filter(r->r.method==method,rows)
            fields = (:seconds,:bytes,:residual_seconds,:residual_local_seconds,:residual_global_seconds,:sweeps,:global_checks,:rank)
            vals = [median(getproperty.(group,key)) for key in fields]
            println(io,join([string(method);string.(vals)],','))
            println(method," ",NamedTuple{fields}(Tuple(vals)))
        end
    end
    for (method,info) in ((:global,oldinfo),(:local,newinfo))
        open(joinpath(out,"history_$method.csv"),"w") do io
            println(io,"sweep,estimated_relative_residual,global_checked,relative_residual_bound")
            for h in info.history
                println(io,join((h.sweep,h.relative_residual_estimate,h.global_residual_checked,h.relative_residual_bound),','))
            end
        end
    end
    open(joinpath(out,"environment.txt"),"w") do io
        println(io,"Julia: ",VERSION,"; CPU: ",Sys.CPU_NAME,"; Julia threads: ",Threads.nthreads(),
            "; BLAS threads: ",BLAS.get_num_threads(),"; task threading: false")
        println(io,"8 bath modes, 10 spatial leaves, 42 generator terms, hierarchy dimensions 3, split M=2")
        println(io,"duration=0.2 (dimensionless), time_points=5, linear rtol=1e-8, maxrank=24, kickrank=3")
        println(io,"Five alternating warmed trials, GC enabled, explicit GC before the timer; setup excluded.")
    end
end
main()
