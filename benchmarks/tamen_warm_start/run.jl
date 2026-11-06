# From TTNDynamics: OPENBLAS_NUM_THREADS=1 julia --threads=4 --project=. benchmarks/tamen_warm_start/run.jl
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
    times = collect(0.:0.2:1.)
    work = Dict(flag=>TAMEnWorkspace() for flag in (false,true))
    solve(flag) = heom_tamen_evolve(system,initial,times;warm_start=flag,
        time_points=5,rtol=1e-8,maxrank=32,maxsweeps=16,
        workspace=work[flag],threaded=false)
    cold,warm = solve(false),solve(true)
    difference = ttn_distance(cold.final_state,warm.final_state)/norm(cold.final_state)
    @assert difference < 2e-7
    println((;relative_endpoint_difference=difference,
             cold_sweeps=sum(d.sweeps for d in cold.diagnostics),
             warm_sweeps=sum(d.sweeps for d in warm.diagnostics))); flush(stdout)
    out = joinpath(@__DIR__,"data")
    mkpath(out)
    last_results = Dict(false=>cold,true=>warm)
    rows = NamedTuple[]
    repeats = parse(Int,get(ENV,"TAMEN_WARM_REPEATS","5"))
    open(joinpath(out,"runs.csv"),"w") do io
        for rep in 1:repeats, flag in (isodd(rep) ? (false,true) : (true,false))
            GC.gc()
            timed = @timed solve(flag)
            r = timed.value
            @assert all(d->d.converged && d.history[end].global_residual_checked,r.diagnostics)
            @assert all(d->d.relative_residual_bound<=1e-8,r.diagnostics)
            @assert !first(r.diagnostics).warm_start_used
            @assert all(d->d.warm_start_used==flag,r.diagnostics[2:end])
            @assert all(d->!hasproperty(d,:warm_start) && !hasproperty(d,:solution),r.diagnostics)
            trace_error = maximum(maximum(abs.(d.invariant_values.-1)) for d in r.diagnostics)
            @assert trace_error < 1e-10
            row = (;warm_start=flag,repeat=rep,seconds=timed.time,bytes=timed.bytes,gc_seconds=timed.gctime,
                sweeps=sum(d.sweeps for d in r.diagnostics),global_checks=sum(d.global_checks for d in r.diagnostics),
                als_seconds=sum(d.timings.als_seconds for d in r.diagnostics),
                residual_seconds=sum(d.timings.residual_seconds for d in r.diagnostics),
                warm_start_seconds=sum(d.timings.warm_start_seconds for d in r.diagnostics),
                maxrank=maximum(r.ranks),trace_error,
                relative_endpoint_difference=ttn_distance(r.final_state,cold.final_state)/norm(cold.final_state))
            isempty(rows) && println(io,join(keys(row),','))
            push!(rows,row); println(io,join(values(row),',')); flush(io)
            last_results[flag] = r
        end
    end
    open(joinpath(out,"summary.csv"),"w") do io
        println(io,"warm_start,seconds,bytes,sweeps,global_checks,als_seconds,residual_seconds,warm_start_seconds,maxrank")
        for flag in (false,true)
            group = filter(r->r.warm_start==flag,rows)
            fields = (:seconds,:bytes,:sweeps,:global_checks,:als_seconds,:residual_seconds,:warm_start_seconds,:maxrank)
            vals = [median(getproperty.(group,key)) for key in fields]
            println(io,join([string(flag);string.(vals)],','))
            println(flag," ",NamedTuple{fields}(Tuple(vals)))
        end
    end
    for (label,result) in (("cold",last_results[false]),("warm",last_results[true]))
        open(joinpath(out,"intervals_$label.csv"),"w") do io
            println(io,"start,end,warm_start_used,residual_reused,sweeps,global_checks,relative_residual,trace_error,warm_start_seconds")
            for (j,d) in enumerate(result.diagnostics)
                println(io,join((times[j],times[j+1],d.warm_start_used,d.residual_warm_start_used,
                    d.sweeps,d.global_checks,d.relative_residual_bound,
                    maximum(abs.(d.invariant_values.-1)),d.timings.warm_start_seconds),','))
            end
        end
    end
    open(joinpath(out,"environment.txt"),"w") do io
        println(io,"Julia: ",VERSION,"; CPU: ",Sys.CPU_NAME,"; Julia threads: ",Threads.nthreads(),
            "; BLAS threads: ",BLAS.get_num_threads(),"; threaded=false; local residual in both modes")
        println(io,"Eight bath modes, split M=2, hierarchy dimension3; five intervals0.2 to dimensionless time1.0")
        println(io,"time_points=5, linear rtol=1e-8, maxrank=32, kickrank=3; repeats=",repeats)
        println(io,"Both paths warmed; alternate trials; GC enabled, explicit GC outside timer; model setup excluded.")
    end
end
main()
