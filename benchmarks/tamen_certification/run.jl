# See docs/tamen_certification.md: run the reference package first, then current.
using TTNSolver
push!(LOAD_PATH,normpath(joinpath(@__DIR__,"../..")))
using TTNDynamics, LinearAlgebra, Statistics, Serialization
BLAS.set_num_threads(1)

function certification_problem()
    k=8
    H,V=ComplexF64[0 0.4;0.4 0.1],ComplexF64[1 0;0 0]
    modes=HEOMMode[]
    for j in 1:k÷2
        omega=0.6+0.4j/(k÷2)
        push!(modes,HEOMMode(V,0.3+omega*im,(0.12-0.02im)*2/k,
                           (0.04-0.01im)*2/k;scale=0.4/sqrt(k/2)),
                    HEOMMode(V,0.3-omega*im,(0.04+0.01im)*2/k,
                           (0.12+0.02im)*2/k;scale=0.4/sqrt(k/2)))
    end
    system=HEOMTTNSystem(H,modes,fill(3,k);system_layout=:split)
    return system,heom_initial_state(system,ComplexF64[1 0;0 0])
end

function main()
    label=get(ENV,"TAMEN_CERT_LABEL","optimized")
    label in ("reference","optimized") || error("unknown label")
    repeats=parse(Int,get(ENV,"TAMEN_CERT_REPEATS","5"))
    cache=get(ENV,"TAMEN_CERT_SNAPSHOTS","/tmp/ttn-residual-comparison-states")
    out=joinpath(@__DIR__,"data"); mkpath(out); mkpath(cache)
    system,initial=certification_problem()
    L=heom_generator(system); c=trace_observable(system;tree=initial.tree)
    options=(;time_points=5,rtol=1e-8,maxrank=32,maxsweeps=20,als_sweep=:etter,threaded=true)
    endpoint,info=tamen_step(initial,L,0.2;options...,conserved=c)
    fixture_file=joinpath(cache,"fixture.bin")
    if label=="reference"
        grid=TTNSolver._tamen_grid(5,0.2,:cheb)
        A=TTNSolver._spacetime_operator(L,grid)
        b=TTNSolver._time_lift(initial,grid.d0)
        serialize(fixture_file,(A,info.solution.state,b,1e-8*norm(b)/20,64))
    end
    # Exactly the same accepted space-time state in both package processes.
    fixture=deserialize(fixture_file)
    certify()=TTNSolver._tamen_residual(fixture...)
    certify()
    works=Dict(case=>TAMEnWorkspace() for case in ("interval","trajectory"))
    function solve(case)
        times=case=="interval" ? [0.,0.2] : collect(0.:0.2:1.)
        heom_tamen_evolve(system,initial,times;options...,workspace=works[case],warm_start=true)
    end
    for case in ("interval","trajectory"); solve(case); end
    rows=NamedTuple[]
    open(joinpath(out,"runs_$label.csv"),"w") do io
        for rep in 1:repeats, case in (isodd(rep) ? ("certificate","interval","trajectory") : ("trajectory","interval","certificate"))
            GC.gc()
            measured=@timed case=="certificate" ? certify() : solve(case)
            if case=="certificate"
                residual,bound,dropped=measured.value
                trace_error=NaN; sweeps=0; checks=1
                residual_seconds=global_seconds=measured.time; als_seconds=0.
                relative_residual=bound/norm(fixture[3])
                state=residual
            else
                result=measured.value
                @assert all(d->d.converged && d.history[end].global_residual_checked && d.relative_residual_bound<=1e-8,result.diagnostics)
                trace_error=maximum(maximum(abs.(d.invariant_values.-1)) for d in result.diagnostics)
                @assert trace_error<1e-10
                sweeps=sum(d.sweeps for d in result.diagnostics)
                checks=sum(d.global_checks for d in result.diagnostics)
                residual_seconds=sum(d.timings.residual_seconds for d in result.diagnostics)
                global_seconds=sum(d.timings.residual_global_seconds for d in result.diagnostics)
                als_seconds=sum(d.timings.als_seconds for d in result.diagnostics)
                relative_residual=maximum(d.relative_residual_bound for d in result.diagnostics)
                state=result.final_state
            end
            snapshot=joinpath(cache,"$case.bin")
            if label=="reference"
                serialize(snapshot,state)
            end
            reference=deserialize(snapshot)
            # Tiny residual comparison is absolute, relative to the RHS scale.
            scale=case=="certificate" ? norm(fixture[3]) : norm(reference)
            difference=ttn_distance(reference,state)/scale
            @assert difference < (case=="certificate" ? 1e-9 : 2e-7)
            row=(;label,case,repeat=rep,seconds=measured.time,bytes=measured.bytes,gc_seconds=measured.gctime,
                residual_seconds,global_seconds,als_seconds,sweeps,checks,
                maxrank=maximum(tree_ranks(state)),relative_residual,trace_error,reference_difference=difference)
            isempty(rows) && println(io,join(keys(row),','))
            push!(rows,row); println(io,join(values(row),','));flush(io)
            println((;label,case,rep,seconds=measured.time,bytes=measured.bytes,sweeps,checks,difference));flush(stdout)
        end
    end
    open(joinpath(out,"environment_$label.txt"),"w") do io
        println(io,"Julia ",VERSION,"; CPU ",Sys.CPU_NAME,"; threads ",Threads.nthreads(),"; BLAS ",BLAS.get_num_threads())
        println(io,"Solver source: ",pathof(TTNSolver))
        println(io,"Eight bath modes, split M=2, hierarchy dimension3, h=0.2, nt=5, rtol=1e-8, maxrank32, Etter threaded.")
        println(io,"Trajectory: five intervals to dimensionless time1; accepted warm starts. Repeats: ",repeats)
        println(io,"Certificate: identical serialized reference iterate, cap64, same termwise error budget.")
        println(io,"Model setup, compilation/warm-up and explicit GC excluded; GC enabled during timing.")
    end
end
main()
