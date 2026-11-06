# Separate package processes: instrumented previous cut, then optimized cut.
include("../tamen_residual_threads/common.jl")

function main()
    label=get(ENV,"TAMEN_CUT_LABEL","optimized")
    label in ("baseline","optimized") || error("unknown label")
    repeats=parse(Int,get(ENV,"TAMEN_CUT_REPEATS","5"))
    out=joinpath(@__DIR__,"data"); mkpath(out)
    cache=get(ENV,"TAMEN_CUT_SNAPSHOTS","/tmp/ttn-residual-cut-states");mkpath(cache)
    system,initial=certification_problem()
    options=(;time_points=5,rtol=1e-8,maxrank=32,maxsweeps=20,als_sweep=:etter,threaded=true)
    fixture_path=joinpath(cache,"fixture.bin")
    if label=="baseline"
        _,d=tamen_step(initial,heom_generator(system),0.2;options...,
            conserved=trace_observable(system;tree=initial.tree))
        grid=TTNSolver._tamen_grid(5,0.2,:cheb)
        b=TTNSolver._time_lift(initial,grid.d0)
        serialize(fixture_path,(TTNSolver._spacetime_operator(heom_generator(system),grid),
            d.solution.state,b,1e-8*norm(b)/20,64))
    end
    fixture=deserialize(fixture_path)
    work=TTNSolver._ResidualWorkspace(); evolution_work=TAMEnWorkspace()
    certify()=TTNSolver._tamen_residual(fixture...;workspace=work,threaded=true)
    evolve()=heom_tamen_evolve(system,initial,collect(0.:0.2:1.);options...,
        workspace=evolution_work,warm_start=true)
    certify();evolve()
    open(joinpath(out,"runs_$label.csv"),"w") do io
        firstrow=true
        for rep in 1:repeats, case in (isodd(rep) ? ("certificate","trajectory") : ("trajectory","certificate"))
            before=TTNSolver._residual_times(work)
            GC.gc(); measured=@timed case=="certificate" ? certify() : evolve()
            if case=="certificate"
                state,bound,drop=measured.value
                scale=norm(fixture[3]); relative=bound/scale
                parts=TTNSolver._residual_time_delta(work,before)
                global_seconds=measured.time; sweeps=0; checks=1; trace_error=NaN
                retained=TTNSolver._residual_buffer_bytes(work)
            else
                result=measured.value; state=result.final_state; scale=norm(state)
                @assert all(d->d.converged && d.history[end].global_residual_checked &&
                    d.relative_residual_bound<=1e-8,result.diagnostics)
                relative=maximum(d.relative_residual_bound for d in result.diagnostics)
                global_seconds=sum(d.timings.residual_global_seconds for d in result.diagnostics)
                parts=NamedTuple{keys(first(result.diagnostics).timings.residual_global_parts)}(
                    Tuple(sum(getproperty(d.timings.residual_global_parts,k) for d in result.diagnostics)
                          for k in keys(first(result.diagnostics).timings.residual_global_parts)))
                sweeps=sum(d.sweeps for d in result.diagnostics); checks=sum(d.global_checks for d in result.diagnostics)
                trace_error=maximum(maximum(abs.(d.invariant_values.-1)) for d in result.diagnostics)
                @assert trace_error<1e-10
                retained=TTNSolver._residual_buffer_bytes(evolution_work.certification)
            end
            snapshot=joinpath(cache,"$case.bin")
            label=="baseline" && serialize(snapshot,state)
            reference=deserialize(snapshot)
            difference=ttn_distance(state,reference)/scale
            @assert difference<(case=="certificate" ? 1e-9 : 2e-7)
            @assert sum(values(parts))<=global_seconds*1.001
            row=(;label,case,repeat=rep,seconds=measured.time,bytes=measured.bytes,gc_seconds=measured.gctime,
                global_seconds,parts...,sweeps,checks,maxrank=maximum(tree_ranks(state)),relative,trace_error,
                difference,retained_bytes=retained)
            write_row(io,row,firstrow);firstrow=false
            println(row);flush(stdout)
        end
    end
    open(joinpath(out,"environment_$label.txt"),"w") do io
        println(io,"Julia ",VERSION,"; CPU ",Sys.CPU_NAME,"; workers ",Threads.nthreads(:default),"; GC ",Threads.ngcthreads(),"; BLAS ",BLAS.get_num_threads())
        println(io,"Solver source ",pathof(TTNSolver),"; warmed repeats ",repeats)
        println(io,"Same8bathmode split M2 HEOM; cap32, nt5, h0.2, rtol1e-8; certificate cap64; sequential5intervals with accepted warm starts.")
    end
end
main()
