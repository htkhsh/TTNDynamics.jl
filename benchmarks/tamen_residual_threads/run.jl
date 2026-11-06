# Run previous package first to produce a common accepted HEOM fixture.
include("common.jl")
include("../../../TTNSolver/test/fixtures/tamen_canonical_reference.jl")

function main()
    label=get(ENV,"TAMEN_THREADS_LABEL","current")
    label in ("previous","current") || error("unknown label")
    repeats=parse(Int,get(ENV,"TAMEN_THREADS_REPEATS","5"))
    cache=get(ENV,"TAMEN_THREADS_SNAPSHOTS","/tmp/ttn-residual-thread-states")
    out=joinpath(@__DIR__,"data"); mkpath(out); mkpath(cache)
    nt=Threads.nthreads(:default)
    system,initial=certification_problem()
    options=(;time_points=5,rtol=1e-8,maxrank=32,maxsweeps=20,als_sweep=:etter,threaded=true)
    if label=="previous"
        _,info=tamen_step(initial,heom_generator(system),0.2;options...,
            conserved=trace_observable(system;tree=initial.tree))
        grid=TTNSolver._tamen_grid(5,0.2,:cheb)
        A=TTNSolver._spacetime_operator(heom_generator(system),grid)
        b=TTNSolver._time_lift(initial,grid.d0)
        serialize(joinpath(cache,"fixture.bin"),(A,info.solution.state,b,1e-8*norm(b)/20,64))
    end
    if label=="current"
        open(joinpath(out,"kernels_t$nt.csv"),"w") do io
            firstrow=true
            for (problem,fixture) in (("heom",deserialize(joinpath(cache,"fixture.bin"))),
                                      ("controlled32",controlled_problem()))
                reference=reference_canonical_residual(fixture...)
                work=Dict(key=>TTNSolver._ResidualWorkspace() for key in ("reuse","both"))
                function run(method)
                    method=="previous" && return reference_canonical_residual(fixture...)
                    workspace=get(work,method,nothing)
                    return TTNSolver._tamen_residual(fixture...;workspace,
                        threaded=method in ("threads","both"))
                end
                methods=("previous","neither","reuse","threads","both")
                for method in methods; run(method); end
                for rep in 1:repeats, method in (isodd(rep) ? methods : reverse(methods))
                    GC.gc()
                    measured=@timed run(method)
                    z,bound,drop=measured.value
                    difference=ttn_distance(z,reference[1])/norm(fixture[3])
                    @assert difference<=1e-10+(drop+reference[3])/norm(fixture[3])
                    @assert abs(bound-reference[2])<=1e-10*norm(fixture[3])+2(drop+reference[3])
                    scratch=get(work,method,nothing)
                    retained=isnothing(scratch) ? 0 : TTNSolver._residual_buffer_bytes(scratch)
                    row=(;problem,method,threads=nt,repeat=rep,seconds=measured.time,
                        bytes=measured.bytes,gc_seconds=measured.gctime,retained_bytes=retained,
                        maxrank=maximum(tree_ranks(z)),relative_bound=bound/norm(fixture[3]),
                        compression_bound=drop,reference_difference=difference)
                    write_row(io,row,firstrow); firstrow=false
                    println((;problem,method,rep,seconds=measured.time,bytes=measured.bytes,difference));flush(stdout)
                end
            end
        end
    end
    # This compares the entire solver, with identical threaded Etter ALS.
    # Only the global residual implementation differs in the previous package.
    if nt==4
        work=TAMEnWorkspace()
        solve()=heom_tamen_evolve(system,initial,collect(0.:0.2:1.);options...,workspace=work,warm_start=true)
        solve()
        open(joinpath(out,"trajectory_$label.csv"),"w") do io
            for rep in 1:repeats
                GC.gc(); measured=@timed solve(); result=measured.value
                @assert all(d->d.converged && d.history[end].global_residual_checked &&
                    d.relative_residual_bound<=1e-8,result.diagnostics)
                trace_error=maximum(maximum(abs.(d.invariant_values.-1)) for d in result.diagnostics)
                @assert trace_error<1e-10
                snapshot=joinpath(cache,"trajectory.bin")
                label=="previous" && serialize(snapshot,result.final_state)
                reference=deserialize(snapshot)
                difference=ttn_distance(reference,result.final_state)/norm(reference)
                @assert difference<2e-7
                row=(;label,threads=nt,repeat=rep,seconds=measured.time,bytes=measured.bytes,
                    gc_seconds=measured.gctime,
                    global_seconds=sum(d.timings.residual_global_seconds for d in result.diagnostics),
                    als_seconds=sum(d.timings.als_seconds for d in result.diagnostics),
                    sweeps=sum(d.sweeps for d in result.diagnostics),
                    checks=sum(d.global_checks for d in result.diagnostics),
                    maxrank=maximum(tree_ranks(result.final_state)),trace_error,
                    relative_bound=maximum(d.relative_residual_bound for d in result.diagnostics),
                    reference_difference=difference,
                    retained_bytes=TTNSolver._residual_buffer_bytes(work.certification))
                write_row(io,row,rep==1)
                println((;label,rep,seconds=measured.time,bytes=measured.bytes,difference));flush(stdout)
            end
        end
    end
    open(joinpath(out,"environment_$(label)_t$nt.txt"),"w") do io
        println(io,"Julia ",VERSION,"; CPU ",Sys.CPU_NAME,"; threads ",nt,"; BLAS ",BLAS.get_num_threads())
        println(io,"Solver source: ",pathof(TTNSolver))
        println(io,"Five-interval HEOM: 8 bath modes; split M2; local bath dimension3; h0.2; nt5; rtol1e-8; cap32; Etter; warm starts.")
        println(io,"Controlled case: 32 leaves; physical dimension4; initial rank8; 12 terms; residual cap24; budget1e-8; seeded random tensors, not physical dynamics.")
        println(io,"Same serialized HEOM iterate and RNG seed; warm-up and explicit GC excluded; GC during timing included; repetitions ",repeats)
        println(io,"Previous=canonical sum before buffer/thread optimization. Neither=new kernels, fresh workspace per term. Reuse=serial persistent workspace. Threads=fresh workspace per term, subtree tasks. Both=persistent workspace and tasks.")
    end
end
main()
