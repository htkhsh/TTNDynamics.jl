# Same fixtures, trajectory loop and settings as the augmented scaling study.
# Only the integration method changes. Preserve both historical datasets.
include("../augmented_threads/run.jl")

function benchmark_parallel(repetitions=5;cases=CASES)
    repetitions >= 3 || error("use at least three repetitions")
    BLAS.get_num_threads() == 1 || error("set OPENBLAS_NUM_THREADS=1")
    data = joinpath(@__DIR__,"data"); mkpath(data)
    suffix = "threads$(Threads.nthreads(:default))"
    open(joinpath(data,"environment_$suffix.txt"),"w") do io
        println(io,"Measured: ",Dates.now(),"\nJulia: ",VERSION,"\nCPU: ",first(Sys.cpu_info()).model)
        println(io,"Julia workers: ",Threads.nthreads(:default),"\nBLAS threads: ",BLAS.get_num_threads())
        println(io,"GC threads: ",Threads.ngcthreads(),"\nKrylovKit: ",Base.pkgversion(TTNSolver.KrylovKit))
        println(io,"h: ",STEP,"\nsteps: ",NSTEPS,"\noptions: ",OPTIONS,"\nrepetitions: ",repetitions)
        println(io,"Load at start: ",Sys.loadavg())
        for file in ("parallel_bug.jl","parallel_workspaces.jl","bug.jl","bug_workspaces.jl","contractions.jl","truncation.jl")
            println(io,file," sha256: ",bytes2hex(sha256(read(joinpath(dirname(pathof(TTNSolver)),file)))))
        end
        println(io,"run.jl sha256: ",bytes2hex(sha256(read(@__FILE__))))
        println(io,"shared augmented_threads/run.jl sha256: ",bytes2hex(sha256(read(joinpath(@__DIR__,"../augmented_threads/run.jl")))))
    end
    open(joinpath(data,"runs_$suffix.csv"),"w") do io
        println(io,"case,workers,threaded,repetition,seconds,bytes,gc_seconds,state_distance,density_difference,omitted_difference,ranks_equal,invariant_drift,max_rank,max_augmented_rank,max_local_dimension,truncation_failures,largest_discarded_bound,leaves,terms,full_dimension")
        for name in cases
            problem = scaling_problem(name)
            println("Warming $name"); flush(stdout)
            for threaded in (false,true), _ in 1:2
                trajectory(problem;step=parallel_bug_step,threaded)
            end
            for rep in 1:repetitions
                measured = Dict{Bool,Any}()
                for threaded in (isodd(rep) ? (false,true) : (true,false))
                    GC.gc()
                    measured[threaded] = @timed trajectory(problem;step=parallel_bug_step,threaded)
                end
                a,b = measured[false].value,measured[true].value
                distance = ttn_distance(a.final_state,b.final_state)
                density_difference = maximum(norm(u.density-v.density) for (u,v) in zip(a.observations,b.observations))
                omitted_difference = maximum(maximum(abs.(u.omitted_block_norms-v.omitted_block_norms)) for (u,v) in zip(a.diagnostics,b.diagnostics))
                @assert omitted_difference < 1e-12
                @assert distance < 1e-11 && density_difference < 1e-11
                @assert a.ranks == b.ranks
                @assert [d.truncation.tolerance_met for d in a.diagnostics] ==
                        [d.truncation.tolerance_met for d in b.diagnostics]
                for threaded in (false,true)
                    m = measured[threaded]; r = m.value
                    invariant_drift = maximum(abs(v.invariant-first(r.observations).invariant) for v in r.observations)
                    println(io,join((name,Threads.nthreads(:default),threaded,rep,m.time,m.bytes,m.gctime,
                        distance,density_difference,omitted_difference,a.ranks==b.ranks,invariant_drift,
                        maximum(maximum.(r.ranks)),maximum(maximum(d.augmented_ranks) for d in r.diagnostics),
                        maximum(d.max_local_dimension for d in r.diagnostics),
                        count(d -> !d.truncation.tolerance_met,r.diagnostics),
                        maximum(d.truncation.discarded_bound for d in r.diagnostics),
                        length(problem.initial.tree.leaf_order),length(problem.operator.terms),
                        prod(big.(leafdims(problem.initial)))),','))
                end
                flush(io)
                println("  $rep: serial ",round(measured[false].time;digits=3)," s; threaded ",
                    round(measured[true].time;digits=3)," s; delta ",distance); flush(stdout)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_parallel(isempty(ARGS) ? 5 : parse(Int,only(ARGS)))
end

