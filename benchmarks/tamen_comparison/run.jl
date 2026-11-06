isdefined(@__MODULE__,:taylor_reference) || include("common.jl")

function selected_options(name)
    rows = [split(line,',') for line in readlines(joinpath(@__DIR__,"data","calibration_$name.csv"))[2:end]]
    valid(method) = filter(row -> row[2] == method && parse(Float64,row[6]) <= ERROR_TARGET,rows)
    t = only(valid("tamen")); b = only(valid("parallel_bug"))
    return (;tolerance=parse(Float64,t[3]),steps=parse(Int,b[3]),atol=parse(Float64,b[4]),maxrank=parse(Int,b[12]))
end

function environment(path)
    open(path,"w") do io
        println(io,"Date: ",now(),"\nJulia: ",VERSION,"\nCPU: ",Sys.cpu_info()[1].model)
        println(io,"Logical CPUs: ",Sys.CPU_THREADS,"\nJulia workers: ",Threads.nthreads(:default),
            "\nGC workers: ",Threads.ngcthreads(),"\nBLAS threads: ",BLAS.get_num_threads(),
            "\nStrided threads: ",TTSolver.TensorOperations.Strided.get_num_threads(),
            "\nBLAS: ",BLAS.get_config(),"\nLoad: ",Sys.loadavg())
        println(io,"Final time: ",FINAL_TIME,"\nTarget relative error: ",ERROR_TARGET,
            "\nWarmups: 1 per method\nRepetitions: 3 alternating pairs\nSeed: 20261105")
        for dir in (dirname(pathof(TTSolver)),dirname(pathof(TTNSolver)),dirname(pathof(TTNDynamics)),@__DIR__)
            for path in sort(readdir(dir;join=true))
                endswith(path,".jl") || continue
                println(io,relpath(path,joinpath(@__DIR__,"../../.."))," sha256: ",bytes2hex(sha256(read(path))))
            end
        end
        path = joinpath(@__DIR__,"../parallel_large_trees/fixtures.jl")
        println(io,"fixtures.jl sha256: ",bytes2hex(sha256(read(path))))
    end
end

function benchmark(name)
    Threads.nthreads(:default) == 8 || error("run with eight Julia workers")
    BLAS.get_num_threads() == 1 || error("run with one BLAS thread")
    data = joinpath(@__DIR__,"data")
    environment(joinpath(data,"environment_$name.txt"))
    inputs = deserialize(joinpath(data,"reference_$name.jls"))
    (;A,x,reference,conserved,convergence) = inputs
    @assert convergence < ERROR_TARGET/100
    problem = large_problem(name); heom = startswith(name,"heom")
    @assert relative_error(problem.initial,x) < 1e-6
    settings = selected_options(name)
    solve(method) = method == "tamen" ? run_tamen(A,x;tolerance=settings.tolerance,conserved) :
        run_bug(problem;steps=settings.steps,atol=settings.atol,maxrank=settings.maxrank)
    println("Warming $name: $settings"); flush(stdout)
    for method in ("tamen","parallel_bug")
        result = solve(method)
        @assert relative_error(result.state,reference) <= ERROR_TARGET
    end
    density_reference = tt_density(reference,heom)
    invariant0 = problem.observe(problem.initial).invariant
    open(joinpath(data,"runs_$name.csv"),"w") do io
        println(io,"case,method,repetition,seconds,bytes,gc_seconds,relative_error,density_error,invariant_drift,rank,stored_complex,steps,tolerance,atol,intervals,truncation_misses,discarded_bound,maxrank")
        for rep in 1:3
            for method in (isodd(rep) ? ("tamen","parallel_bug") : ("parallel_bug","tamen"))
                GC.gc()
                measured = @timed solve(method)
                result = measured.value; state = result.state
                err = relative_error(state,reference)
                @assert err <= ERROR_TARGET
                density = method == "tamen" ? tt_density(state,heom) : problem.observe(state).density
                invariant = heom ? tr(density) : norm(state)
                storage = method == "tamen" ? sum(length,state.cores) :
                    sum(length,values(state.cores))+sum(length,values(state.leaves))
                println(io,join((name,method,rep,measured.time,measured.bytes,measured.gctime,err,
                    norm(density-density_reference),abs(invariant-invariant0),result.rank,storage,
                    method == "tamen" ? 0 : settings.steps,method == "tamen" ? settings.tolerance : 0,
                    method == "tamen" ? 0 : settings.atol,method == "tamen" ? result.intervals : 0,
                    method == "tamen" ? 0 : result.misses,method == "tamen" ? 0 : result.discarded,
                    method == "tamen" ? 0 : settings.maxrank),','))
                flush(io)
                println("$name $rep $method: ",measured.time," s error ",err); flush(stdout)
            end
        end
    end
    # One-time TT operator preparation is separate from propagation. Warm and
    # record it too, since it can exceed a short tAMEn trajectory's cost.
    sop_to_mpo(problem.operator)
    open(joinpath(data,"mpo_setup_$name.csv"),"w") do io
        println(io,"case,repetition,seconds,bytes,rank")
        for rep in 1:3
            GC.gc()
            measured = @timed sop_to_mpo(problem.operator)
            println(io,join((name,rep,measured.time,measured.bytes,maximum(tt_ranks(measured.value))),','))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    for name in (isempty(ARGS) ? collect(LARGE_CASES) : ARGS)
        benchmark(name)
    end
end
