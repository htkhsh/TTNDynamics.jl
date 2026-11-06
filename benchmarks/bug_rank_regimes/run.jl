isdefined(@__MODULE__,:rank_problem) || include("common.jl")

const MEASURE_CASES = ("weak6","weak8","moderate8","strong6","strong6_short","system8")

function selected_bug(name,method)
    rows = [split(line,',') for line in readlines(joinpath(@__DIR__,"data","scan_$name.csv"))[2:end]]
    valid = filter(r -> r[2]==method && parse(Float64,r[7])<=ERROR_TARGET,rows)
    isempty(valid) && error("no accepted $method configuration for $name")
    r = first(valid)
    return (;steps=parse(Int,r[4]),maxrank=parse(Int,r[5]))
end

function tune_tamen(name)
    dir = joinpath(@__DIR__,"data")
    (;spec,A,reference) = deserialize(joinpath(dir,"$name.jls"))
    interval = get(spec,:interval,INTERVAL)
    (;train) = deserialize(joinpath(dir,"input_$name.jls"))
    p = rank_problem(spec); conserved = product_to_tt(p.conserved)
    open(joinpath(dir,"time_basis_$name.csv"),"w") do io
        println(io,"case,points,tolerance,trial,seconds,error,rank,intervals")
        for points in (4,8,12)
            accepted = false
            for tolerance in (1e-3,1e-4,1e-5)
                # trial 0 is calibration/warmup, never used for selection.
                result = run_tamen(A,train;points,tolerance,conserved,final_time=interval)
                err = state_error(result.state,reference)
                println(io,join((name,points,tolerance,0,NaN,err,result.rank,result.intervals),',')); flush(io)
                err <= ERROR_TARGET || continue
                for rep in 1:2
                    GC.gc()
                    measured = @timed run_tamen(A,train;points,tolerance,conserved,final_time=interval)
                    r = measured.value; err = state_error(r.state,reference)
                    @assert err <= ERROR_TARGET
                    println(io,join((name,points,tolerance,rep,measured.time,err,r.rank,r.intervals),',')); flush(io)
                end
                accepted = true
                break
            end
            accepted || error("tAMEn points=$points could not reach accuracy for $name")
        end
    end
end

function selected_tamen(name)
    rows = [split(line,',') for line in readlines(joinpath(@__DIR__,"data","time_basis_$name.csv"))[2:end]]
    valid = filter(r -> parse(Int,r[4])>0 && parse(Float64,r[6])<=ERROR_TARGET,rows)
    options = unique((parse(Int,r[2]),parse(Float64,r[3])) for r in valid)
    score(o) = sum(parse(Float64,r[5]) for r in valid if parse(Int,r[2])==o[1] && parse(Float64,r[3])==o[2])/2
    best = options[argmin(score.(options))]
    return (;points=best[1],tolerance=best[2])
end

function write_environment(name;interval=INTERVAL)
    open(joinpath(@__DIR__,"data","environment_$name.txt"),"w") do io
        println(io,"Date: ",now(),"\nJulia: ",VERSION,"\nCPU: ",Sys.cpu_info()[1].model)
        println(io,"Workers: ",Threads.nthreads(:default),"\nBLAS: ",BLAS.get_num_threads(),
            "\nStrided: ",TTSolver.TensorOperations.Strided.get_num_threads(),"\nGC: ",Threads.ngcthreads(),"\nLoad: ",Sys.loadavg())
        println(io,"Interval: ",interval,"\nTarget: ",ERROR_TARGET,"\nRepetitions: 3\nWarmups: 1 per method")
        for dir in (dirname(pathof(TTSolver)),dirname(pathof(TTNSolver)),dirname(pathof(TTNDynamics)),@__DIR__)
            for file in sort(readdir(dir;join=true))
                endswith(file,".jl") || continue
                println(io,relpath(file,joinpath(@__DIR__,"../../.."))," sha256: ",bytes2hex(sha256(read(file))))
            end
        end
        for file in ("../tamen_comparison/common.jl","../parallel_large_trees/fixtures.jl")
            path = joinpath(@__DIR__,file)
            println(io,relpath(path,joinpath(@__DIR__,"../../.."))," sha256: ",bytes2hex(sha256(read(path))))
        end
    end
end

function measure(name)
    @assert Threads.nthreads(:default)==8 && BLAS.get_num_threads()==1
    dir = joinpath(@__DIR__,"data")
    data = deserialize(joinpath(dir,"$name.jls"))
    (;spec,A,reference) = data
    interval = get(spec,:interval,INTERVAL)
    (;train,tree) = deserialize(joinpath(dir,"input_$name.jls"))
    p = rank_problem(spec); conserved = product_to_tt(p.conserved)
    opts = Dict("tamen"=>selected_tamen(name),"augmented"=>selected_bug(name,"augmented"),
                "parallel"=>selected_bug(name,"parallel"))
    solve(method) = method=="tamen" ? run_tamen(A,train;opts[method]...,conserved,final_time=interval) :
        evolve_bug(p,tree;method=Symbol(method),opts[method]...,interval)
    println("Measuring $name: $opts"); flush(stdout)
    for method in ("tamen","augmented","parallel")
        @assert state_error(solve(method).state,reference) <= ERROR_TARGET
    end
    write_environment(name;interval)
    open(joinpath(dir,"runs_$name.csv"),"w") do io
        println(io,"case,method,repetition,seconds,bytes,gc_seconds,error,input_rank,output_rank,input_storage,output_storage,points,tolerance,steps,maxrank,intervals,truncation_misses")
        for rep in 1:3
            for method in circshift(["tamen","augmented","parallel"],rep-1)
                GC.gc()
                m = @timed solve(method)
                r = m.value; err = state_error(r.state,reference)
                @assert err <= ERROR_TARGET
                o = opts[method]; tt = method=="tamen"
                println(io,join((name,method,rep,m.time,m.bytes,m.gctime,err,
                    tt ? maximum(tt_ranks(train)) : maximum(tree_ranks(tree)),r.rank,
                    tt ? storage(train) : storage(tree),storage(r.state),
                    tt ? o.points : 0,tt ? o.tolerance : 0,tt ? 0 : o.steps,tt ? 0 : o.maxrank,
                    tt ? r.intervals : 0,tt ? 0 : r.misses),',')); flush(io)
                println("  $rep $method ",m.time," s error ",err); flush(stdout)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    for name in (isempty(ARGS) ? MEASURE_CASES : ARGS)
        tune_tamen(name)
        measure(name)
    end
end
