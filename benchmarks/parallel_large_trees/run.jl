include("fixtures.jl")

function write_environment(path,repetitions)
    open(path,"w") do io
        println(io,"Measured: ",Dates.now(),"\nJulia: ",VERSION,"\nCPU: ",first(Sys.cpu_info()).model)
        println(io,"Workers: ",Threads.nthreads(:default),"\nBLAS threads: ",BLAS.get_num_threads(),
            "\nGC threads: ",Threads.ngcthreads(),"\nKrylovKit: ",Base.pkgversion(TTNSolver.KrylovKit))
        println(io,"h: ",LARGE_STEP,"\npresteps: ",PRESTEPS,"\noptions: ",LARGE_OPTIONS,
            "\nrepetitions: ",repetitions,"\nwarmups per mode: 1\nLoad: ",Sys.loadavg())
        for file in ("fixtures.jl","run.jl")
            println(io,file," sha256: ",bytes2hex(sha256(read(joinpath(@__DIR__,file)))))
        end
        for file in ("parallel_bug.jl","parallel_workspaces.jl","bug.jl","bug_workspaces.jl","contractions.jl","truncation.jl")
            println(io,file," sha256: ",bytes2hex(sha256(read(joinpath(dirname(pathof(TTNSolver)),file)))))
        end
    end
end

function tree_metadata(path,state,updated)
    tree = state.tree
    physical = Dict(n=>m for (m,n) in enumerate(tree.leaf_order))
    before,after = tree_ranks(state),tree_ranks(updated)
    open(path,"w") do io
        println(io,"node,left,right,parent,physical,dimension,input_rank,output_rank")
        for node in eachindex(tree.children)
            mode = get(physical,node,0)
            println(io,join((node,tree.children[node]...,tree.parent[node],mode,
                mode == 0 ? 0 : size(state.leaves[node],1),before[node],after[node]),','))
        end
    end
end

function benchmark_large(repetitions=3;cases=LARGE_CASES)
    repetitions >= 3 || error("use at least three repetitions")
    BLAS.get_num_threads() == 1 || error("set OPENBLAS_NUM_THREADS=1")
    data = joinpath(@__DIR__,"data"); mkpath(data)
    suffix = "threads$(Threads.nthreads(:default))"
    write_environment(joinpath(data,"environment_$suffix.txt"),repetitions)
    open(joinpath(data,"runs_$suffix.csv"),"w") do io
        println(io,"case,workers,threaded,repetition,seconds,bytes,gc_seconds,state_distance,density_difference,omitted_difference,ranks_equal,invariant_step_drift,input_rank,output_rank,augmented_rank,max_local_dimension,local_solves,skipped_leaf_solves,pretruncation_misses,truncation_met,discarded_bound,leaves,nodes,height,terms,full_dimension")
        flush(io)
        for name in cases
            println("Preparing $name"); flush(stdout)
            problem = large_problem(name)
            state,history = prepared_snapshot(problem)
            meta = case_metadata(problem,state)
            before = problem.observe(state)
            snapshot = copy(state)
            pre_misses = count(d -> !d.truncation.tolerance_met,history)
            println("Warming $name: ",meta,"; pre-misses=",pre_misses); flush(stdout)
            for threaded in (false,true)
                measured_step(problem,state;threaded)
            end
            for rep in 1:repetitions
                measured = Dict{Bool,Any}()
                for threaded in (isodd(rep) ? (false,true) : (true,false))
                    GC.gc()
                    measured[threaded] = @timed measured_step(problem,state;threaded)
                end
                a,da = measured[false].value; b,db = measured[true].value
                distance = ttn_distance(a,b)
                oa,ob = problem.observe(a),problem.observe(b)
                density_difference = norm(oa.density-ob.density)
                omitted_difference = maximum(abs.(da.omitted_block_norms-db.omitted_block_norms))
                @assert distance < 1e-11 && density_difference < 1e-11 && omitted_difference < 1e-12
                @assert da.ranks == db.ranks && da.truncation.tolerance_met == db.truncation.tolerance_met
                @assert da.local_solves == db.local_solves && da.skipped_leaf_solves == db.skipped_leaf_solves
                for threaded in (false,true)
                    m = measured[threaded]; _,d = m.value
                    observation = threaded ? ob : oa
                    println(io,join((name,Threads.nthreads(:default),threaded,rep,m.time,m.bytes,m.gctime,
                        distance,density_difference,omitted_difference,da.ranks==db.ranks,
                        abs(observation.invariant-before.invariant),meta.input_rank,maximum(d.ranks),
                        maximum(d.augmented_ranks),d.max_local_dimension,d.local_solves,d.skipped_leaf_solves,
                        pre_misses,d.truncation.tolerance_met,d.truncation.discarded_bound,
                        meta.leaves,meta.nodes,meta.height,meta.terms,meta.full_dimension),','))
                end
                if rep == 1 && Threads.nthreads(:default) == 1
                    tree_metadata(joinpath(data,"tree_$name.csv"),state,a)
                end
                flush(io)
                println("  $rep: serial ",round(measured[false].time;digits=3)," s; threaded ",
                    round(measured[true].time;digits=3)," s; delta ",distance); flush(stdout)
            end
            @assert ttn_distance(state,snapshot) < 1e-13
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_large(isempty(ARGS) ? 3 : parse(Int,only(ARGS)))
end
