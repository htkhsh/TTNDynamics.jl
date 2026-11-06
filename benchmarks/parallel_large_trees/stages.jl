include("run.jl")
# Reuse the checked isolated instrumentation; do not call the earlier runner.
include("../parallel_threads/stages.jl")

function benchmark_large_stages(repetitions=3;cases=("heom64","tfd16"))
    repetitions >= 3 || error("use at least three repetitions")
    BLAS.get_num_threads() == 1 || error("set OPENBLAS_NUM_THREADS=1")
    data = joinpath(@__DIR__,"data"); mkpath(data)
    suffix = "threads$(Threads.nthreads(:default))"
    environment = joinpath(data,"environment_stages_$suffix.txt")
    write_environment(environment,repetitions)
    open(environment,"a") do io
        for file in ("stages.jl","../parallel_threads/stages.jl","../parallel_bug_optimization/stages.jl")
            println(io,file," sha256: ",bytes2hex(sha256(read(joinpath(@__DIR__,file)))))
        end
    end
    open(joinpath(data,"stages_$suffix.csv"),"w") do io
        println(io,"case,workers,threaded,repetition,prepare,solve,assemble,finish,other,total,solve_sum,state_distance")
        open(joinpath(data,"nodes_$suffix.csv"),"w") do nodes
            println(nodes,"case,workers,threaded,repetition,node,parent,leaf,input_rank,local_dimension,skipped,solve_seconds")
            for name in cases
                println("Preparing phase timings: $name"); flush(stdout)
                problem = large_problem(name)
                state,_ = prepared_snapshot(problem)
                prepared = TTNSolver._prepare_parallel_bug_step(state,problem.operator,problem.conserved,LARGE_OPTIONS.maxrank)
                ranks = tree_ranks(prepared.old)
                resize!(TIMED_NODES,length(state.tree.children)); fill!(TIMED_NODES,0)
                reference,_ = measured_step(problem,state;threaded=false)
                for threaded in (false,true)
                    measured_step(problem,state;step=TIMED_STEP,threaded)
                    for rep in 1:repetitions
                        fill!(TIMED_STAGES,0); fill!(TIMED_NODES,0)
                        GC.gc()
                        m = @timed measured_step(problem,state;step=TIMED_STEP,threaded)
                        distance = ttn_distance(reference,m.value[1])
                        @assert distance < 1e-11 && tree_ranks(reference) == m.value[2].ranks
                        println(io,join((name,Threads.nthreads(:default),threaded,rep,TIMED_STAGES...,
                            m.time-sum(TIMED_STAGES),m.time,sum(TIMED_NODES),distance),','))
                        tree = state.tree
                        for node in eachindex(tree.children)
                            leaf = TTNSolver._isleaf(tree,node)
                            skipped = leaf && node != tree.root &&
                                size(prepared.old.leaves[node],1) == size(prepared.old.leaves[node],2)
                            println(nodes,join((name,Threads.nthreads(:default),threaded,rep,node,
                                tree.parent[node],leaf,ranks[node],length(prepared.initial[node]),
                                skipped,TIMED_NODES[node]),','))
                        end
                        flush(io); flush(nodes)
                    end
                end
                println("Stage timings complete: $name"); flush(stdout)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_large_stages()
end
