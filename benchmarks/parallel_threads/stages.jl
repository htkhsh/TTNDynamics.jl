# Reuse the existing checked phase markers on an isolated source copy.
# The production solver and main scaling runs contain no extra timers.
include("../parallel_bug_optimization/stages.jl")
const PARALLEL_SOURCE = read(joinpath(dirname(pathof(TTNSolver)),"parallel_bug.jl"),String)
const LOCAL_MARKER = "function _parallel_local_solve("
count(LOCAL_MARKER,PARALLEL_SOURCE) == 1 || error("local solve marker changed")
const TIMED_SOURCE = replace(PARALLEL_SOURCE,LOCAL_MARKER=>"function _original_parallel_local_solve(") * """

function _parallel_local_solve(prepared,operator,node,h,krylov_tol,krylovdim,krylov_maxiter)
    started = time_ns()
    value = _original_parallel_local_solve(prepared,operator,node,h,krylov_tol,krylovdim,krylov_maxiter)
    NODE_SOLVES[node] += (time_ns()-started)/1e9
    return value
end
"""
const TIMED = timed_module(:TimedParallel,TIMED_SOURCE)
Core.eval(TIMED,:(const NODE_SOLVES = Float64[]))
const TIMED_STEP = Base.invokelatest(getproperty,TIMED,:parallel_bug_step)
const TIMED_STAGES = Base.invokelatest(getproperty,TIMED,:STAGES)
const TIMED_NODES = Base.invokelatest(getproperty,TIMED,:NODE_SOLVES)

function benchmark_parallel_stages(repetitions=3;cases=CASES)
    repetitions >= 3 || error("use at least three repetitions")
    BLAS.get_num_threads() == 1 || error("set OPENBLAS_NUM_THREADS=1")
    data = joinpath(@__DIR__,"data"); mkpath(data)
    suffix = "threads$(Threads.nthreads(:default))"
    open(joinpath(data,"environment_stages_$suffix.txt"),"w") do io
        println(io,"Measured: ",Dates.now(),"\nJulia: ",VERSION,"\nCPU: ",first(Sys.cpu_info()).model)
        println(io,"Workers: ",Threads.nthreads(:default),"\nGC threads: ",Threads.ngcthreads(),
            "\nBLAS threads: ",BLAS.get_num_threads(),"\nRepetitions: ",repetitions,"\nLoad: ",Sys.loadavg())
        for file in ("run.jl","stages.jl","../augmented_threads/run.jl","../parallel_bug_optimization/stages.jl")
            println(io,file," sha256: ",bytes2hex(sha256(read(joinpath(@__DIR__,file)))))
        end
        println(io,"parallel_bug.jl sha256: ",bytes2hex(sha256(PARALLEL_SOURCE)))
    end
    open(joinpath(data,"stages_$suffix.csv"),"w") do io
        println(io,"case,workers,threaded,repetition,prepare,solve,assemble,finish,other,total,solve_sum,state_distance")
        open(joinpath(data,"nodes_$suffix.csv"),"w") do nodes
            println(nodes,"case,workers,threaded,repetition,node,parent,leaf,solve_seconds")
            for name in cases
                problem = scaling_problem(name)
                resize!(TIMED_NODES,length(problem.initial.tree.children)); fill!(TIMED_NODES,0)
                reference = trajectory(problem;step=parallel_bug_step,threaded=false)
                for threaded in (false,true)
                    trajectory(problem;step=TIMED_STEP,threaded)
                    for rep in 1:repetitions
                        fill!(TIMED_STAGES,0); fill!(TIMED_NODES,0)
                        GC.gc()
                        m = @timed trajectory(problem;step=TIMED_STEP,threaded)
                        distance = ttn_distance(reference.final_state,m.value.final_state)
                        @assert distance < 1e-11 && reference.ranks == m.value.ranks
                        println(io,join((name,Threads.nthreads(:default),threaded,rep,TIMED_STAGES...,
                            m.time-sum(TIMED_STAGES),m.time,sum(TIMED_NODES),distance),','))
                        tree = problem.initial.tree
                        for node in eachindex(tree.children)
                            println(nodes,join((name,Threads.nthreads(:default),threaded,rep,node,
                                tree.parent[node],TTNSolver._isleaf(tree,node),TIMED_NODES[node]),','))
                        end
                    end
                end
                flush(io); flush(nodes)
                println("Stage timings complete: $name"); flush(stdout)
            end
        end
    end
end
