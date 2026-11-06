# Timing exists only in an isolated source copy, never in production solves.
# Each node owns its NODE_SOLVES slot; STAGES is written only by the caller
# before spawning or after joining all tasks. Checked markers fail loudly
# after a source refactor. Ordinary scaling trials use uninstrumented code.
module TimedAugmented
using LinearAlgebra
import TTNSolver
for binding in (:TTNTensor,:SumOfProducts,:_compression_options,:_check_operator,:_check_truncation_weighting,
        :orthogonalize!,:_canonical_conserved,:tree_ranks,:_upward_operators,
        :_isleaf,:_complement_matrix,:_complement_operator,:_parent_product,
        :_leaf_rhs_workspace,:_core_rhs_workspace,:_orth_columns,:_local_action,
        :_project_children,:_basis_matrix,:_subtree_operator,:truncate!,:_subtree_overlaps,
        :_complement_operators,:_leaf_operators,:_subtree_operators)
    @eval const $binding = TTNSolver.$binding
end
const KrylovKit = TTNSolver.KrylovKit
const STAGES = zeros(3)
const NODE_SOLVES = Float64[]
source = read(joinpath(dirname(pathof(TTNSolver)),"bug.jl"),String)
insertions = (
    "    prepared = _prepare_bug_step(state,operator,conserved,maxrank)" =>
    "    phase_started = time_ns()\n    prepared = _prepare_bug_step(state,operator,conserved,maxrank)\n    STAGES[1] += (time_ns()-phase_started)/1e9\n    if length(NODE_SOLVES) != length(state.tree.children)\n        resize!(NODE_SOLVES,length(state.tree.children)); fill!(NODE_SOLVES,0)\n    end",
    "    (;old,c,modes) = prepared" =>
    "    phase_started = time_ns()\n    (;old,c,modes) = prepared",
    "    function integrate(node, rhs, x)" =>
    "    function integrate(node, rhs, x)\n        solve_started = time_ns()",
    "        return restored" =>
    "        NODE_SOLVES[node] += (time_ns()-solve_started)/1e9\n        return restored",
    "    return _finish_bug_step!(result,old,c,stats;atol,maxrank,correct_invariant,truncation)" =>
    "    phase_now = time_ns()\n    STAGES[2] += (phase_now-phase_started)/1e9\n    final = _finish_bug_step!(result,old,c,stats;atol,maxrank,correct_invariant,truncation)\n    STAGES[3] += (time_ns()-phase_now)/1e9\n    return final")
for (marker,replacement) in insertions
    length(findall(marker,source)) == 1 || error("timing marker changed: $marker")
    global source = replace(source,marker=>replacement)
end
include_string(@__MODULE__,source,"TimedAugmented.jl")
end

function benchmark_stages(repetitions=3;cases=CASES)
    repetitions >= 3 || error("use at least three repetitions")
    BLAS.get_num_threads() == 1 || error("set OPENBLAS_NUM_THREADS=1")
    data = joinpath(@__DIR__,"data"); mkpath(data)
    suffix = "threads$(Threads.nthreads(:default))"
    open(joinpath(data,"environment_stages_$suffix.txt"),"w") do io
        println(io,"Measured: ",Dates.now(),"\nJulia: ",VERSION,"\nCPU: ",first(Sys.cpu_info()).model)
        println(io,"Workers: ",Threads.nthreads(:default),"\nGC threads: ",Threads.ngcthreads(),
            "\nBLAS threads: ",BLAS.get_num_threads(),"\nRepetitions: ",repetitions,"\nLoad: ",Sys.loadavg())
        for file in ("run.jl","stages.jl")
            println(io,file," sha256: ",bytes2hex(sha256(read(joinpath(@__DIR__,file)))))
        end
        println(io,"bug.jl sha256: ",bytes2hex(sha256(read(joinpath(dirname(pathof(TTNSolver)),"bug.jl")))))
    end
    open(joinpath(data,"stages_$suffix.csv"),"w") do io
        println(io,"case,workers,threaded,repetition,prepare,update,finish,other,total,solve_sum,root_solve,state_distance")
        open(joinpath(data,"nodes_$suffix.csv"),"w") do nodes
            println(nodes,"case,workers,threaded,repetition,node,parent,leaf,solve_seconds")
            for name in cases
                problem = scaling_problem(name)
                reference = trajectory(problem;threaded=false)
                for threaded in (false,true)
                    trajectory(problem;step=TimedAugmented.augmented_bug_step,threaded)
                    for rep in 1:repetitions
                        fill!(TimedAugmented.STAGES,0); fill!(TimedAugmented.NODE_SOLVES,0)
                        GC.gc()
                        m = @timed trajectory(problem;step=TimedAugmented.augmented_bug_step,threaded)
                        distance = ttn_distance(reference.final_state,m.value.final_state)
                        @assert distance < 1e-11 && reference.ranks == m.value.ranks
                        phases,solves = TimedAugmented.STAGES,TimedAugmented.NODE_SOLVES
                        println(io,join((name,Threads.nthreads(:default),threaded,rep,phases...,
                            m.time-sum(phases),m.time,sum(solves),solves[problem.initial.tree.root],distance),','))
                        tree = problem.initial.tree
                        for node in eachindex(tree.children)
                            println(nodes,join((name,Threads.nthreads(:default),threaded,rep,node,
                                tree.parent[node],TTNSolver._isleaf(tree,node),solves[node]),','))
                        end
                    end
                end
                flush(io); flush(nodes)
                println("Stage timings complete: $name"); flush(stdout)
            end
        end
    end
end
