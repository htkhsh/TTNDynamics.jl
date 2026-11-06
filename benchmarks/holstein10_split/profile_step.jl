include("common.jl")
include("../augmented_threads/stages.jl")
using Profile

function profile_step()
    label = isempty(ARGS) ? "profile_augmented" : ARGS[1]
    dir = joinpath(@__DIR__,"data",label); mkpath(dir)
    p = example_problem()
    checkpoint = deserialize(joinpath(@__DIR__,"data","augmented_h005","latest.jls"))
    x = checkpoint.state
    options = (;atol=5e-5,maxrank=32,conserved=p.conserved,correct_invariant=false,threaded=true)
    open(joinpath(dir,"metadata.txt"),"w") do io
        println(io,"Input: augmented_h005 at ",checkpoint.time_fs," fs; step h=0.05 fs")
        println(io,"terms: ",length(p.operator.terms),"; ranks: ",tree_ranks(x))
        source_metadata(io)
    end
    println("Production warm-up"); flush(stdout)
    reference,_ = augmented_bug_step(x,p.operator,.05;options...)
    println("Instrumented warm-up"); flush(stdout)
    TimedAugmented.augmented_bug_step(x,p.operator,.05;options...)
    fill!(TimedAugmented.STAGES,0); fill!(TimedAugmented.NODE_SOLVES,0)
    Profile.init(n=20_000_000,delay=0.002)
    Profile.clear(); GC.gc()
    println("Profiled step"); flush(stdout)
    measurement = Profile.@profile @timed TimedAugmented.augmented_bug_step(x,p.operator,.05;options...)
    result,diagnostics = measurement.value
    distance = ttn_distance(reference,result)
    @assert distance < 1e-10
    open(joinpath(dir,"stages.csv"),"w") do io
        println(io,"prepare,update,finish,total,gc_seconds,allocated_bytes,solve_sum,max_node_solve,root_solve,state_distance")
        println(io,join((TimedAugmented.STAGES...,measurement.time,measurement.gctime,measurement.bytes,
            sum(TimedAugmented.NODE_SOLVES),maximum(TimedAugmented.NODE_SOLVES),
            TimedAugmented.NODE_SOLVES[x.tree.root],distance),','))
    end
    open(joinpath(dir,"nodes.csv"),"w") do io
        println(io,"node,parent,leaf,input_rank,augmented_rank,solve_seconds")
        ranks = tree_ranks(x)
        for node in eachindex(ranks)
            println(io,join((node,x.tree.parent[node],TTNSolver._isleaf(x.tree,node),ranks[node],
                            diagnostics.augmented_ranks[node],TimedAugmented.NODE_SOLVES[node]),','))
        end
    end
    for cframes in (false,true)
        open(joinpath(dir,"profile_$(cframes ? "all" : "julia").txt"),"w") do io
            Profile.print(io;format=:flat,sortedby=:count,mincount=20,C=cframes)
        end
    end
    println("STAGES ",TimedAugmented.STAGES," total=",measurement.time,
            " allocation_GB=",measurement.bytes/1e9," distance=",distance); flush(stdout)
end

profile_step()
