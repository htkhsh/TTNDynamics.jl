# Benchmark-only source snapshots preserve the implementation before this
# optimization; no timing switches or legacy paths enter the public solvers.
include("common.jl")
using Test, Statistics

for (name,grouped) in ((:BaselineBUG,false),(:GroupedOnlyBUG,true))
    @eval module $name
        using LinearAlgebra
        import TTNSolver
        const KrylovKit = TTNSolver.KrylovKit
    end
    mod = getfield(@__MODULE__,name)
    for binding in (:TTNTensor,:SumOfProducts,:_compression_options,:_check_operator,
            :orthogonalize!,:_canonical_conserved,:tree_ranks,:_isleaf,
            :_child_product,:_local_action,:_orth_columns,:truncate!,
            :_subtree_overlaps,:enrich_basis)
        Core.eval(mod, :(const $binding = TTNSolver.$binding))
    end
    for file in ("contractions.jl","bug_workspaces.jl","bug.jl","parallel_workspaces.jl","parallel_bug.jl")
        path = grouped && file in ("bug_workspaces.jl","parallel_workspaces.jl") ?
            joinpath(dirname(pathof(TTNSolver)),file) : joinpath(@__DIR__,"data","optimization_baseline",file)
        Base.include(mod,path)
    end
end

function measure_steps()
    label = isempty(ARGS) ? "optimized_steps" : ARGS[1]
    selected_method = length(ARGS) >= 2 ? Symbol(ARGS[2]) : :all
    selected_method in (:all,:augmented,:parallel) || error("unknown method")
    dir = joinpath(@__DIR__,"data",label)
    isdir(dir) && error("Output exists: $dir")
    mkpath(dir)
    p = example_problem()
    snapshot = deserialize(joinpath(@__DIR__,"data","augmented_h005","latest.jls"))
    @assert snapshot.time_fs == 1.0
    x = snapshot.state
    open(joinpath(dir,"metadata.txt"),"w") do io
        println(io,"Same input for all: augmented_h005 at 1 fs; cap32; three warmed trials")
        source_metadata(io)
        for file in sort(readdir(joinpath(@__DIR__,"data","optimization_baseline");join=true))
            println(io,"baseline ",basename(file)," sha256: ",bytes2hex(sha256(read(file))))
        end
    end
    open(joinpath(dir,"timings.csv"),"w") do io
        println(io,"method,implementation,trial,seconds,gc_seconds,allocated_bytes,state_difference,density_difference")
        for (method,h,atol) in ((:augmented,.05,5e-5),(:parallel,.02,2e-5))
            selected_method in (:all,method) || continue
            options = (;atol,maxrank=32,conserved=p.conserved,correct_invariant=false,threaded=true)
            symbol = Symbol(method,"_bug_step")
            variants = [("baseline",getfield(BaselineBUG,symbol)),
                        ("grouped_only",getfield(GroupedOnlyBUG,symbol)),
                        ("optimized",getfield(TTNSolver,symbol))]
            reference = nothing
            for (label,step) in variants
                println("warm ",method," ",label); flush(stdout)
                y,_ = step(x,p.operator,h;options...)
                label == "baseline" && (reference=y)
                GC.gc()
            end
            rng = MersenneTwister(104)
            for trial in 1:3, (label,step) in shuffle(rng,variants)
                GC.gc()
                measured = @timed step(x,p.operator,h;options...)
                y = measured.value[1]
                distance = ttn_distance(reference,y)/norm(reference)
                density = norm(TTNDynamics.root_density_matrix(reference,p.system)-
                               TTNDynamics.root_density_matrix(y,p.system))
                @test distance < 1e-9
                println(io,join((method,label,trial,measured.time,measured.gctime,measured.bytes,distance,density),','))
                flush(io)
                println(method," ",label," trial=",trial," seconds=",measured.time,
                        " GB=",measured.bytes/1e9," distance=",distance); flush(stdout)
            end
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    measure_steps()
end
