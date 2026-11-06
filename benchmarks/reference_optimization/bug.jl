# Load the selected package before dependent projects; --project may point to
# an isolated pre-change TTNSolver snapshot. No method replacement is used.
using TTNSolver, LinearAlgebra, Serialization, Statistics, TOML
const TS=TTNSolver
BLAS.set_num_threads(1)
const FIXTURE=get(ENV,"REFERENCE_OPT_FIXTURE",joinpath(@__DIR__,"data","problem.jls"))
if !isfile(FIXTURE)
    include("../holstein_ttn_tamen/common.jl")
    HC=HolsteinTAMEnComparison
    p=HC.HLT.comparison_problem(HC.configuration())
    mkpath(dirname(FIXTURE))
    serialize(FIXTURE,(;p.tree,p.conserved,p.operator))
end

function main()
    label=get(ENV,"REFERENCE_OPT_LABEL","optimized")
    out=get(ENV,"REFERENCE_OPT_OUTPUT",joinpath(@__DIR__,"data"))
    mkpath(out)
    p=deserialize(FIXTURE)
    snapshot=get(ENV,"REFERENCE_OPT_STATE","")
    initial=isempty(snapshot) ? p.tree : deserialize(snapshot)
    steps=isempty(snapshot) ? 4 : 1
    rows=Dict[]
    for (name,step) in (("augmented",augmented_bug_step),("parallel",parallel_bug_step))
        function trajectory()
            x=copy(initial)
            for _ in 1:steps
                x,_=step(x,p.operator,0.05;atol=5e-7,maxrank=64,
                    conserved=p.conserved,correct_invariant=false,threaded=true)
            end
            x
        end
        trajectory() # compile and warm BLAS
        for repeat in 1:3
            GC.gc()
            m=@timed trajectory()
            x=m.value
            refpath=joinpath(out,"baseline_$(name).jls")
            label=="baseline" && serialize(refpath,x)
            difference=isfile(refpath) ? ttn_distance(x,deserialize(refpath))/norm(x) : NaN
            trace_error=abs(dot(p.conserved,x)-dot(p.conserved,initial))
            @assert trace_error<1e-10
            row=Dict("method"=>name,"repeat"=>repeat,"seconds"=>m.time,"bytes"=>m.bytes,
                "maxrank"=>maximum(tree_ranks(x)),"relative_difference"=>difference,
                "trace_error"=>trace_error)
            push!(rows,row);println(row);flush(stdout)
            # Both trajectories use the same physical truncation budget.
            @assert isnan(difference) || difference<2steps*5e-7
        end
    end
    open(joinpath(out,"bug_$label.toml"),"w") do io
        TOML.print(io,Dict("label"=>label,"solver_source"=>pathof(TTNSolver),
            "threads"=>Threads.nthreads(:default),"blas_threads"=>BLAS.get_num_threads(),
            "elapsed_fs"=>0.05steps,"initial_snapshot"=>snapshot,
            "dt_fs"=>0.05,"truncation_per_step"=>5e-7,
            "krylov_tol"=>1e-12,"runs"=>rows))
    end
end
main()
