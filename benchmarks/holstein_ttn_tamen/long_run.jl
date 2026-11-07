# Long-time run of one case: julia long_run.jl NAME END_FS BUDGET_SECONDS.
# Logs every interval to output/long20/NAME$LONG_TAG/log.csv and saves states
# at integer fs (git-ignored). LONG_MAXRANK and LONG_BLAS override the spec;
# LONG_SITES sets the number of sites and LONG_ROOT the output folder.
# LONG_TOL, LONG_DT and LONG_TRUNC override the tolerance, step and
# truncation_rtol; meta.toml records the settings and source hashes.
include(joinpath(@__DIR__,"common.jl"))
using .HolsteinTAMEnComparison, LinearAlgebra, Random, Serialization
const H=HolsteinTAMEnComparison; const HLT=H.HLT; const TS=HLT.TTNSolver
function main()
    name=ARGS[1]; endfs=parse(Float64,ARGS[2]); budget=parse(Float64,ARGS[3])
    out=joinpath(@__DIR__,"output",get(ENV,"LONG_ROOT","long20"),name*get(ENV,"LONG_TAG","")); mkpath(out)
    sites=parse(Int,get(ENV,"LONG_SITES","10"))
    cfg=H.configuration(;end_fs=endfs,sites)
    s=H.specification(cfg,name)
    haskey(ENV,"LONG_MAXRANK") && (s["maxrank"]=parse(Int,ENV["LONG_MAXRANK"]))
    haskey(ENV,"LONG_BLAS") && (s["blas_threads"]=parse(Int,ENV["LONG_BLAS"]))
    haskey(ENV,"LONG_TOL") && (s["tolerance"]=parse(Float64,ENV["LONG_TOL"]))
    haskey(ENV,"LONG_DT") && (s["dt_fs"]=parse(Float64,ENV["LONG_DT"]))
    haskey(ENV,"LONG_TRUNC") && (s["truncation_rtol"]=parse(Float64,ENV["LONG_TRUNC"]))
    HLT.atomic_toml(joinpath(out,"meta.toml"),Dict("spec"=>s,"sites"=>sites,"end_fs"=>endfs,
        "julia_threads"=>Threads.nthreads(),"sources"=>HLT.source_hashes(),
        "env"=>Dict(k=>v for (k,v) in ENV if startswith(k,"LONG_"))))
    q=H.ordered_problem(HLT.comparison_problem(cfg),H.run_order(name);
        bath_layout=Symbol(get(s,"bath_layout","balanced")))
    BLAS.set_num_threads(s["blas_threads"])
    istree=s["method"]=="tree_tamen"
    work=istree ? TS.TAMEnWorkspace() : HLT.TAMEnWorkspace(ComplexF64)
    x0=istree ? q.tree : q.train
    y,d=H.advance(q,deepcopy(x0),s,work,nothing); istree && H.advance(q,y,s,work,d.warm_start)  # warm-up
    x=deepcopy(x0); hint=nothing; total=0.0; GC.gc()
    steps=HLT.grid_count(endfs,s["dt_fs"]); per=HLT.grid_count(1.0,s["dt_fs"])
    open(joinpath(out,"log.csv"),"w") do io
        println(io,"time_fs,cumulative_seconds,step_seconds,maxrank,matvecs,sweeps,gc_seconds,bytes")
        for j in 1:steps
            HLT.Random.seed!(cfg["execution"]["seed"]+j)
            m=@timed H.advance(q,x,s,work,hint); x,d=m.value
            istree && (hint=d.warm_start)
            total+=m.time
            rank=maximum(HLT.ranks_of(x))
            println(io,join((j*s["dt_fs"],total,m.time,rank,istree ? d.matvecs : 0,istree ? d.sweeps : 0,m.gctime,m.bytes),','))
            flush(io)
            if j%per==0
                serialize(joinpath(out,"time_$(j÷per).jls"),x)
                println(name," t=$(j*s["dt_fs"]) fs cum=$(round(total,digits=2)) s rank=$rank"); flush(stdout)
            end
            total>budget && (println(name," stopped by budget at $(j*s["dt_fs"]) fs"); break)
        end
    end

end
main()
