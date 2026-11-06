# Sequential, checkpointed comparison of current BUG methods against tAMEn.
# This is a measurement script; it does not change package defaults or sources.
include("../holstein_longtime/driver.jl")
include("second_order.jl")
using .HolsteinLongTime, .SecondOrderBUG, LinearAlgebra, Serialization
const HLT=HolsteinLongTime

function specification(cfg,name)
    if name in ("tamen","reference","reference_refined")
        return merge(HLT.case_config(cfg,name),Dict("order"=>0))
    end
    method=startswith(name,"augmented") ? "augmented" : "parallel"
    order=endswith(name,"2") ? 2 : 1
    name in ("augmented","parallel","augmented2","parallel2") || error("unknown case: $name")
    h=Dict("augmented"=>0.05,"parallel"=>0.025,"augmented2"=>0.1,"parallel2"=>0.125)[name]
    return merge(HLT.case_config(cfg,method),Dict("order"=>order,"dt_fs"=>h))
end

function step(p,x,s,work)
    if s["order"]!=2
        return HLT.advance(p,x,s,work)
    end
    h=s["dt_fs"]
    return second_order_bug_step(x,p.operator,h;method=Symbol(s["method"]),
        atol=s["truncation_rate"]*h,maxrank=s["maxrank"],krylov_tol=s["krylov_tolerance"],
        conserved=p.conserved,correct_invariant=false,threaded=true,truncation=Symbol(s["truncation"]))
end

function recompare(label,name;end_fs=5.0,repeats=1,wall_seconds=600.0)
    occursin(r"^[A-Za-z0-9_-]+$",label) || error("invalid label")
    repeats>=1 && wall_seconds>0 || error("invalid repetition count/wall budget")
    cfg=HLT.load_config(joinpath(@__DIR__,"../holstein_longtime/config_50fs.toml"))
    cfg["time"]=Dict("end_fs"=>end_fs,"observation_fs"=>1.0,"checkpoint_fs"=>1.0)
    s=specification(cfg,name); h=s["dt_fs"]
    total=HLT.grid_count(end_fs,h); observe=HLT.grid_count(1.0,h)
    HLT.grid_count(end_fs,1.0)>=1 || error("at least 1 fs required")
    root=joinpath(@__DIR__,"output",label); dir=joinpath(root,name)
    ispath(dir) && error("output exists: $dir")
    mkpath(dir)
    for file in ("recompare.jl","second_order.jl")
        cp(joinpath(@__DIR__,file),joinpath(dir,file))
    end
    BLAS.set_num_threads(s["blas_threads"])
    HLT.atomic_toml(joinpath(dir,"metadata.toml"),Dict("sources"=>HLT.source_hashes(),
        "runtime"=>HLT.runtime_info(),"config"=>cfg,"spec"=>s,"repeats"=>repeats,
        "wall_seconds"=>wall_seconds,"scripts"=>Dict(file=>bytes2hex(HLT.sha256(read(joinpath(@__DIR__,file))))
            for file in ("recompare.jl","second_order.jl"))))
    started=time(); p=HLT.comparison_problem(cfg)
    work=HLT.TAMEnWorkspace(ComplexF64)
    init=s["method"]=="tamen" ? p.train : p.tree
    HLT.Random.seed!(cfg["execution"]["seed"]+1)
    warm=@elapsed step(p,deepcopy(init),s,work)
    println("WARM ",name," ",warm," s"); flush(stdout)
    summaries=NamedTuple[]
    for repetition in 1:repeats
        time()-started<wall_seconds || break
        rep=joinpath(dir,"rep$repetition"); mkpath(rep)
        x=deepcopy(init); seconds=0.0; bytes=0; gc=0.0; laststep=0; peak=maximum(HLT.ranks_of(x)); misses=0
        GC.gc()
        open(joinpath(rep,"steps.csv"),"w") do steps
            open(joinpath(rep,"observables.csv"),"w") do obs
                println(steps,"step,time_fs,seconds,cumulative_seconds,allocated_bytes,gc_seconds,maxrank,augmented_rank,max_local_dimension,truncation_met,discarded_bound")
                println(obs,HLT.observation_header(cfg["model"]["sites"]))
                HLT.record_observation(obs,p,x,0,h)
                serialize(joinpath(rep,"time_0.jls"),x)
                for j in 1:total
                    time()-started<wall_seconds || break
                    HLT.Random.seed!(cfg["execution"]["seed"]+j)
                    m=@timed step(p,x,s,work); x,d=m.value
                    seconds+=m.time; bytes+=m.bytes; gc+=m.gctime; laststep=j
                    rank=maximum(HLT.ranks_of(x)); peak=max(peak,rank)
                    istt=s["method"]=="tamen"
                    met=istt ? "NA" : d.truncation.tolerance_met
                    misses+=(met===false)
                    augmented=istt ? rank : maximum(d.augmented_ranks)
                    dimension=istt ? 0 : d.max_local_dimension
                    discarded=istt ? "NA" : d.truncation.discarded_bound
                    println(steps,join((j,j*h,m.time,seconds,m.bytes,m.gctime,rank,augmented,dimension,met,discarded),',')); flush(steps)
                    if j%observe==0
                        HLT.record_observation(obs,p,x,j,h)
                        serialize(joinpath(rep,"time_$(j÷observe).jls"),x)
                        println(name," rep=",repetition," t=",j*h," fs; seconds=",seconds," rank=",rank," misses=",misses); flush(stdout)
                    end
                end
            end
        end
        serialize(joinpath(rep,"latest.jls"),(;state=x,step=laststep,time_fs=laststep*h,seconds))
        push!(summaries,(;repetition,complete=laststep==total,time_fs=laststep*h,seconds,bytes,gc,
            peak_rank=peak,truncation_misses=misses))
        HLT.write_rows(joinpath(dir,"runs.csv"),summaries)
        println(laststep==total ? "COMPLETE " : "PARTIAL ",name," ",last(summaries)); flush(stdout)
        laststep==total || break
    end
    HLT.atomic_toml(joinpath(dir,"status.toml"),Dict("wall_seconds"=>time()-started,
        "warmup_seconds"=>warm,"completed_repetitions"=>count(r->r.complete,summaries)))
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)>=2 || error("Usage: recompare.jl LABEL CASE [end_fs=5] [repeats=1] [wall_seconds=600]")
    recompare(ARGS[1],ARGS[2];end_fs=length(ARGS)>=3 ? parse(Float64,ARGS[3]) : 5.0,
        repeats=length(ARGS)>=4 ? parse(Int,ARGS[4]) : 1,
        wall_seconds=length(ARGS)>=5 ? parse(Float64,ARGS[5]) : 600.0)
end
