# Finish one softly stopped repetition without changing its parameters.
# Original logs/checkpoint are copied before appending. Run only after its
# original Julia process has exited; no concurrently active writer is allowed.
include("recompare.jl")
using TOML

function readrows(path)
    lines=readlines(path); keys=split(first(lines),',')
    [Dict(zip(keys,split(line,','))) for line in lines[2:end]]
end

function resume_case(label,name,wall_seconds)
    all(occursin(r"^[A-Za-z0-9_-]+$",x) for x in (label,name)) || error("invalid label/name")
    isfinite(wall_seconds) && wall_seconds>0 || error("invalid wall budget")
    dir=joinpath(@__DIR__,"output",label,name)
    meta=TOML.parsefile(joinpath(dir,"metadata.toml")); cfg=meta["config"]; s=meta["spec"]
    meta["sources"]==HLT.source_hashes() || error("source/dependency mismatch")
    all(bytes2hex(HLT.sha256(read(joinpath(@__DIR__,file))))==sha for (file,sha) in meta["scripts"]) || error("script changed")
    BLAS.set_num_threads(s["blas_threads"])
    HLT.runtime_info()==meta["runtime"] || error("runtime mismatch")
    runs=readrows(joinpath(dir,"runs.csv")); previous=last(runs)
    previous["complete"]=="false" || error("last repetition already complete")
    repetition=parse(Int,previous["repetition"]); rep=joinpath(dir,"rep$repetition")
    saved=deserialize(joinpath(rep,"latest.jls")); rows=readrows(joinpath(rep,"steps.csv"))
    all(parse(Int,r["step"])==j for (j,r) in enumerate(rows)) || error("step gap")
    length(rows)==saved.step && parse(Float64,last(rows)["cumulative_seconds"])==saved.seconds || error("checkpoint/log mismatch")
    backup=joinpath(rep,"before_resume"); ispath(backup) && error("resume already attempted")
    mkpath(backup)
    for file in ("steps.csv","observables.csv","latest.jls")
        cp(joinpath(rep,file),joinpath(backup,file))
    end
    cp(joinpath(dir,"runs.csv"),joinpath(backup,"runs.csv"))
    cp(joinpath(dir,"status.toml"),joinpath(backup,"status.toml"))
    cp(@__FILE__,joinpath(backup,"recompare_resume.jl"))
    started=time(); p=HLT.comparison_problem(cfg); work=HLT.TAMEnWorkspace(ComplexF64)
    h=s["dt_fs"]; total=HLT.grid_count(cfg["time"]["end_fs"],h); observe=HLT.grid_count(1.0,h)
    x=saved.state; seconds=saved.seconds; laststep=saved.step
    HLT.Random.seed!(cfg["execution"]["seed"]+laststep+1)
    warm=@elapsed step(p,deepcopy(x),s,work)
    GC.gc()
    println("RESUME ",name," from ",laststep*h," fs; warmup=",warm); flush(stdout)
    open(joinpath(rep,"steps.csv"),"a") do steps
        open(joinpath(rep,"observables.csv"),"a") do obs
            for j in laststep+1:total
                time()-started<wall_seconds || break
                HLT.Random.seed!(cfg["execution"]["seed"]+j)
                m=@timed step(p,x,s,work); x,d=m.value; seconds+=m.time; laststep=j
                istt=s["method"]=="tamen"; rank=maximum(HLT.ranks_of(x))
                met=istt ? "NA" : d.truncation.tolerance_met
                augmented=istt ? rank : maximum(d.augmented_ranks)
                dimension=istt ? 0 : d.max_local_dimension
                discarded=istt ? "NA" : d.truncation.discarded_bound
                println(steps,join((j,j*h,m.time,seconds,m.bytes,m.gctime,rank,augmented,dimension,met,discarded),',')); flush(steps)
                if j%observe==0
                    HLT.record_observation(obs,p,x,j,h)
                    serialize(joinpath(rep,"time_$(j÷observe).jls"),x)
                    println(name," resumed t=",j*h," fs; seconds=",seconds," rank=",rank); flush(stdout)
                end
            end
        end
    end
    serialize(joinpath(rep,"latest.jls"),(;state=x,step=laststep,time_fs=laststep*h,seconds))
    rows=readrows(joinpath(rep,"steps.csv"))
    rows2=[(;repetition=parse(Int,r["repetition"]),complete=r["complete"]=="true",time_fs=parse(Float64,r["time_fs"]),
        seconds=parse(Float64,r["seconds"]),bytes=parse(Int,r["bytes"]),gc=parse(Float64,r["gc"]),
        peak_rank=parse(Int,r["peak_rank"]),truncation_misses=parse(Int,r["truncation_misses"])) for r in runs[1:end-1]]
    push!(rows2,(;repetition,complete=laststep==total,time_fs=laststep*h,seconds,
        bytes=sum(parse(Int,r["allocated_bytes"]) for r in rows),gc=sum(parse(Float64,r["gc_seconds"]) for r in rows),
        peak_rank=maximum(parse(Int,r["maxrank"]) for r in rows),
        truncation_misses=count(r->r["truncation_met"]=="false",rows)))
    HLT.write_rows(joinpath(dir,"runs.csv"),rows2)
    HLT.atomic_toml(joinpath(dir,"resume.toml"),Dict("first_step"=>saved.step,"last_step"=>laststep,
        "wall_seconds"=>time()-started,"warmup_seconds"=>warm,"complete"=>laststep==total,
        "script_sha256"=>bytes2hex(HLT.sha256(read(@__FILE__)))))
    status=TOML.parsefile(joinpath(backup,"status.toml"))
    status["wall_seconds"]+=time()-started
    status["warmup_seconds"]+=warm
    status["completed_repetitions"]=count(r->r.complete,rows2)
    HLT.atomic_toml(joinpath(dir,"status.toml"),status)
    println(laststep==total ? "COMPLETE " : "PARTIAL ",name," ",last(rows2)); flush(stdout)
end

length(ARGS)==3 || error("Usage: recompare_resume.jl LABEL CASE WALL_SECONDS")
resume_case(ARGS[1],ARGS[2],parse(Float64,ARGS[3]))
