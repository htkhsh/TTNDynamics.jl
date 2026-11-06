module HolsteinTAMEnComparison
include("../holstein_longtime/driver.jl")
using .HolsteinLongTime, LinearAlgebra, Serialization, TOML
const HLT=HolsteinLongTime
const TS=HLT.TTNSolver

function configuration(;sites=10,local_size=5,end_fs=5.0)
    cfg=HLT.load_config(joinpath(@__DIR__,"../holstein_longtime/config_50fs.toml"))
    cfg["model"]["sites"]=sites
    cfg["model"]["hierarchy_local_size"]=local_size
    cfg["time"]=Dict("end_fs"=>end_fs,"observation_fs"=>1.0,"checkpoint_fs"=>1.0)
    return cfg
end

function specification(cfg,name)
    if name in ("reference","reference_refined","tt","tt_serial")
        s=HLT.case_config(cfg,startswith(name,"tt") ? "tamen" : name)
        name=="tt_serial" && (s["blas_threads"]=1)
        return s
    end
    name in ("tree","tree_tight","tree_loose") || error("unknown run $name")
    return Dict{String,Any}("method"=>"tree_tamen","dt_fs"=>0.5,"points"=>8,
        "tolerance"=>(name=="tree_tight" ? 1e-5 : name=="tree_loose" ? 5e-4 : 1e-4),
        "maxrank"=>96,"sweeps"=>20,"kick_rank"=>4,"blas_threads"=>1,
        "als_sweep"=>"etter","threaded"=>true)
end

function advance(p,x,s,work,hint)
    s["method"]=="tamen" && return HLT.advance(p,x,s,work)
    return TS.tamen_step(x,p.operator,s["dt_fs"];workspace=work,
        time_points=s["points"],rtol=s["tolerance"],maxrank=s["maxrank"],
        maxsweeps=s["sweeps"],kickrank=s["kick_rank"],conserved=p.conserved,
        als_sweep=Symbol(s["als_sweep"]),threaded=s["threaded"],
        warm_start=hint,store_warm_start=true,estimate_time_error=true)
end

function run_case(name;root=joinpath(@__DIR__,"output","five_fs"),wall_seconds=1200.0,repeats=1,
                  cfg=configuration(),settings=specification(cfg,name))
    s=copy(settings)
    isfinite(wall_seconds) && wall_seconds>=0 && repeats>=1 || error("invalid budget/repetitions")
    dir=joinpath(root,name);ispath(dir) && error("output already exists: $dir")
    mkpath(dir)
    start=time();setup=0.0;warm=0.0
    BLAS.set_num_threads(s["blas_threads"])
    HLT.atomic_toml(joinpath(dir,"metadata.toml"),Dict("sources"=>HLT.source_hashes(),
        "script_hash"=>bytes2hex(HLT.sha256(read(@__FILE__))),"runtime"=>HLT.runtime_info(),
        "config"=>cfg,"spec"=>s,"wall_seconds"=>wall_seconds))
    status=Dict{String,Any}("phase"=>"setup","failure"=>"","time_fs"=>0.0,
        "warmup_seconds"=>0.0,"setup_seconds"=>0.0,"wall_seconds"=>0.0,"completed_repetitions"=>0)
    function save_status(phase)
        status["phase"]=phase;status["wall_seconds"]=time()-start
        HLT.atomic_toml(joinpath(dir,"status.toml"),status)
    end
    save_status("setup")
    if time()-start>=wall_seconds
        save_status("timeout");return
    end
    p=try
        built=@timed HLT.comparison_problem(cfg)
        setup=built.time;built.value
    catch error
        status["failure"]=sprint(showerror,error);save_status("setup_failed");rethrow()
    end
    status["setup_seconds"]=setup
    if time()-start>=wall_seconds
        save_status("timeout");return
    end
    BLAS.set_num_threads(s["blas_threads"])
    istree=s["method"]=="tree_tamen"
    work=istree ? TS.TAMEnWorkspace() : HLT.TAMEnWorkspace(ComplexF64)
    initial=istree ? p.tree : p.train
    HLT.Random.seed!(cfg["execution"]["seed"]+1)
    save_status("warmup")
    warm=try
        @elapsed advance(p,deepcopy(initial),s,work,nothing)
    catch error
        status["failure"]=sprint(showerror,error);save_status("warmup_failed");rethrow()
    end
    status["warmup_seconds"]=warm
    if time()-start>=wall_seconds
        save_status("timeout");return
    end
    println("WARM ",name," ",warm," s; setup ",setup," s");flush(stdout)
    total=HLT.grid_count(cfg["time"]["end_fs"],s["dt_fs"])
    observe=HLT.grid_count(1.0,s["dt_fs"])
    summaries=NamedTuple[]
    for repetition in 1:repeats
        time()-start<wall_seconds || break
        status["time_fs"]=0.0;status["repetition"]=repetition
        save_status("running")
        repdir=joinpath(dir,"rep$repetition");mkpath(repdir)
        x=deepcopy(initial);hint=nothing;seconds=0.0;bytes=0;gc=0.0;laststep=0;peak=1;failure=""
        GC.gc()
        open(joinpath(repdir,"steps.csv"),"w") do io
            println(io,"step,time_fs,seconds,cumulative_seconds,bytes,gc_seconds,maxrank,sweeps,global_checks,relative_bound,global_seconds,addition_seconds,weight_qr_seconds,svd_seconds,basis_update_seconds,final_qr_seconds,time_defect")
            serialize(joinpath(repdir,"time_0.jls"),x)
            for j in 1:total
                time()-start<wall_seconds || break
                HLT.Random.seed!(cfg["execution"]["seed"]+j)
                measured=try
                    @timed advance(p,x,s,work,hint)
                catch error
                    error isa InterruptException && rethrow()
                    failure=sprint(showerror,error)
                    println("FAILED ",name," at accepted t=",laststep*s["dt_fs"],": ",failure);flush(stdout)
                    break
                end
                x,d=measured.value
                if istree
                    @assert d.converged && d.history[end].global_residual_checked
                    hint=d.warm_start
                end
                seconds+=measured.time;bytes+=measured.bytes;gc+=measured.gctime;laststep=j
                rank=maximum(HLT.ranks_of(x));peak=max(peak,rank)
                status["time_fs"]=j*s["dt_fs"]
                parts=istree ? values(d.timings.residual_global_parts) : ntuple(_->0.0,5)
                println(io,join((j,j*s["dt_fs"],measured.time,seconds,measured.bytes,measured.gctime,rank,
                    istree ? d.sweeps : 0,istree ? d.global_checks : 0,
                    istree ? d.relative_residual_bound : NaN,
                    istree ? d.timings.residual_global_seconds : 0.,parts...,
                    istree ? d.time_error.absolute : NaN),','));flush(io)
                serialize(joinpath(repdir,"latest.jls"),(;state=x,step=j,time_fs=j*s["dt_fs"],seconds))
                if j%observe==0
                    serialize(joinpath(repdir,"time_$(j÷observe).jls"),x)
                    println(name," t=",j*s["dt_fs"]," fs; seconds=",seconds," rank=",rank);flush(stdout)
                end
                save_status("running")
            end
        end
        push!(summaries,(;repetition,complete=laststep==total,time_fs=laststep*s["dt_fs"],seconds,bytes,gc,peak_rank=peak))
        HLT.write_rows(joinpath(dir,"runs.csv"),summaries)
        status["failure"]=failure
        status["completed_repetitions"]=count(r->r.complete,summaries)
        save_status(laststep==total ? "complete" : isempty(failure) ? "timeout" : "solve_failed")
        println(last(summaries));flush(stdout)
        laststep==total || break
    end
end

function csvrows(path)
    lines=readlines(path);header=split(first(lines),',')
    return [Dict(zip(header,split(line,','))) for line in lines[2:end]]
end

function analyze(;root=joinpath(@__DIR__,"output","five_fs"))
    names=filter(n->isfile(joinpath(root,n,"metadata.toml")),readdir(root))
    isempty(names) && error("no attempted runs")
    base=TOML.parsefile(joinpath(root,"reference_refined" in names ? "reference_refined" : first(names),"metadata.toml"))
    cfg=base["config"]
    for name in names
        meta=TOML.parsefile(joinpath(root,name,"metadata.toml"))
        meta["sources"]==base["sources"] || error("source mismatch $name")
        meta["config"]==cfg || error("model mismatch $name")
        meta["script_hash"]==base["script_hash"] || error("script mismatch $name")
        Dict(k=>v for (k,v) in meta["runtime"] if k!="blas_threads")==
            Dict(k=>v for (k,v) in base["runtime"] if k!="blas_threads") || error("runtime mismatch $name")
    end
    total=HLT.grid_count(cfg["time"]["end_fs"],1.)
    common=Int[]
    for t in 1:total
        all(isfile(joinpath(root,n,"rep1","time_$t.jls")) for n in ("reference","reference_refined")) || break
        push!(common,t)
    end
    statuses=NamedTuple[]
    for name in names
        path=joinpath(root,name,"status.toml")
        status=isfile(path) ? TOML.parsefile(path) : Dict("phase"=>"interrupted","time_fs"=>0.,"completed_repetitions"=>0)
        push!(statuses,(;name,phase=status["phase"],time_fs=status["time_fs"],
            completed_repetitions=status["completed_repetitions"],reference_checked=false))
    end
    HLT.write_rows(joinpath(root,"run_status.csv"),statuses)
    if isempty(common)
        println("No common reference checkpoints; every attempted run remains unqualified.")
        return statuses
    end
    p=HLT.comparison_problem(cfg)
    BLAS.set_num_threads(8)
    checks=NamedTuple[];maxstate=0.;maxrho=0.
    for t in common
        a=deserialize(joinpath(root,"reference","rep1","time_$t.jls"))
        b=deserialize(joinpath(root,"reference_refined","rep1","time_$t.jls"))
        maxstate=max(maxstate,HLT.comparison_error(a,b))
        maxrho=max(maxrho,norm(HLT.density_of(a,p)-HLT.density_of(b,p)))
        push!(checks,(;time_fs=t,max_state=maxstate,max_density=maxrho,qualified=maxstate<1e-4 && maxrho<1e-4))
    end
    HLT.write_rows(joinpath(root,"reference_convergence.csv"),checks)
    reference_checked=length(checks)==total && last(checks).qualified
    HLT.write_rows(joinpath(root,"run_status.csv"),[merge(r,(;reference_checked)) for r in statuses])
    rows=NamedTuple[]
    for name in filter(n->!(n in ("reference","reference_refined")),names)
        isfile(joinpath(root,name,"runs.csv")) || continue
        for run in csvrows(joinpath(root,name,"runs.csv"))
            rep=parse(Int,run["repetition"]);maxstate=0.;maxrho=0.;maxtrace=0.
            steps=csvrows(joinpath(root,name,"rep$rep","steps.csv"))
            for t in common
                path=joinpath(root,name,"rep$rep","time_$t.jls");isfile(path) || break
                x=deserialize(path);ref=deserialize(joinpath(root,"reference_refined","rep1","time_$t.jls"))
                rho=HLT.density_of(x,p)
                maxstate=max(maxstate,HLT.comparison_error(x,ref))
                maxrho=max(maxrho,norm(rho-HLT.density_of(ref,p)));maxtrace=max(maxtrace,abs(tr(rho)-1))
                lastrow=last(filter(r->parse(Float64,r["time_fs"])<=t+1e-12,steps))
                push!(rows,(;name,repetition=rep,time_fs=t,seconds=parse(Float64,lastrow["cumulative_seconds"]),
                    max_state_error=maxstate,max_density_error=maxrho,max_trace_error=maxtrace,
                    qualified=checks[t].qualified && maxstate<1e-3 && maxrho<1e-3,
                    maxrank=parse(Int,lastrow["maxrank"])))
            end
        end
    end
    isempty(rows) || HLT.write_rows(joinpath(root,"comparison.csv"),rows)
    foreach(println,rows)
end
end
