module PS2Bench
include("../holstein_ttn_tamen/common.jl")
include("ps2.jl")
using .PS2Trial, LinearAlgebra, Serialization, TOML
const HTC=HolsteinTAMEnComparison
const HLT=HTC.HLT
const TS=HTC.TS
const ROOT=joinpath(@__DIR__,"output",get(ENV,"PS2_OUTPUT","holstein5_eager"))
const REFERENCES=joinpath(@__DIR__,"../holstein_ttn_tamen/output/five_fs")

function atomic_state(path,value)
    serialize(path*".tmp",value);mv(path*".tmp",path;force=true)
end
function checked_problem()
    cfg=HTC.configuration()
    meta=TOML.parsefile(joinpath(REFERENCES,"reference_refined","metadata.toml"))
    meta["sources"]==HLT.source_hashes() || error("reference sources have changed")
    wrapper="../holstein_ttn_tamen/common.jl"
    meta["script_hash"]==bytes2hex(HLT.sha256(read(joinpath(@__DIR__,wrapper)))) || error("reference wrapper changed")
    meta["config"]==cfg || error("reference model differs")
    runtime=HLT.runtime_info()
    all(runtime[k]==v for (k,v) in meta["runtime"] if k!="blas_threads") || error("reference runtime differs")
    checks=HTC.csvrows(joinpath(REFERENCES,"reference_convergence.csv"))
    length(checks)==5 && all(r->r["qualified"]=="true",checks) || error("references not qualified")
    scripts=Dict(f=>bytes2hex(HLT.sha256(read(joinpath(@__DIR__,f)))) for f in ("ps2.jl","common.jl","batch.jl",wrapper))
    reference_files=Dict(joinpath(name,"rep1","time_$t.jls")=>
        bytes2hex(HLT.sha256(read(joinpath(REFERENCES,name,"rep1","time_$t.jls"))))
        for name in ("reference","reference_refined") for t in 1:5)
    experiment_path=joinpath(ROOT,"experiment.toml")
    if isfile(experiment_path)
        previous=TOML.parsefile(experiment_path)
        previous["script_hashes"]==scripts || error("trial scripts changed; use a new output directory")
        previous["reference_files"]==reference_files || error("saved reference states changed")
        previous["sources"]==meta["sources"] && previous["config"]==cfg || error("experiment model/sources changed")
    end
    measured=@timed HLT.comparison_problem(cfg)
    mkpath(ROOT)
    isfile(experiment_path) || HLT.atomic_toml(experiment_path,Dict("config"=>cfg,"sources"=>meta["sources"],
        "reference_root"=>abspath(REFERENCES),"reference_checks"=>checks,"setup_seconds"=>measured.time,
        "script_hashes"=>scripts,"reference_files"=>reference_files,
        "runtime"=>runtime))
    return measured.value,cfg
end

function specification(;h=0.1,rate=1e-5,maxrank=64,padding=2,method="ps2")
    return Dict{String,Any}("method"=>method,"dt_fs"=>h,"truncation_rate"=>rate,"maxrank"=>maxrank,
        "rank_padding"=>padding,"krylov_tol"=>1e-12,"krylov_eager"=>true,"blas_threads"=>8)
end
function run_case(p,cfg,name,s;stop_fs=1.,repeats=1,seconds=600.,deadline=Inf,root=ROOT)
    s=s["method"]=="tt" ? HTC.specification(cfg,"tt") : copy(s)
    dir=joinpath(root,name);ispath(dir) && error("output already exists $name")
    mkpath(dir);started=time();deadline=min(deadline,started+seconds)
    BLAS.set_num_threads(s["blas_threads"])
    HLT.atomic_toml(joinpath(dir,"metadata.toml"),Dict("spec"=>s,"stop_fs"=>stop_fs,"runtime"=>HLT.runtime_info()))
    status=Dict{String,Any}("phase"=>"warmup","time_fs"=>0.,"failure"=>"","wall_seconds"=>0.,"warmup_seconds"=>0.)
    function save(phase)
        status["phase"]=phase;status["wall_seconds"]=time()-started
        HLT.atomic_toml(joinpath(dir,"status.toml"),status)
    end
    save("warmup")
    istree=s["method"]=="ps2"
    initial=istree ? p.tree : p.train
    ws=istree ? nothing : HLT.TAMEnWorkspace(ComplexF64)
    function advance(x)
        return istree ? ps2_step(x,p.operator,s["dt_fs"];atol=s["truncation_rate"]*s["dt_fs"],
            maxrank=s["maxrank"],conserved=p.conserved,krylov_tol=s["krylov_tol"],
            krylov_eager=s["krylov_eager"],rank_padding=s["rank_padding"]) :
            HLT.advance(p,x,s,ws)
    end
    try
        HLT.Random.seed!(cfg["execution"]["seed"]+1)
        status["warmup_seconds"]=@elapsed advance(deepcopy(initial))
    catch error
        error isa InterruptException && rethrow()
        status["failure"]=sprint(showerror,error);save("warmup_failed");return
    end
    println("WARM ",name," ",status["warmup_seconds"]);flush(stdout)
    total=HLT.grid_count(stop_fs,s["dt_fs"]);observe=HLT.grid_count(1.,s["dt_fs"])
    for rep in 1:repeats
        time()<deadline || (save("timeout");return)
        repdir=joinpath(dir,"rep$rep");mkpath(repdir)
        status["repetition"]=rep;status["time_fs"]=0.;save("running")
        x=deepcopy(initial);elapsed=0.;last=0;GC.gc()
        atomic_state(joinpath(repdir,"time_0.jls"),x)
        open(joinpath(repdir,"steps.csv"),"w") do io
            println(io,"step,time_fs,seconds,cumulative_seconds,bytes,gc_seconds,maxrank,local_solves,matvecs,max_local_dimension,local_cut_sum,truncation_met,solve_seconds,environment_seconds,split_seconds")
            for j in 1:total
                time()<deadline || break
                HLT.Random.seed!(cfg["execution"]["seed"]+j)
                measured=try
                    @timed advance(x)
                catch error
                    error isa InterruptException && rethrow()
                    status["failure"]=sprint(showerror,error);break
                end
                x,info=measured.value;elapsed+=measured.time;last=j
                rank=maximum(HLT.ranks_of(x));t=j*s["dt_fs"]
                diag=istree ? (info.local_solves,info.matvecs,info.max_local_dimension,info.local_cut_sum,
                    info.truncation_met,values(info.timings)...) : (0,0,0,NaN,true,0.,0.,0.)
                println(io,join((j,t,measured.time,elapsed,measured.bytes,measured.gctime,rank,diag...),','));flush(io)
                atomic_state(joinpath(repdir,"latest.jls"),(;state=x,step=j,time_fs=t,seconds=elapsed))
                if j%observe==0
                    atomic_state(joinpath(repdir,"time_$(j÷observe).jls"),x)
                    println(name," rep=",rep," t=",t," seconds=",elapsed," rank=",rank);flush(stdout)
                end
                status["time_fs"]=t;save("running")
            end
        end
        save(last==total ? "complete" : isempty(status["failure"]) ? "timeout" : "solve_failed")
        last==total || break
    end
end

function analyze(p;finalize=false)
    comparisons=NamedTuple[];statuses=NamedTuple[]
    for name in sort(readdir(ROOT))
        dir=joinpath(ROOT,name);isfile(joinpath(dir,"metadata.toml")) || continue
        meta=TOML.parsefile(joinpath(dir,"metadata.toml"));status=TOML.parsefile(joinpath(dir,"status.toml"))
        if finalize && status["phase"] in ("running","warmup")
            status["phase"]="interrupted";HLT.atomic_toml(joinpath(dir,"status.toml"),status)
        end
        push!(statuses,(;name,phase=status["phase"],time_fs=status["time_fs"],failure=replace(status["failure"],','=>';','\n'=>' ')))
        for repname in sort(filter(n->startswith(n,"rep") && isdir(joinpath(dir,n)),readdir(dir)))
            repdir=joinpath(dir,repname);isfile(joinpath(repdir,"latest.jls")) || continue
            last=deserialize(joinpath(repdir,"latest.jls"))
            steps=filter(r->parse(Int,r["step"])<=last.step,HTC.csvrows(joinpath(repdir,"steps.csv")))
            maxstate=0.;maxrho=0.;maxtrace=0.
            for t in 1:floor(Int,last.time_fs+1e-9)
                file=joinpath(repdir,"time_$t.jls");isfile(file) || break
                x=deserialize(file)
                reference=deserialize(joinpath(REFERENCES,"reference_refined","rep1","time_$t.jls"))
                rho=HLT.density_of(x,p)
                maxstate=max(maxstate,HLT.comparison_error(x,reference))
                maxrho=max(maxrho,norm(rho-HLT.density_of(reference,p)))
                maxtrace=max(maxtrace,abs(tr(rho)-1))
                prefix=filter(r->parse(Float64,r["time_fs"])<=t+1e-9,steps)
                push!(comparisons,(;name,repetition=parse(Int,repname[4:end]),time_fs=t,
                    seconds=parse(Float64,prefix[end]["cumulative_seconds"]),
                    bytes=sum(parse(Int,r["bytes"]) for r in prefix),
                    peak_rank=maximum(parse(Int,r["maxrank"]) for r in prefix),
                    max_local_dimension=maximum(parse(Int,r["max_local_dimension"]) for r in prefix),
                    cut_budget_failures=count(r->r["truncation_met"]=="false",prefix),
                    max_state_error=maxstate,max_density_error=maxrho,max_trace_error=maxtrace,
                    qualified=maxstate<1e-3 && maxrho<1e-3))
            end
        end
    end
    isempty(comparisons) || HLT.write_rows(joinpath(ROOT,"comparison.csv"),comparisons)
    isempty(statuses) || HLT.write_rows(joinpath(ROOT,"run_status.csv"),statuses)
    foreach(println,comparisons);flush(stdout)
    return comparisons
end
end
