module HolsteinLongTime
include("configuration.jl")
include("../holstein10_split/common.jl")

function comparison_problem(cfg)
    m=cfg["model"]
    return example_problem(;sites=m["sites"],local_size=m["hierarchy_local_size"],
        initial_condition=:localized,config_overrides=(;
            initial_site=m["initial_site"],hopping_cm=m["hopping_cm"],
            brownian_frequency_cm=m["frequency_cm"],brownian_damping_cm=m["damping_cm"],
            reorganization_energy_cm=m["reorganization_cm"],temperature_K=m["temperature_K"],
            decomposition_method=:psd,pade_order=m["pade_order"],pade_type=:Nm1,
            operator_tolerance=m["operator_tolerance"]))
end

function advance(p,x,s,workspace)
    h=s["dt_fs"]
    if s["method"]=="tamen"
        st=tt_kron(x,tt_ones(s["points"];T=ComplexF64))
        y,grid,_,_,_=tamen(st,h*p.problem.liouvillian,s["tolerance"];
            time_scheme="cheb",nswp=s["sweeps"],local_iters=s["local_iters"],
            kickrank=s["kick_rank"],time_error_damp=s["time_error_damp"],
            correct_2_norm=false,conserved_observables=[p.problem.trace_observable],
            workspace,verb=0)
        return extract_snapshot(y,grid,1.0,"cheb"),(;intervals=length(grid))
    end
    step=s["method"]=="augmented" ? augmented_bug_step : parallel_bug_step
    return step(x,p.operator,h;atol=s["truncation_rate"]*h,maxrank=s["maxrank"],
        krylov_tol=s["krylov_tolerance"],conserved=p.conserved,correct_invariant=false,
        threaded=true,truncation=Symbol(s["truncation"]))
end

ranks_of(x::TTTensor)=tt_ranks(x)
ranks_of(x::TTNTensor)=tree_ranks(x)
density_of(x::TTTensor,p)=TTDynamics.root_density_matrix(x,p.problem.system)
density_of(x::TTNTensor,p)=TTNDynamics.root_density_matrix(x,p.system)

function source_hashes()
    files=String[]
    # Local bath/noise packages can change without a Manifest revision.
    for mod in (TTSolver,TTNSolver,TTNDynamics,TTDynamics,
                parentmodule(HolsteinExample.NoiseExp),HolsteinExample.QFiND,TTNSolver.KrylovKit)
        src=dirname(pathof(mod))
        for (root,_,names) in walkdir(src), name in names
            endswith(name,".jl") && push!(files,joinpath(root,name))
        end
        manifest=joinpath(dirname(src),"Manifest.toml")
        isfile(manifest) && push!(files,manifest)
    end
    append!(files,[joinpath(@__DIR__,file) for file in ("configuration.jl","driver.jl","analysis.jl")])
    append!(files,[joinpath(@__DIR__,"..",file) for file in
        ("holstein10_split/common.jl","tamen_comparison/common.jl","parallel_large_trees/fixtures.jl")])
    append!(files,[joinpath(HolsteinExample.EXAMPLE,file) for file in ("config.jl","model.jl","utils.jl")])
    for file in ("Project.toml","Manifest.toml")
        path=joinpath(HolsteinExample.EXAMPLE,file)
        isfile(path) && push!(files,path)
    end
    push!(files,joinpath(HolsteinExample.EXAMPLE,"../common/output_utils.jl"))
    base=normpath(joinpath(@__DIR__,"../../.."))
    return Dict(relpath(file,base)=>bytes2hex(sha256(read(file))) for file in files)
end

runtime_info()=Dict("julia"=>string(VERSION),"julia_threads"=>Threads.nthreads(:default),
    "blas_threads"=>BLAS.get_num_threads(),"gc_threads"=>Threads.ngcthreads(),"cpu"=>Sys.cpu_info()[1].model)

function atomic_toml(path,data)
    open(path*".tmp","w") do io
        TOML.print(io,data;sorted=true)
    end
    mv(path*".tmp",path;force=true)
end

function save_checkpoint(dir,progress,identity,sources,runtime,streams,total,archive_every)
    offsets=Dict{String,Int64}()
    for (name,io) in streams
        flush(io); offsets[name]=position(io)
    end
    saved=(;version=1,identity,sources,runtime,progress...,offsets,complete=progress.step==total)
    temp=joinpath(dir,"latest.jls.tmp")
    serialize(temp,saved)
    # An archive is a committed accepted state. latest is the resume authority;
    # an archive beyond latest after a crash is ignored by analysis.
    if progress.step % archive_every == 0 || progress.step==total
        archive=joinpath(dir,"checkpoints","step_$(progress.step).jls")
        cp(temp,archive*".tmp";force=true); mv(archive*".tmp",archive;force=true)
    end
    mv(temp,joinpath(dir,"latest.jls");force=true)
    return saved
end

function restore_logs!(dir,saved)
    # A crash can leave CSV rows newer than latest. Remove only those tails.
    for (name,offset) in saved.offsets
        path=joinpath(dir,name)
        isfile(path) && filesize(path)>=offset || throw(ArgumentError("checkpoint/log mismatch: $path"))
    end
    for (name,offset) in saved.offsets
        open(joinpath(dir,name),"r+") do io
            truncate(io,offset)
        end
    end
end

const STEP_HEADER="step,time_fs,seconds,cumulative_seconds,gc_seconds,allocated_bytes,maxrank,truncation_met,discarded_bound,internal_intervals"
function observation_header(n)
    return join(vcat(["step","time_fs","norm","trace_error","hermiticity_error"],
        ["rho_$(a)_$(b)_$part" for b in 1:n for a in 1:n for part in ("re","im")]),',')
end

function record_observation(io,p,x,j,h)
    rho=density_of(x,p)
    values=Any[j,j*h,norm(x),abs(tr(rho)-1),norm(rho-rho')]
    append!(values,[v for z in vec(rho) for v in (real(z),imag(z))])
    println(io,join(values,',')); flush(io)
end

function run_case(cfg,name,root;resume=false,wall_seconds=3600.0,max_steps=typemax(Int))
    validate_config(cfg)
    (wall_seconds>=0 && !isnan(wall_seconds)) || throw(ArgumentError("invalid wall budget"))
    integer_at_least(max_steps,0) || throw(ArgumentError("invalid max_steps"))
    started=time_ns(); deadline=Float64(wall_seconds)
    s=case_config(cfg,name); h=s["dt_fs"]
    total=grid_count(cfg["time"]["end_fs"],h)
    observe=grid_count(cfg["time"]["observation_fs"],h)
    archive=grid_count(cfg["time"]["checkpoint_fs"],h)
    BLAS.set_num_threads(s["blas_threads"])
    identity=run_identity(cfg,name); sources=source_hashes(); runtime=runtime_info()
    dir=joinpath(root,name); mkpath(root)
    lockdir=dir*".lock"
    ispath(lockdir) && throw(ArgumentError("run locked: $lockdir (remove only after its writer has stopped)"))
    mkdir(lockdir)
    streams=Dict{String,IOStream}()
    try
        saved=nothing
        if resume
            isfile(joinpath(dir,"latest.jls")) || throw(ArgumentError("no checkpoint to resume: $dir"))
            saved=deserialize(joinpath(dir,"latest.jls"))
            saved.version==1 && saved.identity==identity || throw(ArgumentError("model, grid or solver settings changed"))
            saved.sources==sources || throw(ArgumentError("source/dependency files changed; use a new output directory"))
            saved.runtime==runtime || throw(ArgumentError("runtime/thread settings changed"))
            restore_logs!(dir,saved)
        else
            ispath(dir) && throw(ArgumentError("run exists; use --resume or a new output directory"))
            mkpath(joinpath(dir,"checkpoints"))
            atomic_toml(joinpath(dir,"manifest.toml"),Dict("identity"=>identity,"run"=>name,
                "spec"=>run_spec(cfg,name),"sources"=>sources,"runtime"=>runtime,"created"=>string(now())))
        end
        p=comparison_problem(cfg)
        for file in ("steps.csv","observables.csv")
            streams[file]=open(joinpath(dir,file),resume ? "a" : "w")
            resume && seekend(streams[file])
        end
        if isnothing(saved)
            x=s["method"]=="tamen" ? p.train : p.tree
            progress=(;state=x,step=0,seconds=0.0,gc_seconds=0.0,allocated_bytes=0,truncation_misses=0)
            println(streams["steps.csv"],STEP_HEADER)
            println(streams["steps.csv"],join((0,0,0,0,0,0,maximum(ranks_of(x)),"NA","NA",0),','))
            println(streams["observables.csv"],observation_header(cfg["model"]["sites"]))
            record_observation(streams["observables.csv"],p,x,0,h)
            saved=save_checkpoint(dir,progress,identity,sources,runtime,streams,total,archive)
        end
        first_step=saved.step
        progress=(;state=saved.state,step=saved.step,seconds=saved.seconds,gc_seconds=saved.gc_seconds,
            allocated_bytes=saved.allocated_bytes,truncation_misses=saved.truncation_misses)
        work=TAMEnWorkspace(ComplexF64)
        warm_seconds=0.0
        if progress.step<total && max_steps>0 && (time_ns()-started)/1e9<deadline
            Random.seed!(cfg["execution"]["seed"]+progress.step+1)
            warm_seconds=@elapsed advance(p,deepcopy(progress.state),s,work)
            GC.gc()
        end
        while progress.step<total && progress.step-first_step<max_steps && (time_ns()-started)/1e9<deadline
            j=progress.step+1
            # tAMEn uses random enrichment. A step's seed is independent of
            # earlier warmups and restarts; workspace contains scratch only.
            Random.seed!(cfg["execution"]["seed"]+j)
            measured=@timed advance(p,progress.state,s,work)
            x,d=measured.value
            met=s["method"]=="tamen" ? "NA" : d.truncation.tolerance_met
            discarded=s["method"]=="tamen" ? "NA" : d.truncation.discarded_bound
            intervals=s["method"]=="tamen" ? d.intervals : 0
            progress=(;state=x,step=j,seconds=progress.seconds+measured.time,
                gc_seconds=progress.gc_seconds+measured.gctime,allocated_bytes=progress.allocated_bytes+measured.bytes,
                truncation_misses=progress.truncation_misses+(met===false))
            println(streams["steps.csv"],join((j,j*h,measured.time,progress.seconds,measured.gctime,
                measured.bytes,maximum(ranks_of(x)),met,discarded,intervals),',')); flush(streams["steps.csv"])
            if j % observe == 0 || j==total
                record_observation(streams["observables.csv"],p,x,j,h)
                saved=save_checkpoint(dir,progress,identity,sources,runtime,streams,total,archive)
                println(name," t=",j*h," / ",total*h," fs; propagation=",progress.seconds,
                        " s; rank=",maximum(ranks_of(x))); flush(stdout)
            end
        end
        saved=save_checkpoint(dir,progress,identity,sources,runtime,streams,total,archive)
        atomic_toml(joinpath(dir,"status.toml"),Dict("complete"=>saved.complete,"time_fs"=>saved.step*h,
            "target_fs"=>total*h,"propagation_seconds"=>saved.seconds,"truncation_misses"=>saved.truncation_misses))
        sessionpath=joinpath(dir,"sessions.csv")
        fresh=!isfile(sessionpath)
        open(sessionpath,"a") do io
            fresh && println(io,"finished,first_step,last_step,wall_seconds,warmup_seconds,complete")
            println(io,join((now(),first_step,saved.step,(time_ns()-started)/1e9,warm_seconds,saved.complete),','))
        end
        println(saved.complete ? "COMPLETE " : "PAUSED ",name," at ",saved.step*h," fs"); flush(stdout)
        return saved
    finally
        foreach(close,values(streams))
        rm(lockdir)
    end
end

include("analysis.jl")
end
