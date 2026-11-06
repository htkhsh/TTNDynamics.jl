include("../holstein_longtime/driver.jl")
using .HolsteinLongTime, LinearAlgebra, Statistics, Serialization, Test
const HLT=HolsteinLongTime

function experiment(label="trial";repeats=2,wall_seconds=2400)
    repeats>=1 || error("positive repeats required")
    isfinite(wall_seconds) && wall_seconds>0 || error("positive finite wall budget required")
    occursin(r"^[A-Za-z0-9_-]+$",label) || error("invalid label")
    dir=joinpath(@__DIR__,"output",label)
    ispath(dir) && error("output exists: $dir")
    mkpath(dir)
    cp(@__FILE__,joinpath(dir,"experiment.jl"))
    cfg=HLT.load_config(joinpath(@__DIR__,"../holstein_longtime/config_50fs.toml"))
    cfg["time"]=Dict("end_fs"=>1.0,"observation_fs"=>1.0,"checkpoint_fs"=>1.0)
    cfg["execution"]["output_directory"]="."
    HLT.atomic_toml(joinpath(dir,"config.toml"),cfg)
    HLT.atomic_toml(joinpath(dir,"metadata.toml"),Dict("runtime"=>HLT.runtime_info(),
        "sources"=>HLT.source_hashes(),"repeats"=>repeats,"wall_seconds"=>wall_seconds,
        "experiment_sha256"=>bytes2hex(HLT.sha256(read(@__FILE__)))))
    p=HLT.comparison_problem(cfg)
    started=time_ns()
    check_budget()=(time_ns()-started)/1e9<wall_seconds || error("benchmark wall budget reached; completed cases are saved")
    # TT references use the identical generator and axis ordering. Each has
    # its own tolerance, timestep, workspace and reproducible enrichment seed.
    trains=Dict(); train_rows=NamedTuple[]
    for name in ("tamen","reference","reference_refined")
        s=HLT.case_config(cfg,name); BLAS.set_num_threads(s["blas_threads"])
        work=HLT.TAMEnWorkspace(ComplexF64)
        HLT.Random.seed!(cfg["execution"]["seed"]+1)
        HLT.advance(p,deepcopy(p.train),s,work)
        x=deepcopy(p.train); seconds=0.0; bytes=0; peak=maximum(HLT.tt_ranks(x))
        GC.gc()
        for j in 1:HLT.grid_count(1.0,s["dt_fs"])
            check_budget(); HLT.Random.seed!(cfg["execution"]["seed"]+j)
            m=@timed HLT.advance(p,x,s,work)
            x=first(m.value); seconds+=m.time; bytes+=m.bytes
            peak=max(peak,maximum(HLT.tt_ranks(x)))
        end
        trains[name]=x
        push!(train_rows,(;run=name,seconds,allocated_bytes=bytes,peak_rank=peak))
        serialize(joinpath(dir,"$name.jls"),x)
        println("TT ",name," seconds=",seconds," rank=",peak); flush(stdout)
    end
    HLT.write_rows(joinpath(dir,"train_timings.csv"),train_rows)
    ref=trains["reference_refined"]
    ref_rho=HLT.density_of(ref,p)
    check=HLT.relative_error(trains["reference"],ref)
    HLT.atomic_toml(joinpath(dir,"reference_check.toml"),Dict("state_difference"=>check,
        "density_difference"=>norm(HLT.density_of(trains["reference"],p)-ref_rho),
        "tamen_state_difference"=>HLT.relative_error(trains["tamen"],ref),
        "tamen_density_difference"=>norm(HLT.density_of(trains["tamen"],p)-ref_rho)))
    println("REFERENCE state difference=",check); flush(stdout)
    BLAS.set_num_threads(1)
    rows=NamedTuple[]
    for method in (:augmented,:parallel), (variant,truncation,grouping) in
        (("baseline",:local,:balanced),("weighted",:weighted,:balanced),
         ("grouped",:local,:coupling),("both",:weighted,:coupling))
        tree=HLT.heom_tree(p.system;bath_grouping=grouping)
        initial=HLT.heom_initial_state(p.system,HLT.density_of(p.train,p);tree)
        c=HLT.TTNDynamics.trace_observable(p.system;tree)
        step=method==:augmented ? HLT.augmented_bug_step : HLT.parallel_bug_step
        options=(;atol=1e-6,maxrank=64,conserved=c,correct_invariant=false,threaded=true,truncation)
        println("WARM ",method," ",variant); flush(stdout)
        check_budget(); step(initial,p.operator,0.1;options...)
        for repeat in 1:repeats
            x=copy(initial); seconds=0.0; gc=0.0; bytes=0; peak=maximum(HLT.tree_ranks(x)); misses=0
            trace_error=0.0; max_augmented=0; max_local=0; discarded=0.0
            GC.gc()
            open(joinpath(dir,"$(method)_$(variant)_$(repeat)_steps.csv"),"w") do io
                println(io,"time_fs,seconds,cumulative_seconds,gc_seconds,allocated_bytes,maxrank,augmented_rank,max_local_dimension,truncation_met,discarded_bound,trace_error")
                for j in 1:10
                    check_budget()
                    measured=@timed step(x,p.operator,0.1;options...)
                    x,d=measured.value
                    seconds+=measured.time; gc+=measured.gctime; bytes+=measured.bytes
                    peak=max(peak,maximum(d.ranks)); misses+=!d.truncation.tolerance_met
                    max_augmented=max(max_augmented,maximum(d.augmented_ranks))
                    max_local=max(max_local,d.max_local_dimension)
                    discarded+=d.truncation.discarded_bound
                    trace_error=max(trace_error,abs(dot(c,x)-1))
                    println(io,join((j*0.1,measured.time,seconds,measured.gctime,measured.bytes,
                        maximum(d.ranks),maximum(d.augmented_ranks),d.max_local_dimension,
                        d.truncation.tolerance_met,d.truncation.discarded_bound,abs(dot(c,x)-1)),',')); flush(io)
                    println(method," ",variant," #",repeat," t=",j*0.1," seconds=",seconds,
                        " rank=",maximum(d.ranks)); flush(stdout)
                end
            end
            serialize(joinpath(dir,"$(method)_$(variant)_$(repeat).jls"),x)
            state_error=HLT.comparison_error(x,ref)
            density=norm(HLT.TTNDynamics.root_density_matrix(x,p.system)-ref_rho)
            @test isfinite(state_error) && isfinite(density) && trace_error<1e-10
            push!(rows,(;method,variant,repeat,seconds,gc_seconds=gc,allocated_bytes=bytes,
                peak_rank=peak,max_augmented_rank=max_augmented,max_local_dimension=max_local,
                truncation_misses=misses,discarded_bound_sum=discarded,state_error,
                density_error=density,trace_error))
            HLT.write_rows(joinpath(dir,"trials.csv"),rows)
            println("DONE ",last(rows)); flush(stdout)
        end
    end
    println("COMPLETE: ",dir); flush(stdout)
end

experiment(isempty(ARGS) ? "trial" : ARGS[1];repeats=length(ARGS)>=2 ? parse(Int,ARGS[2]) : 2)
