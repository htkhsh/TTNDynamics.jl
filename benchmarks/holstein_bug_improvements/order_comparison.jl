include("../holstein_longtime/driver.jl")
include("second_order.jl")
using .HolsteinLongTime, .SecondOrderBUG, LinearAlgebra, Serialization, Test
const HLT=HolsteinLongTime

function experiment(label="order";repeats=2,wall_seconds=1000,refine=false)
    occursin(r"^[A-Za-z0-9_-]+$",label) || error("invalid label")
    dir=joinpath(@__DIR__,"output",label)
    ispath(dir) && error("output exists")
    mkpath(dir)
    for file in ("second_order.jl","order_comparison.jl")
        cp(joinpath(@__DIR__,file),joinpath(dir,file))
    end
    cfg=HLT.load_config(joinpath(@__DIR__,"../holstein_longtime/config_50fs.toml"))
    cfg["time"]=Dict("end_fs"=>1.0,"observation_fs"=>1.0,"checkpoint_fs"=>1.0)
    HLT.atomic_toml(joinpath(dir,"metadata.toml"),Dict("sources"=>HLT.source_hashes(),
        "runtime"=>HLT.runtime_info(),"config"=>cfg,"repeats"=>repeats))
    p=HLT.comparison_problem(cfg); started=time()
    budget()=time()-started<wall_seconds || error("soft wall budget reached; completed rows retained")
    refs=Dict(); trainrows=NamedTuple[]
    for name in ("tamen","reference","reference_refined")
        s=HLT.case_config(cfg,name); BLAS.set_num_threads(s["blas_threads"])
        w=HLT.TAMEnWorkspace(ComplexF64); x=deepcopy(p.train)
        HLT.Random.seed!(cfg["execution"]["seed"]+1)
        HLT.advance(p,deepcopy(x),s,w)
        seconds=0.0; GC.gc()
        for j in 1:HLT.grid_count(1.0,s["dt_fs"])
            HLT.Random.seed!(cfg["execution"]["seed"]+j)
            m=@timed HLT.advance(p,x,s,w); x=first(m.value); seconds+=m.time
        end
        refs[name]=x
        serialize(joinpath(dir,"$name.jls"),x)
        push!(trainrows,(;method=name,seconds))
    end
    HLT.write_rows(joinpath(dir,"tt_timings.csv"),trainrows)
    ref=refs["reference_refined"]; rho=HLT.density_of(ref,p)
    HLT.atomic_toml(joinpath(dir,"reference_check.toml"),Dict(
        "state_difference"=>HLT.relative_error(refs["reference"],ref),
        "density_difference"=>norm(HLT.density_of(refs["reference"],p)-rho),
        "tamen_state_difference"=>HLT.relative_error(refs["tamen"],ref),
        "tamen_density_difference"=>norm(HLT.density_of(refs["tamen"],p)-rho)))
    BLAS.set_num_threads(1)
    rows=NamedTuple[]
    # Coarser second-order candidates first, then refine without extrapolating.
    cases=refine ? [(1,:parallel,0.025),(1,:parallel,0.0125),(2,:parallel,0.125)] :
        [(order,method,h) for order in (1,2), method in (:augmented,:parallel)
         for h in (order==1 ? (0.1,0.05) : (0.5,0.25,0.1))]
    for (order,method,h) in cases
        budget()
        step=order==1 ? (method==:augmented ? HLT.augmented_bug_step : HLT.parallel_bug_step) :
            ((x,op,h;kw...)->second_order_bug_step(x,op,h;method,kw...))
        options=(;atol=1e-5*h,maxrank=64,conserved=p.conserved,
                  correct_invariant=false,threaded=true,truncation=:weighted)
        println("WARM ",(order,method,h)); flush(stdout)
        step(p.tree,p.operator,h;options...)
        for repetition in 1:repeats
            x=copy(p.tree); seconds=0.0; bytes=0; gc=0.0; peak=0; aug=0; dim=0; misses=0; trace=0.0
            GC.gc()
            path=joinpath(dir,"$(method)_order$(order)_h$(h)_rep$(repetition)")
            open(path*"_steps.csv","w") do io
                println(io,"time_fs,seconds,allocated_bytes,rank,augmented_rank,max_local_dimension,truncation_met,trace_error")
                for j in 1:HLT.grid_count(1.0,h)
                    budget()
                    m=@timed step(x,p.operator,h;options...); x,d=m.value
                    seconds+=m.time; bytes+=m.bytes; gc+=m.gctime
                    peak=max(peak,maximum(d.ranks)); aug=max(aug,maximum(d.augmented_ranks))
                    dim=max(dim,d.max_local_dimension); misses+=!d.truncation.tolerance_met
                    trace=max(trace,abs(HLT.dot(p.conserved,x)-1))
                    println(io,join((j*h,m.time,m.bytes,maximum(d.ranks),maximum(d.augmented_ranks),
                        d.max_local_dimension,d.truncation.tolerance_met,trace),',')); flush(io)
                    println((order,method,h,repetition,t=j*h,seconds,rank=maximum(d.ranks))); flush(stdout)
                end
            end
            serialize(path*".jls",x)
            state_error=HLT.comparison_error(x,ref)
            density_error=norm(HLT.TTNDynamics.root_density_matrix(x,p.system)-rho)
            @test isfinite(state_error) && isfinite(density_error) && trace<1e-10
            push!(rows,(;order,method,h,repetition,seconds,bytes,gc,peak_rank=peak,
                max_augmented_rank=aug,max_local_dimension=dim,truncation_misses=misses,
                state_error,density_error,trace_error=trace))
            HLT.write_rows(joinpath(dir,"trials.csv"),rows)
            println("DONE ",last(rows)); flush(stdout)
        end
    end
    println("COMPLETE")
end
experiment(isempty(ARGS) ? "order" : ARGS[1];refine=length(ARGS)>=2 && ARGS[2]=="refine")
