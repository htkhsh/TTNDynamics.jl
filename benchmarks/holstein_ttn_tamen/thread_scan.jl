# Per-step cost of a TTN case from a saved long-run state, for comparing
# thread settings: julia --threads=T thread_scan.jl NAME STATE.jls STEPS.
# Step 1 starts without a warm start (seconds=NaN) and step 2 compiles the
# warm-start path, so compare from step 3. Every variant runs the same steps
# with the same seeds. Env: SCAN_SITES (default 10),
# SCAN_CHUNKS (core-action groups, default 4), SCAN_BLAS (default 1),
# SCAN_TRUNC (truncation_rtol), SCAN_MAXRANK (default 256),
# SCAN_OUT (CSV appended to), SCAN_LABEL; SCAN_BLOCKED=1 and
# SCAN_BLOCKED_CHUNKS switch on blocked products where TTNSolver has them. Phase times come from the step
# diagnostics (sweep = parallel Etter sweep; als includes truncation and root).
include(joinpath(@__DIR__,"common.jl"))
using .HolsteinTAMEnComparison, LinearAlgebra, Random, Serialization
const H=HolsteinTAMEnComparison; const HLT=H.HLT; const TS=HLT.TTNSolver
function main()
    name,state,steps=ARGS[1],ARGS[2],parse(Int,ARGS[3])
    sites=parse(Int,get(ENV,"SCAN_SITES","10"))
    chunks=parse(Int,get(ENV,"SCAN_CHUNKS","4")); blas=parse(Int,get(ENV,"SCAN_BLAS","1"))
    TS._CORE_CHUNKS[]=chunks
    if haskey(ENV,"SCAN_BLOCKED")
        isdefined(TS,:_CORE_BLOCKED) || error("TTNSolver at $(pathof(TS)) has no blocked products")
        TS._CORE_BLOCKED[]=ENV["SCAN_BLOCKED"]=="1"
        haskey(ENV,"SCAN_BLOCKED_CHUNKS") && (TS._CORE_BLOCKED_CHUNKS[]=parse(Int,ENV["SCAN_BLOCKED_CHUNKS"]))
    end
    println("TTNSolver: ",pathof(TS)); flush(stdout)
    cfg=H.configuration(;end_fs=20.0,sites)
    s=H.specification(cfg,name); s["maxrank"]=parse(Int,get(ENV,"SCAN_MAXRANK","256"))
    haskey(ENV,"SCAN_TRUNC") && (s["truncation_rtol"]=parse(Float64,ENV["SCAN_TRUNC"]))
    q=H.ordered_problem(HLT.comparison_problem(cfg),H.run_order(name);
        bath_layout=Symbol(get(s,"bath_layout","balanced")))
    BLAS.set_num_threads(blas)
    work=TS.TAMEnWorkspace(); x=deserialize(state); hint=nothing
    out=get(ENV,"SCAN_OUT",joinpath(@__DIR__,"output","thread_scan.csv"))
    fresh=!isfile(out)
    open(out,"a") do io
        fresh && println(io,"label,name,sites,threads,chunks,blas,step,seconds,gc_seconds,maxrank,matvecs,sweeps,sweep_s,als_s,residual_local_s,enrichment_s,time_error_s,warm_start_s,environment_s,trace_re,rho11,maxrss_gb")
        for j in 1:steps
            Random.seed!(cfg["execution"]["seed"]+j); GC.gc()
            m=@timed H.advance(q,x,s,work,hint); x,d=m.value; hint=d.warm_start
            rho=H.density(x,q); t=d.timings
            row=(get(ENV,"SCAN_LABEL",""),name,sites,Threads.nthreads(),chunks,blas,j,
                 j==1 ? NaN : m.time,m.gctime,maximum(HLT.ranks_of(x)),d.matvecs,d.sweeps,
                 (round(getfield(t,k),digits=3) for k in (:sweep_seconds,:als_seconds,:residual_local_seconds,
                  :enrichment_seconds,:time_error_seconds,:warm_start_seconds,:environment_seconds))...,
                 repr(real(tr(rho))),repr(real(rho[1,1])),round(Sys.maxrss()/2^30,digits=2))
            println(io,join(row,',')); flush(io)
            println(join(row,' ')); flush(stdout)
        end
    end
end
main()
