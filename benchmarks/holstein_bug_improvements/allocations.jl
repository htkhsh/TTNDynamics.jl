include("../holstein_longtime/driver.jl")
using .HolsteinLongTime, LinearAlgebra, Serialization, Profile, Statistics
const HLT=HolsteinLongTime

# Optional process-local baseline kernel; never rewrites package files.
const KERNEL=length(ARGS)>=2 ? ARGS[2] : "current"
KERNEL in ("current","before") || error("kernel must be current or before")
const BASELINE_SOURCE=read(joinpath(@__DIR__,"reference/truncation.jl"),String)
if KERNEL=="before"
    start=findfirst("function _orth_columns",BASELINE_SOURCE).start
    stop=findfirst("function _subtree_overlaps",BASELINE_SOURCE).start-1
    Base.include_string(HLT.TTNSolver,BASELINE_SOURCE[start:stop],"baseline_orth_columns.jl")
end

function allocation_trial(label;repeats=3)
    occursin(r"^[A-Za-z0-9_-]+$",label) || error("invalid label")
    dir=joinpath(@__DIR__,"output",label)
    ispath(dir) && error("output already exists")
    mkpath(dir)
    cp(@__FILE__,joinpath(dir,"allocations.jl"))
    cfg=HLT.load_config(joinpath(@__DIR__,"../holstein_longtime/config_50fs.toml"))
    HLT.atomic_toml(joinpath(dir,"metadata.toml"),Dict("sources"=>HLT.source_hashes(),
        "runtime"=>HLT.runtime_info(),"repeats"=>repeats,"config"=>cfg,"kernel"=>KERNEL,
        "baseline_sha256"=>bytes2hex(HLT.sha256(BASELINE_SOURCE))))
    p=HLT.comparison_problem(cfg)
    options=(;atol=1e-6,maxrank=64,conserved=p.conserved,correct_invariant=false,
              threaded=true,truncation=:weighted)
    rows=NamedTuple[]
    for (method,step) in ((:augmented,HLT.augmented_bug_step),(:parallel,HLT.parallel_bug_step))
        step(p.tree,p.operator,0.1;options...)
        for repetition in 1:repeats
            x=copy(p.tree); seconds=0.0; bytes=0; gc=0.0; peak=0
            GC.gc()
            for j in 1:10
                m=@timed step(x,p.operator,0.1;options...)
                x,d=m.value; seconds+=m.time; bytes+=m.bytes; gc+=m.gctime
                peak=max(peak,maximum(d.ranks))
                if repetition==1 && j==9
                    serialize(joinpath(dir,"$(method)_input.jls"),x)
                end
            end
            serialize(joinpath(dir,"$(method)_$(repetition).jls"),x)
            push!(rows,(;method,repetition,seconds,bytes,gc,peak_rank=peak,trace_error=abs(HLT.dot(p.conserved,x)-1)))
            HLT.write_rows(joinpath(dir,"timings.csv"),rows)
            println(last(rows)); flush(stdout)
        end
        x=deserialize(joinpath(dir,"$(method)_input.jls"))
        step(x,p.operator,0.1;options...)
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate=0.01 step(x,p.operator,0.1;options...)
        groups=Dict{String,Tuple{Int,Int}}()
        sites=Dict{String,Tuple{Int,Int}}()
        for a in Profile.Allocs.fetch().allocs
            frames=a.stacktrace
            funcs=string.([f.func for f in frames])
            files=string.([f.file for f in frames])
            category=any(==("_orth_columns"),funcs) ? "basis_orthogonalization" :
                any(f->occursin("KrylovKit",f),files) ? "krylov_including_rhs" :
                any(==("orthogonalize!"),funcs) ? "canonicalization" : "other"
            count,bytes=get(groups,category,(0,0)); groups[category]=(count+1,bytes+a.size)
            k=findfirst(f->occursin("TTNSolver/src",f),files)
            if !isnothing(k)
                f=frames[k]; key="$(basename(string(f.file))):$(f.line) $(f.func)"
                count,bytes=get(sites,key,(0,0)); sites[key]=(count+1,bytes+a.size)
            end
        end
        HLT.write_rows(joinpath(dir,"$(method)_allocation_categories.csv"),
            [(;category=k,samples=v[1],sampled_bytes=v[2],estimated_bytes=v[2]/0.01) for (k,v) in sort(collect(groups);by=x->-x.second[2])])
        HLT.write_rows(joinpath(dir,"$(method)_allocation_sites.csv"),
            [(;site=k,samples=v[1],sampled_bytes=v[2],estimated_bytes=v[2]/0.01) for (k,v) in sort(collect(sites);by=x->-x.second[2])])
        Profile.Allocs.clear()
    end
end
allocation_trial(isempty(ARGS) ? "before" : ARGS[1])
