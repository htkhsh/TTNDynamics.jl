include("../holstein_longtime/driver.jl")
using .HolsteinLongTime, LinearAlgebra, Serialization, TOML
const HLT=HolsteinLongTime

function csvrows(path)
    lines=readlines(path); header=split(first(lines),',')
    return [Dict(zip(header,split(line,','))) for line in lines[2:end]]
end
function density(row,n)
    return [complex(parse(Float64,row["rho_$(a)_$(b)_re"]),parse(Float64,row["rho_$(a)_$(b)_im"])) for a in 1:n,b in 1:n]
end

function analyze(label)
    occursin(r"^[A-Za-z0-9_-]+$",label) || error("invalid label")
    root=joinpath(@__DIR__,"output",label)
    names=sort([name for name in readdir(root) if isfile(joinpath(root,name,"runs.csv"))])
    all(name in names for name in ("reference","reference_refined","tamen")) || error("missing TT runs")
    meta=Dict(name=>TOML.parsefile(joinpath(root,name,"metadata.toml")) for name in names)
    base=meta["reference_refined"]
    context(m)=Dict(k=>v for (k,v) in m["runtime"] if k!="blas_threads")
    for name in names
        m=meta[name]
        m["sources"]==base["sources"] || error("source mismatch: $name")
        m["scripts"]==base["scripts"] || error("measurement script mismatch: $name")
        m["config"]==base["config"] || error("problem mismatch: $name")
        context(m)==context(base) || error("runtime mismatch: $name")
        m["runtime"]["blas_threads"]==m["spec"]["blas_threads"] || error("BLAS mismatch")
    end
    n=base["config"]["model"]["sites"]
    target=base["config"]["comparison"]
    total=Int(base["config"]["time"]["end_fs"])
    refs=NamedTuple[]
    maxstate=0.0; maxdensity=0.0
    for t in 1:total
        paths=[joinpath(root,name,"rep1","time_$t.jls") for name in ("reference","reference_refined")]
        all(isfile,paths) || break
        a,b=deserialize.(paths)
        state=HLT.relative_error(a,b)
        # Density observations were saved outside the timed propagation.
        observations=[only(filter(r->parse(Float64,r["time_fs"])==t,
            csvrows(joinpath(root,name,"rep1","observables.csv")))) for name in ("reference","reference_refined")]
        delta=norm(density(observations[1],n)-density(observations[2],n))
        maxstate=max(maxstate,state); maxdensity=max(maxdensity,delta)
        push!(refs,(;time_fs=t,state_difference=state,density_difference=delta,
            max_state_difference=maxstate,max_density_difference=maxdensity,
            checked=maxstate<=target["state_target"]/10 && maxdensity<=target["density_target"]/10))
    end
    isempty(refs) && error("no common reference checkpoint")
    HLT.write_rows(joinpath(root,"reference_convergence.csv"),refs)
    rows=NamedTuple[]
    # Use BLAS8 for untimed TT/TTN error contractions; only one pair at a time.
    BLAS.set_num_threads(8)
    for name in filter(x->!(x in ("reference","reference_refined")),names)
        for run in csvrows(joinpath(root,name,"runs.csv"))
            repetition=parse(Int,run["repetition"]); dir=joinpath(root,name,"rep$repetition")
            steps=csvrows(joinpath(dir,"steps.csv")); obs=csvrows(joinpath(dir,"observables.csv"))
            all(parse(Int,r["step"])==j for (j,r) in enumerate(steps)) || error("step gap")
            maxstate=0.0; maxdensity=0.0; maxtrace=0.0
            for refcheck in refs
                t=refcheck.time_fs; path=joinpath(dir,"time_$t.jls")
                isfile(path) || break
                x=deserialize(path); ref=deserialize(joinpath(root,"reference_refined","rep1","time_$t.jls"))
                state=HLT.comparison_error(x,ref)
                r=only(filter(r->parse(Float64,r["time_fs"])==t,obs))
                rr=only(filter(r->parse(Float64,r["time_fs"])==t,csvrows(joinpath(root,"reference_refined","rep1","observables.csv"))))
                delta=norm(density(r,n)-density(rr,n))
                all(isfinite,(state,delta)) || error("nonfinite error")
                maxstate=max(maxstate,state); maxdensity=max(maxdensity,delta)
                maxtrace=max(maxtrace,parse(Float64,r["trace_error"]))
                prefix=filter(r->parse(Float64,r["time_fs"])<=t+1e-10,steps)
                expected_steps=HLT.grid_count(t,meta[name]["spec"]["dt_fs"])
                length(prefix)==expected_steps || error("incomplete step prefix")
                push!(rows,(;name,repetition,time_fs=t,dt_fs=meta[name]["spec"]["dt_fs"],
                    seconds=parse(Float64,last(prefix)["cumulative_seconds"]),
                    allocated_GiB=sum(parse(Int,r["allocated_bytes"]) for r in prefix)/1024^3,
                    peak_rank=maximum(parse(Int,r["maxrank"]) for r in prefix),
                    truncation_misses=count(r->r["truncation_met"]=="false",prefix),
                    state_difference=state,density_difference=delta,
                    max_state_difference=maxstate,max_density_difference=maxdensity,max_trace_error=maxtrace,
                    reference_checked=refcheck.checked,
                    accuracy_qualified=refcheck.checked && maxstate<=target["state_target"] && maxdensity<=target["density_target"]))
                HLT.write_rows(joinpath(root,"measurements.csv"),rows)
                println(last(rows)); flush(stdout)
            end
        end
    end
end

length(ARGS)==1 || error("Usage: recompare_analysis.jl LABEL")
analyze(only(ARGS))
