"""Contract <train, tree> without forming the full p × nl × b × c temporary.

Chunk the right TT bond to a workspace budget. One slice and the persistent
subtree environments can exceed this budget; it is not a total memory cap.
"""
function comparison_overlap(train::TTTensor, tree::TTNTensor; workspace_bytes=64*1024^2)
    workspace_bytes isa Integer && workspace_bytes>0 || throw(ArgumentError("positive workspace budget required"))
    tt_dims(train)==leafdims(tree) || throw(DimensionMismatch("physical dimensions"))
    env=Dict{Int,Array{ComplexF64,3}}()
    spans=Dict{Int,UnitRange{Int}}()
    for (j,node) in enumerate(tree.tree.leaf_order)
        g=train.cores[j]; a,n,b=size(g); u=tree.leaves[node]
        env[node]=reshape(transpose(u)*reshape(permutedims(conj.(g),(2,1,3)),n,:),size(u,2),a,b)
        spans[node]=j:j
    end
    for node in tree.tree.postorder
        l,r=tree.tree.children[node]
        l==0 && continue
        last(spans[l])+1==first(spans[r]) || throw(ArgumentError("noncontiguous subtree"))
        spans[node]=first(spans[l]):last(spans[r])
        L,R,C=env[l],env[r],tree.cores[node]
        p,nl,nr=size(C); _,a,b=size(L); _,br,c=size(R)
        b==br || throw(DimensionMismatch("TT bond"))
        left=reshape(permutedims(L,(2,1,3)),a,nl*b)
        core=reshape(C,p*nl,nr); right=reshape(R,nr,b*c)
        width=max(1,min(c,div(workspace_bytes,sizeof(ComplexF64)*2p*(nl*b+a))))
        result=Array{ComplexF64}(undef,p,a,c)
        for lo in 1:width:c
            hi=min(c,lo+width-1); count=hi-lo+1
            temp=reshape(core*view(right,:,(lo-1)*b+1:hi*b),p,nl,b,count)
            part=left*reshape(permutedims(temp,(2,3,1,4)),nl*b,p*count)
            result[:,:,lo:hi].=permutedims(reshape(part,a,p,count),(2,1,3))
        end
        env[node]=result
        delete!(env,l); delete!(env,r)
    end
    return only(env[tree.tree.root])
end

function comparison_error(tree::TTNTensor, reference::TTTensor)
    n2=real(dot(reference,reference))
    squared=norm(tree)^2+n2-2real(comparison_overlap(reference,tree))
    squared>=-100eps(Float64)*n2 || error("negative squared distance: $squared")
    return sqrt(max(0.0,squared)/n2)
end
comparison_error(train::TTTensor,reference::TTTensor)=relative_error(train,reference)

function committed_rows(dir,file,saved)
    offset=saved.offsets[file]
    path=joinpath(dir,file)
    filesize(path)>=offset || throw(ArgumentError("short log: $path"))
    data=open(path) do io
        String(read(io,offset))
    end
    lines=split(chomp(data),'\n'); header=split(first(lines),',')
    result=Dict{String,String}[]
    for line in lines[2:end]
        fields=split(line,',')
        length(fields)==length(header) || throw(ArgumentError("invalid CSV row in $path"))
        push!(result,Dict(zip(header,fields)))
    end
    return result
end

function checked_saved(cfg,root,name)
    path=joinpath(root,name,"latest.jls")
    isfile(path) || throw(ArgumentError("missing run: $name"))
    saved=deserialize(path)
    saved.version==1 && saved.identity==run_identity(cfg,name) ||
        throw(ArgumentError("saved model/grid/solver differs: $name"))
    s=case_config(cfg,name)
    0<=saved.step<=grid_count(cfg["time"]["end_fs"],s["dt_fs"]) || throw(ArgumentError("invalid saved step"))
    # Do not retain every run's potentially large final tensor during analysis.
    return (;(key=>getproperty(saved,key) for key in propertynames(saved) if key!=:state)...)
end

function check_compatible_runs(cfg,saved)
    ref=saved[cfg["comparison"]["reference"]]
    context(runtime)=Dict(k=>v for (k,v) in runtime if k!="blas_threads")
    for (name,s) in saved
        s.sources==ref.sources || throw(ArgumentError("source/dependency mismatch between runs: $name"))
        context(s.runtime)==context(ref.runtime) || throw(ArgumentError("CPU/Julia/thread settings differ between runs: $name"))
        s.runtime["blas_threads"]==case_config(cfg,name)["blas_threads"] ||
            throw(ArgumentError("BLAS setting does not match configured run: $name"))
    end
    return nothing
end

function archived_state(cfg,root,name,t)
    j=grid_count(t,case_config(cfg,name)["dt_fs"])
    path=joinpath(root,name,"checkpoints","step_$j.jls")
    isfile(path) || throw(ArgumentError("missing checkpoint: $name at $t fs"))
    saved=deserialize(path)
    saved.step==j && saved.identity==run_identity(cfg,name) || throw(ArgumentError("checkpoint mismatch: $path"))
    return saved.state
end

function write_rows(path,rows)
    isempty(rows) && throw(ArgumentError("no rows for $path"))
    open(path*".tmp","w") do io
        println(io,join(string.(keys(first(rows))),','))
        for row in rows
            println(io,join((isnothing(x) ? "NA" : x for x in values(row)),','))
        end
    end
    mv(path*".tmp",path;force=true)
end

function compare_runs(cfg,root;names=cfg["comparison"]["runs"],allow_partial=false)
    validate_config(cfg)
    isempty(names) && throw(ArgumentError("no comparison runs"))
    c=cfg["comparison"]; reference=c["reference"]; check=c["reference_check"]
    labels=unique([reference;names])
    saved=Dict(name=>checked_saved(cfg,root,name) for name in labels)
    total=cfg["time"]["end_fs"]
    reached(name)=saved[name].step*case_config(cfg,name)["dt_fs"]
    complete=all(name->saved[name].complete && isapprox(reached(name),total;atol=1e-10),labels)
    !allow_partial && !complete && throw(ArgumentError("incomplete trajectory; use --partial explicitly"))
    observation=cfg["time"]["observation_fs"]
    stop=complete ? total : floor((minimum(reached(name) for name in labels)+1e-10)/observation)*observation
    check_ready=false
    if isfile(joinpath(root,check,"latest.jls"))
        saved[check]=checked_saved(cfg,root,check)
        check_ready=complete && saved[check].complete
    end
    check_compatible_runs(cfg,saved)
    compared=unique([names;check_ready ? [check] : String[]])
    metrics=Dict(name=>(;state=0.0,density=0.0,population=0.0) for name in compared)
    # Full-state comparisons keep only one reference/candidate pair in memory.
    checkpoint=cfg["time"]["checkpoint_fs"]
    times=sort(unique([collect(0.0:checkpoint:total);Float64(total)]))
    state_rows=NamedTuple[]
    for t in filter(t->t<=stop+1e-10,times)
        ref=archived_state(cfg,root,reference,t)
        ref isa TTTensor || throw(ArgumentError("reference state is not TT"))
        for name in compared
            x=archived_state(cfg,root,name,t)
            delta=comparison_error(x,ref)
            isfinite(delta) || error("nonfinite state difference: $name at $t")
            m=metrics[name]; metrics[name]=merge(m,(;state=max(m.state,delta)))
            push!(state_rows,(;run=name,reference,time_fs=t,state_relative_difference=delta))
        end
    end
    n=cfg["model"]["sites"]
    rho(row)=[complex(parse(Float64,row["rho_$(a)_$(b)_re"]),parse(Float64,row["rho_$(a)_$(b)_im"])) for a in 1:n,b in 1:n]
    observations=Dict{String,Dict{Int,Dict{String,String}}}()
    for name in unique([reference;compared])
        rows=committed_rows(joinpath(root,name),"observables.csv",saved[name])
        indexed=Dict{Int,Dict{String,String}}()
        for row in rows
            i=grid_count(parse(Float64,row["time_fs"]),observation)
            haskey(indexed,i) && throw(ArgumentError("duplicate observation: $name"))
            indexed[i]=row
        end
        all(haskey(indexed,i) for i in 0:grid_count(stop,observation)) || throw(ArgumentError("missing observations: $name"))
        observations[name]=indexed
    end
    density_rows=NamedTuple[]
    for i in 0:grid_count(stop,observation)
        ref=rho(observations[reference][i])
        for name in compared
            diff=rho(observations[name][i])-ref
            de=norm(diff); pe=maximum(abs,diag(diff))
            isfinite(de) && isfinite(pe) || error("nonfinite density difference")
            m=metrics[name]; metrics[name]=merge(m,(;density=max(m.density,de),population=max(m.population,pe)))
            push!(density_rows,(;run=name,reference,time_fs=i*observation,density_absolute_difference=de,
                                population_max_difference=pe))
        end
    end
    status=!check_ready ? "unchecked" :
        metrics[check].state<=0.1*c["state_target"] && metrics[check].density<=0.1*c["density_target"] ? "checked" : "failed"
    summaries=NamedTuple[]
    for name in names
        s=case_config(cfg,name); rows=committed_rows(joinpath(root,name),"steps.csv",saved[name])
        all(parse(Int,row["step"])==i-1 for (i,row) in enumerate(rows)) || throw(ArgumentError("step log has gaps: $name"))
        j=grid_count(stop,s["dt_fs"])
        length(rows)>=j+1 || throw(ArgumentError("short step log: $name"))
        m=metrics[name]
        qualified=complete && status=="checked" && m.state<=c["state_target"] && m.density<=c["density_target"]
        push!(summaries,(;run=name,method=s["method"],complete,compared_until_fs=stop,
            propagation_seconds=parse(Float64,rows[j+1]["cumulative_seconds"]),
            peak_rank=maximum(parse(Int,r["maxrank"]) for r in rows[1:j+1]),
            truncation_misses=count(r->r["truncation_met"]=="false",rows[1:j+1]),
            max_state_relative_difference=m.state,max_density_difference=m.density,max_population_difference=m.population,
            reference_status=status,accuracy_qualified=qualified,speedup_vs_tamen=nothing))
    end
    baseline=findfirst(r->r.run=="tamen" && r.accuracy_qualified,summaries)
    if !isnothing(baseline)
        seconds=summaries[baseline].propagation_seconds
        summaries=[merge(r,(;speedup_vs_tamen=r.accuracy_qualified && r.propagation_seconds>0 ? seconds/r.propagation_seconds : nothing)) for r in summaries]
    end
    out=joinpath(root,allow_partial ? "comparison_partial" : "comparison"); mkpath(out)
    write_rows(joinpath(out,"state_differences.csv"),state_rows)
    write_rows(joinpath(out,"density_differences.csv"),density_rows)
    write_rows(joinpath(out,"summary.csv"),summaries)
    atomic_toml(joinpath(out,"comparison.toml"),Dict("model"=>cfg["model"],"reference"=>reference,
        "reference_check"=>check,"reference_status"=>status,"runs"=>names,"complete"=>complete,
        "compared_until_fs"=>stop,"requested_until_fs"=>total,"state_target"=>c["state_target"],"density_target"=>c["density_target"]))
    for r in summaries
        println(r.run,": 0–",stop," fs, ",r.propagation_seconds," s, state difference=",
                r.max_state_relative_difference,", reference=",status,", accuracy qualified=",r.accuracy_qualified)
    end
    return summaries
end
