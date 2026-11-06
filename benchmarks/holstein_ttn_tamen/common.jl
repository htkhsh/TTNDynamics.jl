module HolsteinTAMEnComparison
include("../holstein_longtime/driver.jl")
using .HolsteinLongTime, LinearAlgebra, Serialization, TOML
const HLT=HolsteinLongTime
const TS=HLT.TTNSolver
const TTS=HLT.TTSolver
const TD=HLT.TTNDynamics
# Separate output roots keep runs from different solver sources apart.
default_root()=joinpath(@__DIR__,"output",get(ENV,"HOLSTEIN_TTN_TAMEN_OUTPUT","five_fs"))
const TT_ACCEPTANCE=Dict("tree_tt"=>5e-5,"tree_tt_tight"=>3e-5,"tree_tt_loose"=>1e-4,
    "tree_tt_channels"=>5e-5,"tree_tt_channels_tight"=>3e-5,"tree_tt_channels_loose"=>1e-4,
    "tree_center"=>5e-5,"tree_center_tight"=>3e-5,"tree_center_loose"=>1e-4)

# Runs named *center* use heom_tree(...; system_position=:center). Its leaves
# are visited in the physical order `perm` = (baths of the first half, ket,
# bra, baths of the second half); TT references and TT runs for these cases
# use the same core order, so full-state errors remain cheap contractions.
run_order(name)=occursin("center",name) ? :center : :original

"""Physical modes in the order a tree visits its leaves."""
function leaf_sequence(tree)
    modes=Dict(node=>m for (m,node) in enumerate(tree.leaf_order))
    sequence=Int[]
    function visit(node)
        left,right=tree.children[node]
        if left==0
            push!(sequence,modes[node])
        else
            visit(left); visit(right)
        end
    end
    visit(tree.root)
    return sequence
end

"""Problem view for one core/leaf order; `:original` keeps the stored model."""
function ordered_problem(p,order)
    n=length(p.operator.dimensions)
    order==:original && return merge(p,(;order,perm=collect(1:n),ketpos=1,
        mpo=p.problem.liouvillian,trace=p.problem.trace_observable))
    order==:center || error("unknown order $order")
    tree=TD.heom_tree(p.system;bath_grouping=:coupling,system_position=:center)
    perm=leaf_sequence(tree)
    ketpos=findfirst(==(1),perm)
    ketpos<n && perm[ketpos+1]==2 || error("ket and bra must stay adjacent")
    inverse=invperm(perm)
    sop=TS.SumOfProducts(p.operator.dimensions[perm],[TS.ProductTerm(t.coefficient,
        Dict(inverse[m]=>f for (m,f) in t.factors)) for t in p.operator.terms])
    # Only the ket–bra bond of the initial state and trace may exceed one.
    function reorder(x)
        r=TTS.tt_ranks(x)
        all(==(1),r[[1;4:end]]) || error("cannot reorder a TT with bath correlations")
        return TTS.TTTensor([copy(x.cores[m]) for m in perm])
    end
    rho=TD.root_density_matrix(p.tree,p.system)
    x0=TD.heom_initial_state(p.system,rho;tree)
    return merge(p,(;order,perm,ketpos,tree=x0,conserved=TD.trace_observable(p.system;tree),
        train=reorder(p.train),mpo=HLT.sop_to_mpo(sop;tolerance=1e-13),
        trace=reorder(p.problem.trace_observable)))
end

# <train, tree> for a TT whose core j holds physical mode perm[j]. Subtrees
# must be contiguous in that core order (true for the tree defining perm).
function ordered_overlap(train,tree,perm)
    position=Dict(m=>j for (j,m) in enumerate(perm))
    env=Dict{Int,Array{ComplexF64,3}}(); spans=Dict{Int,UnitRange{Int}}()
    for (mode,node) in enumerate(tree.tree.leaf_order)
        j=position[mode]; g=train.cores[j]; a,n,b=size(g); u=tree.leaves[node]
        env[node]=reshape(transpose(u)*reshape(permutedims(conj.(g),(2,1,3)),n,:),size(u,2),a,b)
        spans[node]=j:j
    end
    for node in tree.tree.postorder
        l,r=tree.tree.children[node]; l==0 && continue
        last(spans[l])+1==first(spans[r]) || throw(ArgumentError("noncontiguous subtree"))
        spans[node]=first(spans[l]):last(spans[r])
        L,R,C=env[l],env[r],tree.cores[node]
        p,nl,nr=size(C); _,a,b=size(L); _,_,c=size(R)
        T=reshape(permutedims(L,(2,1,3)),a,nl*b)
        CR=reshape(reshape(C,p*nl,nr)*reshape(R,nr,b*c),p,nl,b,c)
        env[node]=permutedims(reshape(T*reshape(permutedims(CR,(2,3,1,4)),nl*b,p*c),a,p,c),(2,1,3))
        delete!(env,l); delete!(env,r)
    end
    return only(env[tree.tree.root])
end

function state_error(x,reference,q)
    q.order==:original && return HLT.comparison_error(x,reference)
    x isa TTS.TTTensor && return HLT.relative_error(x,reference)
    n2=real(TTS.tt_dot(reference,reference))
    squared=norm(x)^2+n2-2real(ordered_overlap(reference,x,q.perm))
    squared>=-100eps(Float64)*n2 || error("negative squared distance: $squared")
    return sqrt(max(0.0,squared)/n2)
end

function density(x,q)
    (q.order==:original || x isa TS.TTNTensor) && return HLT.density_of(x,q)
    left=ones(ComplexF64,1,1); right=ones(ComplexF64,1,1)
    for k in 1:q.ketpos-1; left=left*x.cores[k][:,1,:]; end
    for k in length(x.cores):-1:q.ketpos+2; right=x.cores[k][:,1,:]*right; end
    K,B=x.cores[q.ketpos],x.cores[q.ketpos+1]; d=size(K,2)
    return [only(left*K[:,i,:]*B[:,j,:]*right) for i in 1:d, j in 1:d]
end

function configuration(;sites=10,local_size=5,end_fs=5.0)
    cfg=HLT.load_config(joinpath(@__DIR__,"../holstein_longtime/config_50fs.toml"))
    cfg["model"]["sites"]=sites
    cfg["model"]["hierarchy_local_size"]=local_size
    cfg["time"]=Dict("end_fs"=>end_fs,"observation_fs"=>1.0,"checkpoint_fs"=>1.0)
    return cfg
end

function specification(cfg,name)
    # TT cases with "_center" anywhere (tt_center_serial, reference_refined_center).
    base=replace(name,"_center"=>"")
    if base!=name && base in ("reference","reference_refined","tt","tt_serial")
        s=specification(cfg,base); s["order"]="center"
        return s
    end
    if name in ("reference","reference_refined","tt","tt_serial")
        s=HLT.case_config(cfg,startswith(name,"tt") ? "tamen" : name)
        name=="tt_serial" && (s["blas_threads"]=1)
        return s
    end
    if haskey(TT_ACCEPTANCE,name)
        # TT-style work: truncate each sweep, accept on the local auxiliary
        # estimate, and solve local systems to 1% of the acceptance target.
        return Dict{String,Any}("method"=>"tree_tamen","dt_fs"=>0.5,"points"=>8,
            "tolerance"=>TT_ACCEPTANCE[name],"maxrank"=>96,"sweeps"=>20,"kick_rank"=>4,
            "blas_threads"=>1,"als_sweep"=>"etter","threaded"=>true,
            "truncation_rtol"=>1e-5,"certify"=>false,"local_tol_factor"=>0.01,
            "local_operator"=>(occursin("channels",name) || occursin("center",name)) ? "channels" : "sop",
            "order"=>String(run_order(name)))
    end
    name in ("tree","tree_tight","tree_loose") || error("unknown run $name")
    return Dict{String,Any}("method"=>"tree_tamen","dt_fs"=>0.5,"points"=>8,
        "tolerance"=>(name=="tree_tight" ? 1e-5 : name=="tree_loose" ? 5e-4 : 1e-4),
        "maxrank"=>96,"sweeps"=>20,"kick_rank"=>4,"blas_threads"=>1,
        "als_sweep"=>"etter","threaded"=>true)
end

function advance(p,x,s,work,hint)
    if s["method"]=="tamen"
        get(s,"order","original")=="center" || return HLT.advance(p,x,s,work)
        h=s["dt_fs"]
        st=TTS.tt_kron(x,TTS.tt_ones(s["points"];T=ComplexF64))
        y,grid,_,_,_=TTS.tamen(st,h*p.mpo,s["tolerance"];time_scheme="cheb",nswp=s["sweeps"],
            local_iters=s["local_iters"],kickrank=s["kick_rank"],time_error_damp=s["time_error_damp"],
            correct_2_norm=false,conserved_observables=[p.trace],workspace=work,verb=0)
        return TTS.extract_snapshot(y,grid,1.0,"cheb"),(;intervals=length(grid))
    end
    return TS.tamen_step(x,p.operator,s["dt_fs"];workspace=work,
        time_points=s["points"],rtol=s["tolerance"],maxrank=s["maxrank"],
        maxsweeps=s["sweeps"],kickrank=s["kick_rank"],conserved=p.conserved,
        als_sweep=Symbol(s["als_sweep"]),threaded=s["threaded"],
        warm_start=hint,store_warm_start=true,estimate_time_error=true,
        truncation_rtol=get(s,"truncation_rtol",0.0),certify=get(s,"certify",true),
        local_tol_factor=get(s,"local_tol_factor",nothing),
        local_operator=Symbol(get(s,"local_operator","sop")))
end

function run_case(name;root=default_root(),wall_seconds=1200.0,repeats=1,
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
        built=@timed ordered_problem(HLT.comparison_problem(cfg),Symbol(get(s,"order","original")))
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
        # Compile both the cold first interval and the warm-started path.
        @elapsed begin
            y,d=advance(p,deepcopy(initial),s,work,nothing)
            istree && advance(p,y,s,work,d.warm_start)
        end
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
            println(io,"step,time_fs,seconds,cumulative_seconds,bytes,gc_seconds,maxrank,sweeps,global_checks,relative_bound,global_seconds,addition_seconds,weight_qr_seconds,svd_seconds,basis_update_seconds,final_qr_seconds,time_defect,relative_estimate,matvecs,als_seconds,environment_seconds,time_error_seconds")
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
                    # Uncertified runs are validated against the refined reference instead.
                    @assert d.converged && (d.history[end].global_residual_checked || !get(s,"certify",true))
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
                    istree ? d.time_error.absolute : NaN,
                    istree ? d.relative_residual_estimate : NaN,istree ? d.matvecs : 0,
                    istree ? d.timings.als_seconds : 0.,istree ? d.timings.environment_seconds : 0.,
                    istree ? d.timings.time_error_seconds : 0.),','));flush(io)
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

function analyze(;root=default_root())
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
    # Each order has its own pair of references: reference(_center) and
    # reference_refined(_center). A run is compared with its order's pair.
    suffix(order)=order==:original ? "" : "_center"
    refnames(order)=("reference"*suffix(order),"reference_refined"*suffix(order))
    orders=[o for o in (:original,:center) if any(n->run_order(n)==o,names)]
    common=Dict{Symbol,Vector{Int}}()
    for order in orders
        common[order]=Int[]
        for t in 1:total
            all(isfile(joinpath(root,n,"rep1","time_$t.jls")) for n in refnames(order)) || break
            push!(common[order],t)
        end
    end
    statuses=NamedTuple[]
    for name in names
        path=joinpath(root,name,"status.toml")
        status=isfile(path) ? TOML.parsefile(path) : Dict("phase"=>"interrupted","time_fs"=>0.,"completed_repetitions"=>0)
        push!(statuses,(;name,phase=status["phase"],time_fs=status["time_fs"],
            completed_repetitions=status["completed_repetitions"],reference_checked=false))
    end
    HLT.write_rows(joinpath(root,"run_status.csv"),statuses)
    if all(o->isempty(common[o]),orders)
        println("No common reference checkpoints; every attempted run remains unqualified.")
        return statuses
    end
    base_problem=HLT.comparison_problem(cfg)
    views=Dict(order=>ordered_problem(base_problem,order) for order in orders)
    BLAS.set_num_threads(8)
    checks=Dict{Symbol,Vector{NamedTuple}}(); convergence=NamedTuple[]; checked=Dict{Symbol,Bool}()
    for order in orders
        q=views[order]; coarse,fine=refnames(order)
        checks[order]=NamedTuple[]; maxstate=0.;maxrho=0.
        for t in common[order]
            a=deserialize(joinpath(root,coarse,"rep1","time_$t.jls"))
            b=deserialize(joinpath(root,fine,"rep1","time_$t.jls"))
            maxstate=max(maxstate,state_error(a,b,q))
            maxrho=max(maxrho,norm(density(a,q)-density(b,q)))
            row=(;order=String(order),time_fs=t,max_state=maxstate,max_density=maxrho,
                qualified=maxstate<1e-4 && maxrho<1e-4)
            push!(checks[order],row); push!(convergence,row)
        end
        checked[order]=length(checks[order])==total && last(checks[order]).qualified
    end
    HLT.write_rows(joinpath(root,"reference_convergence.csv"),convergence)
    HLT.write_rows(joinpath(root,"run_status.csv"),
        [merge(r,(;reference_checked=get(checked,run_order(r.name),false))) for r in statuses])
    rows=NamedTuple[]
    references=Set(n for o in orders for n in refnames(o))
    for name in filter(n->!(n in references),names)
        isfile(joinpath(root,name,"runs.csv")) || continue
        order=run_order(name); q=views[order]; fine=last(refnames(order))
        for run in csvrows(joinpath(root,name,"runs.csv"))
            rep=parse(Int,run["repetition"]);maxstate=0.;maxrho=0.;maxtrace=0.
            steps=csvrows(joinpath(root,name,"rep$rep","steps.csv"))
            for t in common[order]
                path=joinpath(root,name,"rep$rep","time_$t.jls");isfile(path) || break
                x=deserialize(path);ref=deserialize(joinpath(root,fine,"rep1","time_$t.jls"))
                rho=density(x,q)
                maxstate=max(maxstate,state_error(x,ref,q))
                maxrho=max(maxrho,norm(rho-density(ref,q)));maxtrace=max(maxtrace,abs(tr(rho)-1))
                lastrow=last(filter(r->parse(Float64,r["time_fs"])<=t+1e-12,steps))
                push!(rows,(;name,repetition=rep,time_fs=t,seconds=parse(Float64,lastrow["cumulative_seconds"]),
                    max_state_error=maxstate,max_density_error=maxrho,max_trace_error=maxtrace,
                    qualified=checks[order][t].qualified && maxstate<1e-3 && maxrho<1e-3,
                    maxrank=parse(Int,lastrow["maxrank"])))
            end
        end
    end
    isempty(rows) || HLT.write_rows(joinpath(root,"comparison.csv"),rows)
    foreach(println,rows)
end
end
