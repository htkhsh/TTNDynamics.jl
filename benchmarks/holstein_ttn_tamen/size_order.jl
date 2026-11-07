# 5 fs timing for an order where the initially excited site's baths sit next to
# the system: [sites N/2..1, ket, bra, sites N..N/2+1] (TT and a TTN with
# the same leaf order: (left half, (system, right half)), halves balanced).
include(joinpath(@__DIR__,"common.jl"))
using .HolsteinTAMEnComparison, LinearAlgebra, Random, Serialization
const H=HolsteinTAMEnComparison; const HLT=H.HLT; const TS=HLT.TTNSolver; const TD=HLT.TTNDynamics; const TTS=HLT.TTSolver
function main()
    N=parse(Int,ARGS[1]); method=ARGS[2]
    cfg=H.configuration(;sites=N)
    p=HLT.comparison_problem(cfg); n=length(p.operator.dimensions); sys=p.system
    site=[findfirst(j->m.coupling[j,j]!=0,1:N) for m in sys.modes]
    modes_of(j)=[2+k for k in eachindex(site) if site[k]==j]
    h=N÷2
    left=reduce(vcat,[modes_of(j) for j in h:-1:1]); right=reduce(vcat,[modes_of(j) for j in N:-1:h+1])
    perm=[left;1;2;right]
    # Tree with this in-order: (balanced(left pairs), ((ket,bra), balanced(right pairs)))
    children=fill((0,0),n)
    join!(a,b)=(push!(children,(a,b)); length(children))
    function bal(nodes)
        length(nodes)==1 && return nodes[1]
        m=cld(length(nodes),2); join!(bal(nodes[1:m]),bal(nodes[m+1:end]))
    end
    lp=[join!(modes_of(j)...) for j in h:-1:1]; rp=[join!(modes_of(j)...) for j in N:-1:h+1]
    a=bal(lp); s=join!(1,2); c=bal(rp); join!(a,join!(s,c))
    tree=TS.BinaryTree(children;leaf_order=collect(1:n))
    @assert H.leaf_sequence(tree)==perm
    inverse=invperm(perm)
    sop=TS.SumOfProducts(p.operator.dimensions[perm],[TS.ProductTerm(t.coefficient,Dict(inverse[m]=>f for (m,f) in t.factors)) for t in p.operator.terms])
    reorder(x)=TTS.TTTensor([copy(x.cores[m]) for m in perm])
    rho=TD.root_density_matrix(p.tree,sys); x0=TD.heom_initial_state(sys,rho;tree)
    q=merge(p,(;order=:custom,perm,ketpos=findfirst(==(1),perm),tree=x0,conserved=TD.trace_observable(sys;tree),
        train=reorder(p.train),mpo=HLT.sop_to_mpo(sop;tolerance=1e-13),trace=reorder(p.problem.trace_observable)))
    println("N=$N MPO max rank (near order) = ",maximum(size(g,1) for g in q.mpo.cores))
    name=method=="tt" ? "tt_center" : "tree_center"
    s=H.specification(cfg,name); haskey(ENV,"BLAS") && (s["blas_threads"]=parse(Int,ENV["BLAS"])); s["maxrank"]=256
    BLAS.set_num_threads(s["blas_threads"])
    istree=method!="tt"
    work=istree ? TS.TAMEnWorkspace() : HLT.TAMEnWorkspace(ComplexF64)
    xs=istree ? q.tree : q.train
    y,d=H.advance(q,deepcopy(xs),s,work,nothing); istree && H.advance(q,y,s,work,d.warm_start)
    x=deepcopy(xs); hint=nothing; total=0.0; GC.gc()
    for j in 1:10
        HLT.Random.seed!(cfg["execution"]["seed"]+j)
        m=@timed H.advance(q,x,s,work,hint); x,d=m.value; istree && (hint=d.warm_start); total+=m.time
    end
    out=joinpath(@__DIR__,"output","sizes_near"); mkpath(out)
    serialize(joinpath(out,"N$(N)_$(method).jls"),(;x,perm))
    println("N=$N $method threads=$(Threads.nthreads()) blas=$(s["blas_threads"]) 5 fs: $(round(total,digits=2)) s rank=$(maximum(HLT.ranks_of(x)))")
end
main()
