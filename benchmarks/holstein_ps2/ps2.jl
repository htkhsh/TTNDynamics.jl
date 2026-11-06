module PS2Trial
using TTNSolver, LinearAlgebra
const TS=TTNSolver
export ps2_step

# Arrays keep the original node leg order throughout the moving-centre sweep.
# A zero leg denotes the physical index; all other legs are adjacent node IDs.
function network(x)
    tree=x.tree
    legs=Vector{Vector{Int}}(undef,length(tree.children))
    data=Vector{Array{ComplexF64}}(undef,length(legs))
    modes=zeros(Int,length(legs))
    for (mode,node) in enumerate(tree.leaf_order);modes[node]=mode;end
    for node in eachindex(legs)
        parent=tree.parent[node]
        if TS._isleaf(tree,node)
            legs[node]=parent==0 ? [0] : [0,parent]
            data[node]=parent==0 ? vec(copy(x.leaves[node])) : copy(x.leaves[node])
        else
            legs[node]=parent==0 ? collect(tree.children[node]) : [parent;collect(tree.children[node])]
            data[node]=parent==0 ? reshape(copy(x.cores[node]),size(x.cores[node])[2:3]) : copy(x.cores[node])
        end
    end
    return (;tree,legs,data,modes)
end

function unpack(net)
    leaves=Dict(n=>reshape(copy(net.data[n]),size(net.data[n],1),:) for n in net.tree.leaf_order)
    cores=Dict(n=>(n==net.tree.root ? reshape(copy(net.data[n]),1,size(net.data[n])...) : copy(net.data[n]))
        for n in net.tree.postorder if !TS._isleaf(net.tree,n))
    return TTNTensor(net.tree,leaves,cores)
end

others(A,axis)=[i for i in 1:ndims(A) if i!=axis]
unfold(A,axis)=reshape(permutedims(A,[others(A,axis);axis]),:,size(A,axis))
function fold(M,shape,axis)
    order=[i for i in eachindex(shape) if i!=axis]
    return permutedims(reshape(M,(shape[order]...,size(M,2))),invperm([order;axis]))
end
function mode_product(A,F,axis)
    isnothing(F) && return A
    order=[axis;others(A,axis)]
    input=permutedims(A,order)
    output=F*reshape(input,size(A,axis),:)
    return permutedims(reshape(output,(size(F,1),size(input)[2:end]...)),invperm(order))
end

mutable struct Stats
    solves::Int
    matvecs::Int
    maxdim::Int
    drop::Float64
    krylov_error::Float64
    solve_seconds::Float64
    environment_seconds::Float64
    split_seconds::Float64
end
Stats()=Stats(0,0,0,0.,0.,0.,0.,0.)

function ps2_step(state::TTNTensor,op::SumOfProducts,h::Real;
                  atol=1e-8,maxrank=64,conserved=nothing,krylov_tol=1e-12,
                  krylovdim=30,krylov_maxiter=100,krylov_eager=false,rank_padding=2,full_leaves=true)
    isfinite(h) && h>0 || throw(ArgumentError("positive finite step required"))
    TS._check_bug_options(atol,maxrank,krylov_tol,krylovdim,krylov_maxiter)
    TS._check_operator(op,state)
    rank_padding isa Integer && rank_padding>=1 || throw(ArgumentError("positive rank_padding required"))
    full_leaves && maximum(leafdims(state))>maxrank && throw(ArgumentError("maxrank must hold full physical leaf bases"))
    c=TS._canonical_conserved(state,conserved)
    if !isnothing(c)
        maximum(tree_ranks(c))<=maxrank || throw(ArgumentError("maxrank cannot hold protected directions"))
    end
    x=isnothing(c) ? orthogonalize!(copy(state)) : enrich_basis(state,c)
    if full_leaves && !TS._isleaf(x.tree,x.tree.root)
        for node in x.tree.leaf_order
            U=x.leaves[node];p=x.tree.parent[node]
            axis=x.tree.children[p][1]==node ? 2 : 3
            x.cores[p]=TS._child_product(x.cores[p],U,axis)
            x.leaves[node]=Matrix{ComplexF64}(I,size(U,1),size(U,1))
        end
    end
    net=network(x);cn=isnothing(c) ? nothing : network(c)
    env=Dict{Tuple{Int,Int},Vector{Union{Nothing,Matrix{ComplexF64}}}}()
    overlap=Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    # Reserve half the cut budget for enforcing the final rank cap: a protected
    # gauge move can append zero-weight invariant directions beyond that cap.
    stats=Stats();budget=Float64(atol)/max(1,4(length(net.data)-1))

    function factors(node,excluded,k)
        return [leg==excluded ? nothing : leg==0 ? get(op.terms[k].factors,net.modes[node],nothing) :
            env[(leg,node)][k] for leg in net.legs[node]]
    end
    function projected_conserved(node,target)
        C=cn.data[node]
        for (axis,leg) in enumerate(net.legs[node])
            (leg==0 || leg==target) && continue
            C=mode_product(C,overlap[(leg,node)],axis)
        end
        return unfold(C,findfirst(==(target),net.legs[node]))
    end
    function message!(node,target)
        t=time_ns();axis=findfirst(==(target),net.legs[node]);A=net.data[node]
        Q=unfold(A,axis)
        cache=Dict{Tuple,Matrix{ComplexF64}}()
        output=Vector{Union{Nothing,Matrix{ComplexF64}}}(undef,length(op.terms))
        for k in eachindex(op.terms)
            fs=factors(node,target,k)
            if all(isnothing,fs)
                output[k]=nothing
            else
                output[k]=get!(cache,Tuple(fs)) do
                    acted=A
                    for j in eachindex(fs);acted=mode_product(acted,fs[j],j);end
                    Q'*unfold(acted,axis)
                end
            end
        end
        env[(node,target)]=output
        isnothing(c) || (overlap[(node,target)]=Q'*projected_conserved(node,target))
        stats.environment_seconds+=(time_ns()-t)*1e-9
    end
    for node in net.tree.postorder
        node==net.tree.root || message!(node,net.tree.parent[node])
    end

    # A matrix-free local operator; Arnoldi owns each returned vector. Scratch
    # buffers are reused only inside this serial action closure.
    function action(nodes,exclude,shape)
        sets=[factors(node,exclude[i],k)[findall(!=(exclude[i]),net.legs[node])]
              for k in eachindex(op.terms), (i,node) in enumerate(nodes)]
        indices=Dict{Tuple,Int}();keys=Tuple[];coefficients=ComplexF64[]
        for k in eachindex(op.terms)
            key=Tuple(vcat((sets[k,i] for i in eachindex(nodes))...))
            if haskey(indices,key)
                coefficients[indices[key]]+=op.terms[k].coefficient
            else
                push!(keys,key);push!(coefficients,op.terms[k].coefficient);indices[key]=length(keys)
            end
        end
        terms=[(coefficients[i],keys[i]) for i in eachindex(keys) if !iszero(coefficients[i])]
        current=zeros(ComplexF64,shape);input=similar(current);product=similar(current)
        permutations=[[axis;[i for i in eachindex(shape) if i!=axis]] for axis in eachindex(shape)]
        inverses=invperm.(permutations)
        function rhs(A)
            stats.matvecs+=1
            result=zeros(ComplexF64,shape)
            for (coefficient,fs) in terms
                acted=A
                for axis in eachindex(fs)
                    F=fs[axis];isnothing(F) && continue
                    order=permutations[axis];dimensions=Tuple(shape[i] for i in order)
                    permutedims!(reshape(input,dimensions),acted,order)
                    mul!(reshape(product,shape[axis],:),F,reshape(input,shape[axis],:))
                    permutedims!(current,reshape(product,dimensions),inverses[axis])
                    acted=current
                end
                result .+= coefficient.*acted
            end
            return result
        end
        return rhs
    end
    function integrate(A,nodes,exclude,dt)
        t=time_ns();stats.solves+=1;stats.maxdim=max(stats.maxdim,length(A))
        scale=norm(A)
        isfinite(scale) || throw(TS.LocalSolveFailure(:nonfinite,Inf))
        iszero(scale) && return copy(A)
        rhs=action(nodes,exclude,size(A))
        y,info=TS.KrylovKit.exponentiate(rhs,Float64(dt),A/scale;ishermitian=false,issymmetric=false,
            tol=Float64(krylov_tol),krylovdim=min(krylovdim,length(A)),maxiter=krylov_maxiter,eager=krylov_eager,verbosity=0)
        info.converged==1 || throw(TS.LocalSolveFailure(:nonconvergence,Float64(info.normres)))
        y .*= scale
        all(isfinite,y) || throw(TS.LocalSolveFailure(:nonfinite,Inf))
        stats.krylov_error=max(stats.krylov_error,scale*info.normres)
        stats.solve_seconds+=(time_ns()-t)*1e-9
        return y
    end
    function site!(node,dt)
        net.data[node]=integrate(net.data[node],[node],[-1],dt)
    end
    function protected(node,target,rows)
        return isnothing(c) ? zeros(ComplexF64,rows,0) : TS._orth_columns(projected_conserved(node,target))
    end
    function gauge!(a,b)
        t=time_ns();ia=findfirst(==(b),net.legs[a]);ib=findfirst(==(a),net.legs[b])
        A=unfold(net.data[a],ia);B=unfold(net.data[b],ib)
        P=protected(a,b,size(A,1))
        # A gauge move must keep the old null directions too: removing them
        # here would undo PS2's deliberate rank padding before propagation.
        Q=Matrix(qr(A).Q)[:,1:min(size(A)...)]
        U=full_leaves && TS._isleaf(net.tree,a) ? Matrix{ComplexF64}(I,size(A,1),size(A,1)) : TS._orth_columns(P,Q)
        # A zero state still needs a nonempty isometric frame.
        isempty(U) && (U=Matrix{ComplexF64}(I,size(A,1),1))
        R=U'*A
        net.data[a]=fold(U,size(net.data[a]),ia)
        net.data[b]=fold(B*transpose(R),size(net.data[b]),ib)
        stats.split_seconds+=(time_ns()-t)*1e-9
        message!(a,b)
    end
    function cutspace(M,P,leaf)
        p=size(P,2);p<=maxrank || throw(ArgumentError("protected rank exceeds cap"))
        if full_leaves && leaf
            return Matrix{ComplexF64}(I,size(M,1),size(M,1)),0.
        end
        residual=p==0 ? copy(M) : M-P*(P'*M)
        fac=svd!(residual)
        count=0
        while count<length(fac.S) && norm(view(fac.S,count+1:length(fac.S)))>budget;count+=1;end
        available=min(size(M,1)-p,length(fac.S),maxrank-p)
        count=min(count,available)
        target=min(size(M,1),maxrank,max(p+1,rank_padding*(p+count)))
        # Retain the significant directions first; add orthogonal null-space
        # directions only as rank-padding, as motivated by the HEOM PS2 paper.
        U=TS._orth_columns(P,view(fac.U,:,1:min(length(fac.S),max(count,target-p))))
        for i in 1:size(M,1)
            size(U,2)>=target && break
            unit=zeros(ComplexF64,size(M,1),1);unit[i]=1
            U=TS._orth_columns(U,unit)
        end
        U=U[:,1:min(target,size(U,2))]
        return U,norm(view(fac.S,count+1:length(fac.S)))
    end
    function pair!(a,b,dt)
        ia=findfirst(==(b),net.legs[a]);ib=findfirst(==(a),net.legs[b])
        sa,sb=size(net.data[a]),size(net.data[b])
        A=unfold(net.data[a],ia);B=unfold(net.data[b],ib)
        shape=(sa[others(net.data[a],ia)]...,sb[others(net.data[b],ib)]...)
        M=reshape(A*transpose(B),shape)
        M=integrate(M,[a,b],[b,a],dt)
        t=time_ns();matrix=reshape(M,size(A,1),size(B,1))
        P=protected(a,b,size(matrix,1))
        U,drop=cutspace(matrix,P,TS._isleaf(net.tree,a));stats.drop+=drop
        net.data[a]=fold(U,sa,ia)
        net.data[b]=fold(Matrix(transpose(U'*matrix)),sb,ib)
        stats.split_seconds+=(time_ns()-t)*1e-9
        message!(a,b)
    end
    function forward(node)
        TS._isleaf(net.tree,node) && return
        for child in net.tree.children[node]
            gauge!(node,child);forward(child)
            pair!(child,node,h/2)
            # At the turnaround the adjacent root solves (-h/2,h,-h/2)
            # have the same frozen generator and cancel exactly.
            (node == net.tree.root && child == last(net.tree.children[node])) || site!(node,-h/2)
        end
    end
    function backward(node)
        TS._isleaf(net.tree,node) && return
        for child in reverse(net.tree.children[node])
            (node == net.tree.root && child == last(net.tree.children[node])) || site!(node,-h/2)
            pair!(node,child,h/2)
            backward(child);gauge!(child,node)
        end
    end
    forward(net.tree.root)
    TS._isleaf(net.tree,net.tree.root) && site!(net.tree.root,h)
    backward(net.tree.root)
    y=unpack(net)
    if maximum(tree_ranks(y))>maxrank
        t=time_ns()
        cut=truncate!(y;atol=atol/2,maxrank,conserved=c,
            fixed_leaves=full_leaves ? y.tree.leaf_order : Int[])
        stats.drop+=cut.discarded_bound
        stats.split_seconds+=(time_ns()-t)*1e-9
    end
    return y,(;ranks=tree_ranks(y),local_solves=stats.solves,matvecs=stats.matvecs,
        max_local_dimension=stats.maxdim,local_cut_sum=stats.drop,truncation_met=stats.drop<=atol,
        max_krylov_error=stats.krylov_error,
        timings=(;solve_seconds=stats.solve_seconds,environment_seconds=stats.environment_seconds,
                  split_seconds=stats.split_seconds))
end
end
