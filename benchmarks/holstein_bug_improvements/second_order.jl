module SecondOrderBUG
using TTNSolver, LinearAlgebra
const TS=TTNSolver
export second_order_bug_step

# Experimental autonomous-linear implementation of Kusch–Sulz,
# arXiv:2610.11747, sections 5.1.1–5.1.4 and 5.2. Physical tensors are never
# expanded. Pre-augmentation changes bases only, not the represented state.
function prepare(state,op,conserved,maxrank)
    base=TS._prepare_parallel_bug_step(state,op,conserved,maxrank)
    old,c=base.old,base.c
    tree=old.tree; count=length(tree.children); root=tree.root
    hat=copy(old)
    overlap=Vector{Matrix{ComplexF64}}(undef,count)
    mixed=similar(base.up); uphat=similar(base.up)
    for node in tree.postorder
        if TS._isleaf(tree,node)
            U=old.leaves[node]
            if node==root
                return merge(base,(;complements=Vector{Matrix{ComplexF64}}(undef,count)))
            end
            rhs=TS._leaf_rhs_workspace(op,base.modes[node],base.boundary[node],base.initial[node])
            Q=TS._orth_columns(U,rhs(base.initial[node]))
            hat.leaves[node]=Q
            overlap[node]=Q'U
            mixed[node]=TS._leaf_operators(op,base.modes[node],Q,U)
            uphat[node]=TS._leaf_operators(op,base.modes[node],Q)
        else
            l,r=tree.children[node]
            oldcore=TS._project_children(old.cores[node],overlap[l],overlap[r])
            if node==root
                hat.cores[node]=oldcore
                uphat[node]=TS._subtree_operators(oldcore,uphat[l],uphat[r])
                continue
            end
            dims=(size(base.initial[node],1),size(overlap[l],1),size(overlap[r],1))
            derivative=TS._parallel_core_block(op,base.boundary[node],mixed[l],mixed[r],base.initial[node],dims)
            U=TS._basis_matrix(oldcore)
            Q=TS._orth_columns(U,TS._basis_matrix(derivative))
            hat.cores[node]=reshape(Matrix(transpose(Q)),size(Q,2),dims[2],dims[3])
            overlap[node]=Q'U
            mixed[node]=TS._mixed_subtree_operators(Q,old.cores[node],mixed[l],mixed[r])
            uphat[node]=TS._subtree_operators(hat.cores[node],uphat[l],uphat[r])
        end
    end
    embedded=similar(base.initial); initial=similar(base.initial)
    boundary=similar(base.boundary)
    complements=Vector{Matrix{ComplexF64}}(undef,count)
    embedded[root]=old.cores[root]; boundary[root]=base.boundary[root]
    for node in reverse(tree.postorder)
        if TS._isleaf(tree,node)
            initial[node]=embedded[node]
            continue
        end
        l,r=tree.children[node]
        D=embedded[node]
        initial[node]=TS._project_children(D,overlap[l],overlap[r])
        derivative=TS._parallel_core_block(op,boundary[node],mixed[l],mixed[r],D,size(initial[node]))
        for (child,sibling,mode,othermode) in ((l,r,2,3),(r,l,3,2))
            complement=TS._complement_matrix(TS._child_product(D,overlap[sibling],othermode),mode)
            fac=qr(complement); k=min(size(complement)...)
            Q0=Matrix(fac.Q)[:,1:k]
            restricted=TS._child_product(derivative,overlap[child]',mode)
            Q=TS._orth_columns(Q0,TS._complement_matrix(restricted,mode))
            complements[child]=Q
            R=Q'complement
            boundary[child]=TS._complement_operators(Q,boundary[node],uphat[sibling])
            embedded[child]=TS._isleaf(tree,child) ? old.leaves[child]*transpose(R) :
                TS._parent_product(old.cores[child],R)
        end
    end
    return (;old=hat,c,initial,boundary,modes=base.modes,up=uphat,complements)
end

# Project an evolved child onto its new directions and unfold the augmented
# complement back into the parent's coordinates. This uses the evolved block,
# not the first-order h*F(old) cross block.
function lift_cross(block,Q,parent,sibling,mode)
    data=reshape(block*transpose(Q),size(block,1),parent,sibling)
    return permutedims(data,mode==2 ? (2,1,3) : (2,3,1))
end

function parallel_update(prepared,op,h;atol,maxrank,krylov_tol,krylovdim,krylov_maxiter,
                         correct_invariant,threaded,truncation)
    (;old,c,initial,boundary,modes,up,complements)=prepared
    tree=old.tree; root=tree.root; count=length(tree.children)
    solves=Vector{NamedTuple}(undef,count)
    use_threads=threaded && Threads.nthreads(:default)>1
    solve(node)=TS._parallel_local_solve(prepared,op,node,h,krylov_tol,krylovdim,krylov_maxiter)
    if use_threads
        failures=Vector{Any}(nothing,count)
        @sync for node in tree.postorder
            Threads.@spawn try
                solves[node]=solve(node)
            catch e
                failures[node]=e
            end
        end
        for e in failures
            !isnothing(e) && !(e isa TS.LocalSolveFailure) && throw(e)
        end
        for e in failures
            isnothing(e) || throw(e)
        end
    else
        for node in tree.postorder; solves[node]=solve(node); end
    end
    result=copy(old)
    overlaps=Vector{Matrix{ComplexF64}}(undef,count)
    cross=similar(overlaps); newcols=Vector{UnitRange{Int}}(undef,count)
    for node in tree.postorder
        if TS._isleaf(tree,node)
            if node==root
                result.leaves[node]=solves[node].value
                continue
            end
            U=old.leaves[node]; oldrank=size(U,2)
            Q=TS._orth_columns(U,solves[node].value)
            result.leaves[node]=Q; overlaps[node]=Q'U
            newcols[node]=oldrank+1:size(Q,2)
            cross[node]=Q[:,newcols[node]]'solves[node].value
        else
            l,r=tree.children[node]
            evolved=TS._project_children(solves[node].value,overlaps[l],overlaps[r])
            core=copy(evolved)
            p,rl,rr=size(initial[node])
            if !isempty(newcols[l])
                block=lift_cross(cross[l],complements[l],p,rr,2)
                view(core,:,newcols[l],:) .+= TS._child_product(block,overlaps[r],3)
            end
            if !isempty(newcols[r])
                block=lift_cross(cross[r],complements[r],p,rl,3)
                view(core,:,:,newcols[r]) .+= TS._child_product(block,overlaps[l],2)
            end
            if node==root
                result.cores[node]=core
                continue
            end
            U=TS._basis_matrix(TS._project_children(old.cores[node],overlaps[l],overlaps[r]))
            Q=TS._orth_columns(U,TS._basis_matrix(core)); oldrank=size(U,2)
            result.cores[node]=reshape(Matrix(transpose(Q)),size(Q,2),size(core,2),size(core,3))
            overlaps[node]=Q'U; newcols[node]=oldrank+1:size(Q,2)
            cross[node]=Q[:,newcols[node]]'TS._basis_matrix(evolved)
        end
    end
    stats=(;local_solves=sum(s.solves for s in solves),skipped_leaf_solves=sum(s.skipped for s in solves),
        max_local_dimension=maximum(s.dimension for s in solves),krylov_error_estimate=sum(s.error for s in solves),
        method=:parallel2,threaded=use_threads)
    return TS._finish_bug_step!(result,old,c,stats;atol,maxrank,correct_invariant,truncation)
end

function second_order_bug_step(state,op,h::Real;method=:augmented,atol=1e-10,
    maxrank=typemax(Int),conserved=nothing,krylov_tol=1e-12,krylovdim=30,
    krylov_maxiter=100,correct_invariant=true,threaded=true,truncation=:weighted)
    method in (:augmented,:parallel) || throw(ArgumentError("unknown method"))
    isfinite(h) && h>0 && isfinite(Float64(h)) && Float64(h)>0 || throw(ArgumentError("invalid h"))
    TS._check_bug_options(atol,maxrank,krylov_tol,krylovdim,krylov_maxiter)
    TS._check_truncation_weighting(truncation)
    prepared=prepare(state,op,conserved,maxrank)
    update=method==:augmented ? TS._prepared_bug_step : parallel_update
    y,d=update(prepared,op,h;atol,maxrank,krylov_tol,krylovdim,krylov_maxiter,
               correct_invariant,threaded,truncation)
    return y,merge(d,(;method=method==:augmented ? :augmented2 : :parallel2,
                       preaugmented_ranks=tree_ranks(prepared.old)))
end
end
