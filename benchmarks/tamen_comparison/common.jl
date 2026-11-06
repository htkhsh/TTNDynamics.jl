using TTSolver, Random, Serialization
# Both solvers export a workspace type. This benchmark's train solves use TT.
using TTSolver: TAMEnWorkspace
include("../parallel_large_trees/fixtures.jl")

# Keep TT contractions serial; only the BUG node ODEs use Julia workers.
TTSolver.TensorOperations.Strided.set_num_threads(1)

const FINAL_TIME = 0.25
const ERROR_TARGET = 1e-4

# The physical axes have the same meaning in both formats. Only dense
# vectorization differs: TT uses the last physical axis fastest.
function sop_to_mpo(op; tolerance=1e-14)
    result = nothing
    for term in op.terms
        cores = [reshape(get(term.factors,j,Matrix{ComplexF64}(I,n,n)),1,n,n,1)
                 for (j,n) in enumerate(op.dimensions)]
        piece = term.coefficient * TTMatrix(cores)
        result = isnothing(result) ? piece : tt_round(result+piece,tolerance)
    end
    isnothing(result) && error("benchmark expects a nonempty generator")
    return result
end

function product_to_tt(x::TTNTensor)
    all(==(1),tree_ranks(x)) || error("expected a product state")
    cores = [reshape(copy(x.leaves[node]),1,size(x.leaves[node],1),1)
             for node in x.tree.leaf_order]
    cores[1] .*= prod(only(c) for c in values(x.cores); init=1.0+0im)
    return TTTensor(cores)
end

function physical_initial(problem)
    problem.system isa HEOMTTNSystem && return problem.initial
    return tfd_initial_state(problem.system,diagm([0.85,0.05,0.05,0.05]))
end

# <train, tree>, with no dense expansion or conversion to potentially huge
# TT ranks. The fixture tree subtrees must be contiguous in physical order.
function cross_overlap(train::TTTensor, tree::TTNTensor)
    tt_dims(train) == leafdims(tree) || throw(DimensionMismatch("physical dimensions"))
    env = Dict{Int,Array{ComplexF64,3}}()
    spans = Dict{Int,UnitRange{Int}}()
    for (j,node) in enumerate(tree.tree.leaf_order)
        g = train.cores[j]; a,n,b = size(g)
        u = tree.leaves[node]
        env[node] = reshape(transpose(u)*reshape(permutedims(conj.(g),(2,1,3)),n,:),size(u,2),a,b)
        spans[node] = j:j
    end
    for node in tree.tree.postorder
        l,r = tree.tree.children[node]
        l == 0 && continue
        last(spans[l])+1 == first(spans[r]) || throw(ArgumentError("noncontiguous subtree"))
        spans[node] = first(spans[l]):last(spans[r])
        L,R,C = env[l],env[r],tree.cores[node]
        p,nl,nr = size(C); _,a,b = size(L); _,br,c = size(R)
        b == br || throw(DimensionMismatch("TT bond"))
        temp = reshape(reshape(C,p*nl,nr)*reshape(R,nr,b*c),p,nl,b,c)
        result = reshape(permutedims(L,(2,1,3)),a,nl*b) *
                 reshape(permutedims(temp,(2,3,1,4)),nl*b,p*c)
        env[node] = permutedims(reshape(result,a,p,c),(2,1,3))
        delete!(env,l); delete!(env,r)
    end
    return only(env[tree.tree.root])
end

tt_distance(a,b) = norm(tt_round(a-b,0.0))
function relative_error(tree::TTNTensor, reference::TTTensor)
    n2 = real(dot(reference,reference))
    squared = norm(tree)^2+n2-2real(cross_overlap(reference,tree))
    squared >= -100eps(Float64)*n2 || error("negative squared distance: $squared")
    return sqrt(max(0.0,squared)/n2)
end
relative_error(train::TTTensor, reference::TTTensor) = tt_distance(train,reference)/norm(reference)

function tt_density(x::TTTensor, heom::Bool)
    m = isqrt(size(x.cores[1],2))
    if heom
        tail = ones(ComplexF64,1)
        for g in reverse(x.cores[2:end])
            tail = g[:,1,:]*tail
        end
        return reshape(reshape(x.cores[1],m*m,:)*tail,m,m)
    end
    tail = ones(ComplexF64,1,1)
    for g in reverse(x.cores[2:end])
        left = zeros(ComplexF64,size(g,1),size(g,1))
        for n in axes(g,2)
            left .+= g[:,n,:]*tail*g[:,n,:]'
        end
        tail = left
    end
    u = reshape(x.cores[1],m*m,:)
    rho = reshape(u*tail*u',m,m,m,m) # ket,ancilla,bra,ancilla
    return sum(rho[:,a,:,a] for a in 1:m)
end

# Independent reference: compressed Taylor action, with step/rounding
# refinement. This does not use the tAMEn or BUG evolution algorithms.
function taylor_reference(A,x; steps=10, tolerance=1e-12, order=24, final_time=FINAL_TIME, verbose=false)
    h = final_time/steps
    state = x
    for step in 1:steps
        term = state; result = state
        absolute_tolerance = tolerance*norm(state)
        converged = false
        for k in 1:order
            raw = (h/k)*(A*term)
            term = tt_round(raw,min(0.1,absolute_tolerance/max(norm(raw),absolute_tolerance)))
            result = tt_round(result+term,tolerance)
            if k >= 6 && norm(term) < absolute_tolerance*0.01
                converged = true
                break
            end
        end
        converged || error("Taylor series did not converge within $order terms")
        state = result
        if verbose
            println("  reference step $step/$steps rank ",maximum(tt_ranks(state))); flush(stdout)
        end
    end
    return state
end

function run_tamen(A,x; tolerance=1e-5, points=8, final_time=FINAL_TIME, seed=20261105, conserved=nothing)
    Random.seed!(seed)
    space_time = tt_kron(x,tt_ones(points;T=ComplexF64))
    result,grid,_,_,_ = tamen(space_time,final_time*A,tolerance;
        time_scheme="cheb",correct_2_norm=false,verb=0,nswp=30,
        conserved_observables=isnothing(conserved) ? nothing : [conserved],
        workspace=TAMEnWorkspace(ComplexF64))
    final = tt_round(extract_snapshot(result,grid,1.0,"cheb"),1e-12)
    return (;state=final,intervals=length(grid),rank=maximum(tt_ranks(final)))
end

function run_bug(problem; steps=5, atol=1e-8, maxrank=64, final_time=FINAL_TIME)
    state = copy(problem.initial)
    misses = 0; discarded = 0.0; peak_rank = maximum(tree_ranks(state))
    for _ in 1:steps
        state,d = parallel_bug_step(state,problem.operator,final_time/steps;
            atol,maxrank,conserved=problem.conserved,correct_invariant=false,threaded=true)
        misses += !d.truncation.tolerance_met
        discarded += d.truncation.discarded_bound
        peak_rank = max(peak_rank,maximum(d.ranks))
    end
    return (;state,misses,discarded,rank=maximum(tree_ranks(state)),peak_rank)
end
