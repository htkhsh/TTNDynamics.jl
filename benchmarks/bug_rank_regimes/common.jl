include("../tamen_comparison/common.jl")
using SparseArrays

const RANK_CASES = (
    (;name="weak6",m=4,k=6,strength=1.0,prep=0.25),
    (;name="weak8",m=4,k=8,strength=1.0,prep=0.25),
    (;name="moderate8",m=4,k=8,strength=4.0,prep=0.25),
    (;name="strong6",m=4,k=6,strength=16.0,prep=0.5),
    (;name="strong6_short",m=4,k=6,strength=16.0,prep=0.5,interval=0.005),
    (;name="strong8",m=4,k=8,strength=16.0,prep=0.5),
    (;name="long8",m=4,k=8,strength=16.0,prep=1.0),
    (;name="system8",m=8,k=4,strength=16.0,prep=0.5))
const INTERVAL = 0.05

function rank_problem(spec)
    (;m,k,strength) = spec
    rng = MersenneTwister(119)
    H = diagm(-1=>fill(0.25-0.07im,m-1),0=>ComplexF64.(range(-0.2,0.2;length=m)),
              1=>fill(0.25+0.07im,m-1))
    modes = HEOMMode[]
    for j in 1:k
        Z = randn(rng,ComplexF64,m,m)
        V = (Z+Z')/2; V ./= opnorm(V)
        c = strength*(0.08+0.01im)*8/k
        push!(modes,HEOMMode(V,0.8+0.7*(j-0.5)/k,c,conj(c)))
    end
    system = HEOMTTNSystem(H,modes,fill(3,k))
    rho = Matrix{ComplexF64}(I,m,m)*(0.05/m); rho[1,1] += 0.95
    initial = heom_initial_state(system,rho)
    conserved = trace_observable(system;tree=initial.tree)
    return (;system,initial,conserved,operator=heom_generator(system))
end

function sparse_generator(op)
    N = prod(op.dimensions)
    A = spzeros(ComplexF64,N,N)
    for term in op.terms
        product = sparse(ones(ComplexF64,1,1))
        for j in reverse(eachindex(op.dimensions))
            n = op.dimensions[j]
            f = haskey(term.factors,j) ? sparse(term.factors[j]) : spdiagm(0=>ones(ComplexF64,n))
            product = kron(product,f)
        end
        A += term.coefficient*product
    end
    return A
end

function exact_action(A,x,t;tol=1e-12,dimension=50)
    y,info = TTNSolver.KrylovKit.exponentiate(A,t,x;tol,krylovdim=dimension,maxiter=1000)
    info.converged >= 1 || error("global reference did not converge")
    return y
end

# Singular-value truncation shared by the two independent decompositions.
function keep_rank(s,budget)
    r = length(s)
    tail2 = 0.0
    while r > 1 && tail2+abs2(s[r]) <= budget^2
        tail2 += abs2(s[r]); r -= 1
    end
    return r
end

# Input vector has Julia physical-axis order, first mode fastest. TT cores
# still use the same named physical axes; tt_full uses reverse vectorization.
function dense_to_train(v,dims;tolerance=1e-8)
    work = reshape(v,dims[1],:)
    cores = Array{ComplexF64,3}[]
    left = 1
    budget = tolerance*norm(v)/sqrt(max(1,length(dims)-1))
    for j in 1:length(dims)-1
        fac = svd(reshape(work,left*dims[j],:))
        r = keep_rank(fac.S,budget)
        push!(cores,reshape(fac.U[:,1:r],left,dims[j],r))
        work = Diagonal(fac.S[1:r])*fac.Vt[1:r,:]
        left = r
    end
    push!(cores,reshape(work,left,dims[end],1))
    return TTTensor(cores)
end

train_vector(x) = vec(permutedims(reshape(tt_full(x),Tuple(reverse(tt_dims(x)))),
                                  reverse(1:length(tt_dims(x)))))

# Hierarchical SVD, only for these small/medium reference fixtures. Each cut
# basis is built from the complete state, then projected into child bases.
function dense_to_tree(v,dims,tree;tolerance=1e-8)
    tensor = reshape(v,Tuple(dims))
    modes = Dict(node=>[j] for (j,node) in enumerate(tree.leaf_order))
    bases = Dict{Int,Matrix{ComplexF64}}()
    leaves = Dict{Int,Matrix{ComplexF64}}()
    cores = Dict{Int,Array{ComplexF64,3}}()
    budget = tolerance*norm(v)/sqrt(max(1,length(tree.children)-1))
    for node in tree.postorder
        l,r = tree.children[node]
        l != 0 && (modes[node] = [modes[l];modes[r]])
        js = modes[node]
        issorted(js) && last(js)-first(js)+1 == length(js) || error("requires contiguous subtrees")
        if node == tree.root
            U = reshape(v,:,1)
        else
            others = setdiff(collect(eachindex(dims)),js)
            matrix = reshape(permutedims(tensor,[js;others]),prod(dims[js]),:)
            fac = svd(matrix)
            U = fac.U[:,1:keep_rank(fac.S,budget)]
        end
        bases[node] = U
        if l == 0
            leaves[node] = U
        else
            L,R = bases[l],bases[r]
            dl,rl = size(L); dr,rr = size(R); rp = size(U,2)
            temp = reshape(L'*reshape(U,dl,dr*rp),rl,dr,rp)
            projected = reshape(permutedims(temp,(1,3,2)),rl*rp,dr)*conj(R)
            cores[node] = permutedims(reshape(projected,rl,rp,rr),(2,1,3))
            delete!(bases,l); delete!(bases,r)
        end
    end
    return TTNTensor(tree,leaves,cores)
end

function evolve_bug(problem,state;method=:parallel,steps=1,maxrank=32,atol=1e-8,threaded=true,interval=INTERVAL)
    x = copy(state); misses = 0; peak = maximum(tree_ranks(x))
    step = method == :parallel ? parallel_bug_step : augmented_bug_step
    for _ in 1:steps
        x,d = step(x,problem.operator,interval/steps;atol,maxrank,threaded,
                   conserved=problem.conserved,correct_invariant=false)
        misses += !d.truncation.tolerance_met
        peak = max(peak,maximum(d.ranks))
    end
    return (;state=x,rank=maximum(tree_ranks(x)),misses,peak)
end

state_vector(x::TTTensor) = train_vector(x)
state_vector(x::TTNTensor) = vec(dense(x))
state_error(x,ref) = norm(state_vector(x)-ref)/norm(ref)
storage(x::TTTensor) = sum(length,x.cores)
storage(x::TTNTensor) = sum(length,values(x.cores))+sum(length,values(x.leaves))
