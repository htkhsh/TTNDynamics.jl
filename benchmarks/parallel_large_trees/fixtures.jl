using TTNDynamics, TTNSolver, LinearAlgebra, Dates, SHA

const LARGE_CASES = ("heom16","heom32","heom64","tfd16")
const LARGE_STEP = 0.05
const PRESTEPS = 4
const LARGE_OPTIONS = (;atol=1e-10,maxrank=64,correct_invariant=false)

# M and total coupling weight stay fixed as the number of modes grows.
# Coupling types/sites are interleaved, then the standard balanced bath tree
# is used. These are controlled performance fixtures, not fitted bath models.
function large_problem(name)
    heom = startswith(name,"heom")
    k = parse(Int,name[(heom ? 5 : 4):end])
    k >= 8 && k % 8 == 0 || error("mode count must be a multiple of eight")
    m = 4
    if heom
        H = diagm(-1=>fill(0.25-0.07im,m-1),0=>ComplexF64.(range(-0.2,0.2;length=m)),
                  1=>fill(0.25+0.07im,m-1))
        grid = ComplexF64.(range(-0.3,1.;length=m))
        modes = [HEOMMode(diagm(grid.^mod1(j,4)),0.8+0.7*(j-0.5)/k,
                         (0.08+0.01im)*8/k,(0.08-0.01im)*8/k) for j in 1:k]
        system = HEOMTTNSystem(H,modes,fill(3,k))
        rho = Matrix{ComplexF64}(I,m,m)*(0.05/m); rho[1,1] += 0.95
        initial = heom_initial_state(system,rho)
        conserved = trace_observable(system;tree=initial.tree)
        observe = x -> (;density=root_density_matrix(x,system),invariant=dot(conserved,x))
        return (;system,initial,operator=heom_generator(system),conserved,observe)
    end
    r = k ÷ 8
    # Four sites, r physical oscillators per site, two signed modes each.
    H = diagm(ComplexF64.(range(0.15,-0.15;length=m)))
    for j in 1:m
        H[j,mod1(j+1,m)] = H[mod1(j+1,m),j] = -0.4
    end
    modes = TFDMode[]; pairs = Tuple{Int,Int,Float64}[]
    for frequency in 1:r, site in 1:m
        omega = 0.6+0.4*(frequency-0.5)/r
        nbar = inv(expm1(2omega))
        V = zeros(ComplexF64,m,m); V[site,site] = 1
        g = 0.3/sqrt(r)
        push!(modes,TFDMode(V,omega,g*sqrt(nbar+1)),TFDMode(V,-omega,g*sqrt(nbar)))
        push!(pairs,(length(modes)-1,length(modes),nbar))
    end
    system = TFDTTNSystem(H,modes,fill(3,k);thermal_pairs=pairs)
    vacuum = tfd_initial_state(system,diagm([0.85,0.05,0.05,0.05]))
    operator = tfd_hamiltonian(system)
    # Number terms annihilate this vacuum exactly. Omitting them from the
    # one-time seed avoids allocating their zero blocks in exact SoP apply.
    # No term is omitted from the propagation generator.
    active = SumOfProducts(operator.dimensions,filter(t -> haskey(t.factors,1),operator.terms))
    initial = enrich_basis(vacuum,active*vacuum)
    observe = x -> (;density=tfd_density_matrix(x,system),invariant=norm(x))
    return (;system,initial,operator=TTNDynamics._tfd_generator(operator),conserved=nothing,observe)
end

function prepared_snapshot(problem;presteps=PRESTEPS)
    state = copy(problem.initial)
    diagnostics = NamedTuple[]
    for _ in 1:presteps
        state,d = parallel_bug_step(state,problem.operator,LARGE_STEP;
            LARGE_OPTIONS...,conserved=problem.conserved,threaded=false)
        push!(diagnostics,d)
    end
    return state,diagnostics
end

function measured_step(problem,state;threaded,step=parallel_bug_step)
    return step(state,problem.operator,LARGE_STEP;LARGE_OPTIONS...,conserved=problem.conserved,threaded)
end

function case_metadata(problem,state)
    tree = state.tree
    depth(node) = tree.children[node] == (0,0) ? 0 : 1+maximum(depth(c) for c in tree.children[node])
    return (;leaves=length(tree.leaf_order),nodes=length(tree.children),height=depth(tree.root),
        terms=length(problem.operator.terms),full_dimension=prod(big.(leafdims(state))),
        input_rank=maximum(tree_ranks(state)))
end
