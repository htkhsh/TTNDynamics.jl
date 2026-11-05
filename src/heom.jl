function _hermitian_matrix(matrix::AbstractMatrix, name)
    size(matrix, 1) == size(matrix, 2) || throw(DimensionMismatch("$name must be square"))
    !isempty(matrix) || throw(ArgumentError("$name must be nonempty"))
    result = Matrix{ComplexF64}(matrix)
    all(isfinite, result) || throw(ArgumentError("$name must be finite in ComplexF64"))
    ishermitian(result) || throw(ArgumentError("$name must be Hermitian"))
    return result
end

"""
    HEOMMode(V, rate, forward, backward; scale=nothing)

One exponential bath mode with Hermitian system coupling V. `rate` has
nonnegative real part, and `forward` / `backward` are independent coefficients
for left / right lowering connections in the scaled ADO equations.
The default scale is sqrt(max(abs(forward), abs(backward), eps(Float64))).
For complex conjugate poles, supply the paired backward coefficient explicitly.
V is copied. Treat stored coupling data as read-only.
"""
struct HEOMMode
    coupling::Matrix{ComplexF64}
    rate::ComplexF64
    forward::ComplexF64
    backward::ComplexF64
    scale::Float64

    function HEOMMode(V::AbstractMatrix, rate::Number, forward::Number,
                      backward::Number; scale=nothing)
        coupling = _hermitian_matrix(V, "coupling")
        gamma, cf, cb = ComplexF64(rate), ComplexF64(forward), ComplexF64(backward)
        all(isfinite, (gamma, cf, cb)) || throw(ArgumentError("mode coefficients must be finite"))
        real(gamma) >= 0 || throw(ArgumentError("rate must have nonnegative real part"))
        selected = isnothing(scale) ? sqrt(max(abs(cf), abs(cb), eps(Float64))) : scale
        selected isa Real && isfinite(selected) && selected > 0 ||
            throw(ArgumentError("scale must be finite, real and positive"))
        s = Float64(selected)
        isfinite(s) && s > 0 || throw(ArgumentError("scale must be positive and finite in Float64"))
        new(coupling, gamma, cf, cb, s)
    end
end

"""
    HEOMTTNSystem(H, modes, hierarchy_sizes; system_layout=:fused)

A scaled HEOM model with one auxiliary leaf per mode. `system_layout=:fused`
uses one vec(rho) leaf; `:split` uses separate ket and bra leaves, in that
physical order. `hierarchy_sizes[k]` counts n=0:size-1 (independent box cutoffs).
H must be Hermitian; all mode couplings match H. Zero bath modes are allowed.
Inputs are copied and stored arrays should be treated as read-only. Units
must be consistent with hbar=1. No terminator or residual bath correction.
"""
struct HEOMTTNSystem
    hamiltonian::Matrix{ComplexF64}
    modes::Vector{HEOMMode}
    hierarchy_sizes::Vector{Int}
    system_layout::Symbol

    function HEOMTTNSystem(H::AbstractMatrix, modes::AbstractVector,
                           hierarchy_sizes::AbstractVector; system_layout=:fused)
        system_layout isa Symbol && system_layout in (:fused,:split) ||
            throw(ArgumentError("system_layout must be :fused or :split"))
        hamiltonian = _hermitian_matrix(H, "Hamiltonian")
        length(modes) == length(hierarchy_sizes) ||
            throw(DimensionMismatch("provide one hierarchy size per mode"))
        all(n -> n isa Integer && n >= 1, hierarchy_sizes) ||
            throw(ArgumentError("hierarchy sizes must be positive integers"))
        all(mode -> mode isa HEOMMode, modes) || throw(ArgumentError("modes must contain HEOMMode values"))
        owned = HEOMMode[]
        for mode in modes
            size(mode.coupling) == size(hamiltonian) ||
                throw(DimensionMismatch("mode coupling and Hamiltonian dimensions differ"))
            push!(owned, HEOMMode(mode.coupling, mode.rate, mode.forward, mode.backward; scale=mode.scale))
        end
        new(hamiltonian, owned, Int.(hierarchy_sizes), system_layout)
    end
end

_heom_system_leaves(system::HEOMTTNSystem) = system.system_layout == :fused ? 1 : 2

"""Physical dimensions `[M^2, sizes...]` (fused) or `[M, M, sizes...]` (split).
Ket runs fastest in both layouts, so dense vector order is identical.
"""
function heom_dimensions(system::HEOMTTNSystem)
    d = size(system.hamiltonian,1)
    return [system.system_layout == :fused ? [d^2] : [d,d]; system.hierarchy_sizes]
end

"""Default tree: a system branch and balanced bath subtree under the root.
The system branch is one fused leaf or two sibling ket/bra leaves.
"""
function heom_tree(system::HEOMTTNSystem)
    count = length(system.modes)
    system_tree = balanced_tree(_heom_system_leaves(system))
    count == 0 && return system_tree
    offset = length(system_tree.children)
    bath = balanced_tree(count)
    children = [system_tree.children; [(left == 0 ? 0 : left+offset, right == 0 ? 0 : right+offset)
                         for (left, right) in bath.children]]
    push!(children, (system_tree.root, bath.root+offset))
    return BinaryTree(children)
end

"""
    heom_generator(system) -> SumOfProducts

Construct the scaled box-cutoff HEOM generator with hbar=1, Julia vec(rho)
ordering, and stored ADO rho[n] / prod(scale[k]^n[k] * sqrt(n[k]!)).
No global dense matrix, TTNO compression or coefficient dropping is used.
"""
function heom_generator(system::HEOMTTNSystem)
    d = size(system.hamiltonian, 1)
    fused = system.system_layout == :fused
    identity = fused ? Matrix{ComplexF64}(I,d,d) : nothing
    left(A) = kron(identity, A)
    right(A) = kron(transpose(A), identity)
    terms = fused ?
        [ProductTerm(-im,Dict(1 => left(system.hamiltonian)-right(system.hamiltonian)))] :
        [ProductTerm(-im,Dict(1 => system.hamiltonian)),
         ProductTerm(im,Dict(2 => transpose(system.hamiltonian)))]
    for (k, mode) in enumerate(system.modes)
        n = system.hierarchy_sizes[k]
        number = diagm(0 => ComplexF64.(0:n-1))
        lower = zeros(ComplexF64, n, n)
        for occupation in 1:n-1
            lower[occupation, occupation+1] = sqrt(occupation)
        end
        bath_mode = k+_heom_system_leaves(system)
        push!(terms, ProductTerm(-mode.rate,Dict(bath_mode => number)))
        if fused
            Vleft,Vright = left(mode.coupling),right(mode.coupling)
            push!(terms,ProductTerm(-im*mode.scale,Dict(1 => Vleft-Vright,bath_mode => lower)))
            push!(terms,ProductTerm(-im/mode.scale,
                Dict(1 => mode.forward*Vleft-mode.backward*Vright,bath_mode => lower')))
        else
            V = mode.coupling
            push!(terms,ProductTerm(-im*mode.scale,Dict(1 => V,bath_mode => lower)))
            push!(terms,ProductTerm(im*mode.scale,Dict(2 => transpose(V),bath_mode => lower)))
            push!(terms,ProductTerm(-im*mode.forward/mode.scale,Dict(1 => V,bath_mode => lower')))
            push!(terms,ProductTerm(im*mode.backward/mode.scale,Dict(2 => transpose(V),bath_mode => lower')))
        end
    end
    return SumOfProducts(heom_dimensions(system), terms)
end

function _check_heom_tree(system::HEOMTTNSystem, tree::BinaryTree)
    length(tree.leaf_order) == length(system.modes)+_heom_system_leaves(system) ||
        throw(DimensionMismatch("HEOM tree leaf count differs from the system layout and mode count"))
end

"""
    heom_initial_state(system, rho; tree=heom_tree(system))

Place a finite system matrix at the hierarchy vacuum. Does not normalize or
require Hermiticity/positivity, allowing later response-function states.
Fused position 1 is vec(rho); split positions 1 and 2 are ket and bra.
Subsequent positions follow system.modes. Split states retain full M-dimensional
ket/bra bases, also needed to protect the trace during evolution; arbitrary
trees are supported without expanding the hierarchy.
"""
function heom_initial_state(system::HEOMTTNSystem, rho::AbstractMatrix;
                            tree::BinaryTree=heom_tree(system))
    _check_heom_tree(system, tree)
    size(rho) == size(system.hamiltonian) || throw(DimensionMismatch("system matrix dimensions differ"))
    density = Matrix{ComplexF64}(rho)
    all(isfinite, density) || throw(ArgumentError("system matrix must be finite in ComplexF64"))
    system.system_layout == :split && return _split_heom_initial_state(system,density,tree)
    vectors = [vec(density)]
    for size in system.hierarchy_sizes
        vacuum = zeros(ComplexF64, size)
        vacuum[1] = 1
        push!(vectors, vacuum)
    end
    return product_state(tree, vectors)
end

"""Construct vec(I) times auxiliary vacua, using the selected system layout.
Split ket/bra leaf ranks are M; protecting this functional requires maxrank >= M.
"""
function trace_observable(system::HEOMTTNSystem; tree::BinaryTree=heom_tree(system))
    d = size(system.hamiltonian, 1)
    return heom_initial_state(system, Matrix{ComplexF64}(I, d, d); tree)
end

"""
    root_density_matrix(state, system)

Extract the zero-tier ADO by contracting bath leaves at occupation zero.
The full hierarchy tensor is never expanded. Works on any compatible tree.
"""
function root_density_matrix(state::TTNTensor, system::HEOMTTNSystem)
    leafdims(state) == heom_dimensions(system) || throw(DimensionMismatch("state dimensions differ from HEOM model"))
    tree = state.tree
    system_nodes = tree.leaf_order[1:_heom_system_leaves(system)]
    environments = Vector{Matrix{ComplexF64}}(undef, length(tree.children))
    physical_order = Vector{Vector{Int}}(undef,length(tree.children))
    for node in tree.postorder
        if tree.children[node] == (0, 0)
            position = findfirst(==(node),system_nodes)
            environments[node] = isnothing(position) ? state.leaves[node][1:1,:] : state.leaves[node]
            physical_order[node] = isnothing(position) ? Int[] : [position]
        else
            left, right = tree.children[node]
            core = state.cores[node]
            environments[node] = kron(environments[right], environments[left]) *
                transpose(reshape(core, size(core, 1), :))
            physical_order[node] = [physical_order[left];physical_order[right]]
        end
    end
    d = size(system.hamiltonian, 1)
    density = reshape(environments[tree.root][:,1],d,d)
    return physical_order[tree.root] == [2,1] ? permutedims(density) : density
end

# Correlate the two system leaves at their least common ancestor. A subtree
# containing exactly one system index carries that index unchanged; bath-only
# subtrees and subtrees containing both system indices have rank one initially.
function _split_heom_initial_state(system,density,tree)
    d = size(density,1)
    leaves = Dict{Int,Matrix{ComplexF64}}()
    cores = Dict{Int,Array{ComplexF64,3}}()
    indices = zeros(Int,length(tree.children)) # 0: bath, 1: ket, 2: bra, 3: both
    ranks = ones(Int,length(tree.children))
    for (position,node) in enumerate(tree.leaf_order)
        if position <= 2
            leaves[node] = Matrix{ComplexF64}(I,d,d)
            indices[node] = position
            ranks[node] = d
        else
            vacuum = zeros(ComplexF64,system.hierarchy_sizes[position-2],1)
            vacuum[1] = 1
            leaves[node] = vacuum
        end
    end
    for node in tree.postorder
        tree.children[node] == (0,0) && continue
        left,right = tree.children[node]
        a,b = indices[left],indices[right]
        indices[node] = a | b
        ranks[node] = indices[node] in (1,2) ? d : 1
        core = zeros(ComplexF64,ranks[node],ranks[left],ranks[right])
        if (a,b) == (1,2)
            core[1,:,:] = density
        elseif (a,b) == (2,1)
            core[1,:,:] = transpose(density)
        elseif indices[node] in (1,2)
            for j in 1:d
                core[j,a == 0 ? 1 : j,b == 0 ? 1 : j] = 1
            end
        else
            core[1,1,1] = 1
        end
        cores[node] = core
    end
    return TTNTensor(tree,leaves,cores)
end
