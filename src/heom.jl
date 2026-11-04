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
    HEOMTTNSystem(H, modes, hierarchy_sizes)

A scaled HEOM model with a fused vec(rho) system leaf and one auxiliary leaf
per mode. `hierarchy_sizes[k]` counts n=0:size-1 (independent box cutoffs).
H must be Hermitian; all mode couplings match H. Zero bath modes are allowed.
Inputs are copied and stored arrays should be treated as read-only. Units
must be consistent with hbar=1. No terminator or residual bath correction.
"""
struct HEOMTTNSystem
    hamiltonian::Matrix{ComplexF64}
    modes::Vector{HEOMMode}
    hierarchy_sizes::Vector{Int}

    function HEOMTTNSystem(H::AbstractMatrix, modes::AbstractVector,
                           hierarchy_sizes::AbstractVector)
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
        new(hamiltonian, owned, Int.(hierarchy_sizes))
    end
end

"""Physical dimensions `[M^2, hierarchy_sizes...]`, with ket fastest in vec(rho)."""
heom_dimensions(system::HEOMTTNSystem) = [size(system.hamiltonian, 1)^2; system.hierarchy_sizes]

"""Default tree: system leaf under the root and a balanced bath subtree."""
function heom_tree(system::HEOMTTNSystem)
    count = length(system.modes)
    count == 0 && return balanced_tree(1)
    bath = balanced_tree(count)
    children = [(0, 0); [(left == 0 ? 0 : left+1, right == 0 ? 0 : right+1)
                         for (left, right) in bath.children]]
    push!(children, (1, bath.root+1))
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
    identity = Matrix{ComplexF64}(I, d, d)
    left(A) = kron(identity, A)
    right(A) = kron(transpose(A), identity)
    terms = [ProductTerm(-im, Dict(1 => left(system.hamiltonian)-right(system.hamiltonian)))]
    for (k, mode) in enumerate(system.modes)
        n = system.hierarchy_sizes[k]
        number = diagm(0 => ComplexF64.(0:n-1))
        lower = zeros(ComplexF64, n, n)
        for occupation in 1:n-1
            lower[occupation, occupation+1] = sqrt(occupation)
        end
        Vleft, Vright = left(mode.coupling), right(mode.coupling)
        push!(terms, ProductTerm(-mode.rate, Dict(k+1 => number)))
        push!(terms, ProductTerm(-im*mode.scale, Dict(1 => Vleft-Vright, k+1 => lower)))
        push!(terms, ProductTerm(-im/mode.scale,
            Dict(1 => mode.forward*Vleft-mode.backward*Vright, k+1 => lower')))
    end
    return SumOfProducts(heom_dimensions(system), terms)
end

function _check_heom_tree(system::HEOMTTNSystem, tree::BinaryTree)
    length(tree.leaf_order) == length(system.modes)+1 ||
        throw(DimensionMismatch("HEOM tree must have one system leaf and one leaf per mode"))
end

"""
    heom_initial_state(system, rho; tree=heom_tree(system))

Place a finite system matrix at the hierarchy vacuum. Does not normalize or
require Hermiticity/positivity, allowing later response-function states.
Physical position 1 is vec(rho); subsequent positions follow system.modes.
"""
function heom_initial_state(system::HEOMTTNSystem, rho::AbstractMatrix;
                            tree::BinaryTree=heom_tree(system))
    _check_heom_tree(system, tree)
    size(rho) == size(system.hamiltonian) || throw(DimensionMismatch("system matrix dimensions differ"))
    density = Matrix{ComplexF64}(rho)
    all(isfinite, density) || throw(ArgumentError("system matrix must be finite in ComplexF64"))
    vectors = [vec(density)]
    for size in system.hierarchy_sizes
        vacuum = zeros(ComplexF64, size)
        vacuum[1] = 1
        push!(vectors, vacuum)
    end
    return product_state(tree, vectors)
end

"""Construct the linear trace functional vec(I) times all auxiliary vacua."""
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
    system_node = tree.leaf_order[1]
    environments = Vector{Matrix{ComplexF64}}(undef, length(tree.children))
    for node in tree.postorder
        if tree.children[node] == (0, 0)
            environments[node] = node == system_node ? state.leaves[node] : state.leaves[node][1:1, :]
        else
            left, right = tree.children[node]
            core = state.cores[node]
            environments[node] = kron(environments[right], environments[left]) *
                transpose(reshape(core, size(core, 1), :))
        end
    end
    d = size(system.hamiltonian, 1)
    return reshape(environments[tree.root][:, 1], d, d)
end
