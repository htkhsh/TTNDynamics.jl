"""
    TFDMode(V, frequency, amplitude)

An already thermally transformed bosonic mode: `frequency*N + amplitude*V*x`,
with `x=(a+a†)/sqrt(2)`. Frequency and amplitude are finite real numbers;
negative frequencies are allowed. V is Hermitian and copied. In particular,
amplitude is not a squared spectral weight. No temperature factor is applied.
"""
struct TFDMode
    coupling::Matrix{ComplexF64}
    frequency::Float64
    amplitude::Float64

    function TFDMode(V::AbstractMatrix, frequency::Real, amplitude::Real)
        coupling = _hermitian_matrix(V, "coupling")
        omega, g = Float64(frequency), Float64(amplitude)
        all(isfinite, (omega,g)) || throw(ArgumentError("frequency and amplitude must be finite in Float64"))
        new(coupling, omega, g)
    end
end

"""
    TFDTTNSystem(H, modes, boson_sizes; thermal_pairs=[])

Electronic Hamiltonian and explicit signed thermofield modes, in consistent
units with hbar=1. First leaf fuses electron (fastest) and inactive electronic
ancilla. Bath leaves count `0:size-1`. Inputs are copied; treat stored arrays
as read-only. Zero bath modes and size-one explicit modes are allowed.

Optional `(positive_mode, negative_mode, nbar)` tuples identify inverse
Bogoliubov observables for actual thermal pairs. Holstein's builder supplies
these automatically. General signed modes, such as an ID fit, need not pair
and should omit this metadata. No extra thermal transformation is performed.
"""
struct TFDTTNSystem
    hamiltonian::Matrix{ComplexF64}
    modes::Vector{TFDMode}
    boson_sizes::Vector{Int}
    thermal_pairs::Vector{Tuple{Int,Int,Float64}}

    function TFDTTNSystem(H::AbstractMatrix, modes::AbstractVector,
                          boson_sizes::AbstractVector; thermal_pairs=[])
        hamiltonian = _hermitian_matrix(H, "Hamiltonian")
        length(modes) == length(boson_sizes) || throw(DimensionMismatch("provide one boson size per mode"))
        all(n -> n isa Integer && n >= 1, boson_sizes) || throw(ArgumentError("boson sizes must be positive integers"))
        all(m -> m isa TFDMode, modes) || throw(ArgumentError("modes must contain TFDMode values"))
        owned = TFDMode[]
        for mode in modes
            size(mode.coupling) == size(hamiltonian) || throw(DimensionMismatch("coupling and Hamiltonian dimensions differ"))
            push!(owned, TFDMode(mode.coupling, mode.frequency, mode.amplitude))
        end
        pairs = Tuple{Int,Int,Float64}[]
        used = Set{Int}()
        for pair in thermal_pairs
            length(pair) == 3 || throw(ArgumentError("thermal pairs must contain (positive,negative,nbar)"))
            p,q,n = pair
            p isa Integer && q isa Integer && 1 <= p <= length(owned) && 1 <= q <= length(owned) &&
                p != q && !(p in used) && !(q in used) || throw(ArgumentError("thermal pairs require distinct, disjoint valid mode indices"))
            n isa Real && isfinite(n) && n >= 0 && isfinite(Float64(n)) || throw(ArgumentError("nbar must be finite and nonnegative"))
            owned[p].frequency > 0 && owned[q].frequency ≈ -owned[p].frequency ||
                throw(ArgumentError("thermal pairs require opposite frequencies with the positive mode first"))
            owned[p].coupling ≈ owned[q].coupling || throw(ArgumentError("paired modes must have the same coupling operator"))
            owned[q].amplitude ≈ owned[p].amplitude*sqrt(Float64(n)/(Float64(n)+1)) ||
                throw(ArgumentError("paired amplitudes do not match nbar"))
            min(boson_sizes[p],boson_sizes[q]) >= 2 || throw(ArgumentError("thermal pairs require boson sizes >= 2"))
            push!(pairs, (Int(p),Int(q),Float64(n)))
            union!(used, (p,q))
        end
        new(hamiltonian, owned, Int.(boson_sizes), pairs)
    end
end

"""
    holstein_tfd_system(site_energies, hopping, frequencies, couplings;
                        beta, boson_size=3, periodic=true)

Single-excitation Holstein model with one independent physical oscillator per
site. Electronic nearest-neighbor matrix elements are `-hopping`; the closing
bond is included only for more than two sites. Frequencies must be positive;
couplings multiply `|site><site|*(a+a†)/sqrt(2)`. Arrays have one entry per site.

`beta>0` (or Inf at zero temperature) is inverse energy in the supplied units.
Each oscillator becomes adjacent (+omega,-omega) vacuum modes with amplitudes
`g*sqrt(nbar+1)` and `g*sqrt(nbar)`, where `nbar=1/expm1(beta*omega)`.
`boson_size>=2` is the common transformed Fock cutoff, not a HEOM tier cutoff.
There is no damping, reorganization counterterm or automatic unit conversion.
"""
function holstein_tfd_system(site_energies::AbstractVector, hopping::Real,
                             frequencies::AbstractVector, couplings::AbstractVector;
                             beta::Real, boson_size::Integer=3, periodic::Bool=true)
    count = length(site_energies)
    count > 0 || throw(ArgumentError("provide at least one site"))
    length(frequencies) == length(couplings) == count || throw(DimensionMismatch("provide one frequency and coupling per site"))
    all(x -> x isa Real && isfinite(x) && isfinite(Float64(x)), site_energies) || throw(ArgumentError("site energies must be finite real numbers"))
    isfinite(Float64(hopping)) || throw(ArgumentError("hopping must be finite in Float64"))
    beta > 0 && (isfinite(beta) || beta == Inf) || throw(ArgumentError("beta must be positive (Inf for zero temperature)"))
    beta64 = Float64(beta)
    beta64 > 0 && (isfinite(beta64) || beta == Inf) || throw(ArgumentError("beta is not representable in Float64"))
    boson_size >= 2 || throw(ArgumentError("thermal modes require boson_size >= 2"))
    H = Matrix{ComplexF64}(diagm(Float64.(site_energies)))
    for k in 1:count-1
        H[k,k+1] = H[k+1,k] = -Float64(hopping)
    end
    periodic && count > 2 && (H[1,count] = H[count,1] = -Float64(hopping))
    modes = TFDMode[]
    pairs = Tuple{Int,Int,Float64}[]
    for k in 1:count
        omega, g = frequencies[k], couplings[k]
        omega isa Real && isfinite(omega) && omega > 0 && isfinite(Float64(omega)) && Float64(omega) > 0 ||
            throw(ArgumentError("physical frequencies must be positive and finite in Float64"))
        g isa Real && isfinite(g) && isfinite(Float64(g)) || throw(ArgumentError("couplings must be finite real numbers"))
        n = isinf(beta64) ? 0.0 : inv(expm1(beta64*Float64(omega)))
        isfinite(n) || throw(ArgumentError("thermal occupation is not finite in Float64"))
        V = zeros(ComplexF64,count,count)
        V[k,k] = 1
        push!(modes, TFDMode(V, omega, Float64(g)*sqrt(n+1)), TFDMode(V,-omega,Float64(g)*sqrt(n)))
        push!(pairs, (2k-1,2k,n))
    end
    return TFDTTNSystem(H,modes,fill(Int(boson_size),2count); thermal_pairs=pairs)
end

tfd_dimensions(system::TFDTTNSystem) = [size(system.hamiltonian,1)^2; system.boson_sizes]

"""System leaf under the root; a balanced subtree of modes in supplied order."""
function tfd_tree(system::TFDTTNSystem)
    count = length(system.modes)
    count == 0 && return balanced_tree(1)
    bath = balanced_tree(count)
    children = [(0,0); [(l == 0 ? 0 : l+1, r == 0 ? 0 : r+1) for (l,r) in bath.children]]
    push!(children, (1,bath.root+1))
    return BinaryTree(children)
end

_tfd_annihilation(n) = diagm(1 => ComplexF64.(sqrt.(Float64.(1:n-1))))

"""Hermitian SumOfProducts in physical order `[electron+ancilla, modes...]`."""
function tfd_hamiltonian(system::TFDTTNSystem)
    m = size(system.hamiltonian,1)
    identity = Matrix{ComplexF64}(I,m,m)
    terms = [ProductTerm(1,Dict(1 => kron(identity,system.hamiltonian)))]
    for (k,mode) in enumerate(system.modes)
        a = _tfd_annihilation(system.boson_sizes[k])
        push!(terms, ProductTerm(mode.frequency,Dict(k+1 => a'*a)))
        push!(terms, ProductTerm(mode.amplitude,Dict(1 => kron(identity,mode.coupling), k+1 => (a+a')/sqrt(2))))
    end
    return SumOfProducts(tfd_dimensions(system),terms)
end

function _tfd_generator(H::SumOfProducts)
    return SumOfProducts(H.dimensions,[ProductTerm(-im*t.coefficient,t.factors) for t in H.terms])
end

"""Schrodinger generator `-im*H`, without a norm correction."""
tfd_generator(system::TFDTTNSystem) = _tfd_generator(tfd_hamiltonian(system))

"""
    tfd_initial_state(system, rho; tree=tfd_tree(system))

Purify a positive semidefinite electronic matrix as `vec(sqrt(rho))` and put
every thermally transformed bath mode in vacuum. Does not normalize rho.
The electronic ancilla is inactive; this is a factorized initial state, not
an interacting thermal equilibrium state. Only roundoff-negative eigenvalues
(within 100eps times the matrix spectral scale) are clipped.
"""
function tfd_initial_state(system::TFDTTNSystem, rho::AbstractMatrix;
                           tree::BinaryTree=tfd_tree(system))
    length(tree.leaf_order) == length(system.modes)+1 || throw(DimensionMismatch("TFD tree needs one fused electronic leaf and one leaf per mode"))
    size(rho) == size(system.hamiltonian) || throw(DimensionMismatch("electronic density dimensions differ"))
    density = _hermitian_matrix(rho,"initial density")
    values, vectors = eigen(Hermitian(density))
    minimum(values) >= -100eps(Float64)*maximum(abs,values) || throw(ArgumentError("initial density must be positive semidefinite"))
    sqrt_rho = vectors*Diagonal(sqrt.(max.(values,0)))*vectors'
    locals = [vec(sqrt_rho)]
    for n in system.boson_sizes
        vacuum = zeros(ComplexF64,n)
        vacuum[1] = 1
        push!(locals,vacuum)
    end
    return product_state(tree,locals)
end

function _check_tfd_state(state,system)
    leafdims(state) == tfd_dimensions(system) || throw(DimensionMismatch("state dimensions differ from TFD model"))
end

"""
    tfd_density_matrix(state, system)

Trace every bath mode and the inactive electronic ancilla using local
contractions. Does not extract a vacuum slice or expand the bath tensor.
Returns the unnormalized electronic density matrix on any compatible tree.
"""
function tfd_density_matrix(state::TTNTensor, system::TFDTTNSystem)
    _check_tfd_state(state,system)
    identity = SumOfProducts(tfd_dimensions(system),[ProductTerm(1,Dict())])
    env = operator_environments(identity,state)
    node = state.tree.leaf_order[1]
    U = state.leaves[node]
    fused = U*transpose(env.downward[node][1])*U'
    m = size(system.hamiltonian,1)
    rho = zeros(ComplexF64,m,m)
    for ancilla in 0:m-1
        block = (1:m) .+ m*ancilla
        rho .+= view(fused,block,block)
    end
    return rho
end

function _tfd_observer(system, H=tfd_hamiltonian(system))
    dims = tfd_dimensions(system)
    annihilators = _tfd_annihilation.(system.boson_sizes)
    number_ops = [SumOfProducts(dims,[ProductTerm(1,Dict(k+1 => a'*a))]) for (k,a) in enumerate(annihilators)]
    physical_ops = SumOfProducts[]
    for (p,q,n) in system.thermal_pairs
        a,b = annihilators[p],annihilators[q]
        cross = sqrt(n)*sqrt(n+1)
        # Project A†A after normal ordering in the infinite oscillator space.
        # Multiplying truncated b*b' would lose n*cutoff at the top Fock state.
        terms = [ProductTerm(n+1,Dict(p+1 => a'*a)), ProductTerm(n,Dict(q+1 => b'*b+I)),
                 ProductTerm(cross,Dict(p+1 => a',q+1 => b')),
                 ProductTerm(cross,Dict(p+1 => a,q+1 => b))]
        push!(physical_ops,SumOfProducts(dims,terms))
    end
    return state -> (; density=tfd_density_matrix(state,system), norm=norm(state),
        energy=real(dot(state,H,state)),
        mode_occupations=[real(dot(state,N,state)) for N in number_ops],
        physical_occupations=[real(dot(state,N,state)) for N in physical_ops])
end

"""
    tfd_observables(state, system)

Return density, wavefunction norm, energy, transformed-mode occupations and
inverse-transformed physical occupations (one per thermal_pairs entry).
Physical number is projected after normal ordering: `(nbar+1)Nplus +
nbar*(Nminus+I) + sqrt(nbar*(nbar+1))*(a†b†+ab)`.
All expectations are raw `<psi|O|psi>`; no division by norm or repair of drift.
For explicit unpaired modes physical_occupations is empty.
"""
function tfd_observables(state::TTNTensor,system::TFDTTNSystem)
    _check_tfd_state(state,system)
    return _tfd_observer(system)(state)
end
