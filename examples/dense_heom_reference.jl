# Validation only: direct density-matrix equations for every ADO. This does
# not call the product-sum generator or construct Liouville superoperators.
function dense_heom_reference_rhs(system, state)
    d = size(system.hamiltonian, 1)
    dimensions = (d, d, system.hierarchy_sizes...)
    size(state) == dimensions || throw(DimensionMismatch("reference ADO shape differs"))
    result = zeros(ComplexF64, dimensions)
    for index in CartesianIndices(Tuple(system.hierarchy_sizes))
        indices = Tuple(index)
        rho = view(state, :, :, indices...)
        derivative = view(result, :, :, indices...)
        H = system.hamiltonian
        derivative .= -im .* (H*rho - rho*H)
        for (k, mode) in enumerate(system.modes)
            n = indices[k] - 1
            derivative .-= n * mode.rate .* rho
            V = mode.coupling
            if n + 1 < system.hierarchy_sizes[k]
                plus = Base.setindex(indices, indices[k]+1, k)
                neighbor = view(state, :, :, plus...)
                derivative .-= im * mode.scale * sqrt(n+1) .* (V*neighbor - neighbor*V)
            end
            if n > 0
                minus = Base.setindex(indices, indices[k]-1, k)
                neighbor = view(state, :, :, minus...)
                derivative .-= im * sqrt(n) / mode.scale .*
                    (mode.forward * V*neighbor - mode.backward * neighbor*V)
            end
        end
    end
    return result
end

function dense_heom_reference_matrix(system)
    d = size(system.hamiltonian, 1)
    dimensions = (d, d, system.hierarchy_sizes...)
    total = prod(dimensions)
    result = zeros(ComplexF64, total, total)
    basis = zeros(ComplexF64, dimensions)
    for column in 1:total
        fill!(basis, 0)
        basis[column] = 1
        result[:, column] = vec(dense_heom_reference_rhs(system, basis))
    end
    return result
end
