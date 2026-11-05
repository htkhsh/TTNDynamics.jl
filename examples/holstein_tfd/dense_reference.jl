# Small validation problems only. Independent basis-index assembly: no SoP.
using LinearAlgebra

function dense_tfd_reference_matrix(system)
    m = size(system.hamiltonian, 1)
    dims = (m*m, system.boson_sizes...)
    indices = CartesianIndices(dims)
    linear = LinearIndices(dims)
    H = zeros(ComplexF64, length(indices), length(indices))
    for ket in indices
        col = linear[ket]
        electron = mod(ket[1]-1, m)+1
        ancilla = div(ket[1]-1, m)
        for bra_e in 1:m
            bra = Base.setindex(Tuple(ket), bra_e+m*ancilla, 1)
            H[linear[bra...], col] += system.hamiltonian[bra_e,electron]
        end
        for (k, mode) in enumerate(system.modes)
            occupation = ket[k+1]-1
            H[col,col] += mode.frequency*occupation
            for delta in (-1, 1)
                target = occupation+delta
                0 <= target < system.boson_sizes[k] || continue
                amplitude = mode.amplitude*sqrt(max(occupation,target)/2)
                for bra_e in 1:m
                    bra = Base.setindex(Tuple(ket), bra_e+m*ancilla, 1)
                    bra = Base.setindex(bra, target+1, k+1)
                    H[linear[bra...],col] += amplitude*mode.coupling[bra_e,electron]
                end
            end
        end
    end
    return H
end

function dense_tfd_density(vector, m)
    amplitudes = reshape(vector, m, :)
    return amplitudes*amplitudes'
end

# Apply the physical annihilator into a space with one extra negative-mode
# level, then form A†A. This independently keeps transitions outside the state
# cutoff when projecting the physical number operator back into that space.
function dense_tfd_physical_numbers(system)
    dims = [size(system.hamiltonian,1)^2; system.boson_sizes]
    result = Matrix{ComplexF64}[]
    for (p,q,n) in system.thermal_pairs
        enlarged = copy(dims)
        enlarged[q+1] += 1
        rows,cols = LinearIndices(Tuple(enlarged)),LinearIndices(Tuple(dims))
        physical = zeros(ComplexF64,prod(enlarged),prod(dims))
        for ket in CartesianIndices(Tuple(dims))
            if ket[p+1] > 1
                bra = Base.setindex(Tuple(ket),ket[p+1]-1,p+1)
                physical[rows[bra...],cols[ket]] += sqrt(n+1)*sqrt(ket[p+1]-1)
            end
            bra = Base.setindex(Tuple(ket),ket[q+1]+1,q+1)
            physical[rows[bra...],cols[ket]] += sqrt(n)*sqrt(ket[q+1])
        end
        push!(result, physical'*physical)
    end
    return result
end
