# Diagnostic prototype only. The production solver is unchanged.
module GroupedKernels
using LinearAlgebra

struct LocalTerm
    coefficient::ComplexF64
    factors::NTuple{3,Matrix{ComplexF64}}
end

function combine(terms,axis)
    others = filter(!=(axis),1:3)
    lookup = Dict{Tuple{Matrix{ComplexF64},Matrix{ComplexF64}},Int}()
    groups = Vector{Int}[]
    for (i,term) in enumerate(terms)
        key = (term.factors[others[1]],term.factors[others[2]])
        index = get!(lookup,key) do
            push!(groups,Int[])
            length(groups)
        end
        push!(groups[index],i)
    end
    result = LocalTerm[]
    for indices in groups
        first_term = terms[first(indices)]
        if length(indices)==1
            push!(result,first_term)
            continue
        end
        summed = zeros(ComplexF64,size(first_term.factors[axis]))
        for i in indices
            summed .+= terms[i].coefficient .* terms[i].factors[axis]
        end
        factors = ntuple(i->i==axis ? summed : first_term.factors[i],3)
        push!(result,LocalTerm(1.0+0im,factors))
    end
    return result
end

function compile(operator,boundary,left,right)
    terms = [LocalTerm(t.coefficient,(boundary[k],left[k],right[k]))
             for (k,t) in enumerate(operator.terms) if !iszero(t.coefficient)]
    while true
        old_count = length(terms)
        for axis in (2,3,1)
            terms = combine(terms,axis)
        end
        length(terms)==old_count && return terms
    end
end

function workspace(operator,boundary,left,right,initial)
    terms = compile(operator,boundary,left,right)
    p,l,r = size(initial)
    left_input = Array{ComplexF64}(undef,l,p,r)
    left_output = similar(left_input)
    projected_left = similar(initial)
    right_input = Array{ComplexF64}(undef,r,p,l)
    right_output = similar(right_input)
    projected_right = similar(initial)
    acted = similar(initial)
    left_in,left_out = reshape(left_input,l,:),reshape(left_output,l,:)
    right_in,right_out = reshape(right_input,r,:),reshape(right_output,r,:)
    parent_in,parent_out = reshape(projected_right,p,:),reshape(acted,p,:)
    function action(x)
        result = zeros(ComplexF64,size(x))
        for term in terms
            B,L,R = term.factors
            permutedims!(left_input,x,(2,1,3))
            mul!(left_out,L,left_in)
            permutedims!(projected_left,left_output,(2,1,3))
            permutedims!(right_input,projected_left,(3,1,2))
            mul!(right_out,R,right_in)
            permutedims!(projected_right,right_output,(2,3,1))
            mul!(parent_out,B,parent_in)
            result .+= term.coefficient .* acted
        end
        return result
    end
    return action,length(terms)
end
end
