include("common.jl")
include("GroupedKernels.jl")
using Test

module GroupedBug
using LinearAlgebra
import TTNSolver
import ..GroupedKernels
for binding in (:TTNTensor,:SumOfProducts,:_compression_options,:_check_operator,:_check_truncation_weighting,
        :orthogonalize!,:_canonical_conserved,:tree_ranks,:_upward_operators,
        :_isleaf,:_complement_matrix,:_complement_operator,:_parent_product,
        :_leaf_rhs_workspace,:_orth_columns,:_local_action,
        :_project_children,:_basis_matrix,:_subtree_operator,:truncate!,:_subtree_overlaps,
        :_complement_operators,:_leaf_operators,:_subtree_operators)
    @eval const $binding = TTNSolver.$binding
end
const KrylovKit = TTNSolver.KrylovKit
const COUNTS = NTuple{5,Int}[]
const LOCK = ReentrantLock()
function _core_rhs_workspace(op,b,l,r,x)
    action,n = GroupedKernels.workspace(op,b,l,r,x)
    lock(LOCK) do
        push!(COUNTS,(size(x)...,length(op.terms),n))
    end
    return action
end
include_string(@__MODULE__,read(joinpath(dirname(pathof(TTNSolver)),"bug.jl"),String),"GroupedBug.jl")
end

@testset "Grouped local action against dense Kronecker operator" begin
    rng = MersenneTwister(1203)
    B = randn(rng,ComplexF64,3,3); L = randn(rng,ComplexF64,4,4); R = randn(rng,ComplexF64,2,2)
    triples = [(copy(B),randn(rng,ComplexF64,4,4),copy(R)) for _ in 1:4]
    append!(triples,[(copy(B),copy(L),randn(rng,ComplexF64,2,2)) for _ in 1:3])
    append!(triples,[(randn(rng,ComplexF64,3,3),copy(L),copy(R)) for _ in 1:2])
    coefficients = randn(rng,ComplexF64,length(triples))
    op = (;terms=[(;coefficient=c) for c in coefficients])
    boundary,left,right = ([t[j] for t in triples] for j in 1:3)
    x = randn(rng,ComplexF64,3,4,2)
    originals = deepcopy((boundary,left,right,x))
    f,count = GroupedKernels.workspace(op,boundary,left,right,x)
    A = sum(c*kron(t[3],kron(t[2],t[1])) for (c,t) in zip(coefficients,triples))
    @test count==3
    @test vec(f(x)) ≈ A*vec(x) rtol=1e-12
    first_result = f(x); first_saved = copy(first_result)
    y = randn(rng,ComplexF64,size(x)...)
    @test vec(f(y)) ≈ A*vec(y) rtol=1e-12
    @test first_result==first_saved
    @test (boundary,left,right,x)==originals
    # Similar but unequal factors must remain distinct; grouping is not a
    # numerical-rank approximation and has no comparison tolerance.
    near = deepcopy(triples[1][1]); near[1,1] += 1e-10
    tinyop = (;terms=[(;coefficient=1.0+0im),(;coefficient=-1.0+0im)])
    g,n = GroupedKernels.workspace(tinyop,[B,near],[L,2L],[R,2R],x)
    @test n==2
    @test vec(g(x)) ≈ (kron(R,kron(L,B))-kron(2R,kron(2L,near)))*vec(x) rtol=1e-12
end

function probe()
    dir = joinpath(@__DIR__,"data","grouped_probe"); mkpath(dir)
    p = example_problem()
    x = deserialize(joinpath(@__DIR__,"data","augmented_h005","latest.jls")).state
    options = (;atol=5e-5,maxrank=32,conserved=p.conserved,correct_invariant=false,threaded=true)
    open(joinpath(dir,"metadata.txt"),"w") do io
        println(io,"Same input: augmented_h005 at 1 fs; h=.05; cap32; no production source changes")
        source_metadata(io)
    end
    println("Baseline warm-up"); flush(stdout)
    augmented_bug_step(x,p.operator,.05;options...)
    GC.gc()
    baseline = @timed augmented_bug_step(x,p.operator,.05;options...)
    println("BASELINE seconds=",baseline.time); flush(stdout)
    println("Grouped warm-up"); flush(stdout)
    GroupedBug.augmented_bug_step(x,p.operator,.05;options...)
    empty!(GroupedBug.COUNTS); GC.gc()
    grouped = @timed GroupedBug.augmented_bug_step(x,p.operator,.05;options...)
    distance = ttn_distance(baseline.value[1],grouped.value[1])/norm(baseline.value[1])
    density_distance = norm(TTNDynamics.root_density_matrix(baseline.value[1],p.system)-
                            TTNDynamics.root_density_matrix(grouped.value[1],p.system))
    open(joinpath(dir,"timings.csv"),"w") do io
        println(io,"implementation,seconds,gc_seconds,allocated_bytes,state_difference,density_difference")
        for (label,m) in (("baseline",baseline),("grouped",grouped))
            println(io,join((label,m.time,m.gctime,m.bytes,label=="baseline" ? 0.0 : distance,
                            label=="baseline" ? 0.0 : density_distance),','))
        end
    end
    open(joinpath(dir,"terms.csv"),"w") do io
        println(io,"parent,left,right,before,after")
        for row in sort(GroupedBug.COUNTS)
            println(io,join(row,','))
        end
    end
    println("GROUPED seconds=",grouped.time," distance=",distance," density=",density_distance); flush(stdout)
    @test distance < 1e-9
end

probe()
