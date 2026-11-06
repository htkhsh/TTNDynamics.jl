using Test
using Random
using LinearAlgebra
using TTNSolver
using TTNDynamics

include("../examples/heom/common/dense_heom_reference.jl")

@testset "TTNDynamics" begin
    include("heom.jl")
    include("heom_layout.jl")
    include("heom_grouped_tree.jl")
    include("evolution.jl")
    include("adaptive_evolution.jl")
    include("tfd.jl")
    include("tfd_fixed_evolution.jl")
    include("parallel_evolution.jl")
    include("truncation_policy.jl")
    include("augmented_threads.jl")
    include("tamen_evolution.jl")
    include("tamen_adaptive_evolution.jl")
    include("tamen_warm_start.jl")
end
