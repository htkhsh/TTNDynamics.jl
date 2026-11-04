using Test
using Random
using LinearAlgebra
using TTNSolver
using TTNDynamics

include("../examples/dense_heom_reference.jl")

@testset "TTNDynamics" begin
    include("heom.jl")
end
