# Run with the same Julia package environment as the comparison benchmarks.
using Test
module ComparisonEnvironment
include("common.jl")
end
@testset "TT workspace binding in mixed TT/TTN benchmarks" begin
    @test ComparisonEnvironment.TAMEnWorkspace === ComparisonEnvironment.TTSolver.TAMEnWorkspace
    @test ComparisonEnvironment.TAMEnWorkspace(ComplexF64) isa ComparisonEnvironment.TTSolver.TAMEnWorkspace
end
