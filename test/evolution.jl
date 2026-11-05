@testset "TTN HEOM evolution" begin
    H = ComplexF64[0.2 0.3+0.1im; 0.3-0.1im -0.4]
    V = ComplexF64[1 0.2im; -0.2im -0.5]
    modes = [HEOMMode(V, 0.8, 0.7+0.2im, 0.7-0.2im),
             HEOMMode(V, 0.4+1.2im, 0.3+0.1im, 0.2+0.05im; scale=0.7),
             HEOMMode(V, 0.4-1.2im, 0.2-0.05im, 0.3-0.1im; scale=0.7)]
    system = HEOMTTNSystem(H, modes, [3, 3, 3])
    rho = ComplexF64[0.7 0.1im; -0.1im 0.3]
    initial = heom_initial_state(system, rho)
    before = dense(initial)
    L = dense_heom_reference_matrix(system)
    reference = exp(0.2L)*vec(before)
    errors = Float64[]
    for h in (0.02, 0.01)
        result = heom_evolve(system, initial, 0:h:0.2; atol=1e-12, correct_trace=false)
        push!(errors, norm(vec(dense(result.final_state))-reference))
        @test maximum(abs.(result.traces .- 1)) < 1e-11
        @test result.times == collect(0:h:0.2)
        @test length(result.densities) == length(result.times)
        @test result.densities[1] == rho
        @test maximum(result.ranks) > 1
        @test all(info -> info.truncation.tolerance_met, result.diagnostics)
    end
    @test errors[2] < errors[1]
    @test errors[2] < 1e-3
    @test dense(initial) == before
    # Forced active compression: trace direction survives even a tight cap,
    # and accuracy loss is visible in diagnostics without root correction.
    capped = heom_evolve(system, initial, 0:0.02:0.1; atol=1e-14, maxrank=2, correct_trace=false)
    @test maximum(abs.(capped.traces .- 1)) < 1e-11
    @test maximum(capped.ranks) <= 2
    @test any(info -> !info.truncation.tolerance_met, capped.diagnostics)
    @test all(info -> info.correction_norm == 0, capped.diagnostics)
    emptybath = HEOMTTNSystem(H, HEOMMode[], Int[])
    initial0 = heom_initial_state(emptybath, 2rho)
    result0 = heom_evolve(emptybath, initial0, [1.0, 1.1, 1.3])
    U = exp(-0.3im*H)
    @test last(result0.densities) ≈ U*(2rho)*U' atol=1e-12
    @test result0.traces ≈ fill(2, 3) atol=1e-12
    @test_throws ArgumentError heom_evolve(system, initial, Float64[])
    @test_throws ArgumentError heom_evolve(system, initial, [0, 0])
    @test_throws ArgumentError heom_evolve(system, initial, [0, Inf])
    @test_throws ArgumentError heom_evolve(system, initial, [1, 0])
    @test_throws DimensionMismatch heom_evolve(system, initial0, [0, 1])
end
