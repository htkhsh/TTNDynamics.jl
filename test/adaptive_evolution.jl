@testset "Adaptive HEOM evolution" begin
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
    reference = exp(0.3L)*vec(before)
    saveat = [0.03, 0.1, 0.3]
    results = [heom_evolve(system, initial; tspan=(0.,0.3), dt=0.2, saveat,
                           atol=1e-8, rtol, correct_trace=false)
               for rtol in (1e-3, 1e-5)]
    errors = [norm(vec(dense(r.final_state))-reference) for r in results]
    @test errors[2] < errors[1]
    @test errors[2] < 1e-3
    @test results[2].stats.rejected > 0
    @test any(d -> d.reason == :error, results[2].diagnostics)
    for result in results
        @test result.times == [0.; saveat]
        @test result.densities[1] == rho
        @test length(result.densities) == 4
        @test maximum(abs.(result.traces .- 1)) < 1e-11
        @test size(result.ranks, 2) == 4
        @test result.stats.attempts == result.stats.accepted + result.stats.rejected
        kept = filter(d -> d.accepted, result.diagnostics)
        @test all(d -> d.error_ratio <= 1, kept)
        @test all(d -> d.coarse.truncation.tolerance_met &&
                       d.half1.truncation.tolerance_met && d.half2.truncation.tolerance_met, kept)
        @test [d.time for d in kept][2:end] == [d.t_end for d in kept][1:end-1]
        @test last(kept).t_end == 0.3
        @test all(d -> d.half2.correction_norm == 0, kept)
    end
    @test dense(initial) == before
    @test_throws AdaptiveIntegrationError heom_evolve(system, initial;
        tspan=(0., 0.3), dt=0.1, maxrank=1, maxreject=3)
    @test_throws ArgumentError heom_evolve(system, initial;
        tspan=(0., 0.), dt=0.1, krylovdim=0)
end
