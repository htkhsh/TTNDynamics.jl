@testset "HEOM space-time TTN AMEn" begin
    H = ComplexF64[0.2 0.3+0.1im;0.3-0.1im -0.4]
    V = ComplexF64[1 0;0 -0.5]
    modes = [HEOMMode(V,0.4+1.2im,0.3+0.1im,0.2+0.05im;scale=0.7),
             HEOMMode(V,0.4-1.2im,0.2-0.05im,0.3-0.1im;scale=0.7)]
    rho = ComplexF64[1.4 0.1im;-0.1im 0.6]
    for layout in (:fused,:split), als_sweep in (:serial,:etter)
        system = HEOMTTNSystem(H,modes,[2,2];system_layout=layout)
        x = heom_initial_state(system,rho)
        L = dense_heom_reference_matrix(system)
        result = heom_tamen_evolve(system,x,[1.0,1.1,1.3];time_points=8,
                                  rtol=1e-9,maxrank=16,maxsweeps=14,als_sweep,threaded=true)
        exact = exp(0.3L)*vec(dense(x))
        @test norm(vec(dense(result.final_state))-exact) < 2e-8
        @test result.times == [1.0,1.1,1.3]
        @test result.densities[1] == rho
        @test maximum(abs.(result.traces .- 2)) < 1e-10
        @test all(d->d.converged,result.diagnostics)
        @test all(d->d.als_sweep==als_sweep,result.diagnostics)
        @test all(d->maximum(abs.(d.invariant_values.-2)) < 1e-10,result.diagnostics)
        @test !hasproperty(first(result.diagnostics),:solution) # default history is compact
        @test dense(x) == dense(heom_initial_state(system,rho))
        if layout == :split
            @test_throws ArgumentError heom_tamen_evolve(system,x,[0.,0.1];maxrank=1)
        end
        saved = heom_tamen_evolve(system,x,[0.,0.1];time_points=6,
                                 rtol=1e-8,maxrank=16,store_solutions=true)
        @test hasproperty(first(saved.diagnostics),:solution)
        @test length(first(saved.diagnostics).solution.nodes) == 7
        single = heom_tamen_evolve(system,x,[2.])
        @test isempty(single.diagnostics)
        @test single.final_state !== x
        @test_throws ArgumentError heom_tamen_evolve(system,x,Float64[])
        @test_throws ArgumentError heom_tamen_evolve(system,x,[0.,0.])
        @test_throws ArgumentError heom_tamen_evolve(system,x,[0.,Inf])
        @test_throws ArgumentError heom_tamen_evolve(system,x,[0.,0.1];strict=false)
        # Physics-owned trace protection cannot be disabled through forwarded keywords.
        @test_throws ArgumentError heom_tamen_evolve(system,x,[0.,0.1];conserved=nothing)
    end
end
