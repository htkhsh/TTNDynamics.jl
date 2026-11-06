@testset "HEOM adaptive tAMEn" begin
    H = ComplexF64[0 0.4;0.4 0.1]
    V = ComplexF64[1 0;0 0]
    modes = [HEOMMode(V,0.3+0.8im,0.12-0.02im,0.04-0.01im;scale=0.4),
             HEOMMode(V,0.3-0.8im,0.04+0.01im,0.12+0.02im;scale=0.4)]
    for layout in (:fused,:split)
        system = HEOMTTNSystem(H,modes,[2,2];system_layout=layout)
        initial = heom_initial_state(system,ComplexF64[2 0;0 0])
        result = heom_tamen_evolve(system,initial;tspan=(0.,0.4),dt=0.4,
            saveat=[0.,0.07,0.23,0.4],rtol=1e-5,atol=1e-9,time_points=3,
            time_error_damp=10,maxrank=16,threaded=true)
        @test result.times == [0.,0.07,0.23,0.4]
        @test result.densities[1] == ComplexF64[2 0;0 0]
        @test maximum(abs.(result.traces.-2)) < 1e-9
        L = dense_heom_reference_matrix(system)
        @test norm(vec(dense(result.final_state))-exp(0.4L)*vec(dense(initial))) < 1e-4
        @test all(d->!d.accepted || maximum(abs.(d.solve.invariant_values.-2))<1e-9,result.diagnostics)
        @test result.accepted>=1
        @test_throws ArgumentError heom_tamen_evolve(system,initial;tspan=(0.,0.1),dt=0.1,conserved=nothing)
        @test_throws ArgumentError heom_tamen_evolve(system,initial;tspan=(0.,0.1),dt=0.1,conserved_observables=[])
        @test_throws ArgumentError heom_tamen_evolve(system,initial;tspan=(0.,0.1),dt=0.1,observe=identity)
        @test_throws ArgumentError heom_tamen_evolve(system,initial,[0.,0.1];conserved_observables=[])
    end
end
