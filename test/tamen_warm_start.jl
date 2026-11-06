@testset "HEOM interval warm starts" begin
    H = ComplexF64[0 0.4;0.4 0.1]
    V = ComplexF64[1 0;0 0]
    modes = [HEOMMode(V,0.3+0.8im,0.12-0.02im,0.04-0.01im;scale=0.4),
             HEOMMode(V,0.3-0.8im,0.04+0.01im,0.12+0.02im;scale=0.4)]
    for layout in (:fused,:split)
        system = HEOMTTNSystem(H,modes,[2,2];system_layout=layout)
        x = heom_initial_state(system,ComplexF64[2 0;0 0])
        times = [0.,0.1,0.25,0.4]
        warm = heom_tamen_evolve(system,x,times;time_points=7,rtol=1e-9,
            maxrank=16,warm_start=true,store_solutions=true)
        cold = heom_tamen_evolve(system,x,times;time_points=7,rtol=1e-9,
            maxrank=16,warm_start=false)
        @test !first(warm.diagnostics).warm_start_used
        @test all(d->d.warm_start_used,warm.diagnostics[2:end])
        @test all(d->!d.warm_start_used,cold.diagnostics)
        @test ttn_distance(warm.final_state,cold.final_state) < 1e-8
        @test maximum(abs.(warm.traces.-2)) < 1e-10
        @test all(d->maximum(abs.(d.invariant_values.-2)) < 1e-10,warm.diagnostics)
        @test all(d->!hasproperty(d,:warm_start),warm.diagnostics)
        @test all(d->hasproperty(d,:solution),warm.diagnostics)
        @test norm(vec(dense(warm.final_state))-exp(0.4dense_heom_reference_matrix(system))*vec(dense(x))) < 1e-8
        @test_throws ArgumentError heom_tamen_evolve(system,x,times;store_warm_start=true)
    end
end
