@testset "Parallel BUG model evolution" begin
    H = ComplexF64[0.1 0.3im;-0.3im -0.1]
    V = ComplexF64[1 0.1;0.1 -0.4]
    modes = [HEOMMode(V,0.8,0.1+0.02im,0.1-0.02im)]
    rho = ComplexF64[0.8 0.05im;-0.05im 0.2]
    for layout in (:fused,:split)
        system = HEOMTTNSystem(H,modes,[3];system_layout=layout)
        initial = heom_initial_state(system,rho)
        reference = exp(0.2*dense_heom_reference_matrix(system))*vec(dense(initial))
        errors = Float64[]
        for h in (0.02,0.01)
            result = heom_evolve(system,initial,0:h:0.2;method=:parallel,
                threaded=false,atol=1e-13,correct_trace=false)
            push!(errors,norm(vec(dense(result.final_state))-reference))
            @test maximum(abs.(result.traces.-1)) < 2e-12
            @test all(d -> d.method == :parallel,result.diagnostics)
            @test all(d -> d.correction_norm == 0,result.diagnostics)
        end
        @test errors[2] < 0.7errors[1]
        @test errors[2] < 2e-3
        serial = heom_evolve(system,initial;tspan=(0.,0.2),dt=0.05,method=:parallel,
            threaded=false,atol=1e-8,rtol=1e-5,correct_trace=false,record_timing=true)
        threaded = heom_evolve(system,initial;tspan=(0.,0.2),dt=0.05,method=:parallel,
            threaded=true,atol=1e-8,rtol=1e-5,correct_trace=false)
        @test norm(vec(dense(serial.final_state))-reference) < 2e-4
        @test ttn_distance(serial.final_state,threaded.final_state) < 1e-12
        @test serial.stats == threaded.stats
        @test serial.timings.coarse_bug > 0
        @test maximum(abs.(serial.traces.-1)) < 2e-12
        @test_throws ArgumentError heom_evolve(system,initial,[0.,0.1];method=:unknown)
        @test_throws ArgumentError heom_evolve(system,initial;tspan=(0.,0.0),dt=0.1,method=:unknown)
    end
    system = holstein_tfd_system([0.1,-0.1],0.4,[0.7,0.9],[0.3,0.2];beta=2.0,boson_size=2)
    initial = tfd_initial_state(system,rho)
    reference = exp(-0.2im*dense_tfd_reference_matrix(system))*vec(dense(initial))
    errors = Float64[]
    for h in (0.02,0.01)
        result = tfd_evolve(system,initial,0:h:0.2;method=:parallel,atol=1e-13)
        push!(errors,norm(vec(dense(result.final_state))-reference))
        @test maximum(abs.(result.norms.-1)) < 2e-3
        @test all(d -> d.method == :parallel,result.diagnostics)
    end
    @test errors[2] < 0.7errors[1]
    @test errors[2] < 5e-3
    result = tfd_evolve(system,initial;tspan=(0.,0.2),dt=0.05,method=:parallel,
        atol=1e-8,rtol=1e-5)
    @test norm(vec(dense(result.final_state))-reference) < 5e-4
    @test_throws ArgumentError tfd_evolve(system,initial,[0,0.1];method=:unknown)
end
