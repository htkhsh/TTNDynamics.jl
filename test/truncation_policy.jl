@testset "Model drivers accept both truncation policies" begin
    H=ComplexF64[0.1 0.2;0.2 -0.1]
    V=ComplexF64[1 0;0 -0.4]
    rho=ComplexF64[0.8 0.05im;-0.05im 0.2]
    heom=HEOMTTNSystem(H,[HEOMMode(V,0.8,0.1+0.02im,0.1-0.02im)],[2];system_layout=:split)
    tfd=holstein_tfd_system([0.1,-0.1],0.2,[0.7,0.9],[0.1,0.2];beta=2.0,boson_size=2)
    for (system,initial,evolve,generator) in (
        (heom,heom_initial_state(heom,rho),heom_evolve,dense_heom_reference_matrix(heom)),
        (tfd,tfd_initial_state(tfd,rho),tfd_evolve,-im*dense_tfd_reference_matrix(tfd)))
        exact=exp(0.01*generator)*vec(dense(initial))
        before=dense(initial)
        for method in (:augmented,:parallel), truncation in (:weighted,:local)
            fixed=evolve(system,initial,0.0:0.001:0.01;method,truncation,atol=1e-12,threaded=false)
            adaptive=evolve(system,initial;tspan=(0.0,0.01),dt=0.01,method,truncation,
                            atol=1e-8,rtol=1e-5,threaded=false)
            @test norm(vec(dense(fixed.final_state))-exact)<1e-4
            @test norm(vec(dense(adaptive.final_state))-exact)<1e-4
        end
        @test_throws ArgumentError evolve(system,initial,[0.0,0.01];truncation=:invalid)
        @test_throws ArgumentError evolve(system,initial;tspan=(0.0,0.0),dt=0.01,truncation=:invalid)
        @test dense(initial)==before
    end
end
