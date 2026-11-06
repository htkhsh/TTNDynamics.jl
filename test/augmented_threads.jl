@testset "Augmented threading in model drivers" begin
    H = ComplexF64[0.1 0.3im;-0.3im -0.1]
    V = ComplexF64[1 0.1;0.1 -0.4]
    modes = [HEOMMode(V,0.8,0.1+0.02im,0.1-0.02im)]
    rho = ComplexF64[0.8 0.05im;-0.05im 0.2]
    systems = Any[HEOMTTNSystem(H,modes,[3];system_layout=l) for l in (:fused,:split)]
    push!(systems,holstein_tfd_system([0.1,-0.1],0.4,[0.7,0.9],[0.3,0.2];beta=2.,boson_size=2))
    for system in systems
        heom = system isa HEOMTTNSystem
        initial = heom ? heom_initial_state(system,rho) : tfd_initial_state(system,rho)
        evolve = heom ? heom_evolve : tfd_evolve
        opts = heom ? (;correct_trace=false) : (;)
        for adaptive in (false,true)
            run = threaded -> adaptive ?
                evolve(system,initial;tspan=(0.,0.1),dt=0.05,atol=1e-8,rtol=1e-4,threaded,opts...) :
                evolve(system,initial,0:0.02:0.1;atol=1e-12,threaded,opts...)
            a,b = run(false),run(true)
            @test ttn_distance(a.final_state,b.final_state) < 1e-12
            @test a.ranks == b.ranks
            @test a.times == b.times
            @test maximum(norm(x-y) for (x,y) in zip(a.densities,b.densities)) < 1e-12
            da = adaptive ? [d.half2 for d in a.diagnostics if d.accepted] : a.diagnostics
            db = adaptive ? [d.half2 for d in b.diagnostics if d.accepted] : b.diagnostics
            @test all(d -> !d.threaded && d.method == :augmented,da)
            @test all(d -> d.threaded == (Threads.nthreads(:default) > 1),db)
            if heom
                @test maximum(abs.(b.traces.-1)) < 1e-12
                @test all(d -> iszero(d.correction_norm),db)
            end
            adaptive && @test a.stats == b.stats
        end
    end
end
