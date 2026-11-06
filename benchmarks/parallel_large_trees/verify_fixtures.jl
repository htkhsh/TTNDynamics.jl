using Test
include("fixtures.jl")

@testset "Large-tree benchmark fixtures" begin
    for k in (8,16,32,64)
        p = large_problem("heom$k")
        meta = case_metadata(p,p.initial)
        @test length(p.system.modes) == k
        @test meta.leaves == k+1 && meta.nodes == 2k+1
        @test meta.height == 1+Int(log2(k))
        @test sum(m.forward for m in p.system.modes) ≈ 0.64+0.08im
        @test tr(p.observe(p.initial).density) ≈ 1
    end
    for k in (8,16)
        p = large_problem("tfd$k")
        @test length(p.system.modes) == k
        @test length(p.system.thermal_pairs) == k÷2
        weights = [p.system.modes[a].amplitude^2-p.system.modes[b].amplitude^2
                   for (a,b,_) in p.system.thermal_pairs]
        @test sum(weights) ≈ 4*0.3^2
        @test length(p.operator.terms) == 1+2k
        vacuum = tfd_initial_state(p.system,diagm([0.85,0.05,0.05,0.05]))
        @test ttn_distance(p.initial,vacuum) < 1e-13
        if k == 8
            H = tfd_hamiltonian(p.system)
            active = SumOfProducts(H.dimensions,filter(t -> haskey(t.factors,1),H.terms))
            @test ttn_distance(H*vacuum,active*vacuum) < 1e-12
        end
    end
end
