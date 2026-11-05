@testset "Fixed-step TFD evolution" begin
    system = holstein_tfd_system([0.1,-0.1],0.4,[0.7,0.9],[0.3,0.2];
                                beta=2.0,boson_size=2)
    rho = ComplexF64[0.7 0.1im;-0.1im 0.3]
    initial = tfd_initial_state(system,rho)
    before, initial_ranks = dense(initial),tree_ranks(initial)
    H = dense_tfd_reference_matrix(system)
    numbers = dense_tfd_physical_numbers(system)
    reference = exp(-0.2im*H)*vec(before)
    errors = Float64[]
    for h in (0.04,0.02)
        times = 0:h:0.2
        result = tfd_evolve(system,initial,times;atol=1e-12,maxrank=32)
        push!(errors,norm(vec(dense(result.final_state))-reference))
        @test result.times == collect(times)
        @test length(result.diagnostics) == length(times)-1
        @test size(result.ranks) == (length(initial.tree.children),length(times))
        @test first(result.densities) ≈ rho atol=1e-13
        @test last(result.densities) ≈ dense_tfd_density(reference,2) atol=2e-3
        @test size(result.mode_occupations) == (4,length(times))
        @test size(result.physical_occupations) == (2,length(times))
        @test result.physical_occupations[:,end] ≈
              [real(dot(reference,N*reference)) for N in numbers] atol=2e-3
        @test maximum(abs.(result.norms .- norm(initial))) < 1e-10
        @test last(result.energies) ≈ real(dot(reference,H*reference)) atol=1e-9
        @test all(info -> info.truncation.tolerance_met,result.diagnostics)
    end
    @test errors[2] < errors[1]
    @test errors[2] < 2e-3
    capped = tfd_evolve(system,initial,[0,0.1];atol=1e-14,maxrank=1)
    @test maximum(tree_ranks(capped.final_state)) == 1
    @test !only(capped.diagnostics).truncation.tolerance_met
    @test last(capped.norms) < norm(initial)-1e-8
    @test dense(initial) == before
    @test tree_ranks(initial) == initial_ranks

    # Centered mean fields would freeze this product state without enrichment.
    centered = TFDTTNSystem(zeros(2,2),[TFDMode([1 0;0 -1],1,1)],[3])
    centered_initial = tfd_initial_state(centered,fill(0.5,2,2))
    seeded = tfd_evolve(centered,centered_initial,0:0.025:0.5;atol=1e-12)
    centered_ref = exp(-0.5im*dense_tfd_reference_matrix(centered))*vec(dense(centered_initial))
    @test norm(vec(dense(seeded.final_state))-centered_ref) < 2e-3
end

@testset "Fixed-step TFD time grid and snapshots" begin
    bare = TFDTTNSystem([0 0.3im;-0.3im 0],TFDMode[],Int[])
    rho = ComplexF64[1.4 0;0 0.6]
    initial = tfd_initial_state(bare,rho)
    result = tfd_evolve(bare,initial,(1.0,1.1,1.3);enrich_initial=false)
    U = exp(-0.3im*bare.hamiltonian)
    @test last(result.densities) ≈ U*rho*U' atol=1e-12
    @test last(result.norms) ≈ sqrt(2) atol=1e-12
    @test result.energies ≈ fill(real(tr(bare.hamiltonian*rho)),3) atol=1e-12
    @test size(result.mode_occupations) == (0,3)
    @test size(result.physical_occupations) == (0,3)

    system = TFDTTNSystem([0 0.3;0.3 0],[TFDMode([1 0;0 -1],0.7,0.2)],[3])
    initial = tfd_initial_state(system,rho)
    before = dense(initial)
    single = tfd_evolve(system,initial,[2.0])
    @test single.times == [2.0]
    @test single.densities[1] ≈ rho atol=1e-13
    @test size(single.mode_occupations) == (1,1)
    @test size(single.physical_occupations) == (0,1)
    @test isempty(single.diagnostics)
    @test tree_ranks(single.final_state) == tree_ranks(initial)
    single.final_state.leaves[first(initial.tree.leaf_order)][1] += 0.25
    @test dense(initial) == before

    for grid in (Float64[],[0,0],[1,0],[0,Inf],[0,NaN],[-1e308,1e308],[0,1im])
        @test_throws ArgumentError tfd_evolve(system,initial,grid)
    end
    @test_throws DimensionMismatch tfd_evolve(system,tfd_initial_state(bare,rho),[0,1])
end
