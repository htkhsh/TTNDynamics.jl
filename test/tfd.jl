include("../examples/holstein_tfd/dense_reference.jl")

@testset "TFD models and contractions" begin
    H = ComplexF64[0.2 0.3im; -0.3im -0.1]
    V = ComplexF64[1 0.2im; -0.2im -0.4]
    modes = [TFDMode(V, 0.7, 0.4), TFDMode(V, -0.7, -0.2)]
    system = TFDTTNSystem(H, modes, [3,2])
    @test tfd_dimensions(system) == [4,3,2]
    reference = dense_tfd_reference_matrix(system)
    @test dense(tfd_hamiltonian(system)) ≈ reference atol=1e-14
    @test dense(tfd_generator(system)) ≈ -im*reference atol=1e-14
    @test ishermitian(reference)
    # Legacy TTDynamics tt_mkron puts the fused electronic factor first/slowest.
    ids = [Matrix{ComplexF64}(I,n,n) for n in (4,3,2)]
    old = kron(kron(kron(Matrix{ComplexF64}(I,2,2),H),ids[2]),ids[3])
    for (k,mode) in enumerate(system.modes)
        a = diagm(1=>sqrt.(Float64.(1:system.boson_sizes[k]-1)))
        factors = copy(ids)
        factors[k+1] = a'*a
        old += mode.frequency*reduce(kron,factors)
        factors[1] = kron(Matrix{ComplexF64}(I,2,2),mode.coupling)
        factors[k+1] = (a+a')/sqrt(2)
        old += mode.amplitude*reduce(kron,factors)
    end
    permutation = [6*(i[1]-1)+2*(i[2]-1)+i[3] for i in CartesianIndices((4,3,2))][:]
    @test reference ≈ old[permutation,permutation] atol=1e-14
    H[1,1] = 8
    V[1,1] = 8
    @test system.hamiltonian[1,1] == 0.2
    @test system.modes[1].coupling[1,1] == 1
    modes[1].coupling[1,1] = 5
    @test system.modes[1].coupling[1,1] == 1
    rho = ComplexF64[0.7 0.1im; -0.1im 0.3]
    for tree in (tfd_tree(system), BinaryTree([(0,0),(0,0),(0,0),(1,3),(4,2)]; leaf_order=[3,1,2]))
        state = tfd_initial_state(system, rho; tree)
        @test tfd_density_matrix(state,system) ≈ rho atol=1e-14
        @test norm(state)^2 ≈ tr(rho)
        # A complex entangled state probes the conjugation of the bath environment.
        terms = [tfd_hamiltonian(system).terms; ProductTerm(0.2+0.4im,Dict())]
        state = SumOfProducts(tfd_dimensions(system),terms)*state
        @test tfd_density_matrix(state,system) ≈ dense_tfd_density(vec(dense(state)),2) atol=1e-13
        obs = tfd_observables(state,system)
        @test obs.energy ≈ real(dot(vec(dense(state)),reference*vec(dense(state))))
        @test obs.norm ≈ norm(vec(dense(state)))
        @test isempty(obs.physical_occupations)
    end
    @test_throws ArgumentError TFDMode(V, Inf, 1)
    @test_throws ArgumentError TFDMode([0 1; 0 0], 1, 1)
    @test_throws ArgumentError TFDTTNSystem(H, modes, [0,2])
    @test_throws DimensionMismatch TFDTTNSystem(H, modes, [3])
    @test_throws DimensionMismatch TFDTTNSystem(H, [TFDMode(ones(1,1),1,1)], [2])
    @test_throws ArgumentError tfd_initial_state(system, [1 0; 0 -0.1])
    @test_throws ArgumentError tfd_initial_state(system, [1 1; 0 0])
    @test_throws DimensionMismatch tfd_initial_state(system, ones(1,1))
    @test_throws DimensionMismatch tfd_initial_state(system, rho; tree=balanced_tree(2))
    @test tfd_density_matrix(tfd_initial_state(system,2rho),system) ≈ 2rho
    @test iszero(norm(tfd_initial_state(system,zeros(2,2))))
    tiny = TFDTTNSystem(system.hamiltonian,[TFDMode(system.modes[1].coupling,0,1)],[1])
    @test dense(tfd_hamiltonian(tiny)) ≈ kron(Matrix{ComplexF64}(I,2,2),tiny.hamiltonian)
end

@testset "Holstein thermal pairs" begin
    system = holstein_tfd_system([0.1,-0.1], 0.3, [0.7,0.9], [0.2,0.4]; beta=2.0, boson_size=3)
    @test system.hamiltonian ≈ [0.1 -0.3; -0.3 -0.1]
    @test [m.frequency for m in system.modes] == [0.7,-0.7,0.9,-0.9]
    @test tfd_dimensions(system) == [4,3,3,3,3]
    n = 1/expm1(1.4)
    @test system.modes[1].amplitude ≈ 0.2sqrt(n+1)
    @test system.modes[2].amplitude ≈ 0.2sqrt(n)
    @test dense(tfd_hamiltonian(system)) ≈ dense_tfd_reference_matrix(system)
    state = tfd_initial_state(system, [1 0; 0 0])
    obs = tfd_observables(state,system)
    @test obs.physical_occupations ≈ 1 ./ expm1.([1.4,1.8])
    @test obs.mode_occupations == zeros(4)
    numbers = dense_tfd_physical_numbers(system)
    top = product_state(tfd_tree(system), [ComplexF64[1,0,0,0], ComplexF64[1,0,0],
        ComplexF64[0,0,1],ComplexF64[1,0,0],ComplexF64[1,0,0]])
    @test tfd_observables(top,system).physical_occupations[1] ≈ 3n
    # Top-cutoff and pair coherence, with the observable projected after multiplication.
    vectors = [ComplexF64[1,0,0,0], ComplexF64[1,im,2], ComplexF64[1,2im,1],
               ComplexF64[1,1,im], ComplexF64[1,0,2]]
    state = product_state(tfd_tree(system), vectors)
    v = vec(dense(state))
    @test tfd_observables(state,system).physical_occupations ≈ [real(dot(v,N*v)) for N in numbers]
    cold = holstein_tfd_system([0,0],0.3,[0.7,0.9],[0.2,0.4]; beta=Inf)
    @test cold.modes[2].amplitude == cold.modes[4].amplitude == 0
    @test tfd_observables(tfd_initial_state(cold,[1 0;0 0]),cold).physical_occupations == [0,0]
    @test_throws ArgumentError holstein_tfd_system([0],0,[1],[1]; beta=0)
    @test_throws ArgumentError holstein_tfd_system([0],0,[0],[1]; beta=1)
    @test_throws ArgumentError holstein_tfd_system([0],0,[1],[1]; beta=1,boson_size=1)
    @test_throws DimensionMismatch holstein_tfd_system([0,0],0,[1],[1]; beta=1)
    @test_throws ArgumentError TFDTTNSystem(system.hamiltonian,system.modes,system.boson_sizes;
                                           thermal_pairs=[(1,2,-1.0)])
    @test_throws ArgumentError TFDTTNSystem(system.hamiltonian,system.modes,system.boson_sizes;
                                           thermal_pairs=[(1,2,0.0)])
    @test_throws ArgumentError TFDTTNSystem(system.hamiltonian,system.modes,system.boson_sizes;
                                           thermal_pairs=[(1,2,n),(1,2,n)])
end

@testset "Adaptive TFD evolution" begin
    # Pure electronic coherence and centered coupling: all initial mean fields
    # vanish, so a rank-one BUG basis needs an H*psi direction to start growing.
    centered = TFDTTNSystem(zeros(2,2),[TFDMode([1 0;0 -1],1,1)],[3])
    centered_initial = tfd_initial_state(centered,fill(0.5,2,2))
    seeded = tfd_evolve(centered,centered_initial;tspan=(0,1),dt=0.1,rtol=1e-7,atol=1e-10)
    centered_ref = exp(-im*dense_tfd_reference_matrix(centered))*vec(dense(centered_initial))
    @test last(seeded.densities) ≈ dense_tfd_density(centered_ref,2) atol=2e-6
    @test norm(vec(dense(seeded.final_state))-centered_ref) < 3e-6
    @test first(seeded.densities) ≈ fill(0.5,2,2) atol=1e-14
    @test tfd_density_matrix(centered_initial,centered) ≈ fill(0.5,2,2) atol=1e-14
    system = holstein_tfd_system([0.1,-0.1],0.4,[0.7,0.9],[0.3,0.2]; beta=2.0,boson_size=2)
    initial = tfd_initial_state(system, ComplexF64[0.7 0.1im;-0.1im 0.3])
    H = dense_tfd_reference_matrix(system)
    v = vec(dense(initial))
    times = 0:0.1:0.5
    reference = [exp(-im*t*H)*v for t in times]
    numbers = dense_tfd_physical_numbers(system)
    errors = Float64[]
    for rtol in (1e-3,1e-5)
        result = tfd_evolve(system,initial;tspan=(0,0.5),dt=0.1,saveat=times,atol=1e-9,rtol)
        push!(errors, maximum(norm(rho-dense_tfd_density(ref,2)) for (rho,ref) in zip(result.densities,reference)))
        @test result.times ≈ collect(times)
        @test maximum(abs.(result.norms .- norm(initial))) < 1e-4
        @test size(result.mode_occupations) == (4,6)
        @test size(result.physical_occupations) == (2,6)
        @test maximum(abs(result.physical_occupations[j,k]-real(dot(reference[k],N*reference[k])))
                      for (j,N) in enumerate(numbers), k in eachindex(times)) < 1e-3
    end
    @test errors[2] < 3e-5
    @test errors[2] < errors[1]
    bare = TFDTTNSystem([0 0.3im;-0.3im 0],TFDMode[],Int[])
    initial = tfd_initial_state(bare,[1.4 0;0 0.6])
    result = tfd_evolve(bare,initial;tspan=(0,1),dt=0.2,saveat=[0,1],rtol=1e-7,enrich_initial=false)
    U = exp(-im*bare.hamiltonian)
    @test last(result.densities) ≈ U*[1.4 0;0 0.6]*U' atol=1e-12
    @test last(result.norms) ≈ sqrt(2) atol=1e-12
    @test size(result.mode_occupations) == (0,2)
    @test size(result.physical_occupations) == (0,2)
end
