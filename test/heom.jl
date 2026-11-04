function random_heom_state(system, tree=heom_tree(system))
    rng = MersenneTwister(623)
    dims = heom_dimensions(system)
    leaves = Dict(node => randn(rng, ComplexF64, dims[k], node == tree.root ? 1 : 2)
                  for (k, node) in enumerate(tree.leaf_order))
    cores = Dict(node => randn(rng, ComplexF64, node == tree.root ? 1 : 2, 2, 2)
                 for node in eachindex(tree.children) if tree.children[node] != (0,0))
    return TTNTensor(tree, leaves, cores)
end

@testset "Scaled box HEOM" begin
    H = ComplexF64[0.2 0.3+0.1im; 0.3-0.1im -0.4]
    V = ComplexF64[1 0.2im; -0.2im -0.5]
    @testset "Dense ADO action and trace null vector" begin
        real_mode = HEOMMode(V, 0.8, 0.7+0.2im, 0.7-0.2im; scale=0.9)
        pair = [HEOMMode(V, 0.4+1.2im, 0.3+0.1im, 0.2+0.05im; scale=0.7),
                HEOMMode(V, 0.4-1.2im, 0.2-0.05im, 0.3-0.1im; scale=0.7)]
        for system in (HEOMTTNSystem(H, [real_mode], [3]),
                       HEOMTTNSystem(H, [real_mode; pair], [2,2,2]),
                       HEOMTTNSystem(H, [real_mode], [1]),
                       HEOMTTNSystem(H, HEOMMode[], Int[]))
            op = heom_generator(system)
            reference = dense_heom_reference_matrix(system)
            @test dense(op) ≈ reference atol=1e-12 rtol=1e-12
            state = random_heom_state(system)
            input = vec(dense(state))
            @test vec(dense(op*state)) ≈ reference * input atol=1e-12 rtol=1e-12
            observable = trace_observable(system)
            @test norm(vec(dense(observable))' * dense(op)) < 1e-12
            @test abs(dot(observable, op, state)) < 1e-12
            density = reshape(view(dense(state), :, ntuple(_->1, length(system.modes))...), 2, 2)
            @test root_density_matrix(state, system) ≈ density atol=1e-12
            @test dot(observable, state) ≈ tr(density) atol=1e-12
            tree = heom_tree(system)
            @test isempty(system.modes) || tree.children[tree.root][1] == tree.leaf_order[1]
            @test heom_dimensions(system) == [4; system.hierarchy_sizes]
        end
    end

    @testset "Initial state, general matrices and custom trees" begin
        mode = HEOMMode(V, 0.8, 0.7+0.2im, 0.7-0.2im)
        system = HEOMTTNSystem(H, [mode, mode], [2,3])
        rho = ComplexF64[2 im; 0 -1] # response-like matrix: not normalized or Hermitian
        tree = BinaryTree([(2,3),(0,0),(4,5),(0,0),(0,0)]; root=1, leaf_order=[5,2,4])
        state = heom_initial_state(system, rho; tree)
        @test state.tree === tree
        @test root_density_matrix(state, system) == rho
        @test dot(trace_observable(system; tree), state) ≈ tr(rho)
        reference = zeros(ComplexF64, 4, 2, 3)
        reference[:,1,1] = vec(rho)
        @test dense(state) == reference
        op = heom_generator(system)
        @test vec(dense(op*state)) ≈ dense_heom_reference_matrix(system) * vec(reference) atol=1e-12
        arbitrary = random_heom_state(system, tree)
        @test root_density_matrix(arbitrary, system) ≈ reshape(dense(arbitrary)[:,1,1],2,2) atol=1e-12
        @test abs(dot(trace_observable(system; tree), op, arbitrary)) < 1e-12
        rho .= 0
        @test root_density_matrix(state, system) == ComplexF64[2 im; 0 -1]
        @test_throws DimensionMismatch heom_initial_state(system, ones(3,3))
        @test_throws ArgumentError heom_initial_state(system, fill(NaN,2,2))
        @test_throws DimensionMismatch heom_initial_state(system, ones(2,2); tree=balanced_tree(2))
        @test_throws DimensionMismatch trace_observable(system; tree=balanced_tree(1))
        @test_throws DimensionMismatch root_density_matrix(product_state(balanced_tree(1), [ones(4)]), system)
    end

    @testset "Scaling and existing ket/bra permutation" begin
        mode = HEOMMode(V, 0.8, 0.7+0.2im, 0.7-0.2im; scale=0.9)
        system = HEOMTTNSystem(H, [mode], [3])
        unit_mode = HEOMMode(V, mode.rate, mode.forward, mode.backward; scale=1.0)
        unit_system = HEOMTTNSystem(H, [unit_mode], [3])
        weights = repeat([mode.scale^n for n in 0:2]; inner=4)
        @test dense(heom_generator(system)) ≈ Diagonal(1 ./ weights) *
            dense(heom_generator(unit_system)) * Diagonal(weights) atol=1e-12

        Is, Ib = Matrix{ComplexF64}(I,2,2), Matrix{ComplexF64}(I,3,3)
        n = diagm(0 => ComplexF64[0,1,2])
        a = diagm(1 => sqrt.(ComplexF64[1,2]))
        ad = a'
        twin = -im * (kron(H,Is,Ib)-kron(Is,transpose(H),Ib)) - mode.rate*kron(Is,Is,n) -
            im*mode.scale*(kron(V,Is,a)-kron(Is,transpose(V),a)) -
            im/mode.scale*(mode.forward*kron(V,Is,ad)-mode.backward*kron(Is,conj(V),ad))
        # TTSolver uses kron(ket,bra,bath); here vec(rho) and bath are column-major.
        permutation = vec(permutedims(reshape(1:12,2,2,3), (3,2,1)))
        @test dense(heom_generator(system))[permutation,permutation] ≈ twin atol=1e-12
    end

    @testset "Validation, independent coefficients and ownership" begin
        vcopy, hcopy = copy(V), copy(H)
        mode = HEOMMode(vcopy, 0.8, 0.2+0.3im, 0.7-0.4im)
        @test mode.scale ≈ sqrt(abs(0.7-0.4im))
        @test mode.backward == 0.7-0.4im
        vcopy .= 0
        @test mode.coupling == V
        modes, sizes = [mode], [2]
        system = HEOMTTNSystem(hcopy, modes, sizes)
        hcopy .= 0
        mode.coupling .= 0
        empty!(modes)
        sizes[1] = 8
        @test system.hamiltonian == H
        @test system.modes[1].coupling == V
        @test heom_dimensions(system) == [4,2]
        @test all(isfinite, dense(heom_generator(HEOMTTNSystem(H,[HEOMMode(V,0.8,0,0)],[2]))))
        for rate in (-0.1, Inf, NaN)
            @test_throws ArgumentError HEOMMode(V,rate,1,1)
        end
        for scale in (0, -1, Inf, NaN, 1im)
            @test_throws ArgumentError HEOMMode(V,0.8,1,1; scale)
        end
        @test_throws ArgumentError HEOMMode(V,0.8,Inf,1)
        @test_throws ArgumentError HEOMMode(V,0.8,1,NaN)
        @test_throws ArgumentError HEOMMode(V,0.8,BigFloat("1e400"),1)
        @test_throws ArgumentError HEOMMode(ComplexF64[0 im; im 0],0.8,1,1)
        @test_throws ArgumentError HEOMMode(fill(Inf,2,2),0.8,1,1)
        @test_throws DimensionMismatch HEOMMode(ones(2,3),0.8,1,1)
        @test_throws ArgumentError HEOMMode(zeros(0,0),0.8,1,1)
        valid = HEOMMode(V,0.8,1,1)
        @test_throws DimensionMismatch HEOMTTNSystem(H,[valid],Int[])
        @test_throws DimensionMismatch HEOMTTNSystem(ones(3,3),[valid],[2])
        for sizes in ([0],[-1],[2.5])
            @test_throws ArgumentError HEOMTTNSystem(H,[valid],sizes)
        end
        @test_throws ArgumentError HEOMTTNSystem(ComplexF64[0 im; im 0],[valid],[2])
        @test_throws ArgumentError HEOMTTNSystem(fill(NaN,2,2),[valid],[2])
    end
end
