@testset "HEOM fused and split layouts" begin
    H = ComplexF64[0.2 0.3+0.1im;0.3-0.1im -0.4]
    V = ComplexF64[1 0.2im;-0.2im -0.5]
    mode = HEOMMode(V,0.8,0.7+0.2im,0.7-0.2im;scale=0.9)
    rho = ComplexF64[2 im;0 -1] # Neither Hermitian nor normalized.
    @test HEOMTTNSystem(H,[mode],[2]).system_layout == :fused
    for invalid in (:other,"split",nothing,missing)
        @test_throws ArgumentError HEOMTTNSystem(H,[mode],[2];system_layout=invalid)
    end
    for sizes in (Int[],[1],[3],[2,3])
        modes = length(sizes) == 2 ?
            [mode,HEOMMode(V,0.4+1.2im,0.3+0.1im,0.2+0.05im;scale=0.7)] :
            fill(mode,length(sizes))
        fused = HEOMTTNSystem(H,modes,sizes;system_layout=:fused)
        split = HEOMTTNSystem(H,modes,sizes;system_layout=:split)
        @test heom_dimensions(split) == [2,2,sizes...]
        @test dense(heom_generator(split)) ≈ dense_heom_reference_matrix(split) atol=1e-12
        @test dense(heom_generator(split)) ≈ dense(heom_generator(fused)) atol=1e-12
        initial = heom_initial_state(split,rho)
        @test vec(dense(initial)) == vec(dense(heom_initial_state(fused,rho)))
        @test root_density_matrix(initial,split) == rho
        c = trace_observable(split)
        @test dot(c,initial) ≈ tr(rho) atol=1e-13
        @test tree_ranks(c)[c.tree.leaf_order[1:2]] == [2,2]
        @test norm(vec(dense(c))'*dense(heom_generator(split))) < 1e-12
        arbitrary = random_heom_state(split)
        expected = reshape(vec(dense(arbitrary)),2,2,prod(sizes))[:,:,1]
        @test root_density_matrix(arbitrary,split) ≈ expected atol=1e-12
        @test dot(c,arbitrary) ≈ tr(expected) atol=1e-12
        @test vec(dense(heom_generator(split)*arbitrary)) ≈
            dense_heom_reference_matrix(split)*vec(dense(arbitrary)) atol=2e-12
        @test_throws DimensionMismatch root_density_matrix(heom_initial_state(fused,rho),split)
    end

    @testset "Split custom trees and ownership" begin
        system = HEOMTTNSystem(H,[mode,mode],[2,3];system_layout=:split)
        children = [(3,4),(0,0),(2,5),(6,7),(0,0),(0,0),(0,0)]
        # Reversed sibling order, then system leaves separated by bath leaves.
        for order in ([5,2,6,7],[5,6,2,7])
            tree = BinaryTree(children;root=1,leaf_order=order)
            input = copy(rho)
            x = heom_initial_state(system,input;tree)
            input .= 0
            reference = zeros(ComplexF64,2,2,2,3)
            reference[:,:,1,1] = rho
            @test x.tree === tree
            @test dense(x) == reference
            @test root_density_matrix(x,system) == rho
            arbitrary = random_heom_state(system,tree)
            @test root_density_matrix(arbitrary,system) ≈ dense(arbitrary)[:,:,1,1] atol=1e-12
            c = trace_observable(system;tree)
            @test dot(c,arbitrary) ≈ tr(dense(arbitrary)[:,:,1,1]) atol=1e-12
            @test abs(dot(c,heom_generator(system),arbitrary)) < 1e-12
            coarse = heom_evolve(system,x,0:0.002:0.02;atol=1e-12,correct_trace=false)
            evolved = heom_evolve(system,x,0:0.001:0.02;atol=1e-12,correct_trace=false)
            expected = exp(0.02*dense_heom_reference_matrix(system))*vec(reference)
            # Reduced BUG equations depend on the tree. Test convergence to
            # the independent reference rather than identical finite-step errors.
            @test norm(vec(dense(evolved.final_state))-expected) <
                0.6norm(vec(dense(coarse.final_state))-expected)
            @test maximum(abs.(evolved.traces.-tr(rho))) < 1e-11
            @test root_density_matrix(x,system) == rho
        end
        @test_throws DimensionMismatch heom_initial_state(system,rho;tree=balanced_tree(3))
        @test_throws ArgumentError heom_initial_state(system,fill(NaN,2,2))
        zero = heom_initial_state(system,zeros(2,2))
        @test iszero(dense(zero))
    end

    @testset "Split fixed and adaptive evolution" begin
        system = HEOMTTNSystem(H,[mode],[2];system_layout=:split)
        initial = heom_initial_state(system,ComplexF64[0.7 0.1im;-0.1im 0.3])
        L = dense_heom_reference_matrix(system)
        target = exp(0.2L)*vec(dense(initial))
        coarse = heom_evolve(system,initial,0:0.02:0.2;atol=1e-12,correct_trace=false)
        fine = heom_evolve(system,initial,0:0.01:0.2;atol=1e-12,correct_trace=false)
        @test norm(vec(dense(fine.final_state))-target) < 0.7norm(vec(dense(coarse.final_state))-target)
        @test maximum(abs.(fine.traces.-1)) < 1e-11
        @test all(d -> d.truncation.tolerance_met,fine.diagnostics)
        adaptive = heom_evolve(system,initial;tspan=(0.,0.2),dt=0.05,saveat=0:0.1:0.2,
            atol=1e-9,rtol=1e-6,correct_trace=false,record_timing=true)
        @test norm(vec(dense(adaptive.final_state))-target) < 1e-4
        @test maximum(abs.(adaptive.traces.-1)) < 1e-11
        @test adaptive.timings.total > 0
        @test_throws ArgumentError heom_evolve(system,initial,[0.,0.1];maxrank=1)

        bare = HEOMTTNSystem(H,HEOMMode[],Int[];system_layout=:split)
        x = heom_initial_state(bare,rho)
        result = heom_evolve(bare,x,[0.,0.1];atol=0,correct_trace=false)
        U = exp(-0.1im*H)
        @test last(result.densities) ≈ U*rho*U' atol=1e-12
        @test root_density_matrix(x,bare) == rho
        scalar = HEOMTTNSystem(ones(1,1),HEOMMode[],Int[];system_layout=:split)
        s = heom_initial_state(scalar,fill(2im,1,1))
        @test last(heom_evolve(scalar,s,[0.,0.1]).densities) ≈ fill(2im,1,1)
    end
end
