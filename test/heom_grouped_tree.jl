using Test, LinearAlgebra, TTNSolver, TTNDynamics

@testset "HEOM trees grouped by coupling" begin
    H=ComplexF64[0 0.4;0.4 0]
    V1=ComplexF64[1 0;0 0]; V2=I-V1
    # Interleaved modes test preservation of physical order when tree order
    # and physical axis order no longer agree.
    modes=[HEOMMode(V,0.8+0.1im,0.2,0.3) for V in (V1,V2,V1,V2)]
    rho=ComplexF64[0.7 0.1im;-0.1im 0.3]
    for layout in (:fused,:split)
        system=HEOMTTNSystem(H,modes,fill(2,4);system_layout=layout)
        tree=heom_tree(system;bath_grouping=:coupling)
        offset=layout==:fused ? 1 : 2
        a,b,c,d=tree.leaf_order[offset+1:end]
        @test tree.parent[a]==tree.parent[c]
        @test tree.parent[b]==tree.parent[d]
        x=heom_initial_state(system,rho;tree)
        baseline=heom_initial_state(system,rho)
        @test dense(x) ≈ dense(baseline)
        @test TTNDynamics.root_density_matrix(x,system) ≈ rho
        @test dense(TTNDynamics.trace_observable(system;tree)) ≈
              dense(TTNDynamics.trace_observable(system))
        op=heom_generator(system)
        @test vec(dense(apply(op,x))) ≈ dense(op)*vec(dense(x)) atol=1e-12
        # An initial product between system and bath can hide coupling
        # directions from all reduced BUG equations. Enrich for this dense
        # integration check; it preserves the input tensor exactly.
        x=enrich_basis(x,apply(op,x))
        @test dense(x) ≈ dense(baseline) atol=1e-12
        conserved=TTNDynamics.trace_observable(system;tree)
        for step in (augmented_bug_step,parallel_bug_step)
            y,_=step(x,op,1e-3;atol=1e-10,truncation=:weighted,conserved,
                     correct_invariant=false,threaded=false)
            @test norm(vec(dense(y))-exp(1e-3*dense(op))*vec(dense(x)))<1e-5
            @test dot(conserved,y) ≈ dot(conserved,x) atol=1e-12
        end
        @test heom_tree(system;bath_grouping=:balanced).children==heom_tree(system).children
        @test_throws ArgumentError heom_tree(system;bath_grouping=:invalid)
    end
    # A full Holstein bath has two consecutive components per site.
    modes=[HEOMMode(diagm(ComplexF64.(1:10 .== site)),0.8,0.2,0.3) for site in 1:10 for _ in 1:2]
    system=HEOMTTNSystem(zeros(10,10),modes,fill(5,20);system_layout=:split)
    tree=heom_tree(system;bath_grouping=:coupling)
    @test length(tree.leaf_order)==22
    for j in 3:2:22
        @test tree.parent[tree.leaf_order[j]]==tree.parent[tree.leaf_order[j+1]]
    end
    empty_system=HEOMTTNSystem(H,HEOMMode[],Int[];system_layout=:split)
    @test heom_tree(empty_system;bath_grouping=:coupling).children==heom_tree(empty_system).children
end
