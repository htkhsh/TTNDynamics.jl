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

@testset "HEOM trees with the system between bath halves" begin
    H=ComplexF64[0 0.4;0.4 0]
    V1=ComplexF64[1 0;0 0]; V2=I-V1
    modes=[HEOMMode(V,0.8+0.1im,0.2,0.3) for V in (V1,V1,V2,V2)]
    rho=ComplexF64[0.7 0.1im;-0.1im 0.3]
    for layout in (:fused,:split), grouping in (:balanced,:coupling)
        system=HEOMTTNSystem(H,modes,fill(2,4);system_layout=layout)
        tree=heom_tree(system;bath_grouping=grouping,system_position=:center)
        nsys=layout==:fused ? 1 : 2
        @test length(tree.leaf_order)==nsys+4
        # Root: (first bath half, (system branch, second bath half)).
        first_half,right=tree.children[tree.root]
        sysbranch,second_half=tree.children[right]
        leaves_below(n)=tree.children[n]==(0,0) ? [n] : vcat(leaves_below.(tree.children[n])...)
        sysleaves=tree.leaf_order[1:nsys]
        @test sort(leaves_below(sysbranch))==sort(sysleaves)
        @test sort(vcat(leaves_below(first_half),leaves_below(second_half)))==sort(tree.leaf_order[nsys+1:end])
        @test length(leaves_below(first_half))==2 && length(leaves_below(second_half))==2
        grouping==:coupling && @test tree.parent[tree.leaf_order[nsys+1]]==tree.parent[tree.leaf_order[nsys+2]]
        x=heom_initial_state(system,rho;tree)
        @test dense(x) ≈ dense(heom_initial_state(system,rho))
        @test TTNDynamics.root_density_matrix(x,system) ≈ rho
        @test dense(TTNDynamics.trace_observable(system;tree)) ≈ dense(TTNDynamics.trace_observable(system))
        op=heom_generator(system)
        result=heom_tamen_evolve(system,x,[0.0,0.2,0.4];time_points=8,rtol=1e-10,maxrank=16,maxsweeps=20)
        exact=exp(0.4dense(op))*vec(dense(x))
        @test norm(vec(dense(result.final_state))-exact) < 1e-7*norm(exact)
        @test abs(last(result.traces)-tr(rho)) < 1e-10
    end
    system=HEOMTTNSystem(H,modes[1:1],[2];system_layout=:split)
    # A single bath group falls back to the root layout.
    @test heom_tree(system;system_position=:center).children==heom_tree(system).children
    @test_throws ArgumentError heom_tree(system;system_position=:middle)
end

@testset "HEOM trees with chained bath halves" begin
    n=6
    H=ComplexF64[abs(i-j)==1 ? 0.4 : 0 for i in 1:n, j in 1:n]
    P(j)=diagm(ComplexF64.(1:n .== j))
    modes=[HEOMMode(P(j),0.8+0.1im,0.2,0.3) for j in 1:n]
    rho=zeros(ComplexF64,n,n); rho[1,1]=0.7; rho[2,2]=0.3; rho[1,2]=0.1im; rho[2,1]=-0.1im
    system=HEOMTTNSystem(H,modes,fill(2,n);system_layout=:split)
    tree=heom_tree(system;bath_grouping=:coupling,system_position=:center,bath_layout=:chain)
    balanced=heom_tree(system;bath_grouping=:coupling,system_position=:center)
    @test tree.children != balanced.children
    @test length(tree.leaf_order)==n+2
    # Chain nodes join one bath leaf to the rest of their half.
    first_half,right=tree.children[tree.root]
    leafs=Set(tree.leaf_order[3:end])
    node=first_half
    while !(node in leafs)
        a,b=tree.children[node]
        @test (a in leafs) || (b in leafs)
        node=a in leafs ? b : a
    end
    x=heom_initial_state(system,rho;tree)
    @test dense(x) ≈ dense(heom_initial_state(system,rho))
    op=heom_generator(system)
    result=heom_tamen_evolve(system,x,[0.0,0.2];time_points=8,rtol=1e-10,maxrank=64,maxsweeps=20)
    exact=exp(0.2dense(op))*vec(dense(x))
    @test norm(vec(dense(result.final_state))-exact) < 1e-7*norm(exact)
    @test_throws ArgumentError heom_tree(system;bath_layout=:chain)
    @test_throws ArgumentError heom_tree(system;system_position=:center,bath_layout=:ring)
end
