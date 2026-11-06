using Test
include("common.jl")

tt_in_tree_order(x) = vec(permutedims(reshape(tt_full(x),Tuple(reverse(tt_dims(x)))),
                                      reverse(1:length(tt_dims(x)))))

@testset "TT / TTN comparison conventions" begin
    Random.seed!(71)
    dims = [2,3,2,3]
    tree = balanced_tree(4)
    x = product_state(tree,[randn(ComplexF64,n) for n in dims])
    y = product_to_tt(x)
    @test tt_in_tree_order(y) ≈ vec(dense(x))
    terms = [ProductTerm(randn(ComplexF64),Dict(j=>randn(ComplexF64,dims[j],dims[j]) for j in js))
             for js in ([],[1],[2,4],[1,3])]
    op = SumOfProducts(dims,terms); mpo = sop_to_mpo(op)
    evolved = op*x
    @test tt_in_tree_order(mpo*y) ≈ vec(dense(evolved)) atol=1e-11
    train = tt_rand(dims;r=3,T=ComplexF64)
    @test cross_overlap(train,evolved) ≈ dot(tt_in_tree_order(train),vec(dense(evolved)))
    @test relative_error(evolved,train) ≈ norm(vec(dense(evolved))-tt_in_tree_order(train))/norm(train)
    @test cross_overlap(y,x) ≈ dot(y,y)
    @test tt_distance(y,2y) ≈ norm(y)
    badtree = BinaryTree(tree.children;root=tree.root,leaf_order=[1,3,2,4])
    bad = product_state(badtree,[ones(ComplexF64,n) for n in dims])
    @test_throws ArgumentError cross_overlap(y,bad)
end

@testset "Independent reference and tAMEn against dense evolution" begin
    for heom in (true,false)
        H = ComplexF64[0 .3+.1im;.3-.1im .2]
        V = ComplexF64[1 0;0 -.2]
        rho = ComplexF64[.9 0;0 .1]
        if heom
            system = HEOMTTNSystem(H,[HEOMMode(V,.8,.08+.01im,.08-.01im),
                                       HEOMMode(V,1.2,.06+.01im,.06-.01im)],[3,2])
            x = heom_initial_state(system,rho); op = heom_generator(system)
        else
            system = TFDTTNSystem(H,[TFDMode(V,.8,.2),TFDMode(V,-.8,.1)],[3,2])
            x = tfd_initial_state(system,rho)
            op = TTNDynamics._tfd_generator(tfd_hamiltonian(system))
        end
        A = sop_to_mpo(op); y = product_to_tt(x)
        ref = taylor_reference(A,y;steps=5)
        exact = exp(FINAL_TIME*dense(op))*vec(dense(x))
        @test norm(tt_in_tree_order(ref)-exact)/norm(exact) < 1e-10
        conserved = heom ? product_to_tt(trace_observable(system;tree=x.tree)) : nothing
        approx = run_tamen(A,y;tolerance=1e-7,conserved).state
        @test relative_error(approx,ref) < 1e-6
        @test tt_density(y,heom) ≈ rho
        @test norm(tt_density(approx,heom)-tt_density(ref,heom)) < 1e-6
        exact_tensor = reshape(exact,4,3,2)
        if heom
            density_exact = reshape(exact_tensor[:,1,1],2,2)
        else
            psi = reshape(exact_tensor,2,2,:)
            density_exact = sum(psi[:,a,b]*psi[:,a,b]' for a in 1:2, b in axes(psi,3))
        end
        @test norm(tt_density(ref,heom)-density_exact) < 1e-10
        @test tt_in_tree_order(y) ≈ vec(dense(x)) # solver did not mutate input
    end
end
