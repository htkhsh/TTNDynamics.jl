using Test, LinearAlgebra, Random, TTNSolver
include("second_order.jl")
using .SecondOrderBUG

@testset "Second-order TTN BUG trial" begin
    rng=MersenneTwister(912)
    tree=balanced_tree(4)
    x=TTNTensor(tree,Dict(n=>randn(rng,ComplexF64,3,2) for n in tree.leaf_order),
        Dict(n=>randn(rng,ComplexF64,n==tree.root ? 1 : 2,2,2)
             for n in tree.postorder if !TTNSolver._isleaf(tree,n)))
    x.cores[tree.root]./=norm(x)
    # This generator is tangent to the rank manifold. Arbitrary entangling
    # generators have a normal-component error in the paper's bound, which
    # must not be mistaken for a failure of temporal order.
    terms=[ProductTerm(-0.08im,Dict(j=>randn(rng,ComplexF64,3,3))) for j in 1:4]
    op=SumOfProducts(fill(3,4),terms)
    before=dense(x)
    prep=SecondOrderBUG.prepare(x,op,nothing,64)
    @test dense(prep.old) ≈ before atol=1e-12
    @test dense(x)==before
    for method in (:augmented,:parallel)
        errors=Float64[]
        for h in (0.1,0.05,0.025)
            y,d=second_order_bug_step(x,op,h;method,atol=0.0,maxrank=64,threaded=false)
            push!(errors,norm(vec(dense(y))-exp(h*dense(op))*vec(before)))
        end
        println((method,local_errors=errors)); flush(stdout)
        @test errors[2]<errors[1]/5
        @test errors[3]<errors[2]/5
        global_errors=Float64[]
        for h in (0.1,0.05,0.025)
            y=copy(x)
            for _ in 1:round(Int,0.2/h)
                y,_=second_order_bug_step(y,op,h;method,atol=1e-13,maxrank=2,threaded=false)
            end
            push!(global_errors,norm(vec(dense(y))-exp(0.2*dense(op))*vec(before)))
        end
        println((method,global_errors)); flush(stdout)
        @test global_errors[2]<global_errors[1]/3
        @test global_errors[3]<global_errors[2]/3
        serial,_=second_order_bug_step(x,op,0.05;method,atol=1e-10,threaded=false)
        threaded,_=second_order_bug_step(x,op,0.05;method,atol=1e-10,threaded=true)
        @test ttn_distance(serial,threaded)<1e-11
        @test_throws ArgumentError second_order_bug_step(x,op,-0.1;method)
    end
    # Complex non-Hermitian generator annihilates a known left functional.
    c=product_state(tree,[ComplexF64[1,0,0] for _ in 1:4])
    A=randn(rng,ComplexF64,3,3); A[1,:].=0
    invariant_op=SumOfProducts(fill(3,4),[ProductTerm(0.2,Dict(1=>A,3=>A))])
    for method in (:augmented,:parallel)
        y,d=second_order_bug_step(x,invariant_op,0.1;method,atol=1e-7,maxrank=8,
                                  conserved=c,correct_invariant=false)
        @test dot(c,y) ≈ dot(c,x) atol=1e-11
        @test maximum(tree_ranks(y))<=8
        @test d.correction_norm==0
    end
end

@testset "Single leaf and rank-deficient complements" begin
    one=product_state(balanced_tree(1),[ComplexF64[1,im]])
    A=ComplexF64[-0.1 0.3im;0.2 -0.4]
    op=SumOfProducts([2],[ProductTerm(1,Dict(1=>A))])
    for method in (:augmented,:parallel)
        y,_=second_order_bug_step(one,op,0.1;method,correct_invariant=false)
        @test vec(dense(y)) ≈ exp(0.1A)*vec(dense(one)) atol=1e-12
    end
    tree=balanced_tree(4)
    vacuum=product_state(tree,[ComplexF64[1,0] for _ in 1:4])
    excited=product_state(tree,[ComplexF64[0,1] for _ in 1:4])
    redundant=enrich_basis(vacuum,excited)
    # Added basis columns have zero amplitudes, so complementary matrices
    # are deficient even though their stored bond dimensions are two.
    op=SumOfProducts(fill(2,4),[ProductTerm(-im,Dict(1=>ComplexF64[0 1;1 0]))])
    prep=SecondOrderBUG.prepare(redundant,op,nothing,8)
    @test dense(prep.old) ≈ dense(vacuum) atol=1e-12
    for method in (:augmented,:parallel)
        y,_=second_order_bug_step(redundant,op,0.1;method,atol=1e-12)
        @test vec(dense(y)) ≈ exp(0.1*dense(op))*vec(dense(vacuum)) atol=1e-11
        zero=copy(redundant); zero.cores[tree.root].*=0
        y,_=second_order_bug_step(zero,op,0.1;method,atol=1e-12)
        @test norm(y)==0
    end
end
