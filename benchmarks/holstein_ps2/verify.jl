using TTNSolver, LinearAlgebra, Random, Test
include("ps2.jl")
using .PS2Trial
const TS=TTNSolver
Random.seed!(712)

function fullstate(tree,dims)
    x=product_state(tree,[randn(ComplexF64,d) for d in dims])
    for _ in 1:8
        x=TS._tamen_add(x,product_state(tree,[randn(ComplexF64,d) for d in dims]))
    end
    orthogonalize!(x)
    return x
end
function randomop(dims)
    terms=[ProductTerm(0.1randn(ComplexF64),Dict(i=>randn(ComplexF64,d,d))) for (i,d) in enumerate(dims)]
    for i in 1:length(dims)-1
        push!(terms,ProductTerm(0.08+0.03im,Dict(i=>randn(ComplexF64,dims[i],dims[i]),
            i+1=>randn(ComplexF64,dims[i+1],dims[i+1]))))
    end
    SumOfProducts(dims,terms)
end

@testset "PS2 numerical trial" begin
    @testset "nonstandard topology and zero state" begin
        tree=BinaryTree([(2,3),(4,5),(6,7),(0,0),(0,0),(0,0),(0,0)];root=1,leaf_order=[7,4,6,5])
        x=fullstate(tree,fill(2,4));op=randomop(fill(2,4))
        y,_=ps2_step(x,op,0.02;atol=0.,maxrank=16,krylov_tol=1e-13)
        @test norm(vec(dense(y))-exp(0.02dense(op))*vec(dense(x)))/norm(x)<1e-10
        z=product_state(tree,[zeros(2) for _ in 1:4])
        y,info=ps2_step(z,op,0.1;maxrank=16)
        @test norm(y)==0
        @test all(isfinite,dense(y))
    end
    @testset "dense full-space and ownership" begin
        for d in ([3],[2,3],[2,2,2,2])
            tree=balanced_tree(length(d));x=fullstate(tree,d);op=randomop(d)
            before=dense(x)
            y,info=ps2_step(x,op,0.03;atol=0.,maxrank=32,krylov_tol=1e-13)
            @test norm(vec(dense(y))-exp(0.03dense(op))*vec(before))/norm(before)<1e-10
            @test dense(x)==before
            @test info.max_local_dimension>0
            @test info.local_solves == (length(d)==1 ? 1 : 4*(length(tree.children)-1)-2)
        end
    end
    @testset "protected functional and rank growth" begin
        # Column-sum-zero non-Hermitian generator, including a long-range term.
        d=fill(3,4);tree=balanced_tree(4)
        c=product_state(tree,[ones(3) for _ in d])
        x=product_state(tree,[ComplexF64[1,0,0] for _ in d])
        J=ComplexF64[-1 0 0;1 -2 0;0 2 0]
        V=diagm(ComplexF64[1,2,3])
        op=SumOfProducts(d,[ProductTerm(1,Dict(1=>J)),ProductTerm(0.4,Dict(1=>V,4=>J)),
            ProductTerm(0.2,Dict(2=>J,3=>V)),ProductTerm(0.3,Dict(3=>J))])
        exact=exp(0.1dense(op))*vec(dense(x))
        for h in (0.05,0.025)
            y=copy(x)
            for _ in 1:round(Int,0.1/h)
                y,info=ps2_step(y,op,h;atol=1e-9,maxrank=16,conserved=c,krylov_tol=1e-13)
                @test abs(dot(c,y)-dot(c,x))<1e-10
                @test info.truncation_met
            end
            @test norm(vec(dense(y))-exact)<1e-4
            @test maximum(tree_ranks(y))>1
        end
        @test_throws ArgumentError ps2_step(x,op,-0.1)
        @test_throws ArgumentError ps2_step(x,op,0.1;maxrank=0)
    end
    @testset "active truncation and Arnoldi failure" begin
        tree=balanced_tree(4);x=fullstate(tree,fill(2,4))
        c=product_state(tree,[ones(2) for _ in 1:4])
        J=ComplexF64[-1 0;1 0]
        op=SumOfProducts(fill(2,4),[ProductTerm(1,Dict(i=>J)) for i in 1:4])
        before=dense(x)
        y,info=ps2_step(x,op,0.1;conserved=c,maxrank=2,atol=0.)
        @test abs(dot(c,y)-dot(c,x))<1e-10
        @test maximum(tree_ranks(y))<=2
        @test info.local_cut_sum>1e-8
        @test !info.truncation_met
        @test dense(x)==before
        single=product_state(balanced_tree(1),[ComplexF64[1,2,3]])
        op=randomop([3]);before=dense(single)
        @test_throws LocalSolveFailure ps2_step(single,op,1.;maxrank=3,krylovdim=1,krylov_maxiter=1)
        @test dense(single)==before
    end
    @testset "eager Arnoldi keeps accuracy with fewer actions" begin
        tree=balanced_tree(2);x=fullstate(tree,[6,6]);op=randomop([6,6])
        a,ia=ps2_step(x,op,0.01;maxrank=36,atol=1e-12,krylov_eager=false)
        b,ib=ps2_step(x,op,0.01;maxrank=36,atol=1e-12,krylov_eager=true)
        target=exp(0.01dense(op))*vec(dense(x))
        @test norm(vec(dense(b))-target)/norm(target)<1e-10
        @test norm(vec(dense(a))-vec(dense(b)))/norm(target)<1e-10
        @test ib.matvecs<ia.matvecs
    end
end
