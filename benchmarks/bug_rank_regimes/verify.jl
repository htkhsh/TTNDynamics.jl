using Test
include("common.jl")
include("../../examples/heom/common/dense_heom_reference.jl")

@testset "Dense rank-profile helpers" begin
    Random.seed!(14)
    dims = [4,3,2,3]
    v = randn(ComplexF64,prod(dims))
    train = dense_to_train(v,dims;tolerance=1e-12)
    tree = dense_to_tree(v,dims,balanced_tree(length(dims));tolerance=1e-12)
    @test state_error(train,v) < 1e-12
    @test state_error(tree,v) < 1e-12
    @test cross_overlap(train,tree) ≈ dot(v,v)
    product = product_state(balanced_tree(4),[randn(ComplexF64,n) for n in dims])
    @test maximum(tt_ranks(dense_to_train(vec(dense(product)),dims))) == 1
    @test maximum(tree_ranks(dense_to_tree(vec(dense(product)),dims,product.tree))) == 1
    spec = (;name="test",m=2,k=2,strength=3.0,prep=0.25)
    p = rank_problem(spec); L = sparse_generator(p.operator)
    @test Matrix(L) ≈ dense_heom_reference_matrix(p.system)
    initial = vec(dense(p.initial))
    @test exact_action(L,initial,.25) ≈ exp(.25*Matrix(L))*initial atol=1e-11
    @test norm(transpose(vec(dense(p.conserved)))*L) < 1e-12
end
