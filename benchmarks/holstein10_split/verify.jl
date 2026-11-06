using Test
include("common.jl")

function train_vector(x)
    return vec(permutedims(reshape(tt_full(x),Tuple(reverse(tt_dims(x)))),reverse(1:length(tt_dims(x)))))
end

@testset "Holstein example to split TTN" begin
    p = example_problem(;sites=2,local_size=2)
    @test tt_dims(p.train) == leafdims(p.tree) == [2,2,2,2,2,2]
    @test train_vector(p.train) ≈ vec(dense(p.tree))
    @test generator_agreement(p) < 1e-10
    @test train_vector(p.problem.liouvillian*p.train) ≈ dense(p.operator)*vec(dense(p.tree)) atol=1e-11
    @test any(abs(m.backward-conj(m.forward))>1e-4 for m in p.system.modes)
    @test [m.scale for m in p.system.modes] ≈ sqrt.(p.problem.system.noise.abs_coeff)
    exact = exp(0.5*dense(p.operator))*vec(dense(p.tree))
    actual,_ = train_step(p,p.train,0.5;scheme="cheb",points=8,tolerance=1e-9,sweeps=30,local_iters=100)
    @test norm(train_vector(actual)-exact)/norm(exact) < 1e-7
    @test cross_overlap(p.train,p.tree) ≈ dot(p.train,p.train)
    @test TTNDynamics.root_density_matrix(p.tree,p.system) ≈ TTDynamics.root_density_matrix(p.train,p.problem.system)
end
