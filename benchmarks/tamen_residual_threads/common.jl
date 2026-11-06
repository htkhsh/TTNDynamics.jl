using TTNSolver
push!(LOAD_PATH,normpath(joinpath(@__DIR__,"../..")))
using TTNDynamics, LinearAlgebra, Statistics, Serialization, Random
BLAS.set_num_threads(1)

function certification_problem()
    k=8
    H,V=ComplexF64[0 0.4;0.4 0.1],ComplexF64[1 0;0 0]
    modes=HEOMMode[]
    for j in 1:k÷2
        omega=0.6+0.4j/(k÷2)
        push!(modes,HEOMMode(V,0.3+omega*im,(0.12-0.02im)*2/k,
                           (0.04-0.01im)*2/k;scale=0.4/sqrt(k/2)),
                    HEOMMode(V,0.3-omega*im,(0.04+0.01im)*2/k,
                           (0.12+0.02im)*2/k;scale=0.4/sqrt(k/2)))
    end
    system=HEOMTTNSystem(H,modes,fill(3,k);system_layout=:split)
    return system,heom_initial_state(system,ComplexF64[1 0;0 0])
end

function random_state(tree,rank,rng)
    leaves=Dict(n=>randn(rng,ComplexF64,4,rank) for n in tree.leaf_order)
    cores=Dict(n=>randn(rng,ComplexF64,n==tree.root ? 1 : rank,rank,rank)
        for n in tree.postorder if !TTNSolver._isleaf(tree,n))
    x=orthogonalize!(TTNTensor(tree,leaves,cores))
    x.cores[tree.root]./=norm(x)
    return x
end

function controlled_problem()
    rng=MersenneTwister(722)
    tree=balanced_tree(32)
    x,b=random_state(tree,8,rng),random_state(tree,8,rng)
    terms=[ProductTerm(0.1im,Dict(j=>randn(rng,ComplexF64,4,4),
                                 33-j=>randn(rng,ComplexF64,4,4))) for j in 1:12]
    return SumOfProducts(fill(4,32),terms),x,b,1e-8,24
end

function write_row(io,row,firstrow)
    firstrow && println(io,join(keys(row),','))
    println(io,join(values(row),',')); flush(io)
end
