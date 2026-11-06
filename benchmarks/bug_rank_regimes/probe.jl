include("common.jl")

function probe(spec)
    name = spec.name
    println("Preparing $name $spec"); flush(stdout)
    p = rank_problem(spec); dims = leafdims(p.initial)
    interval = get(spec,:interval,INTERVAL)
    L = sparse_generator(p.operator)
    v0 = vec(dense(p.initial))
    @time initial = exact_action(L,v0,spec.prep)
    reference = exact_action(L,initial,interval)
    check = exact_action(L,initial,interval;tol=1e-13,dimension=70)
    convergence = norm(reference-check)/norm(check)
    @assert convergence < 1e-9
    train = dense_to_train(initial,dims;tolerance=1e-7)
    tree = dense_to_tree(initial,dims,p.initial.tree;tolerance=1e-7)
    train_error,tree_error = state_error(train,initial),state_error(tree,initial)
    @assert max(train_error,tree_error) < 2e-7
    A = sop_to_mpo(p.operator)
    println("$name TT rank=",maximum(tt_ranks(train))," TTN rank=",maximum(tree_ranks(tree)),
            " MPO rank=",maximum(tt_ranks(A))," input errors ",(train_error,tree_error)); flush(stdout)
    dir = joinpath(@__DIR__,"data"); mkpath(dir)
    serialize(joinpath(dir,"$name.jls"),(;spec,train,tree,reference,A,convergence))
    open(joinpath(dir,"profile_$name.csv"),"w") do io
        println(io,"case,m,modes,strength,prep,interval,dimension,tt_rank,ttn_rank,mpo_rank,tt_storage,ttn_storage,tt_input_error,ttn_input_error,reference_convergence")
        println(io,join((name,spec.m,spec.k,spec.strength,spec.prep,interval,length(initial),
            maximum(tt_ranks(train)),maximum(tree_ranks(tree)),maximum(tt_ranks(A)),storage(train),storage(tree),
            train_error,tree_error,convergence),','))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    for spec in RANK_CASES
        (isempty(ARGS) || spec.name in ARGS) && probe(spec)
    end
end
