# Run sequential processes with --threads=1,2,4,8; OPENBLAS_NUM_THREADS=1.
# Whole HEOM convergence and synthetic one-sweep scaling are separate cases.
using TTNDynamics, TTNSolver, LinearAlgebra, Random, Statistics
BLAS.set_num_threads(1)
const REPEATS = parse(Int,get(ENV,"TAMEN_ETTER_REPEATS","3"))
const CASES = split(get(ENV,"TAMEN_ETTER_CASES","heom8,sweep16,sweep64"),',')
const OUTPUT = joinpath(@__DIR__,"data")

function heom_problem(k)
    H,V = ComplexF64[0 0.4;0.4 0.1],ComplexF64[1 0;0 0]
    modes = HEOMMode[]
    for j in 1:k÷2
        omega = 0.6+0.4j/(k÷2)
        push!(modes,HEOMMode(V,0.3+omega*im,(0.12-0.02im)*2/k,
                           (0.04-0.01im)*2/k;scale=0.4/sqrt(k/2)),
                    HEOMMode(V,0.3-omega*im,(0.04+0.01im)*2/k,
                           (0.12+0.02im)*2/k;scale=0.4/sqrt(k/2)))
    end
    system = HEOMTTNSystem(H,modes,fill(3,k);system_layout=:split)
    x = heom_initial_state(system,ComplexF64[1 0;0 0])
    return x,heom_generator(system),trace_observable(system;tree=x.tree)
end

function sweep_problem(d,rank=24)
    rng = MersenneTwister(7301)
    tree = balanced_tree(d)
    leaves = Dict(node=>Matrix{ComplexF64}(I,4,4) for node in tree.leaf_order)
    ranks = Dict(node=>4 for node in tree.leaf_order)
    cores = Dict{Int,Array{ComplexF64,3}}()
    for node in tree.postorder
        haskey(leaves,node) && continue
        left,right = tree.children[node]
        a,b = ranks[left],ranks[right]
        r = node==tree.root ? 1 : min(rank,a*b)
        Q = Matrix(qr(randn(rng,ComplexF64,a*b,r)).Q)[:,1:r]
        cores[node] = reshape(Matrix(transpose(Q)),r,a,b)
        ranks[node] = r
    end
    x = TTNTensor(tree,leaves,cores)
    terms = ProductTerm[]
    for j in 1:d
        push!(terms,ProductTerm(-1/d,Dict(j=>diagm(ComplexF64.(0:3)))))
        push!(terms,ProductTerm(0.1/d,Dict(j=>randn(rng,ComplexF64,4,4),
             mod1(j+1,d)=>randn(rng,ComplexF64,4,4))))
    end
    L = SumOfProducts(fill(4,d),terms)
    grid = TTNSolver._tamen_grid(5,0.2,:cheb)
    A = TTNSolver._spacetime_operator(L,grid)
    st = TTNSolver._time_lift(x,ones(5);full_time_basis=true)
    b = TTNSolver._time_lift(x,grid.d0)
    return A,st,b,length(tree.children)+1
end

function record(io,row)
    println(io,join(values(row),',')); flush(io)
end

function heom_trials(io)
    x,L,c = heom_problem(8)
    configurations = ((:serial,false),(:etter,false),(:etter,true))
    works = Dict(config=>TAMEnWorkspace() for config in configurations)
    solve(config) = tamen_step(x,L,0.2;time_points=5,rtol=1e-8,maxrank=32,
        maxsweeps=20,conserved=c,als_sweep=config[1],threaded=config[2],workspace=works[config])
    baselines = Dict(config=>solve(config) for config in configurations) # warm all paths
    ref = baselines[(:serial,false)][1]
    etter_ref = baselines[(:etter,false)][1]
    for rep in 1:REPEATS, config in (isodd(rep) ? configurations : reverse(configurations))
        GC.gc()
        t = @timed solve(config)
        y,d = t.value
        delta = ttn_distance(y,ref)/norm(ref)
        same_algorithm_delta = config[1]==:etter ? ttn_distance(y,etter_ref)/norm(etter_ref) : 0.0
        trace_error = maximum(abs.(d.invariant_values.-1))
        @assert d.converged && d.relative_residual_bound<=1e-8
        @assert delta<2e-7 && same_algorithm_delta<1e-10 && trace_error<1e-10
        record(io,(;case="heom8",threads=Threads.nthreads(),als_sweep=config[1],threaded=config[2],
            repeat=rep,seconds=t.time,gc_seconds=t.gctime,bytes=t.bytes,
            sweep_seconds=d.timings.sweep_seconds,als_seconds=d.timings.als_seconds,
            residual_seconds=d.timings.residual_seconds,sweeps=d.sweeps,matvecs=d.matvecs,
            rank=maximum(tree_ranks(y)),spawned_tasks=d.als_parallel_tasks,
            relative_residual=d.relative_residual_bound,trace_error,reference_difference=delta,same_algorithm_delta))
        println((;case="heom8",config,repeat=rep,seconds=t.time,sweeps=d.sweeps,delta)); flush(stdout)
    end
end

function sweep_trials(io,d)
    A,x,b,time_leaf = sweep_problem(d)
    works = Dict(flag=>TAMEnWorkspace() for flag in (false,true))
    function solve(flag)
        y = copy(x)
        stats = TTNSolver._tamen_etter_sweep!(y,A,b,time_leaf;
            tol=1e-9,krylovdim=40,krylov_maxiter=100,workspace=works[flag],threaded=flag)
        return y,stats
    end
    ref,s = solve(false)
    p,ps = solve(true)
    @assert ttn_distance(ref,p)/norm(ref)<1e-10 && s.matvecs==ps.matvecs
    for rep in 1:REPEATS, flag in (isodd(rep) ? (false,true) : (true,false))
        GC.gc()
        t = @timed solve(flag)
        y,stats = t.value
        delta = ttn_distance(y,ref)/norm(ref)
        @assert delta<1e-10 && stats.matvecs==s.matvecs
        @assert y.leaves[time_leaf]==x.leaves[time_leaf]
        record(io,(;case="sweep$d",threads=Threads.nthreads(),als_sweep=:etter,threaded=flag,
            repeat=rep,seconds=t.time,gc_seconds=t.gctime,bytes=t.bytes,
            sweep_seconds=t.time,als_seconds=t.time,residual_seconds=0.0,
            sweeps=1,matvecs=stats.matvecs,rank=maximum(tree_ranks(y)),
            spawned_tasks=stats.spawned_tasks,relative_residual=NaN,trace_error=NaN,
            reference_difference=delta,same_algorithm_delta=delta))
        println((;case="sweep$d",flag,repeat=rep,seconds=t.time,delta)); flush(stdout)
    end
end

function main()
    mkpath(OUTPUT)
    open(joinpath(OUTPUT,"runs_threads$(Threads.nthreads()).csv"),"w") do io
        println(io,"case,threads,als_sweep,threaded,repeat,seconds,gc_seconds,bytes,sweep_seconds,als_seconds,residual_seconds,sweeps,matvecs,rank,spawned_tasks,relative_residual,trace_error,reference_difference,same_algorithm_delta")
        "heom8" in CASES && heom_trials(io)
        for d in (16,64)
            "sweep$d" in CASES && sweep_trials(io,d)
        end
    end
    open(joinpath(OUTPUT,"environment_threads$(Threads.nthreads()).txt"),"w") do io
        println(io,"Julia ",VERSION,"; CPU ",Sys.CPU_NAME,"; Julia threads ",Threads.nthreads(),
            "; BLAS threads ",BLAS.get_num_threads(),"; repeats ",REPEATS)
        println(io,"Warmed paths, alternating order, GC enabled; explicit GC and setup excluded from timing.")
        println(io,"HEOM: split M=2, 8 modes, n=3, h=0.2, nt=5, rtol=1e-8, rank<=32, local residual.")
        println(io,"Synthetic: 16/64 spatial leaves, n=4, rank<=24, 2d+1 space-time terms, h=0.2, nt=5.")
        println(io,"Synthetic measures ONE sweep with local tolerance1e-9, not a converged propagation.")
    end
end
main()
