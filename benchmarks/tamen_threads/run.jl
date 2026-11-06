# Run separate processes with --threads=1,2,4,8 and OPENBLAS_NUM_THREADS=1.
# Environment-only scaling is deliberately separate from converged solves.
using TTNDynamics, TTNSolver, LinearAlgebra, Random, Statistics

BLAS.set_num_threads(1)
const REPEATS = parse(Int,get(ENV,"TAMEN_THREAD_REPEATS","5"))
const OUTPUT = joinpath(@__DIR__,"data")
mkpath(OUTPUT)

function heom_problem(k)
    H = ComplexF64[0 0.4;0.4 0.1]
    V = ComplexF64[1 0;0 0]
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

function write_row(io,row)
    println(io,join(row,',')); flush(io)
end

function measure_solve(io,k)
    x,L,c = heom_problem(k)
    works = [TAMEnWorkspace(),TAMEnWorkspace()]
    # Pin the original residual path so the historical threading comparison
    # remains reproducible after local residual updates became the default.
    solve(threaded) = tamen_step(x,L,0.2;time_points=5,rtol=1e-8,residual_method=:global,
        maxrank=24,maxsweeps=16,conserved=c,threaded,workspace=works[1+threaded])
    # Compile both paths and check identical accepted work before timing.
    y,serial = solve(false)
    yp,parallel = solve(true)
    @assert serial.converged && parallel.converged
    @assert serial.sweeps == parallel.sweeps
    @assert serial.matvecs == parallel.matvecs
    discrepancy = ttn_distance(y,yp)/norm(y)
    @assert discrepancy < 1e-10
    println((;case="heom$k",threads=Threads.nthreads(),sweeps=serial.sweeps,
             rank=maximum(tree_ranks(y)),discrepancy)); flush(stdout)
    for rep in 1:REPEATS
        # Alternate order to reduce first/second-run bias. Collect outside timing;
        # keep GC enabled during every measurement.
        for threaded in (isodd(rep) ? (false,true) : (true,false))
            GC.gc()
            measured = @timed solve(threaded)
            endpoint,d = measured.value
            @assert d.converged && d.sweeps == serial.sweeps && d.matvecs == serial.matvecs
            @assert ttn_distance(endpoint,y)/norm(y) < 1e-10
            t = d.timings
            write_row(io,("heom$k",Threads.nthreads(),threaded,rep,
                length(x.tree.leaf_order),length(L.terms),maximum(tree_ranks(endpoint)),
                measured.time,measured.gctime,measured.bytes,t.environment_seconds,
                t.als_seconds,t.residual_seconds,t.enrichment_seconds,
                d.sweeps,d.matvecs,d.relative_residual_bound,discrepancy))
        end
    end
end

function contraction_problem(d,rank)
    rng = MersenneTwister(7301)
    tree = balanced_tree(d)
    leaves = Dict(node=>Matrix{ComplexF64}(I,4,4) for node in tree.leaf_order)
    ranks = Dict(node=>4 for node in tree.leaf_order)
    cores = Dict{Int,Array{ComplexF64,3}}()
    for node in tree.postorder
        haskey(leaves,node) && continue
        left,right = tree.children[node]
        a,b = ranks[left],ranks[right]
        r = node == tree.root ? 1 : min(rank,a*b)
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
    return A,st,b
end

function measure_environment(io,d,rank)
    A,x,b = contraction_problem(d,rank)
    contract(threaded) = TTNSolver._tamen_upward(A,x,b;threaded)
    serial,parallel = contract(false),contract(true)
    discrepancy = maximum(norm(serial[1][j][q]-parallel[1][j][q])
        for j in eachindex(serial[1]) for q in eachindex(serial[1][j]))
    discrepancy = max(discrepancy,maximum(norm.(serial[2].-parallel[2])))
    @assert discrepancy < 1e-12
    println((;case="environment$d",rank,threads=Threads.nthreads(),discrepancy)); flush(stdout)
    for rep in 1:REPEATS
        for threaded in (isodd(rep) ? (false,true) : (true,false))
            GC.gc()
            measured = @timed contract(threaded)
            write_row(io,("environment$d",Threads.nthreads(),threaded,rep,
                length(x.tree.leaf_order),length(A.terms),maximum(tree_ranks(x)),
                measured.time,measured.gctime,measured.bytes,discrepancy))
        end
    end
end

open(joinpath(OUTPUT,"solves_threads$(Threads.nthreads()).csv"),"w") do io
    println(io,"case,threads,threaded,repeat,spatial_leaves,terms,rank,seconds,gc_seconds,bytes,environment_seconds,als_seconds,residual_seconds,enrichment_seconds,sweeps,matvecs,relative_residual,serial_difference")
    for k in (2,8)
        measure_solve(io,k)
    end
end
open(joinpath(OUTPUT,"environments_threads$(Threads.nthreads()).csv"),"w") do io
    println(io,"case,threads,threaded,repeat,leaves_including_time,terms,rank,seconds,gc_seconds,bytes,serial_difference")
    for d in (16,64)
        measure_environment(io,d,32)
    end
end
