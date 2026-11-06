using TTNDynamics, TTNSolver, LinearAlgebra, Dates, SHA

const CASES = ("heom4_8_fused","heom4_8_split","tfd4_3")
const STEP = 0.05
const NSTEPS = 5
const OPTIONS = (;atol=1e-10,maxrank=64,correct_invariant=false)

function scaling_problem(name)
    if startswith(name,"heom")
        parts = split(name,'_')
        m,k = parse(Int,parts[1][5:end]),parse(Int,parts[2])
        H = diagm(-1=>fill(0.25-0.07im,m-1),0=>ComplexF64.(range(-0.2,0.2;length=m)),
                  1=>fill(0.25+0.07im,m-1))
        modes = [HEOMMode(diagm(ComplexF64.(range(-0.3,1.;length=m)).^j),
                    0.7+0.1j,0.08+0.01im,0.08-0.01im) for j in 1:k]
        system = HEOMTTNSystem(H,modes,fill(3,k);system_layout=Symbol(parts[3]))
        rho = Matrix{ComplexF64}(I,m,m)*(0.05/m); rho[1,1] += 0.95
        initial = heom_initial_state(system,rho)
        c = trace_observable(system;tree=initial.tree)
        observe = x -> (;density=root_density_matrix(x,system),invariant=dot(c,x))
        return (;system,initial,operator=heom_generator(system),conserved=c,observe)
    end
    m,n = parse.(Int,split(name[4:end],'_'))
    system = holstein_tfd_system(collect(range(0.15,-0.15;length=m)),0.4,fill(0.8,m),fill(0.3,m);
                                beta=2.,boson_size=n)
    rho = fill(0.15/(m-1),m); rho[1] = 0.85
    initial = tfd_initial_state(system,diagm(rho))
    H = tfd_hamiltonian(system)
    initial = enrich_basis(initial,H*initial)
    observer = TTNDynamics._tfd_observer(system,H)
    observe = x -> begin
        value = observer(x)
        (;density=value.density,invariant=value.norm)
    end
    return (;system,initial,operator=TTNDynamics._tfd_generator(H),conserved=nothing,observe)
end

# Problem construction and TFD's one-time basis enrichment are excluded.
# Each timed trajectory includes all BUG preparation/solves/truncation and
# the same observable contractions after every step. No dense state is built.
function trajectory(problem;threaded=false,step=augmented_bug_step,nsteps=NSTEPS)
    x = copy(problem.initial)
    observations = [problem.observe(x)]
    ranks = [tree_ranks(x)]
    diagnostics = NamedTuple[]
    for _ in 1:nsteps
        x,d = step(x,problem.operator,STEP;OPTIONS...,conserved=problem.conserved,threaded)
        push!(diagnostics,d)
        push!(observations,problem.observe(x))
        push!(ranks,d.ranks)
    end
    return (;final_state=x,observations,ranks,diagnostics)
end

function benchmark(repetitions=5;cases=CASES)
    repetitions >= 3 || error("use at least three repetitions")
    BLAS.get_num_threads() == 1 || error("set OPENBLAS_NUM_THREADS=1")
    data = joinpath(@__DIR__,"data"); mkpath(data)
    suffix = "threads$(Threads.nthreads(:default))"
    open(joinpath(data,"environment_$suffix.txt"),"w") do io
        println(io,"Measured: ",Dates.now(),"\nJulia: ",VERSION,"\nCPU: ",first(Sys.cpu_info()).model)
        println(io,"Julia workers: ",Threads.nthreads(:default),"\nBLAS threads: ",BLAS.get_num_threads())
        println(io,"GC threads: ",Threads.ngcthreads(),"\nKrylovKit: ",Base.pkgversion(TTNSolver.KrylovKit))
        println(io,"h: ",STEP,"\nsteps: ",NSTEPS,"\noptions: ",OPTIONS,"\nrepetitions: ",repetitions)
        println(io,"Load at start: ",Sys.loadavg())
        for file in ("bug.jl","bug_workspaces.jl","contractions.jl","truncation.jl")
            println(io,file," sha256: ",bytes2hex(sha256(read(joinpath(dirname(pathof(TTNSolver)),file)))))
        end
        println(io,"run.jl sha256: ",bytes2hex(sha256(read(@__FILE__))))
    end
    open(joinpath(data,"runs_$suffix.csv"),"w") do io
        println(io,"case,workers,threaded,repetition,seconds,bytes,gc_seconds,state_distance,density_difference,ranks_equal,invariant_drift,max_rank,max_augmented_rank,max_local_dimension,truncation_failures,largest_discarded_bound,leaves,terms,full_dimension")
        for name in cases
            problem = scaling_problem(name)
            println("Warming $name"); flush(stdout)
            for threaded in (false,true), _ in 1:2
                trajectory(problem;threaded)
            end
            for rep in 1:repetitions
                measured = Dict{Bool,Any}()
                for threaded in (isodd(rep) ? (false,true) : (true,false))
                    GC.gc()
                    measured[threaded] = @timed trajectory(problem;threaded)
                end
                a,b = measured[false].value,measured[true].value
                distance = ttn_distance(a.final_state,b.final_state)
                density_difference = maximum(norm(u.density-v.density) for (u,v) in zip(a.observations,b.observations))
                @assert distance < 1e-11 && density_difference < 1e-11
                @assert a.ranks == b.ranks
                @assert [d.truncation.tolerance_met for d in a.diagnostics] ==
                        [d.truncation.tolerance_met for d in b.diagnostics]
                for threaded in (false,true)
                    m = measured[threaded]; r = m.value
                    invariant_drift = maximum(abs(v.invariant-first(r.observations).invariant) for v in r.observations)
                    println(io,join((name,Threads.nthreads(:default),threaded,rep,m.time,m.bytes,m.gctime,
                        distance,density_difference,a.ranks==b.ranks,invariant_drift,
                        maximum(maximum.(r.ranks)),maximum(maximum(d.augmented_ranks) for d in r.diagnostics),
                        maximum(d.max_local_dimension for d in r.diagnostics),
                        count(d -> !d.truncation.tolerance_met,r.diagnostics),
                        maximum(d.truncation.discarded_bound for d in r.diagnostics),
                        length(problem.initial.tree.leaf_order),length(problem.operator.terms),
                        prod(big.(leafdims(problem.initial)))),','))
                end
                flush(io)
                println("  $rep: serial ",round(measured[false].time;digits=3)," s; threaded ",
                    round(measured[true].time;digits=3)," s; delta ",distance); flush(stdout)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark(isempty(ARGS) ? 5 : parse(Int,only(ARGS)))
end
