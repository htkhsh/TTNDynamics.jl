using TTNDynamics, TTNSolver, LinearAlgebra, Dates, SHA
include("../parallel_bug/run.jl")
include("reference/Before.jl")
include("stages.jl")

function optimization_case(name)
    name == "tfd3" && return tfd_case()
    if name == "tfd4_large"
        system = holstein_tfd_system([0.15,0.05,-0.05,-0.15],0.4,fill(0.8,4),fill(0.3,4);
                                    beta=2.0,boson_size=3)
        return system,tfd_initial_state(system,diagm([0.85,0.05,0.05,0.05]))
    end
    parts = split(name,'_')
    return heom_case(parse(Int,parts[1][5:end]),parse(Int,parts[2]),Symbol(parts[3]))
end

# Identical caller/observation costs for both step implementations. Model
# setup, TFD enrichment, conservation, grid and tolerances match the old report.
function trajectory(step,system,initial,h;threaded=false)
    if system isa HEOMTTNSystem
        L = heom_generator(system)
        c = trace_observable(system;tree=initial.tree)
        x = copy(initial)
        observe = x -> (;density=root_density_matrix(x,system),invariant=dot(c,x))
    else
        H = tfd_hamiltonian(system)
        L,c = TTNDynamics._tfd_generator(H),nothing
        x = enrich_basis(initial,H*initial)
        tfd_observe = TTNDynamics._tfd_observer(system,H)
        observe = x -> begin
            value = tfd_observe(x)
            (;density=value.density,invariant=value.norm)
        end
    end
    observations = [observe(x)]
    ranks = [tree_ranks(x)]
    diagnostics = NamedTuple[]
    for _ in 1:round(Int,0.5/h)
        x,d = step(x,L,h;conserved=c,correct_invariant=false,atol=1e-10,maxrank=64,threaded)
        push!(diagnostics,d)
        push!(observations,observe(x))
        push!(ranks,tree_ranks(x))
    end
    return (;final_state=x,observations,ranks,diagnostics)
end

function run_optimization(repetitions=5;resume=false,
        cases=("heom4_6_fused","heom4_6_split","heom16_2_fused","heom16_2_split","tfd3","tfd4_large"))
    repetitions >= 3 || error("use at least three repetitions")
    BLAS.get_num_threads() == 1 || error("set OPENBLAS_NUM_THREADS=1")
    suffix = "threads$(Threads.nthreads(:default))"
    data = joinpath(@__DIR__,"data"); mkpath(data)
    source = read(joinpath(dirname(pathof(TTNSolver)),"parallel_bug.jl"),String)
    before_source = read(joinpath(@__DIR__,"reference/parallel_bug_before.jl"),String)
    timed_before = timed_module(:BeforeTimed,before_source)
    timed_after = timed_module(:AfterTimed,source)
    open(joinpath(data,"environment_$suffix.txt"),resume ? "a" : "w") do io
        println(io,"Measured: ",Dates.now(),"\nJulia: ",VERSION,"\nCPU: ",first(Sys.cpu_info()).model)
        println(io,"Worker threads: ",Threads.nthreads(:default),"\nBLAS threads: ",BLAS.get_num_threads())
        println(io,"Repetitions: ",repetitions,"\nKrylovKit: ",Base.pkgversion(TTNSolver.KrylovKit))
        println(io,"Baseline parallel_bug.jl sha256: ",bytes2hex(sha256(before_source)))
        for file in ("parallel_bug.jl","parallel_workspaces.jl","bug.jl","contractions.jl")
            println(io,file," sha256: ",bytes2hex(sha256(read(joinpath(dirname(pathof(TTNSolver)),file)))))
        end
    end
    open(joinpath(data,"runs_$suffix.csv"),resume ? "a" : "w") do io
        resume || println(io,"case,threaded,repetition,version,seconds,bytes,gc_seconds,state_distance,density_difference,omitted_difference,max_rank,ranks_equal,trace_or_norm_drift,truncation_failures")
        open(joinpath(data,"stages_$suffix.csv"),resume ? "a" : "w") do phases
            resume || println(phases,"case,threaded,repetition,version,prepare,solve,assemble,finish,other,total")
            for name in cases
                system,x = optimization_case(name)
                println("Warming $name"); flush(stdout)
                # Both ordinary and instrumented paths warm up outside timing.
                for threaded in (false,true), step in (Before.parallel_bug_step,parallel_bug_step)
                    trajectory(step,system,x,0.05;threaded)
                    trajectory(step,system,x,0.05;threaded)
                end
                for threaded in (false,true)
                    for rep in 1:repetitions
                        measured = Dict{Symbol,Any}()
                        for version in (isodd(rep) ? (:before,:after) : (:after,:before))
                            step = version == :before ? Before.parallel_bug_step : parallel_bug_step
                            GC.gc()
                            measured[version] = @timed trajectory(step,system,x,0.05;threaded)
                        end
                        a,b = measured[:before].value,measured[:after].value
                        distance = ttn_distance(a.final_state,b.final_state)
                        density_difference = maximum(norm(u.density-v.density) for (u,v) in zip(a.observations,b.observations))
                        omitted_difference = maximum(maximum(abs.(u.omitted_block_norms-v.omitted_block_norms)) for
                                                     (u,v) in zip(a.diagnostics,b.diagnostics))
                        @assert distance < 1e-8 && density_difference < 1e-8
                        @assert [d.truncation.tolerance_met for d in a.diagnostics] ==
                                [d.truncation.tolerance_met for d in b.diagnostics]
                        for version in (:before,:after)
                            m = measured[version]; result = m.value
                            invariant_drift = maximum(abs(v.invariant-first(result.observations).invariant) for v in result.observations)
                            println(io,join((name,threaded,rep,version,m.time,m.bytes,m.gctime,distance,density_difference,
                                omitted_difference,maximum(maximum(r) for r in result.ranks),a.ranks==b.ranks,invariant_drift,
                                count(d -> !d.truncation.tolerance_met,result.diagnostics)),','))
                        end
                        flush(io)
                        println("  threaded=$threaded $rep: ",round(measured[:before].time;digits=3)," -> ",
                            round(measured[:after].time;digits=3)," s; delta=",distance); flush(stdout)
                    end
                    for version in (:before,:after)
                        mod = version == :before ? timed_before : timed_after
                        step = Base.invokelatest(getproperty,mod,:parallel_bug_step)
                        stage_times = Base.invokelatest(getproperty,mod,:STAGES)
                        Base.invokelatest(trajectory,step,system,x,0.05;threaded)
                        for rep in 1:repetitions
                            fill!(stage_times,0)
                            GC.gc()
                            timing = @timed Base.invokelatest(trajectory,step,system,x,0.05;threaded)
                            other = timing.time-sum(stage_times)
                            println(phases,join((name,threaded,rep,version,stage_times...,other,timing.time),','))
                        end
                    end
                    flush(phases)
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_optimization(isempty(ARGS) ? 5 : parse(Int,only(ARGS)))
end
