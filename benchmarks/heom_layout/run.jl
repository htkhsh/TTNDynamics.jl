using TTNDynamics, TTNSolver, LinearAlgebra, Dates, SHA
include("../../examples/heom/common/dense_heom_reference.jl")
include("../../examples/common/model_tree_output.jl")

function layout_fixture(d,layout)
    H = diagm(-1=>fill(0.25-0.07im,d-1),0=>ComplexF64.(range(-0.2,0.2;length=d)),
               1=>fill(0.25+0.07im,d-1))
    V = diagm(-1=>fill(-0.1im,d-1),0=>ComplexF64.(range(0.6,1.2;length=d)),1=>fill(0.1im,d-1))
    W = diagm(-1=>fill(0.07+0im,d-1),0=>ComplexF64.(range(-0.3,0.7;length=d)),1=>fill(0.07+0im,d-1))
    modes = [HEOMMode(V,0.8,0.05+0.01im,0.05-0.01im),
             HEOMMode(W,1.2,0.025+0.005im,0.025-0.005im)]
    system = HEOMTTNSystem(H,modes,[2,2];system_layout=layout)
    rho = Matrix{ComplexF64}(I,d,d)*(0.05/d)
    rho[1,1] += 0.95
    return system,heom_initial_state(system,rho)
end

function layout_reference(system,initial)
    # Direct ADO equations, independent of the product-sum generator. The
    # largest case is 1024-dimensional; all dense work stays in this benchmark.
    L = dense_heom_reference_matrix(system)
    transition = exp(0.025L)
    states = [vec(dense(initial))]
    for _ in 1:40
        push!(states,transition*last(states))
    end
    return (;L,states)
end

function evolve_layout(system,initial,method;h=0.05)
    method == :fixed && return heom_evolve(system,initial,0:h:1;
        atol=1e-10,maxrank=64,correct_trace=false)
    return heom_evolve(system,initial;tspan=(0.,1.),dt=0.1,saveat=0:0.1:1,
        atol=1e-8,rtol=1e-4,maxrank=64,correct_trace=false)
end

function layout_errors(result,reference,d)
    errors = Float64[]
    for (t,rho) in zip(result.times,result.densities)
        index = round(Int,t/0.025)+1
        @assert isapprox(t,0.025*(index-1);atol=1e-14)
        target = reshape(reference.states[index][1:d^2],d,d)
        push!(errors,norm(rho-target))
    end
    final_error = norm(vec(dense(result.final_state))-last(reference.states))/norm(last(reference.states))
    drift = maximum(abs.(result.traces.-first(result.traces)))
    @assert isfinite(final_error) && drift < 1e-10
    return (;density_error=maximum(errors),final_error,drift)
end

function run_layout_benchmark(repetitions=5)
    repetitions >= 3 || error("use at least three repetitions")
    data = joinpath(@__DIR__,"data")
    trees = joinpath(@__DIR__,"trees")
    mkpath(data)
    mkpath(trees)
    open(joinpath(data,"environment.txt"),"w") do io
        println(io,"Measured: ",Dates.now(),"\nJulia: ",VERSION)
        println(io,"CPU: ",first(Sys.cpu_info()).model)
        println(io,"Julia threads: ",Threads.nthreads(),"\nBLAS threads: ",BLAS.get_num_threads())
        println(io,"BLAS: ",BLAS.get_config(),"\nRepetitions: ",repetitions)
        println(io,"KrylovKit: ",Base.pkgversion(TTNSolver.KrylovKit))
        for (label,path) in (("model",joinpath(dirname(pathof(TTNDynamics)),"heom.jl")),
                             ("BUG",joinpath(dirname(pathof(TTNSolver)),"bug.jl")),
                             ("workspaces",joinpath(dirname(pathof(TTNSolver)),"bug_workspaces.jl")))
            println(io,label," sha256: ",bytes2hex(sha256(read(path))))
        end
    end
    open(joinpath(data,"runs.csv"),"w") do io
        println(io,"dimension,method,repetition,layout,wall_seconds,bytes,gc_seconds,accepted,rejected,max_rank,system_bath_rank,operator_terms,operator_factor_bytes,density_error,final_relative_error,trace_drift,layout_density_difference,layout_state_difference")
        open(joinpath(data,"convergence.csv"),"w") do checks
            println(checks,"dimension,layout,generator_relative_error,coarse_density_error,fine_density_error,coarse_final_error,fine_final_error")
            for d in (2,4,8,16)
                fixtures = Dict(layout=>layout_fixture(d,layout) for layout in (:fused,:split))
                reference = layout_reference(fixtures[:fused]...)
                operators = Dict(layout=>heom_generator(first(fixtures[layout])) for layout in (:fused,:split))
                for layout in (:fused,:split)
                    generator_error = norm(dense(operators[layout])-reference.L)/norm(reference.L)
                    @assert generator_error < 1e-13
                    # Independent step refinement is not part of measured runs.
                    a = layout_errors(evolve_layout(fixtures[layout]...,:fixed),reference,d)
                    b = layout_errors(evolve_layout(fixtures[layout]...,:fixed;h=0.025),reference,d)
                    println(checks,join((d,layout,generator_error,a.density_error,b.density_error,a.final_error,b.final_error),','))
                end
                flush(checks)
                methods = d == 4 ? (:fixed,:adaptive) : (:fixed,)
                for method in methods
                    println("Warming M=$d, $method, both layouts")
                    for _ in 1:2, layout in (:fused,:split)
                        evolve_layout(fixtures[layout]...,method)
                    end
                    last_results = Dict{Symbol,Any}()
                    for repetition in 1:repetitions
                        measured = Dict{Symbol,Any}()
                        order = isodd(repetition) ? (:fused,:split) : (:split,:fused)
                        for layout in order
                            system,initial = fixtures[layout]
                            GC.gc() # Collection itself is outside the timed call.
                            measured[layout] = @timed evolve_layout(system,initial,method)
                        end
                        fused,split = measured[:fused].value,measured[:split].value
                        @assert fused.times == split.times
                        difference = maximum(norm(a-b) for (a,b) in zip(fused.densities,split.densities))
                        state_difference = norm(vec(dense(fused.final_state))-vec(dense(split.final_state)))
                        for layout in (:fused,:split)
                            m = measured[layout]
                            result = m.value
                            error = layout_errors(result,reference,d)
                            accepted,rejected = method == :fixed ? (length(result.diagnostics),0) :
                                (result.stats.accepted,result.stats.rejected)
                            tree = result.final_state.tree
                            system_node = tree.children[tree.root][1]
                            op = operators[layout]
                            factor_bytes = sum(sizeof(A) for term in op.terms for A in values(term.factors))
                            println(io,join((d,method,repetition,layout,m.time,m.bytes,m.gctime,accepted,rejected,
                                maximum(result.ranks),maximum(result.ranks[system_node,:]),length(op.terms),factor_bytes,
                                error.density_error,error.final_error,error.drift,difference,state_difference),','))
                            last_results[layout] = result
                        end
                        flush(io)
                        println("  $repetition: fused=",round(measured[:fused].time;digits=4),
                            " s, split=",round(measured[:split].time;digits=4)," s; density difference=",difference)
                    end
                    if d == 4 && method == :fixed
                        for layout in (:fused,:split)
                            system,initial = fixtures[layout]
                            for (label,state) in (("initial",initial),("final",last_results[layout].final_state))
                                write_model_tree(joinpath(trees,"$(layout)_$label"),system,state;
                                    title="HEOM · $layout · $label",subtitle="M = 4 · hierarchy sizes = [2, 2]")
                            end
                        end
                    end
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || error("usage: run.jl [repetitions]")
    run_layout_benchmark(isempty(ARGS) ? 5 : parse(Int,only(ARGS)))
end
