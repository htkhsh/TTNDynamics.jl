using TTNDynamics, TTNSolver, LinearAlgebra, Dates, SHA
include("../../examples/heom/common/dense_heom_reference.jl")
include("../../examples/holstein_tfd/dense_reference.jl")

function heom_case(d,baths,layout)
    H = diagm(-1=>fill(0.25-0.07im,d-1),0=>ComplexF64.(range(-0.2,0.2;length=d)),
               1=>fill(0.25+0.07im,d-1))
    modes = [HEOMMode(diagm(ComplexF64.(range(-0.3,1.0;length=d)).^k),
        0.7+0.1k,0.08+0.01im,0.08-0.01im) for k in 1:baths]
    system = HEOMTTNSystem(H,modes,fill(2,baths);system_layout=layout)
    rho = Matrix{ComplexF64}(I,d,d)*(0.05/d); rho[1,1] += 0.95
    return system,heom_initial_state(system,rho)
end

function tfd_case()
    system = holstein_tfd_system([0.1,0.0,-0.1],0.4,fill(0.8,3),fill(0.3,3);
                                beta=2.0,boson_size=2)
    initial = tfd_initial_state(system,diagm([0.9,0.05,0.05]))
    return system,initial
end

const VARIANTS = ((:augmented,false),(:parallel,false),(:parallel,true))
label(method,threaded) = method == :augmented ? "augmented" : threaded ? "parallel_threads" : "parallel_serial"
evolve(system::HEOMTTNSystem,x,h,method,threaded) =
    heom_evolve(system,x,0:h:0.5;method,threaded,atol=1e-10,maxrank=64,correct_trace=false)
evolve(system::TFDTTNSystem,x,h,method,threaded) =
    tfd_evolve(system,x,0:h:0.5;method,threaded,atol=1e-10,maxrank=64)
generator(system::HEOMTTNSystem) = dense_heom_reference_matrix(system)
generator(system::TFDTTNSystem) = -im*dense_tfd_reference_matrix(system)
drift(result,::HEOMTTNSystem) = maximum(abs.(result.traces.-first(result.traces)))
drift(result,::TFDTTNSystem) = maximum(abs.(result.norms.-first(result.norms)))

function benchmark(repetitions=5)
    repetitions >= 3 || error("use at least three repetitions")
    BLAS.get_num_threads() == 1 || error("set OPENBLAS_NUM_THREADS=1")
    data = joinpath(@__DIR__,"data"); mkpath(data)
    suffix = "threads$(Threads.nthreads(:default))"
    open(joinpath(data,"environment_$suffix.txt"),"w") do io
        println(io,"Measured: ",Dates.now(),"\nJulia: ",VERSION)
        println(io,"CPU: ",first(Sys.cpu_info()).model)
        println(io,"Julia worker threads: ",Threads.nthreads(:default),"\nBLAS threads: ",BLAS.get_num_threads())
        println(io,"Repetitions: ",repetitions,"\nKrylovKit: ",Base.pkgversion(TTNSolver.KrylovKit))
        for file in ("bug.jl","parallel_bug.jl","contractions.jl")
            println(io,file," sha256: ",bytes2hex(sha256(read(joinpath(dirname(pathof(TTNSolver)),file)))))
        end
    end
    open(joinpath(data,"runs_$suffix.csv"),"w") do io
        println(io,"case,variant,repetition,threads,seconds,bytes,gc_seconds,relative_error,invariant_drift,max_rank,max_local_dimension,omitted_block_max,serial_thread_distance")
        open(joinpath(data,"convergence_$suffix.csv"),"w") do conv
            println(conv,"case,variant,h,relative_error,invariant_drift,max_rank")
            for name in ("heom4_6_fused","heom4_6_split","heom16_2_fused","heom16_2_split","tfd3")
                system,x = if name == "tfd3"
                    tfd_case()
                else
                    parts = split(name,'_')
                    heom_case(parse(Int,parts[1][5:end]),parse(Int,parts[2]),Symbol(parts[3]))
                end
                reference = exp(0.5generator(system))*vec(dense(x))
                error(result) = norm(vec(dense(result.final_state))-reference)/norm(reference)
                println("Warming and checking $name")
                for (method,threaded) in VARIANTS
                    for h in (0.05,0.025)
                        result = evolve(system,x,h,method,threaded)
                        println(conv,join((name,label(method,threaded),h,error(result),drift(result,system),maximum(result.ranks)),','))
                    end
                    evolve(system,x,0.05,method,threaded)
                end
                flush(conv)
                for repetition in 1:repetitions
                    results = Dict{String,Any}()
                    order = circshift(collect(VARIANTS),repetition%3)
                    for (method,threaded) in order
                        GC.gc()
                        results[label(method,threaded)] = @timed evolve(system,x,0.05,method,threaded)
                    end
                    agreement = ttn_distance(results["parallel_serial"].value.final_state,
                                             results["parallel_threads"].value.final_state)
                    @assert agreement < 1e-11
                    for (method,threaded) in VARIANTS
                        variant = label(method,threaded)
                        m = results[variant]; result = m.value
                        omitted = method == :parallel ? maximum(maximum(d.omitted_block_norms) for d in result.diagnostics) : NaN
                        @assert isfinite(error(result))
                        @assert all(d.truncation.tolerance_met for d in result.diagnostics)
                        println(io,join((name,variant,repetition,Threads.nthreads(:default),m.time,m.bytes,m.gctime,
                            error(result),drift(result,system),maximum(result.ranks),
                            maximum(d.max_local_dimension for d in result.diagnostics),omitted,agreement),','))
                    end
                    flush(io)
                    println("  $repetition: ",join(["$v=$(round(results[v].time*1000;digits=1)) ms" for v in
                        ("augmented","parallel_serial","parallel_threads")],", "))
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark(isempty(ARGS) ? 5 : parse(Int,only(ARGS)))
end
