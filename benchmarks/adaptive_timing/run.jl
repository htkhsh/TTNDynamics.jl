# From TTNDynamics: OPENBLAS_NUM_THREADS=1 julia --project=examples benchmarks/adaptive_timing/run.jl [repetitions]
using TTNDynamics, TTNSolver, LinearAlgebra, Dates
include("../../examples/holstein_tfd/dynamics.jl")
include("../../examples/heom/common/dense_heom_reference.jl")

function timing_cases()
    H = ComplexF64[0.2 0.3+0.1im;0.3-0.1im -0.4]
    V = ComplexF64[1 0.2im;-0.2im -0.5]
    modes = [HEOMMode(V,0.8,0.7+0.2im,0.7-0.2im),
             HEOMMode(V,0.4+1.2im,0.3+0.1im,0.2+0.05im;scale=0.7),
             HEOMMode(V,0.4-1.2im,0.2-0.05im,0.3-0.1im;scale=0.7)]
    heom = HEOMTTNSystem(H,modes,[3,3,3])
    initial = heom_initial_state(heom,ComplexF64[0.7 0.1im;-0.1im 0.3])
    reference = exp(10*dense_heom_reference_matrix(heom))*vec(dense(initial))
    cases = [(name="HEOM_3m",dimension=108,
        run=timing -> heom_evolve(heom,initial;tspan=(0.,10.),dt=0.2,
            saveat=0:0.2:10,atol=1e-8,rtol=1e-5,correct_trace=false,record_timing=timing),
        reference)]
    # Fock size 5 exercises larger ranks; its accuracy is not checked against
    # a dense model here. Instrumentation equivalence is checked for every run.
    tfd_cases = [tfd_timing_case(size) for size in (3,5)]
    return [cases;tfd_cases]
end

function tfd_timing_case(boson_size)
    system,initial = holstein_tfd_problem(merge(holstein_config,(;boson_size)))
    reference = if boson_size == 3
        E,U = eigen(Hermitian(dense_tfd_reference_matrix(system)))
        U*(exp.(-50im.*E).*(U'*vec(dense(initial))))
    else
        nothing
    end
    return (name="TFD_Fock$boson_size",dimension=prod(tfd_dimensions(system)),
        run=timing -> tfd_evolve(system,initial;tspan=(0.,50.),dt=2.,saveat=0:1:50,
            atol=1e-9,rtol=1e-5,record_timing=timing),reference)
end

function measured_run(run, timing)
    GC.gc()  # Outside the measured interval; GC remains enabled during it.
    started = time_ns()
    result = run(timing)
    return result,Float64(time_ns()-started)*1e-9
end

function benchmark_case(case,repetitions,runs_io,attempts_io)
    println("Warming $(case.name) (timing off and on)...")
    case.run(false)
    case.run(true)
    for repetition in 1:repetitions
        timed = plain = nothing
        elapsed = baseline = 0.0
        # Alternate order to reduce drift bias in the timing-overhead check.
        for enabled in (isodd(repetition) ? (true,false) : (false,true))
            result,seconds = measured_run(case.run,enabled)
            if enabled
                timed,elapsed = result,seconds
            else
                plain,baseline = result,seconds
            end
        end
        @assert timed.stats == plain.stats && timed.times == plain.times
        @assert [(d.dt,d.accepted,d.reason) for d in timed.diagnostics] ==
                [(d.dt,d.accepted,d.reason) for d in plain.diagnostics]
        distance = ttn_distance(timed.final_state,plain.final_state)
        @assert distance < 1e-11*max(1,norm(plain.final_state))
        @assert plain.timings === nothing
        tm = timed.timings
        dense_error = isnothing(case.reference) ? NaN :
            norm(vec(dense(timed.final_state))-case.reference)/norm(case.reference)
        augmented_rank = maximum(maximum(info.augmented_ranks)
            for d in timed.diagnostics for info in (d.coarse,d.half1,d.half2) if !isnothing(info))
        row = (case.name,repetition,case.dimension,timed.stats.accepted,timed.stats.rejected,
            maximum(timed.ranks),augmented_rank,elapsed,baseline,tm.total,tm.coarse_bug,
            tm.accepted_update,tm.rejected_update,tm.difference,tm.candidate_norm,
            tm.observations,tm.other,distance,dense_error)
        println(runs_io,join(row,','))
        flush(runs_io)
        for (attempt,d) in enumerate(timed.diagnostics)
            t = d.timings
            println(attempts_io,join((case.name,repetition,attempt,d.time,d.dt,Int(d.accepted),
                d.reason,t.coarse_bug,t.fine_bug,t.difference,t.candidate_norm),','))
        end
        flush(attempts_io)
        println("  repeat $repetition: driver=$(round(tm.total;digits=4)) s, ",timed.stats,
                ", rank=$(maximum(timed.ranks))")
    end
end

function run_timing_benchmark(repetitions=5)
    repetitions >= 3 || throw(ArgumentError("use at least 3 repetitions"))
    data_dir = joinpath(@__DIR__,"data")
    mkpath(data_dir)
    open(joinpath(data_dir,"environment.txt"),"w") do io
        println(io,"Measured: ",Dates.now())
        println(io,"Julia: ",VERSION)
        println(io,"CPU: ",first(Sys.cpu_info()).model)
        println(io,"Julia threads: ",Threads.nthreads())
        println(io,"BLAS threads: ",BLAS.get_num_threads())
        println(io,"BLAS: ",BLAS.get_config())
        println(io,"Repetitions: ",repetitions)
        println(io,"Two full warmups per case; alternating timed/untimed runs; GC enabled.")
    end
    open(joinpath(data_dir,"runs.csv"),"w") do runs_io
        println(runs_io,"case,repetition,dimension,accepted,rejected,max_rank,max_augmented_rank,wall_seconds,untimed_wall_seconds,total,coarse_bug,accepted_update,rejected_update,difference,candidate_norm,observations,other,timing_state_difference,dense_relative_error")
        open(joinpath(data_dir,"attempts.csv"),"w") do attempts_io
            println(attempts_io,"case,repetition,attempt,time,dt,accepted,reason,coarse_bug,fine_bug,difference,candidate_norm")
            for case in timing_cases()
                benchmark_case(case,repetitions,runs_io,attempts_io)
            end
        end
    end
    println("Timing data: $data_dir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || error("usage: run.jl [repetitions]")
    run_timing_benchmark(isempty(ARGS) ? 5 : parse(Int,only(ARGS)))
end
