# Compare a saved pre-optimization TTNSolver source directory with the current
# package, in the same process. No source files are overwritten by this script.
using TTNDynamics, TTNSolver, LinearAlgebra, Dates, SHA
length(ARGS) in (1,2) || error("usage: run.jl BASELINE_SRC [repetitions]")
# A source-loaded module has no package dependency context. Expose the current
# solver's project so the baseline imports the same installed KrylovKit.
push!(LOAD_PATH,dirname(dirname(pathof(TTNSolver))))

module Baseline
include(joinpath(abspath(ARGS[1]),"TTNSolver.jl"))
module TTNDynamics
using LinearAlgebra
using ..TTNSolver
# The model wrappers are identical; only their solver dependency differs.
for file in ("heom.jl","evolution.jl","adaptive_evolution.jl","tfd.jl","tfd_evolution.jl")
    include(joinpath(@__DIR__,"../../src",file))
end
end
end

function comparison_cases(D)
    H = ComplexF64[0.2 0.3+0.1im;0.3-0.1im -0.4]
    V = ComplexF64[1 0.2im;-0.2im -0.5]
    modes = [D.HEOMMode(V,0.8,0.7+0.2im,0.7-0.2im),
             D.HEOMMode(V,0.4+1.2im,0.3+0.1im,0.2+0.05im;scale=0.7),
             D.HEOMMode(V,0.4-1.2im,0.2-0.05im,0.3-0.1im;scale=0.7)]
    heom = D.HEOMTTNSystem(H,modes,[3,3,3])
    initial = D.heom_initial_state(heom,ComplexF64[0.7 0.1im;-0.1im 0.3])
    cases = [(name="HEOM_adaptive",run=() -> D.heom_evolve(heom,initial;
        tspan=(0.,10.),dt=0.2,saveat=0:0.2:10,atol=1e-8,rtol=1e-5,
        correct_trace=false,record_timing=true)),
        (name="HEOM_fixed",run=() -> D.heom_evolve(heom,initial,0:0.1:10;
            atol=1e-10,correct_trace=false))]
    return [cases;[comparison_tfd_case(D,size,:adaptive) for size in (3,5)];
            [comparison_tfd_case(D,3,:fixed)]]
end

function comparison_tfd_case(D,size,method)
    icm2ifs = 2pi*299792458.0*1e2*1e-15
    beta = (6.62607015e-34/(2pi))*1e15/(1.380649e-23*300)
    system = D.holstein_tfd_system([0.,0.],200*icm2ifs,
        [160.,160.].*icm2ifs,[80.,80.].*icm2ifs;beta,boson_size=size,periodic=true)
    initial = D.tfd_initial_state(system,ComplexF64[1 0;0 0])
    run = method == :adaptive ?
        () -> D.tfd_evolve(system,initial;tspan=(0.,50.),dt=2.,saveat=0:1:50,
            atol=1e-9,rtol=1e-5,record_timing=true) :
        () -> D.tfd_evolve(system,initial,0:0.5:50;atol=1e-10)
    return (name="TFD$(size)_$(method)",run)
end

function measure(run)
    GC.gc()
    return @timed run()
end

function compare_results(before,after)
    @assert before.times == after.times
    @assert before.ranks == after.ranks
    if hasproperty(before,:stats)
        @assert before.stats == after.stats
        @assert [(d.dt,d.accepted,d.reason) for d in before.diagnostics] ==
                [(d.dt,d.accepted,d.reason) for d in after.diagnostics]
    end
    error = maximum(norm(a-b) for (a,b) in zip(before.densities,after.densities))
    @assert error <= 1e-11
    for field in (:traces,:norms,:energies,:mode_occupations,:physical_occupations)
        if hasproperty(before,field)
            @assert getproperty(before,field) ≈ getproperty(after,field) atol=1e-11 rtol=1e-11
        end
    end
    a,b = Baseline.TTNSolver.dense(before.final_state),TTNSolver.dense(after.final_state)
    final_error = norm(a-b)/max(1,norm(a))
    @assert final_error <= 1e-11
    return error,final_error
end

function run_comparison(repetitions)
    repetitions >= 3 || error("use at least three repetitions")
    data = joinpath(@__DIR__,"data")
    mkpath(data)
    open(joinpath(data,"environment.txt"),"w") do io
        println(io,"Measured: ",Dates.now(),"\nJulia: ",VERSION)
        println(io,"CPU: ",first(Sys.cpu_info()).model)
        println(io,"Julia threads: ",Threads.nthreads(),"\nBLAS threads: ",BLAS.get_num_threads())
        println(io,"BLAS: ",BLAS.get_config(),"\nRepetitions: ",repetitions)
        println(io,"KrylovKit: ",Base.pkgversion(TTNSolver.KrylovKit))
        for (label,path) in (("before",abspath(ARGS[1])),("after",dirname(pathof(TTNSolver))))
            for name in ("TTNSolver.jl","bug.jl","adaptive.jl","bug_workspaces.jl")
                file = joinpath(path,name)
                println(io,label," ",name," sha256: ",isfile(file) ? bytes2hex(sha256(read(file))) : "absent")
            end
        end
    end
    before_cases,after_cases = comparison_cases(Baseline.TTNDynamics),comparison_cases(TTNDynamics)
    open(joinpath(data,"runs.csv"),"w") do io
        println(io,"case,repetition,version,wall_seconds,bytes,gc_seconds,accepted,rejected,max_rank,coarse_bug,accepted_update,rejected_update,difference,density_difference,final_state_difference")
        for (old,new) in zip(before_cases,after_cases)
            @assert old.name == new.name
            println("Warming ",new.name)
            for _ in 1:2
                compare_results(old.run(),new.run())
            end
            for repetition in 1:repetitions
                measured = Dict{Symbol,Any}()
                order = isodd(repetition) ? (:before,:after) : (:after,:before)
                for version in order
                    measured[version] = measure(version == :before ? old.run : new.run)
                end
                before,after = measured[:before].value,measured[:after].value
                density_error,state_error = compare_results(before,after)
                for version in (:before,:after)
                    m = measured[version]
                    r = m.value
                    accepted,rejected = hasproperty(r,:stats) ? (r.stats.accepted,r.stats.rejected) :
                        (length(r.diagnostics),0)
                    phases = hasproperty(r,:timings) ?
                        (r.timings.coarse_bug,r.timings.accepted_update,r.timings.rejected_update,r.timings.difference) :
                        (NaN,NaN,NaN,NaN)
                    println(io,join((new.name,repetition,version,m.time,m.bytes,m.gctime,
                        accepted,rejected,maximum(r.ranks),phases...,density_error,state_error),','))
                end
                flush(io)
                println("  ",repetition,": ",round(measured[:before].time;digits=4)," → ",
                        round(measured[:after].time;digits=4)," s; state difference=",state_error)
            end
        end
    end
end

run_comparison(length(ARGS) == 2 ? parse(Int,ARGS[2]) : 5)
