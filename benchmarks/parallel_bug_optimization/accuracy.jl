# Compare the time needed to meet a common final-state error target. This is
# a finite search on fixed grids, not a claim of an optimal error/work curve.
include("run.jl")
function augmented_comparison_step(x,L,h;threaded=false,kwargs...)
    return augmented_bug_step(x,L,h;kwargs...)
end

function compare_accuracy(repetitions=5;target=1e-4)
    repetitions >= 3 || error("use at least three repetitions")
    BLAS.get_num_threads() == 1 || error("set OPENBLAS_NUM_THREADS=1")
    data = joinpath(@__DIR__,"data"); mkpath(data)
    variants = ("augmented","parallel_serial","parallel_threads")
    open(joinpath(data,"accuracy_scan.csv"),"w") do scan
        println(scan,"case,variant,h,relative_error,meets_target")
        open(joinpath(data,"accuracy_runs.csv"),"w") do io
            println(io,"case,variant,repetition,workers,target,h,relative_error,seconds,bytes")
            for name in ("heom4_6_fused","tfd3")
                system,x = optimization_case(name)
                exact = exp(0.5generator(system))*vec(dense(x))
                error(result) = norm(vec(dense(result.final_state))-exact)/norm(exact)
                selected = Dict{String,Float64}()
                for variant in variants
                    step = variant == "augmented" ? augmented_comparison_step : parallel_bug_step
                    threaded = variant == "parallel_threads"
                    for j in 0:8
                        h = 0.05/2^j
                        result = trajectory(step,system,x,h;threaded)
                        e = error(result)
                        println(scan,join((name,variant,h,e,e<=target),',')); flush(scan)
                        if e <= target
                            selected[variant] = h
                            trajectory(step,system,x,h;threaded)
                            break
                        end
                    end
                    haskey(selected,variant) || error("no grid met target for $name/$variant")
                end
                println("$name selected h: ",selected); flush(stdout)
                for rep in 1:repetitions
                    for variant in circshift(collect(variants),rep%3)
                        step = variant == "augmented" ? augmented_comparison_step : parallel_bug_step
                        threaded = variant == "parallel_threads"
                        h = selected[variant]
                        GC.gc()
                        m = @timed trajectory(step,system,x,h;threaded)
                        e = error(m.value)
                        @assert e <= target
                        println(io,join((name,variant,rep,Threads.nthreads(:default),target,h,e,m.time,m.bytes),','))
                    end
                    flush(io)
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare_accuracy(isempty(ARGS) ? 5 : parse(Int,only(ARGS)))
end
