include("optimization_probe.jl")

function compare_trajectories()
    label = isempty(ARGS) ? "optimized_trajectories" : ARGS[1]
    dir = joinpath(@__DIR__,"data",label)
    isdir(dir) && error("Output exists: $dir")
    mkpath(dir)
    p = example_problem()
    mature = deserialize(joinpath(@__DIR__,"data","augmented_h005","latest.jls")).state
    reference = deserialize(joinpath(@__DIR__,"data","pilot_reference.jls"))[3]
    open(joinpath(dir,"metadata.txt"),"w") do io
        println(io,"Localized initial state, 0–1 fs, same settings before/after.")
        println(io,"One whole-trajectory trial per variant; warmup, observables and saving excluded.")
        source_metadata(io)
        for file in sort(readdir(joinpath(@__DIR__,"data","optimization_baseline");join=true))
            println(io,"baseline ",basename(file)," sha256: ",bytes2hex(sha256(read(file))))
        end
    end
    open(joinpath(dir,"summary.csv"),"w") do summary
        println(summary,"method,implementation,seconds,gc_seconds,allocated_bytes,state_difference,state_error,density_error,maxrank,truncation_misses")
        for (method,h,atol) in ((:augmented,.05,5e-5),(:parallel,.02,2e-5))
            options = (;atol,maxrank=32,conserved=p.conserved,correct_invariant=false,threaded=true)
            symbol = Symbol(method,"_bug_step")
            baseline = nothing
            for (label,mod) in (("baseline",BaselineBUG),("optimized",TTNSolver))
                step = getfield(mod,symbol)
                println("warm trajectory ",method," ",label); flush(stdout)
                step(mature,p.operator,h;options...)
                step(p.tree,p.operator,h;options...)
                x = copy(p.tree)
                seconds,gc,bytes,misses = 0.0,0.0,0,0
                GC.gc()
                open(joinpath(dir,"$(method)_$(label)_steps.csv"),"w") do io
                    println(io,"time_fs,seconds,gc_seconds,allocated_bytes,maxrank,truncation_met")
                    for n in 1:round(Int,1/h)
                        measured = @timed step(x,p.operator,h;options...)
                        x,stats = measured.value
                        seconds += measured.time; gc += measured.gctime; bytes += measured.bytes
                        met = stats.truncation.tolerance_met
                        misses += !met
                        println(io,join((n*h,measured.time,measured.gctime,measured.bytes,
                                         maximum(tree_ranks(x)),met),',')); flush(io)
                        n % 10 == 0 && (println(method," ",label," t=",n*h," s=",seconds); flush(stdout))
                    end
                end
                label=="baseline" && (baseline=x)
                distance = ttn_distance(baseline,x)/norm(baseline)
                @test distance < 1e-7
                err = relative_error(x,reference)
                rhoerr = norm(TTNDynamics.root_density_matrix(x,p.system)-
                              TTDynamics.root_density_matrix(reference,p.problem.system))
                println(summary,join((method,label,seconds,gc,bytes,distance,err,rhoerr,
                                      maximum(tree_ranks(x)),misses),',')); flush(summary)
                serialize(joinpath(dir,"$(method)_$(label)_final.jls"),x)
                println("DONE ",method," ",label," seconds=",seconds," difference=",distance); flush(stdout)
            end
        end
    end
end

compare_trajectories()
