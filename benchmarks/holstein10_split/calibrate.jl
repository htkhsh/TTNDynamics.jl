include("common.jl")

function calibrate()
    p = example_problem()
    refs = deserialize(joinpath(@__DIR__,"data","pilot_reference.jls"))
    target = length(refs)-1
    target==2 || error("expected reference through 1 fs")
    open(joinpath(@__DIR__,"data","calibration.csv"),"w") do io
        println(io,"method,h,seconds,maxrank,state_error,density_error,truncation_misses")
        flush(io)
        measured_total = 0.0
        for (method,steps) in ((:parallel,20),(:parallel,50),(:parallel,100))
            x = p.tree; total = 0.; misses = 0
            atol = 1e-3/steps
            tree_step(p,x,1/steps;method,atol,maxrank=32) # warmup
            GC.gc()
            for j in 1:steps
                m = @timed tree_step(p,x,1/steps;method,atol,maxrank=32)
                x,d = m.value; total += m.time; misses += !d.truncation.tolerance_met
                measured_total += m.time
                println(method," h=",1/steps," step=",j,"/",steps," seconds=",m.time,
                        " cumulative=",total," rank=",maximum(tree_ranks(x)),
                        " truncation_met=",d.truncation.tolerance_met); flush(stdout)
                serialize(joinpath(@__DIR__,"data","calibration_latest.jls"),
                          (;state=x,time_fs=j/steps,stepsize=1/steps,total,misses))
                if measured_total>240
                    println("Calibration time budget reached at ",j/steps," fs"); flush(stdout)
                    return
                end
            end
            err = relative_error(x,last(refs))
            de = norm(TTNDynamics.root_density_matrix(x,p.system)-TTDynamics.root_density_matrix(last(refs),p.problem.system))
            row = (method,1/steps,total,maximum(tree_ranks(x)),err,de,misses)
            println(io,join(row,',')); flush(io)
            println(join(row,',')); flush(stdout)
        end
    end
end

calibrate()
