isdefined(@__MODULE__,:taylor_reference) || include("common.jl")

function calibrate(name)
    data = joinpath(@__DIR__,"data"); mkpath(data)
    println("Preparing $name"); flush(stdout)
    fixture_setup = @timed large_problem(name)
    problem = fixture_setup.value
    setup = @timed sop_to_mpo(problem.operator)
    A = setup.value; x = product_to_tt(physical_initial(problem))
    conserved = isnothing(problem.conserved) ? nothing : product_to_tt(problem.conserved)
    println("MPO rank ",maximum(tt_ranks(A))," setup seconds ",setup.time); flush(stdout)
    println("Computing independent reference"); flush(stdout)
    coarse = @timed taylor_reference(A,x;steps=10,tolerance=1e-9,order=20,verbose=true)
    println("Coarse reference ",coarse.time," s rank ",maximum(tt_ranks(coarse.value))); flush(stdout)
    fine = @timed taylor_reference(A,x;steps=20,tolerance=1e-10,order=24,verbose=true)
    reference = fine.value
    convergence = relative_error(coarse.value,reference)
    println("Fine reference ",fine.time," s rank ",maximum(tt_ranks(reference))," convergence ",convergence); flush(stdout)
    @assert convergence < ERROR_TARGET/100
    serialize(joinpath(data,"reference_$name.jls"),(;A,x,conserved,reference,convergence))
    open(joinpath(data,"reference_$name.txt"),"w") do io
        println(io,"case=$name\nMPO rank=$(maximum(tt_ranks(A)))\nsetup seconds=$(setup.time)")
        println(io,"fixture setup seconds=$(fixture_setup.time) (includes first-call compilation and TFD enrichment)")
        println(io,"coarse seconds=$(coarse.time)\nfine seconds=$(fine.time)\nreference rank=$(maximum(tt_ranks(reference)))\nconvergence=$convergence")
    end
    open(joinpath(data,"calibration_$name.csv"),"w") do io
        println(io,"case,method,parameter,atol,seconds,relative_error,density_error,rank,misses,discarded,intervals,maxrank")
        tamen_passed = false
        for tolerance in (1e-3,1e-4,1e-5,1e-6)
            run = @timed run_tamen(A,x;tolerance,conserved)
            err = relative_error(run.value.state,reference)
            density = norm(tt_density(run.value.state,startswith(name,"heom"))-tt_density(reference,startswith(name,"heom")))
            println(io,join((name,"tamen",tolerance,0,run.time,err,density,run.value.rank,0,0,run.value.intervals,0),',')); flush(io)
            println("tAMEn tol=$tolerance: ",run.time," s error ",err," rank ",run.value.rank); flush(stdout)
            if isfinite(err) && err <= ERROR_TARGET
                tamen_passed = true
                break
            end
        end
        tamen_passed || error("no tAMEn configuration reached the target for $name")
        bug_passed = false
        for maxrank in (8,16,32,64)
          for steps in (5,10,20,40,80,160)
            # A stepwise truncation budget comfortably below the global target.
            atol = min(1e-7,ERROR_TARGET/(100steps))
            run = @timed run_bug(problem;steps,atol,maxrank)
            err = relative_error(run.value.state,reference)
            density = norm(problem.observe(run.value.state).density-tt_density(reference,startswith(name,"heom")))
            println(io,join((name,"parallel_bug",steps,atol,run.time,err,density,run.value.rank,run.value.misses,run.value.discarded,0,maxrank),',')); flush(io)
            println("BUG steps=$steps atol=$atol cap=$maxrank: ",run.time," s error ",err," rank ",run.value.rank," misses ",run.value.misses); flush(stdout)
            if isfinite(err) && err <= ERROR_TARGET
                bug_passed = true
                break
            end
          end
          bug_passed && break
        end
        bug_passed || error("no BUG configuration reached the target for $name")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    for name in (isempty(ARGS) ? ["heom16"] : ARGS)
        calibrate(name)
    end
end
