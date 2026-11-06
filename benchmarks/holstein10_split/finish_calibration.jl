include("common.jl")

function finish_calibration()
    dir = joinpath(@__DIR__,"data")
    saved = deserialize(joinpath(dir,"calibration_latest.jls"))
    p = example_problem()
    refs = deserialize(joinpath(dir,"pilot_reference.jls"))
    h = saved.stepsize; steps = round(Int,1/h)
    x = saved.state; total = saved.total; misses = saved.misses
    isapprox(h,0.02) || error("expected the interrupted h=0.02 calibration")
    start = round(Int,saved.time_fs/h)
    open(joinpath(dir,"calibration_metadata.txt"),"w") do io
        println(io,"parallel: maxrank=32, per-step atol=1e-3*h, threaded=true, BLAS=1")
        println(io,"resumed h=",h," at t=",saved.time_fs,"; warm-up excluded")
        println(io,"short tAMEn: Chebyshev12, h=0.5, BLAS=8, three repeated 0–1 fs runs")
        source_metadata(io)
    end
    tree_step(p,x,h;method=:parallel,atol=1e-3*h,maxrank=32) # compile; discard
    GC.gc()
    for j in start+1:steps
        m = @timed tree_step(p,x,h;method=:parallel,atol=1e-3*h,maxrank=32)
        x,d = m.value; total += m.time; misses += !d.truncation.tolerance_met
        println("RESUMED parallel t=",j*h," seconds=",m.time," total=",total); flush(stdout)
        serialize(joinpath(dir,"calibration_latest.jls"),(;state=x,time_fs=j*h,stepsize=h,total,misses))
    end
    err = relative_error(x,last(refs))
    de = norm(TTNDynamics.root_density_matrix(x,p.system)-TTDynamics.root_density_matrix(last(refs),p.problem.system))
    row = (:parallel,h,total,maximum(tree_ranks(x)),err,de,misses)
    open(joinpath(dir,"calibration.csv"),"a") do io
        println(io,join(row,','))
    end
    serialize(joinpath(dir,"parallel_h002_final.jls"),x)
    println("COMPLETE ",join(row,',')); flush(stdout)

    BLAS.set_num_threads(8)
    open(joinpath(dir,"short_tamen.csv"),"w") do io
        println(io,"tolerance,sweeps,local_iters,repeat,seconds,maxrank,state_error,density_error,intervals")
        for (tol,sweeps,iters) in ((5e-3,3,5),(1e-3,5,10))
            function propagate()
                y = p.train; w = TAMEnWorkspace(ComplexF64); intervals = 0
                for _ in 1:2
                    y,n = train_step(p,y,.5;scheme="cheb",points=12,tolerance=tol,sweeps,local_iters=iters,workspace=w)
                    intervals += n
                end
                return y,intervals
            end
            Random.seed!(20261105); propagate()
            for rep in 1:3
                Random.seed!(20261105); GC.gc()
                m = @timed propagate(); y,n = m.value
                err = relative_error(y,last(refs))
                de = norm(TTDynamics.root_density_matrix(y,p.problem.system)-TTDynamics.root_density_matrix(last(refs),p.problem.system))
                row = (tol,sweeps,iters,rep,m.time,maximum(tt_ranks(y)),err,de,n)
                println(io,join(row,',')); flush(io)
                println(join(row,',')); flush(stdout)
            end
        end
    end
end

finish_calibration()
