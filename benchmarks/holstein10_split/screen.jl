include("common.jl")

function screen_train()
    p = example_problem()
    refs = deserialize(joinpath(@__DIR__,"data","pilot_reference.jls"))
    open(joinpath(@__DIR__,"data","train_screen.csv"),"w") do io
        println(io,"tolerance,sweeps,local_iters,seconds,maxrank,state_error,density_error,intervals")
        for (tol,sweeps,iters) in ((5e-3,3,5),(1e-3,5,10),(1e-4,10,20))
            x = p.train; w = TAMEnWorkspace(ComplexF64); total=0.; intervals=0
            train_step(p,x,.5;scheme="cheb",points=12,tolerance=tol,sweeps,local_iters=iters)
            for j in 1:2
                m = @timed train_step(p,x,.5;scheme="cheb",points=12,tolerance=tol,sweeps,
                                     local_iters=iters,workspace=w)
                x,n = m.value; total += m.time; intervals += n
            end
            err = relative_error(x,last(refs))
            de = norm(TTDynamics.root_density_matrix(x,p.problem.system)-TTDynamics.root_density_matrix(last(refs),p.problem.system))
            row = (tol,sweeps,iters,total,maximum(tt_ranks(x)),err,de,intervals)
            println(io,join(row,',')); flush(io)
            println(join(row,',')); flush(stdout)
        end
    end
end

screen_train()
include("trajectory.jl")
trajectory(["tamen_cheb_blas1","tamen","0.5","0.001","128","cheb","12","5","10","1"])
