include("common.jl")

function reference_check()
    dir = joinpath(@__DIR__,"data")
    target = joinpath(dir,"reference_check.csv")
    ispath(target) && error("reference check already exists")
    Random.seed!(20261105)
    p = example_problem()
    previous = deserialize(joinpath(dir,"pilot_reference.jls"))
    y = p.train; work = TAMEnWorkspace(ComplexF64)
    refined = TTTensor[p.train]
    open(joinpath(dir,"reference_check_metadata.txt"),"w") do io
        println(io,"Refinement: Chebyshev12, h=0.5 fs, tolerance=1e-6, sweeps=20, local_iters=20")
        println(io,"Compared with cached pilot reference at tolerance=1e-5; not a timing benchmark.")
        source_metadata(io)
    end
    open(target,"w") do io
        println(io,"time_fs,seconds,maxrank,state_relative_difference,density_absolute_difference")
        for j in 1:2
            m = @timed train_step(p,y,.5;scheme="cheb",points=12,tolerance=1e-6,
                                  sweeps=20,local_iters=20,workspace=work)
            y = m.value[1]; push!(refined,y)
            delta = relative_error(y,previous[j+1])
            de = norm(TTDynamics.root_density_matrix(y,p.problem.system)-
                      TTDynamics.root_density_matrix(previous[j+1],p.problem.system))
            row = (.5j,m.time,maximum(tt_ranks(y)),delta,de)
            println(io,join(row,',')); flush(io)
            serialize(joinpath(dir,"reference_check.jls"),refined)
            println(join(row,',')); flush(stdout)
        end
    end
end

reference_check()
