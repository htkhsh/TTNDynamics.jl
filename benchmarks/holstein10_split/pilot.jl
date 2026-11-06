include("common.jl")

function pilot()
    Random.seed!(20261105)
    dir = joinpath(@__DIR__,"data"); mkpath(dir)
    p = example_problem()
    open(joinpath(dir,"model.txt"),"w") do io
        println(io,"config: ",p.config,"\ndecomposition: ",p.decomposition)
        println(io,"layout: split\ninitial condition: localized, site 1\nrequested duration fs: ",TARGET_TIME_FS)
        println(io,"dimensions: ",leafdims(p.tree),"\nfull dimension: ",prod(big.(leafdims(p.tree))))
        println(io,"leaves: ",length(p.tree.tree.leaf_order),"\nnodes: ",length(p.tree.tree.children))
        println(io,"MPO ranks: ",tt_ranks(p.problem.liouvillian))
        println(io,"generator relative Frobenius difference: ",generator_agreement(p))
        source_metadata(io)
    end
    println("MODEL READY"); flush(stdout)
    # Pilot only; neither these compilation times nor its short duration are
    # presented as the requested 100 fs benchmark.
    ref = p.train; work = TAMEnWorkspace(ComplexF64)
    references = TTTensor[p.train]
    for j in 1:2
        m = @timed train_step(p,ref,0.5;scheme="cheb",points=12,tolerance=1e-5,sweeps=20,local_iters=20,workspace=work)
        ref = m.value[1]; push!(references,ref)
        serialize(joinpath(dir,"pilot_reference.jls"),references)
        println("REF t=",0.5j," seconds=",m.time," rank=",maximum(tt_ranks(ref))," intervals=",m.value[2]); flush(stdout)
    end
    serialize(joinpath(dir,"pilot_reference.jls"),references)
    open(joinpath(dir,"pilot.csv"),"w") do io
        println(io,"method,dt,time_fs,seconds,rank,error,density_error,truncation_met")
        for method in (:tamen,:augmented,:parallel)
            x = method==:tamen ? p.train : p.tree
            w = TAMEnWorkspace(ComplexF64)
            Random.seed!(20261105)
            for j in 1:2
                m = @timed (method==:tamen ? train_step(p,x,0.5;workspace=w) : tree_step(p,x,0.5;method))
                x,d = m.value
                rank = method==:tamen ? maximum(tt_ranks(x)) : maximum(tree_ranks(x))
                err = relative_error(x,references[j+1])
                rho = method==:tamen ? TTDynamics.root_density_matrix(x,p.problem.system) : TTNDynamics.root_density_matrix(x,p.system)
                de = norm(rho-TTDynamics.root_density_matrix(references[j+1],p.problem.system))
                met = method==:tamen ? true : d.truncation.tolerance_met
                println(io,join((method,.5,.5j,m.time,rank,err,de,met),',')); flush(io)
                println(method," t=",.5j," seconds=",m.time," rank=",rank," error=",err); flush(stdout)
            end
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    pilot()
end
