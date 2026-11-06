include("common.jl")

# Compare saved checkpoints without expanding the hierarchy tensor.
function evaluate(reference_label,labels;allow_partial=false)
    base = joinpath(@__DIR__,"data")
    p = example_problem()
    refdir = joinpath(base,reference_label)
    if !allow_partial
        for label in [reference_label;labels]
            for t in 0:10:Int(TARGET_TIME_FS)
                isfile(joinpath(base,label,"state_$(1000t).jls")) ||
                    error("incomplete 100 fs trajectory: $label, missing $t fs")
            end
        end
    end
    open(joinpath(base,"errors_vs_$(reference_label).csv"),"w") do io
        println(io,"method,reference,time_fs,state_relative_error,density_absolute_error,density_relative_error,population_max_error")
        for label in labels
            dir = joinpath(base,label)
            files = sort(filter(f->startswith(f,"state_") && endswith(f,".jls"),readdir(dir)))
            for file in files
                isfile(joinpath(refdir,file)) || continue
                a,b = deserialize(joinpath(dir,file)),deserialize(joinpath(refdir,file))
                a.time_fs==b.time_fs || error("checkpoint time mismatch")
                b.state isa TTTensor || error("reference must be a tensor train")
                rho = a.state isa TTTensor ? TTDynamics.root_density_matrix(a.state,p.problem.system) :
                                            TTNDynamics.root_density_matrix(a.state,p.system)
                rref = TTDynamics.root_density_matrix(b.state,p.problem.system)
                err = relative_error(a.state,b.state)
                de = norm(rho-rref)
                row = (label,reference_label,a.time_fs,err,de,de/norm(rref),maximum(abs,diag(rho-rref)))
                println(io,join(row,',')); flush(io)
                println(join(row,',')); flush(stdout)
            end
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)>=2 || error("supply reference label and compared run labels")
    evaluate(first(ARGS),ARGS[2:end])
end
