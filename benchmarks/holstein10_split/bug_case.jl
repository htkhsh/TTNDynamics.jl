include("trajectory.jl")

# One short BUG trajectory, followed by comparison against the cached pilot
# reference. Keep the full trajectory logs and each completed case separately.
function short_bug_case(label,method,h,cap=32;rate=1e-3,duration=1.0)
    duration in (0.5,1.0) || error("cached reference supports 0.5 or 1 fs")
    x = trajectory([label,string(method),string(h),string(rate*h),string(cap),
                    "CN","3","3","5","1",string(duration)])
    dir = joinpath(@__DIR__,"data",label)
    p = example_problem()
    reference = deserialize(joinpath(@__DIR__,"data","pilot_reference.jls"))[round(Int,2duration)+1]
    state_error = relative_error(x,reference)
    rho = TTNDynamics.root_density_matrix(x,p.system)
    density_error = norm(rho-TTDynamics.root_density_matrix(reference,p.problem.system))
    checkpoint = deserialize(joinpath(dir,"latest.jls"))
    lines = readlines(joinpath(dir,"steps.csv"))
    headers = split(first(lines),',')
    column = findfirst(==("truncation_met"),headers)
    misses = count(line->split(line,',')[column]=="false",lines[2:end])
    row = (method,h,checkpoint.cumulative_seconds,maximum(tree_ranks(x)),
           state_error,density_error,misses)
    open(joinpath(dir,"summary.csv"),"w") do io
        println(io,"method,h,seconds,maxrank,state_error,density_error,truncation_misses")
        println(io,join(row,','))
    end
    println("SUMMARY ",join(row,',')); flush(stdout)
    return row
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)>=3 || error("supply label, method, h [rank cap] [tolerance rate]")
    short_bug_case(ARGS[1],Symbol(ARGS[2]),parse(Float64,ARGS[3]),
                   length(ARGS)>=4 ? parse(Int,ARGS[4]) : 32;
                   rate=length(ARGS)>=5 ? parse(Float64,ARGS[5]) : 1e-3)
end
