include("common.jl")

checkpoint = deserialize(joinpath(@__DIR__,"data",first(ARGS),"latest.jls"))
p = example_problem()
x = checkpoint.state
println("Snapshot at ",checkpoint.time_fs," fs, ranks ",tt_ranks(x)); flush(stdout)
for tol in (1e-12,1e-8,1e-6)
    y = tt_round(x,tol)
    println("rounding=",tol," maxrank=",maximum(tt_ranks(y))," distance=",tt_distance(x,y)/norm(x),
            " trace_change=",abs(dot(p.problem.trace_observable,y)-dot(p.problem.trace_observable,x)))
    flush(stdout)
end
