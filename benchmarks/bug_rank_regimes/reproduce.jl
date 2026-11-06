# Regenerate only the ignored inputs, preserving the published calibration
# CSVs. Run probe.jl first to regenerate the ignored reference caches.
include("run.jl")

function reproduce(name)
    dir = joinpath(@__DIR__,"data")
    data = deserialize(joinpath(dir,"$name.jls"))
    problem = rank_problem(data.spec)
    physical = state_vector(data.train)
    train = dense_to_train(physical,leafdims(problem.initial);tolerance=1e-5)
    tree = dense_to_tree(physical,leafdims(problem.initial),problem.initial.tree;tolerance=1e-5)
    @assert max(state_error(train,physical),state_error(tree,physical)) < 2e-5
    serialize(joinpath(dir,"input_$name.jls"),(;train,tree))
    measure(name)
end

if abspath(PROGRAM_FILE) == @__FILE__
    foreach(reproduce,isempty(ARGS) ? MEASURE_CASES : ARGS)
end
