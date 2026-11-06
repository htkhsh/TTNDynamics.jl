using TTNSolver, LinearAlgebra, Serialization
root=joinpath(@__DIR__,"output")
open(joinpath(root,"allocation_state_differences.csv"),"w") do io
    println(io,"method,relative_state_difference,same_ranks")
    for method in ("augmented","parallel")
        x=deserialize(joinpath(root,"before","$(method)_1.jls"))
        y=deserialize(joinpath(root,"after","$(method)_1.jls"))
        delta=ttn_distance(x,y)/norm(x)
        ranks=tree_ranks(x)==tree_ranks(y)
        println(io,join((method,delta,ranks),','))
        @assert delta<1e-10 && ranks
    end
end
