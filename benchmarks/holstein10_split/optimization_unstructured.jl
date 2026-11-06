include("optimization_probe.jl")

# Characterize setup cost when all factors differ. This is a local kernel
# benchmark, not a claim about complete BUG trajectory performance.
function unstructured_probe()
    path = joinpath(@__DIR__,"data","optimization_unstructured.csv")
    open(path,"w") do io
        println(io,"rank,terms,implementation,kind,seconds,allocated_bytes")
        rng = MersenneTwister(815)
        for n in (2,8,16), count in (2,32)
            op = SumOfProducts([2],[ProductTerm(randn(rng,ComplexF64),Dict()) for _ in 1:count])
            B,L,R = [[randn(rng,ComplexF64,n,n) for _ in 1:count] for _ in 1:3]
            x = randn(rng,ComplexF64,n,n,n)
            for (label,mod) in (("baseline",BaselineBUG),("optimized",TTNSolver))
                solve = () -> begin
                    action = mod._core_rhs_workspace(op,B,L,R,x)
                    for _ in 1:10
                        action(x)
                    end
                end
                block = () -> mod._parallel_core_block(op,B,L,R,x,size(x))
                for (kind,run) in (("setup_plus_10_actions",solve),("one_block",block))
                    run()
                    measurements = [@timed(run()) for _ in 1:21]
                    println(io,join((n,count,label,kind,median(m.time for m in measurements),
                                    median(m.bytes for m in measurements)),','))
                end
            end
        end
    end
end

unstructured_probe()
