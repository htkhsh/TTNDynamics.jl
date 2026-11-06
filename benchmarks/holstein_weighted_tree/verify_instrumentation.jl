using Test, LinearAlgebra, TTNSolver
include("../augmented_threads/stages.jl")
include("../parallel_bug_optimization/stages.jl")

@testset "Existing stage instrumentation supports both truncations" begin
    x=product_state(balanced_tree(2),[ComplexF64[1,0],ComplexF64[0.6,0.8]])
    op=SumOfProducts([2,2],[ProductTerm(-im,Dict(1=>ComplexF64[0 1;1 0]))])
    for truncation in (:local,:weighted)
        fill!(TimedAugmented.STAGES,0)
        a,_=augmented_bug_step(x,op,0.02;truncation,threaded=false)
        b,_=TimedAugmented.augmented_bug_step(x,op,0.02;truncation,threaded=false)
        @test ttn_distance(a,b)<1e-12
        @test all(>(0),TimedAugmented.STAGES)
        @test sum(TimedAugmented.NODE_SOLVES)>0
    end
    current=read(joinpath(dirname(pathof(TTNSolver)),"parallel_bug.jl"),String)
    instrumented=timed_module(:CurrentParallelTiming,current)
    for truncation in (:local,:weighted)
        a,_=parallel_bug_step(x,op,0.02;truncation,threaded=false)
        b,_=Base.invokelatest(instrumented.parallel_bug_step,x,op,0.02;truncation,threaded=false)
        @test ttn_distance(a,b)<1e-12
        @test all(>(0),instrumented.STAGES)
    end
    old_source=read(joinpath(@__DIR__,"../parallel_bug_optimization/reference/parallel_bug_before.jl"),String)
    old=timed_module(:OldParallelTiming,old_source)
    a,_=parallel_bug_step(x,op,0.02;threaded=false)
    b,_=Base.invokelatest(old.parallel_bug_step,x,op,0.02;threaded=false)
    @test ttn_distance(a,b)<1e-12
    @test all(>(0),old.STAGES)
end
