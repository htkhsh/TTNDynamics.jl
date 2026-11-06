include("fixtures.jl")
BLAS.get_num_threads() == 1 || error("set OPENBLAS_NUM_THREADS=1")
for name in (isempty(ARGS) ? LARGE_CASES : ARGS)
    println("Preparing $name"); flush(stdout)
    setup = @timed large_problem(name)
    p = setup.value
    seed = @timed prepared_snapshot(p)
    x,history = seed.value
    println((;name,setup=setup.time,preevolution=seed.time,case_metadata(p,x)...,
        pretruncation_misses=count(d -> !d.truncation.tolerance_met,history))); flush(stdout)
    results = Dict{Bool,Any}()
    for threaded in (false,true)
        GC.gc()
        m = @timed measured_step(p,x;threaded)
        results[threaded] = m.value[1]
        d = m.value[2]
        println((;threaded,seconds=m.time,GB=m.bytes/1e9,rank=maximum(d.ranks),
            augmented=maximum(d.augmented_ranks),max_local_dimension=d.max_local_dimension,
            local_solves=d.local_solves,skipped=d.skipped_leaf_solves,
            truncation_met=d.truncation.tolerance_met)); flush(stdout)
    end
    println("Serial/threaded distance: ",ttn_distance(results[false],results[true])); flush(stdout)
end
