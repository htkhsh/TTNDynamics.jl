isdefined(@__MODULE__,:rank_problem) || include("common.jl")

function scan(name;resume=false)
    dir = joinpath(@__DIR__,"data")
    data = deserialize(joinpath(dir,"$name.jls"))
    (;spec,reference,A) = data
    interval = get(spec,:interval,INTERVAL)
    p = rank_problem(spec)
    # Both inputs approximate the same independently prepared physical state.
    # Compression errors are included in the final comparison to the reference.
    physical = state_vector(data.train)
    train = dense_to_train(physical,leafdims(p.initial);tolerance=1e-5)
    tree = dense_to_tree(physical,leafdims(p.initial),p.initial.tree;tolerance=1e-5)
    @assert max(state_error(train,physical),state_error(tree,physical)) < 2e-5
    conserved = product_to_tt(p.conserved)
    println("Scanning $name input ranks TT=",maximum(tt_ranks(train))," TTN=",maximum(tree_ranks(tree))); flush(stdout)
    serialize(joinpath(dir,"input_$name.jls"),(;train,tree))
    path = joinpath(dir,"scan_$name.csv")
    previous = resume && isfile(path) ? [split(line,',') for line in readlines(path)[2:end]] : []
    passed(method) = any(r -> r[2]==string(method) && parse(Float64,r[7])<=ERROR_TARGET,previous)
    open(path,resume ? "a" : "w") do io
        resume || println(io,"case,method,tolerance,steps,maxrank,seconds,error,rank,storage,intervals,misses")
      if !passed("tamen")
        accepted = false
        for tolerance in (1e-3,1e-4,1e-5)
            println("$name tAMEn tol=$tolerance"); flush(stdout)
            m = @timed run_tamen(A,train;tolerance,conserved,final_time=interval)
            r = m.value; err = state_error(r.state,reference)
            println(io,join((name,"tamen",tolerance,0,0,m.time,err,r.rank,storage(r.state),r.intervals,0),',')); flush(io)
            println("  ",m.time," s error ",err," rank ",r.rank); flush(stdout)
            if err <= ERROR_TARGET
                accepted = true
                break
            end
        end
        accepted || error("tAMEn did not reach target for $name")
      end
        for method in (:augmented,:parallel)
          passed(method) && continue
          accepted = false
          # Reserve space for the conserved vector as well as the state.
          for cap in unique([max(16,maximum(tree_ranks(tree))+1),max(32,2maximum(tree_ranks(tree))+1)])
            for steps in (1,2,4,8,16,32,64,128)
                any(r -> r[2]==string(method) && parse(Int,r[4])==steps && parse(Int,r[5])==cap,previous) && continue
                println("$name $method steps=$steps cap=$cap"); flush(stdout)
                m = @timed evolve_bug(p,tree;method,steps,maxrank=cap,atol=1e-8,interval)
                r = m.value; err = state_error(r.state,reference)
                println(io,join((name,method,0,steps,cap,m.time,err,r.rank,storage(r.state),0,r.misses),',')); flush(io)
                println("  ",m.time," s error ",err," rank ",r.rank); flush(stdout)
                if err <= ERROR_TARGET
                    accepted = true
                    break
                end
                # A rank cap causes an error floor; try more rank before many
                # expensive time refinements at an already stalled error.
                r.misses > 0 && steps >= 8 && err > 3e-4 && break
            end
            accepted && break
          end
          accepted || error("$method did not reach target for $name")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    for name in (isempty(ARGS) ? ["weak6","strong6","system8"] : ARGS)
        scan(name)
    end
end
