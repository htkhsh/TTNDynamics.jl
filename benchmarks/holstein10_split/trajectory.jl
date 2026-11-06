isdefined(@__MODULE__,:example_problem) || include("common.jl")

# Arguments: unique label, method, h, tolerance, rank cap, temporal scheme,
# temporal points, sweeps, local iterations, BLAS threads, duration (default 100 fs).
function trajectory(args=ARGS)
    length(args)>=2 || error("supply a unique run label and method")
    label,method = args[1],Symbol(args[2])
    method in (:tamen,:augmented,:parallel) || error("unknown method")
    occursin(r"^[a-zA-Z0-9_-]+$",label) || error("invalid run label")
    arg(i,default,T) = length(args)>=i ? parse(T,args[i]) : default
    h = arg(3,0.5,Float64); tolerance = arg(4,5e-3,Float64)
    cap = arg(5,128,Int); scheme = length(args)>=6 ? args[6] : "CN"
    points = arg(7,3,Int); sweeps = arg(8,3,Int); local_iters = arg(9,5,Int)
    blas = arg(10,1,Int); duration = arg(11,TARGET_TIME_FS,Float64)
    steps = round(Int,duration/h)
    isapprox(steps*h,duration;atol=1e-10,rtol=0) || error("nonintegral step count")
    dir = joinpath(@__DIR__,"data",label)
    ispath(dir) && error("run directory already exists: $dir")
    mkpath(dir)
    BLAS.set_num_threads(blas)
    Random.seed!(20261105)
    p = example_problem()
    options = (;label,method,h,tolerance,cap,scheme,points,sweeps,local_iters,blas,duration)
    open(joinpath(dir,"metadata.txt"),"w") do io
        println(io,"options: ",options,"\nconfig: ",p.config,"\ndecomposition: ",p.decomposition)
        println(io,"initial_condition: localized, site 1\nlayout: split\nstarted: ",now())
        source_metadata(io)
    end
    advance(x,w) = method==:tamen ? train_step(p,x,h;scheme,points,tolerance,sweeps,local_iters,workspace=w) :
        tree_step(p,x,h;method,atol=tolerance,maxrank=cap)
    x = method==:tamen ? p.train : p.tree
    println("WARMUP ",label); flush(stdout)
    warm = @timed advance(deepcopy(x),TAMEnWorkspace(ComplexF64))
    println("WARMUP DONE seconds=",warm.time); flush(stdout)
    warm = nothing; GC.gc()
    Random.seed!(20261105)
    work = TAMEnWorkspace(ComplexF64)
    cumulative = 0.0
    function checkpoint(x,j)
        target = joinpath(dir,"latest.jls")
        serialize(target*".tmp",(;state=x,step=j,time_fs=j*h,cumulative_seconds=cumulative,options))
        mv(target*".tmp",target;force=true)
        if j==0 || isapprox(j*h/10,round(j*h/10);atol=1e-9) || j==steps
            cp(target,joinpath(dir,"state_$(round(Int,j*h*1000)).jls");force=true)
        end
    end
    open(joinpath(dir,"steps.csv"),"w") do io
        names = ["step","time_fs","seconds","cumulative_seconds","gc_seconds","allocated_bytes",
                 "maxrank","stored_elements","norm","trace_error","hermiticity_error","truncation_met",
                 "discarded_bound","intervals"]
        append!(names,["rho_$(a)_$(b)_$part" for b in 1:10 for a in 1:10 for part in ("re","im")])
        println(io,join(names,','))
        function record(j,m,d)
            rho = method==:tamen ? TTDynamics.root_density_matrix(x,p.problem.system) :
                                  TTNDynamics.root_density_matrix(x,p.system)
            ranks = method==:tamen ? tt_ranks(x) : tree_ranks(x)
            elements = method==:tamen ? sum(length,x.cores) :
                sum(length,values(x.leaves))+sum(length,values(x.cores))
            met = method==:tamen || j==0 ? "NA" : d.truncation.tolerance_met
            discarded = method==:tamen || j==0 ? "NA" : d.truncation.discarded_bound
            intervals = method==:tamen && j>0 ? d : 0
            row = Any[j,j*h,m.time,cumulative,m.gctime,m.bytes,maximum(ranks),elements,norm(x),
                      abs(tr(rho)-1),norm(rho-rho'),met,discarded,intervals]
            append!(row,[v for z in vec(rho) for v in (real(z),imag(z))])
            println(io,join(row,',')); flush(io)
            println(label," t=",j*h," / ",duration," seconds=",m.time,
                    " cumulative=",cumulative," rank=",maximum(ranks)," truncation_met=",met)
            flush(stdout)
        end
        record(0,(;time=0.,gctime=0.,bytes=0),nothing); checkpoint(x,0)
        for j in 1:steps
            m = @timed advance(x,work)
            x,d = m.value
            cumulative += m.time
            record(j,m,d)
            (isapprox(j*h/0.5,round(j*h/0.5);atol=1e-9) || j==steps) && checkpoint(x,j)
        end
    end
    println("COMPLETE ",label," propagation_seconds=",cumulative)
    return x
end

if abspath(PROGRAM_FILE)==@__FILE__
    trajectory()
end
