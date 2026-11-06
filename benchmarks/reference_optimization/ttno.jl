using TTNSolver, LinearAlgebra, Serialization, TOML, Statistics
const TS=TTNSolver
BLAS.set_num_threads(1)

# Conservative size of the largest exact application QR input, not a peak RSS
# estimate (other arrays, QR work and the following residual sum also live).
function application_elements(W,x)
    ranks=zeros(Int,length(x.tree.children));largest=0
    wr,xr=tree_ranks(W),tree_ranks(x)
    for node in x.tree.postorder
        rows=if TS._isleaf(x.tree,node)
            size(x.leaves[node],1)
        else
            l,r=x.tree.children[node];ranks[l]*ranks[r]
        end
        columns=wr[node]*xr[node]
        largest=max(largest,rows*columns)
        ranks[node]=min(rows,columns)
    end
    largest
end

function main()
    fixture=get(ENV,"REFERENCE_OPT_RESIDUAL","/tmp/ttn-residual-comparison-states/fixture.bin")
    A,x,b,budget,cap=deserialize(fixture)
    out=joinpath(@__DIR__,"data");mkpath(out)
    results=Dict[]
    reference,refbound,_=TS._tamen_residual(A,x,b,budget,cap)
    for kind in ("sop","ttno_structural","ttno_compressed")
        build()=kind=="sop" ? A : TreeTensorOperator(A,x.tree;
            atol=kind=="ttno_compressed" ? budget/(100norm(x)) : 0.)
        build() # compilation is excluded from the reported construction cost
        built=@timed build()
        W=built.value
        certify()=TS._tamen_residual(W,x,b,budget,cap;threaded=true)
        certify()
        for repeat in 1:3
            GC.gc();m=@timed certify()
            r,bound,drop=m.value
            difference=ttn_distance(r,reference)
            @assert difference<=drop+refbound+1e-10norm(b)
            row=Dict("backend"=>kind,"repeat"=>repeat,"seconds"=>m.time,"bytes"=>m.bytes,
                "build_seconds"=>built.time,"relative_residual_bound"=>bound/norm(b),
                "relative_difference"=>difference/norm(b),
                "operator_ranks"=>kind=="sop" ? Int[] : tree_ranks(W),
                "operator_discarded_bound"=>kind=="sop" ? 0. : W.discarded_bound)
            push!(results,row);println(row);flush(stdout)
        end
    end
    open(joinpath(out,"ttno.toml"),"w") do io
        TOML.print(io,Dict("fixture"=>fixture,"terms"=>length(A.terms),
            "state_ranks"=>tree_ranks(x),"dimensions"=>leafdims(x),
            "budget"=>budget,"residual_maxrank"=>cap,
            "threads"=>Threads.nthreads(:default),"blas_threads"=>BLAS.get_num_threads(),"runs"=>results))
    end
    problem=joinpath(out,"problem.jls")
    state=joinpath(@__DIR__,"../holstein_ps2/output/holstein5_eager/ps2_five_fs/rep1/time_3.jls")
    if isfile(problem) && isfile(state)
        p=deserialize(problem);mature=deserialize(state)
        TreeTensorOperator(p.operator,p.tree.tree) # warm construction
        m=@timed TreeTensorOperator(p.operator,p.tree.tree)
        W=m.value
        report=Dict("terms"=>length(p.operator.terms),"ranks"=>tree_ranks(W),
            "build_seconds"=>m.time,"build_bytes"=>m.bytes,
            "initial_qr_elements"=>application_elements(W,p.tree),
            "mature_qr_elements"=>application_elements(W,mature),
            "note"=>"QR input size estimate; high-rank full action not run")
        grid=TS._tamen_grid(8,0.5,:cheb)
        st=TS._time_lift(mature,ones(8))
        rhs=TS._time_lift(mature,grid.d0)
        stA=TS._spacetime_operator(p.operator,grid)
        opbudget=1e-4*norm(rhs)/2000/max(norm(st),1.0)
        TreeTensorOperator(stA,st.tree;atol=opbudget)
        cm=@timed TreeTensorOperator(stA,st.tree;atol=opbudget)
        report["spacetime_compressed_ranks"]=tree_ranks(cm.value)
        report["spacetime_operator_budget"]=opbudget
        report["spacetime_operator_discarded_bound"]=cm.value.discarded_bound
        report["spacetime_build_seconds"]=cm.time
        report["spacetime_mature_qr_elements"]=application_elements(cm.value,st)
        open(joinpath(out,"holstein_ttno.toml"),"w") do io;TOML.print(io,report);end
        println(report)
    end
end
main()
