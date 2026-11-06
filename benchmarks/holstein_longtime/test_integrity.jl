using Test, Serialization
include("driver.jl")
const HLT=HolsteinLongTime
@testset "Dependency, comparison and checkpoint integrity" begin
    cfg=HLT.load_config(joinpath(@__DIR__,"config.toml"))
    hashes=HLT.source_hashes()
    @test any(k->startswith(k,"HEOMKit/src/"),keys(hashes))
    @test any(k->startswith(k,"QFiND/src/"),keys(hashes))
    @test haskey(hashes,"TTDynamics/examples/holstein/Manifest.toml")
    mktempdir() do root
        names=[cfg["comparison"]["reference"];cfg["comparison"]["runs"]]
        for name in names
            mkpath(joinpath(root,name))
            runtime=Dict("julia"=>"test","julia_threads"=>8,"gc_threads"=>1,"cpu"=>"test",
                         "blas_threads"=>HLT.case_config(cfg,name)["blas_threads"])
            s=(;version=1,identity=HLT.run_identity(cfg,name),step=0,complete=false,state=zeros(10),
                sources=Dict("file"=>"hash"),runtime)
            serialize(joinpath(root,name,"latest.jls"),s)
        end
        @test !hasproperty(HLT.checked_saved(cfg,root,"tamen"),:state)
        @test isdefined(HLT,:check_compatible_runs)
        if isdefined(HLT,:check_compatible_runs)
            saved=Dict(name=>HLT.checked_saved(cfg,root,name) for name in names)
            @test isnothing(HLT.check_compatible_runs(cfg,saved))
            old=saved["parallel"]
            saved["parallel"]=merge(old,(;sources=Dict("file"=>"changed")))
            @test_throws ArgumentError HLT.check_compatible_runs(cfg,saved)
            saved["parallel"]=merge(old,(;runtime=merge(old.runtime,Dict("julia_threads"=>1))))
            @test_throws ArgumentError HLT.check_compatible_runs(cfg,saved)
            saved["parallel"]=merge(old,(;runtime=merge(old.runtime,Dict("blas_threads"=>8))))
            @test_throws ArgumentError HLT.check_compatible_runs(cfg,saved)
        end
    end
end

@testset "Bounded TT–TTN comparison contraction" begin
    @test isdefined(HLT,:comparison_overlap)
    if isdefined(HLT,:comparison_overlap)
        rng=HLT.MersenneTwister(441)
        tree=HLT.balanced_tree(4)
        leaves=Dict(node=>HLT.randn(rng,ComplexF64,2,2) for node in tree.leaf_order)
        cores=Dict(node=>HLT.randn(rng,ComplexF64,node==tree.root ? 1 : 2,2,2)
                   for node in tree.postorder if !HLT.TTNSolver._isleaf(tree,node))
        x=HLT.TTNTensor(tree,leaves,cores)
        bonds=[1,3,4,3,1]
        y=HLT.TTTensor([HLT.randn(rng,ComplexF64,bonds[i],2,bonds[i+1]) for i in 1:4])
        vector=vec(permutedims(reshape(HLT.tt_full(y),2,2,2,2),(4,3,2,1)))
        expected=HLT.dot(vector,vec(HLT.dense(x)))
        for bytes in (1,256,64*1024^2)
            @test HLT.comparison_overlap(y,x;workspace_bytes=bytes) ≈ expected rtol=1e-12
        end
        @test HLT.comparison_overlap(y,x;workspace_bytes=1) ≈ HLT.cross_overlap(y,x) rtol=1e-12
    end
end
