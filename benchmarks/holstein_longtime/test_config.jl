using Test, TOML

@testset "Long-time comparison configuration" begin
    @test isfile(joinpath(@__DIR__,"configuration.jl"))
    if isfile(joinpath(@__DIR__,"configuration.jl"))
        include("configuration.jl")
        cfg = load_config(joinpath(@__DIR__,"config.toml"))
        @test cfg["model"]["frequency_cm"] == 800
        @test cfg["model"]["hopping_cm"] == 400
        @test cfg["model"]["sites"] == 10
        @test cfg["time"]["end_fs"] == 100
        @test case_config(cfg,"augmented_hhalf")["dt_fs"] == 0.05
        @test case_config(cfg,"parallel_rank")["maxrank"] == 128
        @test !haskey(case_config(cfg,"tamen"),"maxrank")
        @test grid_count(0.3,0.1) == 3
        @test_throws ArgumentError grid_count(0.3,0.2)
        @test_throws ArgumentError grid_count(1.0,0.0)
        @test_throws ArgumentError grid_count(Inf,0.1)
        c=deepcopy(cfg); c["runs"]["tamen"]["typo"]=1
        @test_throws ArgumentError validate_config(c)
        c=deepcopy(cfg); c["runs"]["tamen"]["maxrank"]=32
        @test_throws ArgumentError validate_config(c)
        c=deepcopy(cfg); c["runs"]["parallel"]["maxrank"]=9
        @test_throws ArgumentError validate_config(c)
        c=deepcopy(cfg); c["runs"]["augmented"]["dt_fs"]=0.3
        @test_throws ArgumentError validate_config(c)
        c=deepcopy(cfg); c["model"]["frequency_cm"]=-1
        @test_throws ArgumentError validate_config(c)
        c=deepcopy(cfg); c["runs"]["tamen"]["extends"]="tamen"
        @test_throws ArgumentError validate_config(c)
        c=deepcopy(cfg); c["runs"]["reference_refined"]=Dict("extends"=>"reference")
        @test_throws ArgumentError validate_config(c)
        @test canonical_hash(Dict("a"=>1,"b"=>2)) == canonical_hash(Dict("b"=>2,"a"=>1))
        @test run_identity(cfg,"tamen") != run_identity(cfg,"reference")
        c=deepcopy(cfg); c["model"]["frequency_cm"]=801
        @test run_identity(c,"tamen") != run_identity(cfg,"tamen")
        c=deepcopy(cfg); c["runs"]["parallel"]["maxrank"]=128
        @test run_identity(c,"tamen") == run_identity(cfg,"tamen")
    end
end
