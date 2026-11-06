using Test
@testset "Long-time benchmark driver exists" begin
    @test isfile(joinpath(@__DIR__,"driver.jl"))
end
include("driver.jl")
using .HolsteinLongTime
const HLT = HolsteinLongTime
using LinearAlgebra, Serialization, TOML

function small_config()
    cfg = HLT.load_config(joinpath(@__DIR__,"config.toml"))
    cfg["model"]["sites"]=2; cfg["model"]["hierarchy_local_size"]=2
    cfg["time"] = Dict("end_fs"=>0.2,"observation_fs"=>0.1,"checkpoint_fs"=>0.1)
    cfg["runs"] = Dict(name=>HLT.case_config(cfg,name) for name in
        ("tamen","augmented","parallel","reference","reference_refined"))
    for (name,s) in cfg["runs"]
        s["blas_threads"]=1
        s["dt_fs"] = name=="reference_refined" ? 0.05 : 0.1
        if s["method"]=="tamen"
            s["tolerance"]=1e-8; s["points"]=8; s["sweeps"]=20; s["local_iters"]=30
        else
            s["maxrank"]=16; s["truncation_rate"]=1e-9
        end
    end
    return HLT.validate_config(cfg)
end

@testset "Weighted defaults and checkpoint identity" begin
    cfg=HLT.load_config(joinpath(@__DIR__,"config.toml"))
    legacy=deepcopy(cfg)
    pop!(legacy["runs"]["augmented"],"truncation",nothing)
    @test HLT.case_config(legacy,"augmented")["truncation"]=="weighted"
    @test HLT.case_config(legacy,"parallel")["truncation"]=="weighted"
    @test HLT.run_identity(legacy,"augmented")==HLT.run_identity(cfg,"augmented")
    local_cfg=deepcopy(cfg); local_cfg["runs"]["augmented"]["truncation"]="local"
    @test HLT.validate_config(local_cfg)===local_cfg
    @test HLT.case_config(local_cfg,"parallel")["truncation"]=="local"
    @test HLT.run_identity(local_cfg,"augmented")!=HLT.run_identity(cfg,"augmented")
    invalid=deepcopy(cfg); invalid["runs"]["augmented"]["truncation"]="unknown"
    @test_throws ArgumentError HLT.validate_config(invalid)
end

@testset "Brownian model and resumable trajectories" begin
    cfg = small_config()
    p = HLT.comparison_problem(cfg)
    invalid=HLT.case_config(cfg,"augmented"); invalid["truncation"]="unknown"
    @test_throws ArgumentError HLT.advance(p,p.tree,invalid,nothing)
    @test p.config.hopping_cm == 400
    @test p.config.brownian_frequency_cm == 800
    @test p.config.initial_condition == :localized
    @test p.problem.system.H_sys[1,2] ≈ -400*HLT.HolsteinExample.icm2ifs
    @test p.system.system_layout == :split
    @test HLT.generator_agreement(p) < 1e-10
    @test HLT.TTNDynamics.root_density_matrix(p.tree,p.system) ≈ ComplexF64[1 0;0 0]
    expected = exp(0.2*HLT.dense(p.operator))*vec(HLT.dense(p.tree))
    mktempdir() do root
        for name in ("tamen","augmented","parallel")
            continuous = HLT.run_case(cfg,name,joinpath(root,"continuous");wall_seconds=Inf)
            paused = HLT.run_case(cfg,name,joinpath(root,"resumed");wall_seconds=Inf,max_steps=1)
            @test paused.step == 1
            # Simulate a crash after CSV append, before the next checkpoint.
            for file in ("steps.csv","observables.csv")
                open(joinpath(root,"resumed",name,file),"a") do io
                    println(io,"uncommitted,incomplete,row")
                end
            end
            resumed = HLT.run_case(cfg,name,joinpath(root,"resumed");resume=true,wall_seconds=Inf)
            @test resumed.step == 2
            delta = name=="tamen" ? HLT.tt_distance(continuous.state,resumed.state) : HLT.ttn_distance(continuous.state,resumed.state)
            @test delta < 1e-10
            vector = name=="tamen" ? vec(permutedims(reshape(HLT.tt_full(resumed.state),Tuple(reverse(HLT.tt_dims(resumed.state)))),
                         reverse(1:length(HLT.tt_dims(resumed.state))))) : vec(HLT.dense(resumed.state))
            @test norm(vector-expected)/norm(expected) < (name=="tamen" ? 1e-6 : 1e-3)
            @test length(readlines(joinpath(root,"resumed",name,"steps.csv"))) == 4
            @test !occursin("uncommitted",read(joinpath(root,"resumed",name,"observables.csv"),String))
            @test_throws ArgumentError HLT.run_case(cfg,name,joinpath(root,"resumed"))
            changed=deepcopy(cfg); changed["model"]["frequency_cm"]=801
            @test_throws ArgumentError HLT.run_case(changed,name,joinpath(root,"resumed");resume=true)
            again=HLT.run_case(cfg,name,joinpath(root,"resumed");resume=true,wall_seconds=0)
            @test again.offsets==resumed.offsets
            checkpoint=joinpath(root,"resumed",name,"latest.jls")
            serialize(checkpoint,merge(again,(;sources=Dict("changed"=>"hash"))))
            @test_throws ArgumentError HLT.run_case(cfg,name,joinpath(root,"resumed");resume=true)
            serialize(checkpoint,again)
        end
        HLT.run_case(cfg,"reference",joinpath(root,"continuous");wall_seconds=Inf)
        @test_throws ArgumentError HLT.compare_runs(cfg,joinpath(root,"resumed"))
        # A reference alone permits raw differences, but no accuracy-qualified speed ratios.
        report = HLT.compare_runs(cfg,joinpath(root,"continuous"))
        @test all(r->r.reference_status=="unchecked" && isnothing(r.speedup_vs_tamen),report)
        HLT.run_case(cfg,"reference_refined",joinpath(root,"continuous");wall_seconds=Inf)
        report = HLT.compare_runs(cfg,joinpath(root,"continuous"))
        @test all(r->r.reference_status=="checked",report)
        @test all(r->r.complete,report)
        stopped=HLT.run_case(cfg,"tamen",joinpath(root,"zero_budget");wall_seconds=0)
        @test stopped.step==0
        HLT.run_case(cfg,"reference",joinpath(root,"zero_budget");wall_seconds=0)
        @test_throws ArgumentError HLT.compare_runs(cfg,joinpath(root,"zero_budget");names=["tamen"])
        partial=HLT.compare_runs(cfg,joinpath(root,"zero_budget");names=["tamen"],allow_partial=true)
        @test !only(partial).complete && only(partial).compared_until_fs==0
        @test isnothing(only(partial).speedup_vs_tamen)
    end
end
