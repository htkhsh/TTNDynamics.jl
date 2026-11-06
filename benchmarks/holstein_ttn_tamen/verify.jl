include("common.jl")
using Test, LinearAlgebra
const HTC=HolsteinTAMEnComparison
const HLT=HTC.HLT

@testset "Holstein TT / TTN tAMEn fixture" begin
    cfg=HTC.configuration(;sites=2,local_size=2,end_fs=1.0)
    p=HLT.comparison_problem(cfg)
    @test HLT.generator_agreement(p)<1e-10
    @test HLT.comparison_error(p.tree,p.train)<1e-7
    target=exp(0.2*HLT.dense(p.operator))*vec(HLT.dense(p.tree))
    for name in ("tt","tree")
        s=HTC.specification(cfg,name);s["dt_fs"]=0.2;s["tolerance"]=1e-8
        s["sweeps"]=30
        work=name=="tt" ? HLT.TAMEnWorkspace(ComplexF64) : HTC.TS.TAMEnWorkspace()
        x=name=="tt" ? p.train : p.tree
        y,d=HTC.advance(p,x,s,work,nothing)
        vector=name=="tt" ? vec(permutedims(reshape(HLT.tt_full(y),Tuple(reverse(HLT.tt_dims(y)))),
            reverse(1:length(HLT.tt_dims(y))))) : vec(HLT.dense(y))
        @test norm(vector-target)/norm(target)<1e-6
        @test abs(tr(HLT.density_of(y,p))-1)<1e-9
        name=="tree" && @test d.converged && d.history[end].global_residual_checked
    end
end

@testset "Explicit startup failure and zero-budget status" begin
    cfg=HTC.configuration(;sites=2,local_size=2,end_fs=1.0)
    mktempdir() do root
        HTC.run_case("tt";root,cfg,wall_seconds=0.)
        status=HTC.TOML.parsefile(joinpath(root,"tt","status.toml"))
        @test status["phase"]=="timeout"
        @test status["time_fs"]==0
        @test status["completed_repetitions"]==0
        s=HTC.specification(cfg,"tree");s["sweeps"]=0
        @test_throws ArgumentError HTC.run_case("tree";root,cfg,settings=s)
        failed=HTC.TOML.parsefile(joinpath(root,"tree","status.toml"))
        @test failed["phase"]=="warmup_failed"
        @test !isempty(failed["failure"])
        @test failed["time_fs"]==0
        # Incomplete/missing references cannot silently qualify this result.
        HTC.analyze(;root)
        rows=HTC.csvrows(joinpath(root,"run_status.csv"))
        @test Set(r["name"] for r in rows)==Set(["tt","tree"])
        @test all(r->r["reference_checked"]=="false",rows)
    end
end
