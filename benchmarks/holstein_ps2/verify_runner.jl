include("common.jl")
using .PS2Bench, Test, TOML
const B=PS2Bench
cfg=B.HTC.configuration(;sites=2,local_size=2,end_fs=1.0)
p=B.HLT.comparison_problem(cfg)
@testset "PS2 comparison runner" begin
    mktempdir() do root
        for (name,s) in (("tt",B.specification(;method="tt",h=0.5)),("ps2",B.specification(;h=0.2)))
            B.run_case(p,cfg,name,s;root,stop_fs=1.,seconds=120.)
            dir=joinpath(root,name)
            status=TOML.parsefile(joinpath(dir,"status.toml"))
            @test status["phase"]=="complete"
            @test status["time_fs"]==1.
            @test isfile(joinpath(dir,"rep1","time_1.jls"))
            meta=TOML.parsefile(joinpath(dir,"metadata.toml"))
            @test meta["spec"]==(name=="tt" ? B.HTC.specification(cfg,"tt") : s)
        end
    end
end
