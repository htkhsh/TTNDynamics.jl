include("driver.jl")
using .HolsteinLongTime, Test, TOML
const HLT=HolsteinLongTime

# Real 10-site model, but only 0.2 fs. This tests the driver, not long-time
# accuracy or a representative performance comparison.
cfg=HLT.load_config(joinpath(@__DIR__,"config.toml"))
cfg["time"]=Dict("end_fs"=>0.2,"observation_fs"=>0.1,"checkpoint_fs"=>0.1)
cfg["runs"]=Dict(name=>HLT.case_config(cfg,name) for name in
    ("tamen","augmented","parallel","reference","reference_refined"))
for (name,s) in cfg["runs"]
    s["dt_fs"]=name=="reference_refined" ? 0.05 : 0.1
end
label=isempty(ARGS) ? "smoke" : ARGS[1]
occursin(r"^[A-Za-z0-9_-]+$",label) || error("invalid label")
root=joinpath(@__DIR__,"output",label)
ispath(root) && error("smoke output exists; choose a new label")
mkpath(root)
cfg["execution"]["output_directory"]="."
HLT.atomic_toml(joinpath(root,"config.toml"),cfg)
for name in ("tamen","augmented","parallel","reference","reference_refined")
    result=HLT.run_case(cfg,name,root;wall_seconds=600)
    @test result.complete
end
HLT.compare_runs(cfg,root)
println("SMOKE ONLY: 10 sites, 0–0.2 fs; outputs: ",root)
