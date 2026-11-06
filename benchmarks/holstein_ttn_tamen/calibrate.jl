# Fewer global verifications: keep linear tolerance, tune temporal interval/order.
include("common.jl")
const HTC=HolsteinTAMEnComparison
cfg=HTC.configuration()
settings=HTC.specification(cfg,"tree")
settings["dt_fs"]=1.0
settings["points"]=6
HTC.run_case("tree_interval1";cfg,settings,wall_seconds=1700.0,repeats=1)
HTC.analyze()
