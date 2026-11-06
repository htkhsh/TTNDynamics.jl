# Keep the original interval and accuracy target; parallelize dense kernels.
include("common.jl")
const HTC=HolsteinTAMEnComparison
cfg=HTC.configuration()
settings=HTC.specification(cfg,"tree")
settings["blas_threads"]=8
HTC.run_case("tree_blas8";cfg,settings,wall_seconds=900.0,repeats=1)
HTC.analyze()
