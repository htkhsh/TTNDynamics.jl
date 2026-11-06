include("common.jl")
const HTC=HolsteinTAMEnComparison
isempty(ARGS) && error("Usage: run.jl RUN [wall_seconds=1200] [repeats=1], or analyze")
if first(ARGS)=="analyze"
    HTC.analyze()
else
    HTC.run_case(first(ARGS);wall_seconds=length(ARGS)>=2 ? parse(Float64,ARGS[2]) : 1200.,
        repeats=length(ARGS)>=3 ? parse(Int,ARGS[3]) : 1)
end
