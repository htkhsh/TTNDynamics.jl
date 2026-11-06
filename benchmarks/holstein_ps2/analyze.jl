include("common.jl")
using .PS2Bench
p,cfg=PS2Bench.checked_problem()
PS2Bench.analyze(p;finalize=true)
