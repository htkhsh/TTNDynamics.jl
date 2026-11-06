# Retain the last fully saved accepted state after an interrupted calibration.
include("common.jl")
using .HolsteinTAMEnComparison, Serialization, TOML
const HTC=HolsteinTAMEnComparison
const HLT=HTC.HLT
root=joinpath(@__DIR__,"output","five_fs")
function recover(name)
dir=joinpath(root,name);rep=joinpath(dir,"rep1")
saved=deserialize(joinpath(rep,"latest.jls"))
path=joinpath(rep,"steps.csv")
lines=readlines(path);header=split(first(lines),',')
rows=HTC.csvrows(path)
kept=filter(r->parse(Int,r["step"])<=saved.step,rows)
length(kept)==saved.step || error("accepted checkpoint/log mismatch")
isfile(path*".interrupted") || cp(path,path*".interrupted")
open(path,"w") do io
    println(io,first(lines))
    for r in kept;println(io,join((r[k] for k in header),','));end
end
HLT.write_rows(joinpath(dir,"runs.csv"),[(;repetition=1,complete=false,time_fs=saved.time_fs,
    seconds=saved.seconds,bytes=sum(parse(Int,r["bytes"]) for r in kept),
    gc=sum(parse(Float64,r["gc_seconds"]) for r in kept),
    peak_rank=maximum(parse(Int,r["maxrank"]) for r in kept))])
status=TOML.parsefile(joinpath(dir,"status.toml"))
status["phase"]="calibration_stopped";status["time_fs"]=saved.time_fs
status["failure"]="Stopped an expensive trial; only accepted checkpoints are used for accuracy comparison."
# The final status update predates the interrupted interval. Recovery may run
# much later, so its timestamp must not be counted as solver runtime.
status["wall_seconds_is_complete"]=false
status["recovered_at_unix"]=time()
HLT.atomic_toml(joinpath(dir,"status.toml"),status)
end
foreach(recover,isempty(ARGS) ? ["tree"] : ARGS)
HTC.analyze(;root)
