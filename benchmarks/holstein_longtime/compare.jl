include("driver.jl")
const HLT=HolsteinLongTime
length(ARGS)>=1 || error("Usage: run.sh compare.jl CONFIG.toml [RUN ...] [--partial]")
cfg=HLT.load_config(first(ARGS))
partial="--partial" in ARGS[2:end]
names=filter(!=("--partial"),ARGS[2:end])
isempty(names) && (names=cfg["comparison"]["runs"])
HLT.compare_runs(cfg,HLT.output_root(cfg,first(ARGS));names,allow_partial=partial)
