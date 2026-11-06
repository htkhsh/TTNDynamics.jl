# Re-run references, TT and TT-style TTN acceptance with one set of sources.
# Output goes to output/$HOLSTEIN_TTN_TAMEN_OUTPUT (default tt_acceptance).
haskey(ENV,"HOLSTEIN_TTN_TAMEN_OUTPUT") || (ENV["HOLSTEIN_TTN_TAMEN_OUTPUT"]="tt_acceptance")
include("common.jl")
const HTC=HolsteinTAMEnComparison
started=time()
for name in ("reference","reference_refined","tt","tt_serial","tree_tt",
             "tree_tt_channels","tree_tt_channels_tight","tree_tt_channels_loose")
    remaining=3300-(time()-started)
    remaining>0 || break
    try
        HTC.run_case(name;wall_seconds=min(remaining,600.),
            repeats=name in ("reference","reference_refined") ? 1 : 3)
    catch error
        error isa InterruptException && rethrow()
        println("RUN FAILED ",name,": ",sprint(showerror,error));flush(stdout)
    end
end
HTC.analyze()
