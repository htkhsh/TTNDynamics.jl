# Tree study: references, TT and TTN in the original and centered-system
# orders. Output goes to output/$HOLSTEIN_TTN_TAMEN_OUTPUT (default tree_center).
haskey(ENV,"HOLSTEIN_TTN_TAMEN_OUTPUT") || (ENV["HOLSTEIN_TTN_TAMEN_OUTPUT"]="tree_center")
include("common.jl")
const HTC=HolsteinTAMEnComparison
started=time()
for name in ("reference","reference_refined","reference_center","reference_refined_center",
             "tt","tt_serial","tt_center","tt_center_serial",
             "tree_tt_channels","tree_center","tree_center_tight","tree_center_loose")
    remaining=3300-(time()-started)
    remaining>0 || break
    try
        HTC.run_case(name;wall_seconds=min(remaining,600.),
            repeats=startswith(name,"reference") ? 1 : 3)
    catch error
        error isa InterruptException && rethrow()
        println("RUN FAILED ",name,": ",sprint(showerror,error));flush(stdout)
    end
end
HTC.analyze()
