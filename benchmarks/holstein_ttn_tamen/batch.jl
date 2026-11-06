include("common.jl")
const HTC=HolsteinTAMEnComparison
started=time()
for name in ("reference","reference_refined","tt","tt_serial","tree")
    remaining=3300-(time()-started)
    remaining>0 || break
    try
        HTC.run_case(name;wall_seconds=min(remaining,name=="tree" ? 2300. : 600.),
            repeats=name in ("tt","tt_serial") ? 3 : 1)
    catch error
        error isa InterruptException && rethrow()
        println("RUN FAILED ",name,": ",sprint(showerror,error));flush(stdout)
    end
end
HTC.analyze()
