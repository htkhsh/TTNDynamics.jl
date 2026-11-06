include("driver.jl")
const HLT=HolsteinLongTime

function main(args=ARGS)
    length(args)>=2 || error("Usage: run.sh run.jl CONFIG.toml RUN [--resume] [--wall-seconds 3600] [--max-steps N]")
    config,name=args[1:2]
    resume=false; wall=3600.0; steps=typemax(Int); i=3
    while i<=length(args)
        if args[i]=="--resume"
            resume=true; i+=1
        elseif args[i] in ("--wall-seconds","--max-steps") && i<length(args)
            if args[i]=="--wall-seconds"
                wall=parse(Float64,args[i+1])
            else
                steps=parse(Int,args[i+1])
            end
            i+=2
        else
            error("unknown or incomplete argument: $(args[i])")
        end
    end
    cfg=HLT.load_config(config)
    HLT.run_case(cfg,name,HLT.output_root(cfg,config);resume,wall_seconds=wall,max_steps=steps)
end
main()
