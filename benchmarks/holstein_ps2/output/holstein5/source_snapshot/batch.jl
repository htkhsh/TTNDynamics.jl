include("common.jl")
using .PS2Bench
const B=PS2Bench
started=time();deadline=started+3300.
p,cfg=B.checked_problem()
B.run_case(p,cfg,"tt",B.specification(;method="tt",h=0.5);stop_fs=5.,repeats=3,seconds=300.,deadline)
settings=Dict()
for (name,h,padding) in (("ps2_h02",0.2,2),("ps2_h01",0.1,2),("ps2_h01_pad1",0.1,1),("ps2_h005",0.05,2))
    time()+100<deadline || break
    settings[name]=B.specification(;h,padding)
    B.run_case(p,cfg,name,settings[name];seconds=300.,deadline)
end
rows=B.analyze(p)
qualified=filter(r->startswith(r.name,"ps2_") && r.time_fs==1 && r.qualified && r.cut_budget_failures==0,rows)
if !isempty(qualified) && time()+120<deadline
    selected=qualified[argmin([r.seconds for r in qualified])]
    println("SELECTED ",selected.name);flush(stdout)
    B.run_case(p,cfg,"ps2_five_fs",settings[selected.name];stop_fs=5.,seconds=deadline-time(),deadline)
    rows=B.analyze(p)
    final=filter(r->r.name=="ps2_five_fs" && r.time_fs==5,rows)
    if !isempty(final) && !only(final).qualified && time()+180<deadline
        # Refine the time step without changing the selected rank policy.
        refined=copy(settings[selected.name]);refined["dt_fs"]/=2
        B.run_case(p,cfg,"ps2_five_fs_refined",refined;stop_fs=5.,seconds=deadline-time(),deadline)
    end
else
    println("No qualified calibration or insufficient remaining budget; no5fs claim.")
end
B.analyze(p;finalize=true)
