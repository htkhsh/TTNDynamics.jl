# Density-matrix errors of the long-time truncation batch (output/long20_trunc).
# 10 sites: every run against the TT reference at tolerance 1e-5, which is
# itself checked against the stored 1e-6 / 1e-7 references (<= 9 fs).
# 20 sites: no reference; runs are compared with the TTN run at 1e-5 and TT.
include(joinpath(@__DIR__,"common.jl"))
using .HolsteinTAMEnComparison, LinearAlgebra, Serialization
const H=HolsteinTAMEnComparison; const HLT=H.HLT
const OUT=joinpath(@__DIR__,"output","long20_trunc")
const STORED=joinpath(@__DIR__,"..","holstein_longtime","output","holstein50")

function stored(dir)
    rows=H.csvrows(joinpath(STORED,dir,"observables.csv")); out=Dict{Int,Matrix{ComplexF64}}()
    for r in rows
        t=parse(Float64,r["time_fs"]); isinteger(t) || continue
        out[Int(t)]=[parse(Float64,r["rho_$(i)_$(j)_re"])+im*parse(Float64,r["rho_$(i)_$(j)_im"]) for i in 1:10, j in 1:10]
    end
    out
end
function densities(dir,q)
    out=Dict{Int,Matrix{ComplexF64}}()
    for t in 1:20
        f=joinpath(OUT,dir,"time_$t.jls"); isfile(f) && (out[t]=H.density(deserialize(f),q))
    end
    out
end
fmt(x)=isnothing(x) ? "" : string(round(x,sigdigits=3))
err(a,b,t)=haskey(a,t) && haskey(b,t) ? opnorm(a[t]-b[t]) : nothing

function main()
    rows=String["sites,run,versus,time_fs,density_error,trace_error"]
    for sites in (10,20)
        cfg=H.configuration(;end_fs=20.0,sites); p=HLT.comparison_problem(cfg)
        q(layout)=H.ordered_problem(p,:center;bath_layout=layout)
        tag="_N$sites"
        runs=Dict{String,Dict{Int,Matrix{ComplexF64}}}()
        for dir in readdir(OUT)
            isdir(joinpath(OUT,dir)) && occursin(tag,dir) || continue
            runs[dir]=densities(dir,q(occursin("chain",dir) ? :chain : :balanced))
        end
        against=sites==10 ? Dict("ref"=>runs["reference_center_N10_tol1e-5"],
                                 "stored_1e-6"=>stored("reference"),"stored_1e-7"=>stored("reference_refined")) :
                            Dict("ttn_1e-5"=>runs["tree_center_N20_trunc1e-5"],"tt"=>runs["tt_center_N20"])
        for name in sort(collect(keys(runs))), vname in sort(collect(keys(against))), t in 1:20
            r,v=runs[name],against[vname]
            e=err(r,v,t); isnothing(e) && continue
            push!(rows,join((sites,name,vname,t,fmt(e),fmt(abs(tr(r[t])-1))),','))
        end
    end
    write(joinpath(OUT,"accuracy.csv"),join(rows,'\n')*"\n")
    # Compact view: errors at selected times.
    for line in rows
        f=split(line,','); f[1]=="sites" && continue
        f[4] in ("5","9","10","15","20") && println(line)
    end
end
main()
