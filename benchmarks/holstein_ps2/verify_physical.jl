include("../holstein_ttn_tamen/common.jl")
include("ps2.jl")
using .PS2Trial, Test, LinearAlgebra
const HTC=HolsteinTAMEnComparison
const HLT=HTC.HLT
const TS=HTC.TS
BLAS.set_num_threads(1)
@testset "PS2 split HEOM physical fixture" begin
    p=HLT.comparison_problem(HTC.configuration(;sites=2,local_size=2,end_fs=1.0))
    @test HLT.generator_agreement(p)<1e-10
    target=exp(HLT.dense(p.operator))*vec(HLT.dense(p.tree))
    errors=Float64[]
    for h in (0.2,0.1,0.05)
        x=copy(p.tree);drops=0.
        for _ in 1:round(Int,1/h)
            x,info=ps2_step(x,p.operator,h;conserved=p.conserved,atol=1e-9*h,maxrank=32,krylov_tol=1e-13,krylov_eager=true)
            @test abs(dot(p.conserved,x)-1)<1e-10
            @test info.truncation_met
            drops+=info.local_cut_sum
        end
        error=norm(vec(HLT.dense(x))-target)/norm(target)
        push!(errors,error)
        println("PHYSICAL h=",h," error=",error," rank=",maximum(TS.tree_ranks(x))," cut=",drops)
    end
    @test last(errors)<1e-5
    @test last(errors)<first(errors) || maximum(errors)<1e-10
    before=HLT.dense(p.tree)
    @test_throws ArgumentError ps2_step(p.tree,p.operator,0.1;conserved=p.conserved,maxrank=1)
    @test HLT.dense(p.tree)==before
end
