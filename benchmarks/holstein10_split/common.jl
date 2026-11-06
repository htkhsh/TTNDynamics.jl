include("../tamen_comparison/common.jl")
import TTDynamics

module HolsteinExample
const EXAMPLE = normpath(joinpath(@__DIR__,"../../../TTDynamics/examples/holstein"))
include(joinpath(EXAMPLE,"config.jl"))
include(joinpath(EXAMPLE,"model.jl"))
include(joinpath(EXAMPLE,"utils.jl"))
end

const TARGET_TIME_FS = 100.0

function example_problem(;sites=10,local_size=5,initial_condition=:localized,config_overrides=(;))
    config = merge(HolsteinExample.DEFAULT_CONFIG,config_overrides,
        (;site_count=sites,site_energies_cm=zeros(sites),hierarchy_local_size=local_size,
          initial_condition,initial_site=get(config_overrides,:initial_site,1),blas_threads=1))
    decomposition = HolsteinExample.decompose_bath(config)
    problem = HolsteinExample.build_problem(config,decomposition)
    noise = problem.system.noise
    # NoiseExp pairs conjugate exponents and defines its own ADO scaling.
    # Reusing c2 and abs_coeff is essential; conj(c1[k]) is generally wrong.
    modes = [HEOMMode(noise.V[k],noise.γ[k],noise.c1[k],noise.c2[k];
                      scale=sqrt(noise.abs_coeff[k])) for k in 1:noise.nterms]
    system = HEOMTTNSystem(problem.system.H_sys,modes,problem.system.nb;system_layout=:split)
    train = HolsteinExample.initial_heom_state(config,problem.system)
    rho = TTDynamics.root_density_matrix(train,problem.system)
    tree = heom_initial_state(system,rho)
    conserved = TTNDynamics.trace_observable(system;tree=tree.tree)
    return (;config,decomposition,problem,system,train,tree,conserved,operator=heom_generator(system))
end

function train_step(p,x,h;scheme="CN",points=3,tolerance=5e-3,sweeps=3,local_iters=5,
                    workspace=TAMEnWorkspace(ComplexF64))
    st = tt_kron(x,tt_ones(points;T=ComplexF64))
    y,grid,_,_,_ = tamen(st,h*p.problem.liouvillian,tolerance;
        time_scheme=scheme,nswp=sweeps,local_iters,kickrank=p.config.kick_rank,
        time_error_damp=100.0,correct_2_norm=false,
        conserved_observables=[p.problem.trace_observable],workspace,verb=0)
    return extract_snapshot(y,grid,1.0,scheme),length(grid)
end

function tree_step(p,x,h;method=:augmented,atol=1e-6,maxrank=64)
    step = method==:augmented ? augmented_bug_step : parallel_bug_step
    return step(x,p.operator,h;atol,maxrank,threaded=true,conserved=p.conserved,correct_invariant=false)
end

matrix_vector(A::TTMatrix) = TTTensor([reshape(g,size(g,1),size(g,2)*size(g,3),size(g,4)) for g in A.cores])
function generator_agreement(p)
    rebuilt = sop_to_mpo(p.operator;tolerance=1e-12)
    a,b = matrix_vector(p.problem.liouvillian),matrix_vector(rebuilt)
    return tt_distance(a,b)/norm(a)
end

function source_metadata(io)
    println(io,"Julia: ",VERSION,"\nCPU: ",Sys.cpu_info()[1].model,"\nWorkers: ",Threads.nthreads(:default),
        "\nBLAS: ",BLAS.get_num_threads(),"\nGC: ",Threads.ngcthreads(),"\nLoad: ",Sys.loadavg())
    for dir in (dirname(pathof(TTSolver)),dirname(pathof(TTNSolver)),dirname(pathof(TTNDynamics)),
                dirname(pathof(TTDynamics)),@__DIR__,HolsteinExample.EXAMPLE)
        for file in sort(readdir(dir;join=true))
            endswith(file,".jl") || continue
            println(io,relpath(file,joinpath(@__DIR__,"../../.."))," sha256: ",bytes2hex(sha256(read(file))))
        end
    end
end
