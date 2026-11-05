# From TTNDynamics: julia --project=examples examples/holstein_tfd/dynamics.jl
# Then: python3 examples/holstein_tfd/plot_dynamics.py
using TTNDynamics, TTNSolver, LinearAlgebra
include("config.jl")
include("dense_reference.jl")
include("../common/model_tree_output.jl")

function holstein_tfd_problem(config=holstein_config)
    # Same angular-frequency and temperature convention as TTDynamics/QFiND.
    icm2ifs = 2pi*299792458.0*1e2*1e-15
    hbar = 6.62607015e-34/(2pi)
    kb = 1.380649e-23
    isfinite(config.temperature_K) && config.temperature_K >= 0 || error("temperature_K must be finite and nonnegative")
    beta = config.temperature_K == 0 ? Inf : hbar*1e15/(kb*config.temperature_K)
    system = holstein_tfd_system(config.site_energies_cm.*icm2ifs,
        config.hopping_cm*icm2ifs, config.frequencies_cm.*icm2ifs,
        config.couplings_cm.*icm2ifs; beta,config.boson_size,config.periodic)
    m = length(config.site_energies_cm)
    rho = zeros(ComplexF64,m,m)
    if config.initial == :localized
        rho[config.initial_site,config.initial_site] = 1
    elseif config.initial == :electronic_gibbs
        E,U = eigen(Hermitian(system.hamiltonian))
        # At T=0 choose the equally weighted ground eigenspace.
        weights = isinf(beta) ? Float64.(E .≈ minimum(E)) : exp.(-beta.*(E.-minimum(E)))
        rho = Matrix(Hermitian(U*Diagonal(weights./sum(weights))*U'))
    else
        error("initial must be :localized or :electronic_gibbs")
    end
    initial = tfd_initial_state(system,rho)
    return system,initial
end

function tfd_example_grid(dt,tmax)
    isfinite(dt) && dt > 0 && isfinite(tmax) && tmax > 0 ||
        throw(ArgumentError("step size and duration must be finite and positive"))
    grid = collect(0:dt:tmax)
    last(grid) < tmax && push!(grid,tmax)
    return grid
end

function run_holstein_tfd(config=holstein_config; output_dir=@__DIR__)
    config.method in (:fixed,:adaptive) || throw(ArgumentError("method must be :fixed or :adaptive"))
    system,initial = holstein_tfd_problem(config)
    m = length(config.site_energies_cm)
    data_dir = joinpath(output_dir,"data")
    figure_dir = joinpath(output_dir,"figures")
    tree_dir = joinpath(output_dir,"trees")
    mkpath(data_dir)
    mkpath(figure_dir)
    mkpath(tree_dir)
    write_model_tree(joinpath(tree_dir,"tree_initial"),system,initial;
        title="Holstein–TFD · initial tree",
        subtitle="$(config.temperature_K) K · before basis enrichment · frequencies in fs⁻¹",print_tree=true)
    runs = config.method == :fixed ?
        [(label="coarse",parameter="dt_fs",value=config.fixed_dt_fs),
         (label="fine",parameter="dt_fs",value=config.fixed_dt_fs/2)] :
        [(;label,parameter="rtol",value=rtol) for (label,rtol) in config.tolerances]

    # Dense assembly/diagonalization is confined to this small validation example.
    dimension = prod(tfd_dimensions(system))
    dimension <= 2048 || error("dense comparison exceeds dimension 2048; use tfd_evolve alone for larger models")
    Hdense = dense_tfd_reference_matrix(system)
    eigenvalues,eigenvectors = eigen(Hermitian(Hdense))
    coefficients = eigenvectors'*vec(dense(initial))
    numbers = dense_tfd_physical_numbers(system)

    println("Holstein–TFD: $m sites, $(length(system.modes)) transformed modes, dense dimension=$dimension")
    println("T=$(config.temperature_K) K, tmax=$(config.tmax_fs) fs, Fock size=$(config.boson_size)")
    summaries = NamedTuple[]
    for run in runs
        label = run.label
        result = if config.method == :fixed
            tfd_evolve(system,initial,tfd_example_grid(run.value,config.tmax_fs);atol=config.fixed_atol)
        else
            tfd_evolve(system,initial;tspan=(0,config.tmax_fs),dt=config.initial_dt_fs,
                saveat=tfd_example_grid(config.save_interval_fs,config.tmax_fs),
                atol=config.adaptive_atol,rtol=run.value)
        end
        reference = [eigenvectors*(exp.(-im*t.*eigenvalues).*coefficients) for t in result.times]
        reference_densities = [dense_tfd_density(v,m) for v in reference]
        reference_numbers = [real(dot(v,N*v)) for N in numbers, v in reference]
        errors = [norm(rho-ref) for (rho,ref) in zip(result.densities,reference_densities)]
        path = joinpath(data_dir,"dynamics_$label.csv")
        open(path,"w") do io
            columns = ["time_fs"; ["p$(j)_ttn" for j in 1:m]; ["p$(j)_dense" for j in 1:m];
                ["n$(j)_ttn" for j in 1:m]; ["n$(j)_dense" for j in 1:m];
                ["thermal_n$(j)" for j in eachindex(system.modes)];
                "density_error"; "occupation_error"; "norm_drift"; "energy_drift_ifs"; "max_rank"]
            println(io,join(columns,','))
            for k in eachindex(result.times)
                row = [result.times[k]; real.(diag(result.densities[k])); real.(diag(reference_densities[k]));
                    result.physical_occupations[:,k]; reference_numbers[:,k]; result.mode_occupations[:,k];
                    errors[k]; maximum(abs.(result.physical_occupations[:,k]-reference_numbers[:,k]));
                    abs(result.norms[k]-result.norms[1]); abs(result.energies[k]-result.energies[1]);
                    maximum(result.ranks[:,k])]
                println(io,join(row,','))
            end
        end
        open(joinpath(data_dir,"steps_$label.csv"),"w") do io
            println(io,"time_fs,dt_fs,accepted,error_ratio")
            if config.method == :fixed
                for k in eachindex(result.diagnostics)
                    println(io,join((result.times[k],result.times[k+1]-result.times[k],1,NaN),','))
                end
            else
                for d in result.diagnostics
                    println(io,join((d.time,d.dt,Int(d.accepted),d.error_ratio),','))
                end
            end
        end
        counts = config.method == :fixed ? (steps=length(result.diagnostics),) : result.stats
        println("$(config.method) / $label $(run.parameter)=$(run.value): ",counts,
            "; max density error=",maximum(errors),
            "; max occupation error=",maximum(abs.(result.physical_occupations-reference_numbers)))
        println("  final state error=",norm(vec(dense(result.final_state))-last(reference)),
            "; max norm drift=",maximum(abs.(result.norms.-first(result.norms))),
            "; max rank=",maximum(result.ranks))
        println("  wrote $path")
        push!(summaries,(;label,density_error=maximum(errors),
            occupation_error=maximum(abs.(result.physical_occupations-reference_numbers)),
            final_state_error=norm(vec(dense(result.final_state))-last(reference))))
        write_model_tree(joinpath(tree_dir,"tree_$label"),system,result.final_state;
            title="Holstein–TFD · $label",
            subtitle="t = $(last(result.times)) fs · $(config.temperature_K) K · $(run.parameter) = $(run.value)")
    end
    open(joinpath(data_dir,"runs.csv"),"w") do io
        println(io,"label,method,parameter,value")
        for run in runs
            println(io,join((run.label,config.method,run.parameter,run.value),','))
        end
    end
    return (;method=config.method,runs=summaries)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || error("usage: dynamics.jl [fixed|adaptive]")
    config = isempty(ARGS) ? holstein_config : merge(holstein_config,(method=Symbol(only(ARGS)),))
    run_holstein_tfd(config)
end
