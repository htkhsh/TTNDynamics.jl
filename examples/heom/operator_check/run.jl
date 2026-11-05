using TTNDynamics
using TTNSolver
using LinearAlgebra

include("../common/dense_heom_reference.jl")
include("tree_diagram.jl")
include("dynamics_diagram.jl")

H = ComplexF64[0.2 0.3+0.1im; 0.3-0.1im -0.4]
V = ComplexF64[1 0.2im; -0.2im -0.5]
modes = [HEOMMode(V, 0.8, 0.7+0.2im, 0.7-0.2im),
         HEOMMode(V, 0.4+1.2im, 0.3+0.1im, 0.2+0.05im; scale=0.7),
         HEOMMode(V, 0.4-1.2im, 0.2-0.05im, 0.3-0.1im; scale=0.7)]
system = HEOMTTNSystem(H, modes, [2, 2, 2])
generator = heom_generator(system)
rho = ComplexF64[0.7 0.1im; -0.1im 0.3]
initial = heom_initial_state(system, rho)

# Include nonzero higher ADOs so the comparison exercises both connections.
probe = product_state(heom_tree(system),
    [vec(rho), ComplexF64[1, 0.2im], ComplexF64[1, 0.3], ComplexF64[1, -0.1im]])
reference = dense_heom_reference_matrix(system)
operator_error = norm(dense(generator) - reference)
action_error = norm(vec(dense(generator * probe)) - reference * vec(dense(probe)))
trace_error = abs(dot(trace_observable(system), generator, probe))
@assert operator_error < 1e-12
@assert action_error < 1e-12
@assert trace_error < 1e-12
@assert root_density_matrix(initial, system) ≈ rho
println("Physical dimensions: ", heom_dimensions(system))
println("Product terms: ", length(generator.terms))
println("Dense operator error: ", operator_error)
println("TTN action error: ", action_error)
println("Trace derivative error: ", trace_error)
println("Initial trace: ", dot(trace_observable(system), initial))
figure_dir = joinpath(@__DIR__,"figures")
mkpath(figure_dir)
diagram_path = joinpath(figure_dir, "heom_operator_check.svg")
write_heom_tree_svg(diagram_path, system, probe;
    operator_error, action_error, trace_error,
    initial_trace=dot(trace_observable(system), initial))
println("Visualization: ", diagram_path)

dynamics_times, populations, trace_history =
    dense_heom_reference_dynamics(system, initial; final_time=5.0, step_size=0.01)
dynamics_path = joinpath(figure_dir, "heom_dynamics.svg")
write_heom_dynamics_svg(dynamics_path, dynamics_times, populations, trace_history;
                        step_size=0.01)
println("Dynamics plot: ", dynamics_path)
println("Maximum trace drift: ", maximum(abs.(trace_history .- trace_history[1])))
