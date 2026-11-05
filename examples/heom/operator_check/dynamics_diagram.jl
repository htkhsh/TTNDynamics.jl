function dense_heom_reference_dynamics(system, initial; final_time=5.0, step_size=0.01)
    isfinite(final_time) && final_time > 0 || throw(ArgumentError("final_time must be positive"))
    isfinite(step_size) && step_size > 0 || throw(ArgumentError("step_size must be positive"))
    steps = round(Int, final_time / step_size)
    isapprox(steps * step_size, final_time; atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("final_time must be a multiple of step_size"))
    d = size(system.hamiltonian, 1)
    dims = (d, d, system.hierarchy_sizes...)
    current = reshape(ComplexF64.(vec(dense(initial))), dims)
    nsteps = Int(steps)
    times = collect(range(0.0; step=step_size, length=nsteps+1))
    populations = Matrix{Float64}(undef, d, nsteps+1)
    traces = Vector{ComplexF64}(undef, nsteps+1)
    vacuum = ntuple(_ -> 1, length(system.modes))
    for sample in 1:nsteps+1
        root = view(current, :, :, vacuum...)
        traces[sample] = tr(root)
        for site in 1:d
            populations[site, sample] = real(root[site, site])
        end
        sample == nsteps+1 && break
        k1 = dense_heom_reference_rhs(system, current)
        k2 = dense_heom_reference_rhs(system, current .+ (step_size/2) .* k1)
        k3 = dense_heom_reference_rhs(system, current .+ (step_size/2) .* k2)
        k4 = dense_heom_reference_rhs(system, current .+ step_size .* k3)
        current .+= (step_size/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    return times, populations, traces
end

function write_heom_dynamics_svg(filename, times, populations, traces; step_size)
    rows, samples = size(populations)
    length(times) == samples == length(traces) ||
        throw(DimensionMismatch("times, populations, and traces must have matching lengths"))
    rows >= 2 || throw(DimensionMismatch("the population plot needs at least two levels"))
    width, height = 1200, 700
    x_left, x_right = 110.0, 1145.0
    xwidth = x_right - x_left
    population_top, population_bottom = 165.0, 370.0
    trace_top, trace_bottom = 465.0, 620.0
    t0, t1 = first(times), last(times)
    xcoord(t) = x_left + (t-t0)/(t1-t0)*xwidth
    ypop(p) = population_bottom - clamp(p, 0, 1)* (population_bottom-population_top)
    trace_error = abs.(traces .- traces[1])
    logtop = ceil(log10(max(2 * maximum(trace_error), 1e-15)))
    logbottom = min(-18.0, logtop-5)
    ytrace(error) = trace_bottom -
        (log10(max(error, 10.0^logbottom))-logbottom)/(logtop-logbottom) *
        (trace_bottom-trace_top)

    elements = String[
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$width\" height=\"$height\" viewBox=\"0 0 $width $height\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"#f5f7fb\"/>",
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#192438}.muted{fill:#63718a}.small{font-size:13px}.label{font-size:14px;font-weight:600}.title{font-size:25px;font-weight:700}.tick{font-size:12px;fill:#63718a}</style>",
        "<text class=\"title\" x=\"35\" y=\"42\">Three-mode HEOM dynamics</text>",
        "<text class=\"muted\" x=\"36\" y=\"69\">Dense ADO reference · classical RK4 · h = $step_size · 0 ≤ t ≤ $(last(times))</text>",
        "<text class=\"small muted\" x=\"36\" y=\"95\">Initial state ρ₀(0) = diag(0.7, 0.3); all higher ADOs start at vacuum</text>",
        "<rect x=\"$x_left\" y=\"$population_top\" width=\"$xwidth\" height=\"$(population_bottom-population_top)\" rx=\"4\" fill=\"white\" stroke=\"#dce3ed\"/>",
        "<rect x=\"$x_left\" y=\"$trace_top\" width=\"$xwidth\" height=\"$(trace_bottom-trace_top)\" rx=\"4\" fill=\"white\" stroke=\"#dce3ed\"/>",
    ]

    for value in 0.0:0.25:1.0
        y = ypop(value)
        push!(elements, "<line x1=\"$x_left\" y1=\"$y\" x2=\"$x_right\" y2=\"$y\" stroke=\"#e5eaf1\"/>")
        push!(elements, "<text class=\"tick\" x=\"$(x_left-15)\" y=\"$(y+4)\" text-anchor=\"end\">$(round(value; digits=2))</text>")
    end
    tick_count = 5
    for index in 0:tick_count
        time = t0 + (t1-t0)*index/tick_count
        x = xcoord(time)
        for (top, bottom) in ((population_top,population_bottom), (trace_top,trace_bottom))
            push!(elements, "<line x1=\"$x\" y1=\"$top\" x2=\"$x\" y2=\"$bottom\" stroke=\"#edf0f5\"/>")
        end
        push!(elements, "<text class=\"tick\" x=\"$x\" y=\"$(trace_bottom+22)\" text-anchor=\"middle\">$(round(time; digits=2))</text>")
    end

    colors = ["#3974c3", "#e38831"]
    for site in 1:2
        points = join(("$(xcoord(times[k])),$(ypop(populations[site,k]))" for k in eachindex(times)), " ")
        push!(elements, "<polyline points=\"$points\" fill=\"none\" stroke=\"$(colors[site])\" stroke-width=\"3\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>")
        legend_x = 135 + (site-1)*125
        push!(elements, "<line x1=\"$legend_x\" y1=\"124\" x2=\"$(legend_x+30)\" y2=\"124\" stroke=\"$(colors[site])\" stroke-width=\"3\"/>")
        push!(elements, "<text class=\"label\" x=\"$(legend_x+38)\" y=\"129\">P($site)</text>")
    end
    push!(elements, "<text class=\"label\" x=\"$x_left\" y=\"$(population_top-15)\">System populations</text>")
    push!(elements, "<text class=\"label\" x=\"$x_left\" y=\"$(trace_top-15)\">Trace drift |Tr ρ₀(t) − Tr ρ₀(0)|  ·  log scale</text>")
    push!(elements, "<text class=\"small muted\" text-anchor=\"end\" x=\"$x_right\" y=\"$(trace_top-15)\">max = $(round(maximum(trace_error); sigdigits=3))</text>")

    errors_y = [ytrace(error) for error in trace_error]
    trace_points = join(("$(xcoord(times[k])),$(errors_y[k])" for k in eachindex(times)), " ")
    push!(elements, "<polyline points=\"$trace_points\" fill=\"none\" stroke=\"#22845c\" stroke-width=\"2.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>")
    for exponent in ceil(Int, logtop):-2:ceil(Int, logbottom)
        error = 10.0^exponent
        y = ytrace(error)
        push!(elements, "<line x1=\"$x_left\" y1=\"$y\" x2=\"$x_right\" y2=\"$y\" stroke=\"#e5eaf1\"/>")
        push!(elements, "<text class=\"tick\" x=\"$(x_left-15)\" y=\"$(y+4)\" text-anchor=\"end\">1e$(exponent)</text>")
    end
    push!(elements, "<text class=\"small muted\" x=\"$x_left\" y=\"665\">Reference trajectory only · population curves use the system (zero-tier) density</text>")
    push!(elements, "</svg>")
    open(filename, "w") do io
        write(io, join(elements, "\n"))
    end
    return filename
end
