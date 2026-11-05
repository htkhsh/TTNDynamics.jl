_svg_escape(value) = replace(string(value),
    "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")

function _tree_height(tree, node)
    left, right = tree.children[node]
    (left, right) == (0, 0) && return 0
    return 1 + max(_tree_height(tree, left), _tree_height(tree, right))
end

function _node_x(tree, node, physical_x)
    left, right = tree.children[node]
    (left, right) == (0, 0) && return physical_x[node]
    return (_node_x(tree, left, physical_x) + _node_x(tree, right, physical_x)) / 2
end

"""Write a standalone SVG tree diagram with the checked HEOM error values."""
function write_heom_tree_svg(filename, system, state; operator_error, action_error,
                             trace_error, initial_trace)
    tree = state.tree
    width = 1200
    height = 650
    left_margin = 110.0
    right_margin = 110.0
    spacing = (width - left_margin - right_margin) / (length(tree.leaf_order) - 1)
    leaf_x = Dict(node => left_margin + (position - 1)*spacing
                  for (position, node) in enumerate(tree.leaf_order))
    xs = Dict(node => _node_x(tree, node, leaf_x) for node in eachindex(tree.children))
    height_in_edges = _tree_height(tree, tree.root)
    leaf_y = 410.0
    ys = Dict(node => (tree.children[node] == (0,0) ? leaf_y :
                       130 + 92*(height_in_edges-_tree_height(tree,node)))
              for node in eachindex(tree.children))
    ranks = tree_ranks(state)
    system_dimension = first(heom_dimensions(system))

    elements = String[
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$width\" height=\"$height\" viewBox=\"0 0 $width $height\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"#f5f7fb\"/>",
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#192438}.muted{fill:#63718a}.small{font-size:13px}.label{font-size:14px;font-weight:600}.rank{font-size:12px;font-weight:700;fill:#3e5f8a}.metric{font-size:23px;font-weight:700}.title{font-size:25px;font-weight:700}</style>",
        "<text class=\"title\" x=\"34\" y=\"40\">HEOM tensor tree · dense verification</text>",
        "<text class=\"muted\" x=\"35\" y=\"68\">Spin-boson · physical order: vec(ρ), n₁, n₂, n₃ · box cutoffs: [2, 2, 2] · all displayed bonds have rank 1</text>",
        "<text class=\"small muted\" x=\"35\" y=\"96\">State topology and rank for the probe passed to the product-sum HEOM generator</text>",
    ]

    # Draw branches first so the nodes and rank badges sit on top.
    for parent in eachindex(tree.children)
        left, right = tree.children[parent]
        (left, right) == (0, 0) && continue
        for child in (left, right)
            push!(elements, "<line x1=\"$(xs[parent])\" y1=\"$(ys[parent]+23)\" x2=\"$(xs[child])\" y2=\"$(ys[child]-29)\" stroke=\"#8b9ab1\" stroke-width=\"2.3\"/>")
            label_x = (xs[parent]+xs[child])/2
            label_y = (ys[parent]+ys[child])/2
            push!(elements, "<rect x=\"$(label_x-25)\" y=\"$(label_y-10)\" width=\"50\" height=\"20\" rx=\"10\" fill=\"white\" stroke=\"#d9e1ed\"/>")
            push!(elements, "<text class=\"rank\" text-anchor=\"middle\" x=\"$label_x\" y=\"$(label_y+4)\">r=$(ranks[child])</text>")
        end
    end

    mode_labels = ["system · vec(ρ)", ("bath mode $k" for k in eachindex(system.modes))...]
    detail = ["fused system axis · M² = $system_dimension",
              ("γ = $(system.modes[k].rate) · n = 0:$(system.hierarchy_sizes[k]-1)"
               for k in eachindex(system.modes))...]
    for (position, node) in enumerate(tree.leaf_order)
        x, y = xs[node], ys[node]
        color = position == 1 ? "#dceafe" : "#ffedd5"
        stroke = position == 1 ? "#3e77c2" : "#d78b37"
        push!(elements, "<rect x=\"$(x-105)\" y=\"$(y-34)\" width=\"210\" height=\"68\" rx=\"13\" fill=\"$color\" stroke=\"$stroke\" stroke-width=\"1.6\"/>")
        push!(elements, "<text class=\"label\" text-anchor=\"middle\" x=\"$x\" y=\"$(y-7)\">$(_svg_escape(mode_labels[position]))</text>")
        push!(elements, "<text class=\"small muted\" text-anchor=\"middle\" x=\"$x\" y=\"$(y+16)\">$(_svg_escape(detail[position]))</text>")
    end
    for node in eachindex(tree.children)
        tree.children[node] == (0,0) && continue
        x, y = xs[node], ys[node]
        fill = node == tree.root ? "#263c5b" : "#dce3ed"
        stroke = node == tree.root ? "#263c5b" : "#8797ad"
        text_fill = node == tree.root ? "white" : "#27354a"
        push!(elements, "<circle cx=\"$x\" cy=\"$y\" r=\"25\" fill=\"$fill\" stroke=\"$stroke\" stroke-width=\"1.5\"/>")
        push!(elements, "<text text-anchor=\"middle\" x=\"$x\" y=\"$(y+5)\" font-size=\"12\" font-weight=\"700\" fill=\"$text_fill\">$(node == tree.root ? "root" : "Cₙ")</text>")
    end

    push!(elements, "<text class=\"small muted\" x=\"35\" y=\"466\">r: virtual bond dimension · blue: fused system leaf · orange: auxiliary hierarchy leaves</text>")
    metrics = [
        ("Dense operator error", operator_error),
        ("TTN action error", action_error),
        ("Trace derivative", trace_error),
        ("Initial trace", initial_trace),
    ]
    card_width = 272.0
    card_gap = 16.0
    card_x = 36.0
    card_y = 500.0
    for (index, (label, value)) in enumerate(metrics)
        x = card_x + (index-1)*(card_width+card_gap)
        displayed = index == length(metrics) ? string(round(real(value); sigdigits=4)) :
                    (iszero(value) ? "0" : string(round(abs(value); sigdigits=3)))
        status = index == length(metrics) ? abs(value-1) < 1e-12 : abs(value) < 1e-12
        color = status ? "#19724a" : "#a33131"
        push!(elements, "<rect x=\"$x\" y=\"$card_y\" width=\"$card_width\" height=\"111\" rx=\"14\" fill=\"white\" stroke=\"#dce3ed\"/>")
        push!(elements, "<text class=\"small muted\" x=\"$(x+18)\" y=\"$(card_y+27)\">$(_svg_escape(label))</text>")
        push!(elements, "<text class=\"metric\" x=\"$(x+18)\" y=\"$(card_y+64)\">$(_svg_escape(displayed))</text>")
        push!(elements, "<text x=\"$(x+card_width-18)\" y=\"$(card_y+63)\" text-anchor=\"end\" font-size=\"12\" font-weight=\"700\" fill=\"$color\">$(status ? "PASS" : "CHECK")</text>")
        push!(elements, "<text class=\"small muted\" x=\"$(x+18)\" y=\"$(card_y+91)\">$(index == length(metrics) ? "target = 1" : "target = 0")</text>")
    end
    push!(elements, "</svg>")
    open(filename, "w") do io
        write(io, join(elements, "\n"))
    end
    return filename
end
