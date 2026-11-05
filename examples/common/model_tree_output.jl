# Model-specific labels live with the examples, not in TTNPlots or the solvers.
using TTNPlots

function model_tree_labels(system::HEOMTTNSystem)
    split = system.system_layout == :split
    labels = [split ? ["System · ket","System · bra"] : ["System · vec(ρ)"];
              ["Hierarchy mode $k" for k in eachindex(system.modes)]]
    details = [split ? ["density row index","density column index"] : ["ket × bra"];
               ["γ = $(round(m.rate;sigdigits=4))\nn = 0:$(system.hierarchy_sizes[k]-1)"
                              for (k,m) in enumerate(system.modes)]]
    colors = [fill("#4263aa",split ? 2 : 1);fill("#b17c38",length(system.modes))]
    return (;leaf_labels=labels,leaf_details=details,leaf_colors=colors)
end

function model_tree_labels(system::TFDTTNSystem)
    labels = ["Electron + ancilla"; ["Thermal mode $k" for k in eachindex(system.modes)]]
    for (k,(p,q,_)) in enumerate(system.thermal_pairs)
        labels[p+1] = "Oscillator $k · +ω"
        labels[q+1] = "Oscillator $k · −ω"
    end
    details = ["physical electron ×\ninactive ancilla";
        ["ω = $(round(m.frequency;sigdigits=4))\nn = 0:$(system.boson_sizes[k]-1)"
         for (k,m) in enumerate(system.modes)]]
    colors = ["#4263aa"; [m.frequency >= 0 ? "#25887e" : "#b17c38" for m in system.modes]]
    return (;leaf_labels=labels,leaf_details=details,leaf_colors=colors)
end

function write_model_tree(stem,system,state;title,subtitle="",print_tree=false)
    labels = model_tree_labels(system)
    diagram = write_tree_svg(stem*".svg",state;title,subtitle,labels...)
    hierarchy = tree_text(state;labels...)
    write(stem*".txt",title*"\n"*subtitle*"\n\n"*hierarchy)
    print_tree && print(hierarchy)
    println("  tree: $diagram")
    return diagram
end
