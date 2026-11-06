using TTNSolver, Serialization
pushfirst!(LOAD_PATH,normpath(joinpath(@__DIR__,"../../../TTNPlots")))
using TTNPlots

label=isempty(ARGS) ? "trial" : first(ARGS)
occursin(r"^[A-Za-z0-9_-]+$",label) || error("invalid label")
root=joinpath(@__DIR__,"output",label)
labels=["System ket","System bra",["Site $j · $k" for j in 1:10 for k in 1:2]...]
palette=["#0072b2","#d55e00","#009e73","#cc79a7","#e69f00",
         "#4d7c8a","#8e5572","#658d2e","#8064a2","#b17c38"]
colors=["#4263aa","#4263aa",[color for color in palette for _ in 1:2]...]
for method in ("augmented","parallel"), variant in ("baseline","weighted","grouped","both")
    x=deserialize(joinpath(root,"$(method)_$(variant)_1.jls"))
    stem=joinpath(root,"tree_$(method)_$variant")
    title="$method BUG · $variant · t=1 fs"
    write_tree_svg(stem*".svg",x;title,leaf_labels=labels,leaf_colors=colors,
        subtitle="Holstein · 10 sites · split · two Brownian components per site")
    write(stem*".txt",tree_text(x;leaf_labels=labels))
end
