# Run with --project=examples; only topology CSVs are read, no state rebuilt.
using TTNPlots
data = joinpath(@__DIR__,"data")
destination = joinpath(@__DIR__,"figures","trees"); mkpath(destination)
for name in ("heom16","heom32","heom64","tfd16")
    rows = [parse.(Int,split(line,',')) for line in readlines(joinpath(data,"tree_$name.csv"))[2:end]]
    children = [(row[2],row[3]) for row in rows]
    root = only(row[1] for row in rows if row[4] == 0)
    physical = sort(filter(row -> row[5] > 0,rows);by=row -> row[5])
    order = [row[1] for row in physical]
    dims = [row[6] for row in physical]
    ranks = [row[7] for row in rows]
    labels = [m == 1 ? "System M=4" : "Mode $(m-1)" for m in eachindex(physical)]
    colors = [m == 1 ? "#4263aa" : "#25887e" for m in eachindex(physical)]
    if startswith(name,"tfd")
        for m in 2:length(physical)
            k = m-1
            pair = cld(k,2)
            labels[m] = "Site $(mod1(pair,4)) · $(isodd(k) ? "+" : "−")ω$(cld(pair,4))"
            colors[m] = isodd(k) ? "#25887e" : "#b17c38"
        end
    end
    path = joinpath(destination,"$name.svg")
    write_tree_svg(path,children;root,leaf_order=order,dimensions=dims,ranks,
        leaf_labels=labels,leaf_colors=colors,title="Parallel BUG: $name",
        subtitle="Input to measured step, t=0.20; $(length(order)) leaves, $(length(rows)) nodes")
    println(path)
end
