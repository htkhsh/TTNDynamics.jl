# Benchmark-only instrumentation of isolated copies. Production code has no
# timers added. Markers are checked so a future refactor cannot silently time
# the wrong region. Ordinary paired timing uses the uninstrumented methods.
function timed_module(name,source)
    # Both the frozen pre-optimization source and the current opt-in API are
    # benchmark inputs. Keep checked markers for each supported finish call.
    suffix = occursin("correct_invariant,truncation)",source) ? ",truncation" : ""
    finish = "_finish_bug_step!(result,old,c,stats;atol,maxrank,correct_invariant$suffix)"
    insertions = (
        "    prepared = _prepare_parallel_bug_step(state,operator,conserved,maxrank)" =>
        "    phase_started = time_ns()\n    prepared = _prepare_parallel_bug_step(state,operator,conserved,maxrank)\n    STAGES[1] += (time_ns()-phase_started)/1e9",
        "    (;old,c,initial,boundary,modes,up) = prepared" =>
        "    phase_started = time_ns()\n    (;old,c,initial,boundary,modes,up) = prepared",
        "    result = copy(old)" =>
        "    phase_now = time_ns()\n    STAGES[2] += (phase_now-phase_started)/1e9\n    phase_started = phase_now\n    result = copy(old)",
        "    return $finish" =>
        "    phase_now = time_ns()\n    STAGES[3] += (phase_now-phase_started)/1e9\n    final = $finish\n    STAGES[4] += (time_ns()-phase_now)/1e9\n    return final")
    for (marker,replacement) in insertions
        length(findall(marker,source)) == 1 || error("stage instrumentation marker changed: $marker")
        source = replace(source,marker=>replacement)
    end
    mod = Module(name)
    Core.eval(mod,:(using LinearAlgebra))
    Core.eval(mod,:(import TTNSolver))
    for binding in (:TTNTensor,:SumOfProducts,:LocalSolveFailure,:tree_ranks,:enrich_basis,
                    :_prepare_bug_step,:_check_bug_options,:_check_truncation_weighting,:_isleaf,:_leaf_rhs_workspace,
                    :_core_rhs_workspace,:_orth_columns,:_local_action,:_project_children,
                    :_child_product,:_core_rhs,:_basis_matrix,:_finish_bug_step!,:_parallel_core_block,
                    :_leaf_operators,:_mixed_subtree_operators)
        Core.eval(mod,:(const $binding = TTNSolver.$binding))
    end
    Core.eval(mod,:(const KrylovKit = TTNSolver.KrylovKit))
    Core.eval(mod,:(const STAGES = zeros(4)))
    Base.include_string(mod,source,string(name,".jl"))
    return mod
end
