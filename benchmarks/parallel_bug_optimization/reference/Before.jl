# Isolated baseline: the pre-optimization parallel step, with the unchanged
# shared TTNSolver primitives. Never patches or extends the live step methods.
module Before
using LinearAlgebra
import TTNSolver
using TTNSolver: TTNTensor, SumOfProducts, LocalSolveFailure, tree_ranks,
    enrich_basis, _prepare_bug_step, _check_bug_options, _isleaf,
    _leaf_rhs_workspace, _core_rhs_workspace, _orth_columns, _local_action,
    _project_children, _child_product, _core_rhs, _basis_matrix, _finish_bug_step!
const KrylovKit = TTNSolver.KrylovKit
include("parallel_bug_before.jl")
end
