# HEOM system layout implementation

The user explicitly requested selectable `system_layout=:fused` / `:split`
and performance comparisons, following the proposed ket/bra sibling tree.
Keep `:fused` as the compatible default. Implement in the current working tree
without committing unrelated existing work.

- [x] Add tests for default/invalid layouts, complex left/right generator
  actions, arbitrary matrices and custom trees, trace contraction, zero modes,
  size-one dimensions/cutoffs, and fixed/adaptive dynamics against dense HEOM.
- [x] Store the layout on `HEOMTTNSystem`; adapt physical dimensions, tree,
  product-sum generator, initial/trace states and density extraction. Use the
  existing BUG APIs. Split initial states use full ket/bra bases and route
  their correlation through the tree to their least common ancestor, never
  expanding the full hierarchy. Extract density with ket-first axis order.
- [x] Update tree labels and document the API and trace-protected rank cost.
- [x] Run complete tests and request an independent source review.
- [x] Benchmark warmed, alternating fused/split runs at several system sizes,
  with the same model/steps/tolerances and independent dense-reference errors.
  Record wall time, cumulative allocations, operator storage, ranks, trace
  drift and adaptive attempts; generate a report and figures.

Limits: split local factors use H and transpose(H), not H adjoint. The trace
functional has rank M across each separated system leaf; its protected spaces
require maxrank >= M. More product terms and nodes can make small systems
slower. Compare accuracy as well as timing: the compression trees differ.

Completed: 336/336 TTNDynamics tests pass, including custom-tree evolution
against independent dense equations. Source review found no blocking issues.
Five alternating pairs at M=2/4/8/16 (fixed) and M=4 (adaptive) are recorded in
`benchmarks/heom_layout/data/`. Split is slower for M=2/4/8 but 1.50x faster
for M=16 on this fixture; it reduces operator storage but increases cumulative
allocations in all measured cases. Report and figures: `docs/heom_layout.md`.
