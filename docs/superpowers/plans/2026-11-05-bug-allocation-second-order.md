# BUG allocation and second-order trials

User authorization: try items 3 and 4 of the preceding recommendations.
Use current weighted truncation and the balanced tree. Preserve existing dirty
work and all old outputs; no commits, no automatic 50 fs run.

1. Profile allocations of both first-order methods after warm-up on the same
   10-site split Brownian Holstein model, 0–1 fs. Save source copies, timings,
   states, ranks and allocation stacks before changing the shared implementation.
2. Reduce measured orthogonalization allocation with local preallocated buffers,
   preserving column priority, two reorthogonalizations, input ownership and
   tiny-amplitude behavior. Add behavioral and warmed allocation tests first.
   Re-run package suites and compare identical trajectories.
3. Trial second-order augmented BUG from Kusch–Sulz arXiv:2610.11747,
   sections 5.1.1–5.1.2 and 5.2, in a separate benchmark module. Pre-augment
   subtree bases, augment complementary environments downward, then reuse the
   recursive Galerkin update. No dense/global derivative in production-sized runs.
   Keep the production method/default unchanged. If practical, also trial the
   second-order parallel assembly from section 5.1.4.
4. Verify complex coefficients, state preservation by pre-augmentation, trace,
   serial/threaded agreement, and convergence against independent dense small
   problems. Compare 0–1 fs Holstein timing/error versus dt and refined TT
   reference. Report cut/rank limitations and do not extrapolate to 50 fs.
5. Independent final code/math review; publish measured tables, plots, source
   references and reproducible commands.

Review focus: ordinary versus conjugate transpose in environment maps;
rank-deficient complement QR; protected trace directions; allocation ownership;
time-discretization convergence separated from truncation/reference error.

Execution is inline, as in the preceding trials. A new benchmark directory
isolates the second-order trial; the shared optimization is limited to an
equivalent orthogonalization kernel if measurement supports it.

Ledger: profile → kernel optimization share identical saved inputs; kernel
optimization → second-order trial share the orthogonalization contract. No
interface conflict; elapsed-time/accuracy comparisons must use unprofiled runs.

## Execution record

All five steps completed on 2026-11-05. Production changes are confined to
`_orth_columns` and its tests; both second-order methods remain in the benchmark
module, with no production API or default-method change.

- Saved old source copies and three warm unprofiled 0–1 fs runs per method.
  Separate 1% allocation samples identify sites without contaminating timings.
- New orthogonalization checks initially passed 8 of 9 assertions; the warmed
  allocation bound failed. With buffered `mul!`, all 9 passed. Full package
  suites: TTNSolver 863, TTNDynamics 481. Final before/after relative state
  differences are below 8e-16 and every bond rank agrees.
- Cumulative allocations fell 22.6% / 31.6% for augmented / parallel BUG.
  About 1% timing changes fall within the observed run-to-run spread.
- Implemented and independently reviewed subtree/complement pre-augmentation,
  augmented Galerkin update and parallel evolved cross blocks from Kusch–Sulz.
  A generic entangling convergence fixture exposed the normal-component term
  in the error bound; a tangent generator isolates the temporal order.
  This restriction and the practical fixed-total-cut-budget comparison are
  disclosed in the report. All 27 prototype assertions pass, including single
  leaf, rank-deficient complements, protected trace and threaded agreement.
- Completed all 26 trajectories (13 settings, two runs each), plus refined TT
  references. At the common 1e-3 full-state target, parallel order 2 takes
  6.38 s versus order 1's 8.52 s; at closer errors there is no speed advantage.
  First-order augmented remains the fastest tested BUG candidate meeting the
  target (5.68 s), and tAMEn remains faster on this 1 fs example.
- Independent review found no important defects. Added the suggested edge-case
  tests. Tables, figures, source hashes and reproduction commands accompany
  `docs/bug_improvements.md`. No 50 fs run or extrapolation was performed.
