# Weighted truncation and grouped bath tree implementation plan

> Execute with superpowers:executing-plans; use test-driven development and independent final review.

**Goal:** Test the two improvements approved by the user: amplitude-weighted
rank truncation and a tree that groups Brownian components belonging to one site.

**Architecture:** Keep the existing local truncation and balanced tree defaults.
Add opt-in weighted truncation to standalone compression and both BUG steps.
Carry complementary amplitudes downward with thin QR, avoiding squared Gram
matrices. Group HEOM modes by identical coupling matrices without permuting
physical axes. Benchmark the four combinations on the same short trajectory.

**Tech stack:** Existing Julia/LinearAlgebra/KrylovKit packages and Python plots.
**Spec:** User-approved items 1 and 2 in the preceding discussion; 10-site split
Holstein, localized initial condition, Brownian 800 and hopping 400 cm^-1.

## Constraints and review focus

- Preserve existing uncommitted work and old benchmark outputs; no commits.
- No normalization or post-hoc trace correction; conserve the supplied functional.
- Complex conjugation in complementary factors must match tensor coordinates.
- Discarded bounds must bound the full-state change, including rank-cap failures.
- Invalid modes/protected spaces must leave the input unchanged.
- Bath grouping changes topology only, including interleaved equal couplings.
- Do not claim matched accuracy from density observations alone.
- Short measurements only; target about one hour for the experiment.

## Tasks

- [x] Add failing weighted-truncation tests: an amplitude-weighted rank fixture,
  complex dense error bounds, protected directions, zero/scaled states, caps,
  and invalid-option nonmutation. Run the focused test and record RED.
- [x] Extend `truncate!(...; weighting=:local)` with `:weighted`; reuse
  `_retained_space` on weighted complementary QR factors. Budget `atol/edges`
  and sum full-state tail bounds; carry amplitudes after each orthogonal cut.
- [x] Add `truncation=:local` / `:weighted` to the augmented and parallel BUG
  step APIs and prepared steps. Check both against a small dense exponential.
- [x] Test and add `heom_tree(system; bath_grouping=:balanced)` with
  `:coupling`: identical mode coupling matrices share a balanced subtree;
  physical leaf ordering and existing defaults remain unchanged.
- [x] Run TTNSolver and TTNDynamics suites with the existing local environment.
- [x] Create a separate factorial benchmark for local/weighted × balanced/
  coupling, both BUG methods, same h=0.1 fs, atol=1e-6, rank cap 64, 0–1 fs.
  Save reference and refined-reference states, errors, allocations, ranks and
  per-step timings. Warm up each path and run timed cases sequentially.
- [x] Independently review correctness, then document measured results and
  limits. Leave the 50 fs production configuration and data unchanged.

## Execution record

The user's request authorizes implementation and measurement of the two
previously proposed changes. Execute inline in the shared development packages
to preserve their existing uncommitted dependencies and source paths; do not
commit or create a parallel checkout with stale packages.

- Weighted test RED: unsupported `weighting` keyword; GREEN: 103 assertions.
- Grouped tree test RED: unsupported `bath_grouping` keyword. Dense integration
  initially exposed the known absent-coupling-direction behavior of an
  unenriched product basis. The diagnostic reproduced identical errors with
  both old/new trees and local/weighted truncation. Enriching with the initial
  derivative preserved the tensor and reduced the discrepancy to ~3e-16.
  Apply enrichment only in the small dense integration test, not the benchmark.
- Full suites: TTNSolver 832/832, TTNDynamics 459/459, with 8 Julia threads.
- Independent read-only reviewer found no defects and passed nine additional
  complex tests with rank-2 protected functionals.
- First benchmark setup failed before timing because TTTensor lacks `copy`;
  use `deepcopy` for TT inputs. Setup-only files are retained separately.
- Production timings run sequentially, with two trials per BUG case and
  strict TT reference/refinement at the common final time of 1 fs.
- Existing source-instrumentation helpers initially failed their checked finish
  marker after adding the truncation keyword (RED). Updated their imports and
  markers, supporting the frozen old parallel source as well. Focused regression
  passed 12/12 assertions against uninstrumented results (GREEN).
- All 16 BUG trajectories completed. Weighted-only reduced median augmented
  time from 144.52 to 4.64 s and parallel time from 66.66 to 3.03 s. Maximum ranks
  fell from 64 to 30/45. Grouping provided no additional benefit with weighted
  truncation. Full-state errors remain about 1.07e-3/1.79e-3, whereas tAMEn
  took 0.229 s with error 8.14e-5; no matched-accuracy speed claim is warranted.
- Exported and inspected the comparison figure, saved all eight final tree
  snapshots, and documented detailed times, errors, references and limitations
  in `docs/weighted_tree.md`. The 50 fs configuration and outputs were untouched.
