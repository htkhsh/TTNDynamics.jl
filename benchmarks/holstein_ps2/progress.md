# PS2 experimental trial —2026-11-06

User approved PS2 + local Arnoldi + protected rank adaptation,1fs then5fs
Holstein split comparisons under~1h benchmark budget. Experimental module only;
no production source/API changes, no commits or branch changes. Existing dirty
dev/als work preserved. No AGENTS.md found.

- ps2.jl implements Chen/Franco Algorithms4–6, with full physical leaves,
  preserved QR null spaces and padded protected SVD cuts. Scalar coefficients
  occur only in local actions. Oriented operator/functional messages update
  immediately when the centre moves. Environments never cross a changed side.
- Scientific checks: generic37/37 in /tmp/ps2-test-final.log; physical75/75 in
  /tmp/ps2-physical-final.log, with h.2,.1,.05 errors5.58e-5,1.40e-5,3.49e-6.
  Explicit cap test initially failed(3>2): protected gauge can add functional
  directions; final weighted protected cut now enforces cap only when needed.
  Half the local cut budget is reserved for that cut. Its test is now green.
  A separate test-file placement typo caused recursion; moved testset out of
  fullstate() and killed only our own failed process. Numerical kernel not
  implicated by that test harness mistake.
- Independent read-only reviewer /root/review_ps2_trial active, requested by
  requesting-code-review skill. No extra agents or heavy calculations by it.
- common.jl validates source/config/runtime against saved refined TT references,
  hashes all reference state files plus ps2/common/batch scripts, and never
  overwrites existing experiment provenance. run_case uses atomic checkpoints,
  per-step timing and explicit failure/timeout statuses. batch calibrates h.2,
  h.1,h.05 at1fs and selects the fastest qualified candidate for5fs. New TT
  baseline has3repeats; refined references are reused only after hash checks.
- Reviewer finished: no mathematical blocker. Confirmed TT copy() missing;
  changed warmup/repetition clones to deepcopy. Effective TT settings now
  recorded/executed identically; reference wrapper hash checked and preserved.
  Runner regression8/8 passed /tmp/ps2-runner.log (both physical TT/PS2 finish1fs).
  Overall current validation37+75+8=120 assertions. Independent reviewer did
  not run heavy solvers or judge the not-yet-completed measurements.
- Benchmark started05:48UTC (see next note for exact process/time). Budget3300s
  internal,3500s external, to retain some recovery/analysis time within~1h.
  Calibrationh.2/.1/.05 with padding2, plus h.1/padding1;300s per case soft
  cap. Fastest qualified/cut-budget-passing1fs candidate extends to5fs; if
  complete but inaccurate, retry half-step if time remains. Script hashes now
  frozen: do not edit ps2.jl/common.jl/batch.jl or the saved-reference wrapper
  until all measurements and analysis finish. Docs/plot scripts may change.
  Active exec session30448 launched05:48:43UTC; hard cap3500s ends06:47:03UTC.
  One-hour comparison budget ends~06:48:43UTC; stop early enough for recovery
  if a long interval is still running. Log output/batch.log. No tests running.
- First calibration stopped session30448 at~05:55UTC viaCtrl-C(exit130): h.2,
  padding2 reached0.4fs in31.2825s/rank45; third interval was expensive. Each
  step used169local exponentials and~5239actions, because KrylovKit defaults
  eager=false and builds30vectors before checking. Original sources copied to
  output/holstein5/source_snapshot, status explicitly calibration_stopped;
  old outputs and their provenance retained. No1fs PS2 accuracy claim there.
- Added optional krylov_eager flag(defaultfalse) and set it true in comparison
  settings, preserving tolerance1e-12 and max dimension30. New output root
  output/holstein5_eager; three new numerical checks compare eager/non-eager
  and denseexpm, require feweractions. Generic/eager tests session1891 log
  /tmp/ps2-eager-test.log, physical/eager session9707 log
  /tmp/ps2-eager-physical.log currently running. No timed calculation active.
  Reordered calibration to h.2/pad1,h.1/pad1,h.2/pad2,h.05/pad1. The absolute
  overall deadline stays06:48:43UTC, not reset. Set PS2_DEADLINE_UNIX for remaining
  budget when launching batch; soft stop06:42:43UTC, hardstop06:44:43UTC reserves
  finalanalysis time. ps2/common/batch will freeze again after validation.
- Eager validation40/40 and physical75/75 passed; time-halving errors unchanged
  at5.58e-5,1.40e-5,3.49e-6. Runner8/8 already passed after review fixes.
  Eager batch launched05:57UTC with absolute softdeadline06:42:43UTC and
  harddeadline06:44:43UTC; final overall budget06:48:43UTC. Active session/time
  recorded next. Log output/batch_eager.log; finalroot output/holstein5_eager.
  No tests or other heavy calculations running. PS2/common/batch now frozen.
  Active session31189 launched05:57:24UTC. By06:00:25UTC TT3repeats complete5fs,
  PS2h.2/pad1 complete1fs5.6250s/rank29;h.1/pad1 complete1fs27.0723s/rank34.
  h.2/pad2 calibration now running, followed byh.05/pad1 then accuracy analysis.
  Do not claim qualification before analyze returns. No code edits to frozen
  sources since launch. Final report/docs/plots and statuses remain pending.
- By06:06UTC all four1fs calibrations completed. h.2/pad1 failed state target
  (1.8753e-3); h.1/pad1 passed at27.07235s/state5.67644e-4/rho8.32628e-5.
  h.05/pad1 passed56.46920s/state1.15556e-4; h.2/pad2 passed266.32445s/
  state3.85051e-4. All cut budgets passed. Selected h.1/pad1 for5fs,
  running in ps2_five_fs. Fresh TT5fs median4.035965s/three reps, state6.176e-4.
  Calibration1fs h.1 uses87.9% in local exponential actions. Added phase output
  to Python summary only (frozen Julia source hashes unchanged). Interim plot
  and calibration doc updated. Final5fs accuracy still pending.
- 06:14UTC: selected5fs run reached2.4fs/485.56085s; rank64 from2.2fs,
  latest largest local vector1,358,280 elements, all cut budgets still true.
  Accuracy at2+fs is pending end-of-run analysis. No concurrent heavy tests.
  Absolute deadlines remain06:42:43soft/06:44:43hard/06:48:43overall UTC.
  Script hashes rechecked unchanged; local docs links and Python syntax checked.
- 06:23:44UTC: selected run completed3fs in1069.198042246s, rank64,
  max localdimension2,271,360; cut budgets all pass. 3fs phase totals:
  solve977.569177s(91.43%), environment66.645446s, split22.895851s,
  allocation188.383517536GB (cumulative, not peak). Accuracy still pending.
  Benchmark continuing within unchanged deadline; no completed5fs claim.
- Final: selected run stopped normally at3.8fs after softdeadline, status timeout,
  38accepted steps,2276.825187664s propagation, no cut-budget failures.
  Batch session31189 exited0 by~06:44UTC; no hardkill/recovery was needed.
  Finalanalyze verified PS2 through3fs: maxstate5.6764397816e-4,
  maxrho2.1686657462e-4, maxtrace1.1134e-14, qualified. 3fs propagation1069.198s
  vsTTmedian1.287946522s (0.884244154–1.311694326), ratio830.157.
  TT3fs maxstate4.4484544821e-4,rho4.0137467010e-5. TT5fs complete;
  PS2 not complete5fs. Summaries and plot regenerated, finalreport updated.
  Numerical validation remains40+75+8=123passed; Julia sources never changed
  during the measured run. No timed or heavy calculation remains running.
- Final artifact verification passed: all source/script/reference state hashes,
  all22comparison rows' checkpoint presence, cumulative timing/allocation sums,
  qualification flags and incomplete5fs labels. All38accepted selected steps
  obey cut budgets and cap64. Python syntax, documentation links, source/doc
  whitespace and gitdiff--check passed. FinalPNG visually inspected (3fs PS2
  endpoint and5fs TT endpoint, calibration markers, target line and legend).
