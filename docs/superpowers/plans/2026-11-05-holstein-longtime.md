# Long-time Holstein TT / TTN comparison

User request: prepare comparison code with a Brownian bath at 800 cm^-1 and
hopping 400 cm^-1. Preserve the established 10-site split ket/bra layout and
site-1 localized initial condition. Default to 100 fs, configurable before
starting. Damping 300 cm^-1, reorganization energy 800 cm^-1, temperature 300 K
and hierarchy size 5 are carried forward and exposed in the configuration.

Use the existing verified physical-model and TT/TTN overlap helpers, plus a
separate `benchmarks/holstein_longtime` driver and TOML configuration. This
keeps earlier benchmark data reproducible. A new solver API or changes to the
Holstein example's own defaults are unnecessary.

- [x] Test and implement configuration validation and inherited solver cases.
- [x] Add explicit model overrides to the existing comparison model helper.
- [x] Implement timed propagation, common observation/checkpoint grids,
  atomic checkpoints, per-step random seeds and validated resume.
- [x] Test interruption/restart, including rollback of uncommitted CSV tails,
  altered model/solver/source rejection and continuous-vs-resumed evolution.
- [x] Compare complete trajectories at shared times, refuse incomplete or
  mismatched models by default, and mark partial results explicitly.
- [x] Provide tighter tAMEn reference and timestep/rank refinement cases;
  expose reference convergence and accuracy qualification before speed ratios.
- [x] Add plots and concise execution/resume instructions.
- [x] Run small dense-reference tests and short smoke runs with the actual
  10-site configuration, then independent read-only review.

This request authorizes preparing and testing code, not launching all long
production trajectories. Use short verification runs in this turn.

Review focus: ket/bra ordering and Brownian c1/c2 scaling; strict time grids;
restart state/RNG/CSV consistency; soft wall limits at accepted-step boundaries;
no rank cap silently passed to tAMEn; incomplete reference comparisons must
never be presented as accuracy-matched full-time speedups.

Validation on 2026-11-05:

- Configuration: 23 assertions passed.
- Integrity and chunked TT–TTN contraction: 14 assertions passed; the complex
  random-tensor contraction agrees with dense contraction at three budgets.
- Model, dense solution, interruption/resume and comparison: 45 assertions
  passed, plus the driver existence check.
- Existing split benchmark: 9 assertions passed; TTNDynamics: 421 passed.
- Actual 10-site model: all three methods and two references completed 0–0.2 fs.
  CLI resume/comparison and SVG/PNG generation succeeded; plot inspected.
- Independent read-only review findings addressed: hash bath dependencies,
  reject mixed code/runtime comparisons, release loaded final states, and
  require actual reference refinement. Each received regression coverage.

No 100 fs production run or long-time speedup claim was made. The short smoke
trajectory validates operation only. Analysis contracts saved states in chunks
to avoid one large four-index temporary; subtree environments still grow with
bond ranks, and checkpoints require storage proportional to the saved states.
