These five Julia files are read-only benchmark reference sources, copied from
the working TTNSolver implementation immediately before the 2026-11-05 exact
term grouping and environment reuse changes. They include prior, uncommitted
BUG threading work; a Git revision alone cannot reproduce this baseline.

`optimization_probe.jl` loads them into an isolated module using the current
TTN data types and unchanged algebra/truncation functions. They are never
loaded by the public solver. Measurement metadata records their hashes.
