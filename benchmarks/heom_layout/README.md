# HEOM system layout comparison

Compare `system_layout=:fused` and `:split` using the same two-mode HEOM
models at system dimensions 2, 4, 8 and 16. Fixed stepping is measured at
all dimensions; M=4 also checks adaptive stepping. Independent dense ADO
equations validate generator actions and dynamics, including h/h2 refinement.

From the TTNDynamics package root:

```sh
OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=examples benchmarks/heom_layout/run.jl 5
python3 benchmarks/heom_layout/plot.py
```

Two warmups precede five alternating pairs per case. GC remains enabled.
No other tests or numerical jobs should run concurrently. Timings include
generator construction and observations, excluding model/initial-state
construction and dense reference work. `data/` contains timing, allocation,
storage, rank, trace and accuracy measurements, plus environment metadata.
`figures/` contains SVG/PNG comparisons; `trees/` holds M=4 initial/final
SVG/text diagrams. See [the report](../../docs/heom_layout.md).
