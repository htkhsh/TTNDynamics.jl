#!/usr/bin/env bash
set -euo pipefail
benchmark_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
dev_dir="$(cd -- "$benchmark_dir/../../.." && pwd)"
export JULIA_DEPOT_PATH="${HOLSTEIN_DEPOT_PATH:-/tmp/ttndynamics-fixed-depot:/tmp/ttdynamics-julia-depot:/home/takahashi/.julia}"
export JULIA_LOAD_PATH="$dev_dir/TTNDynamics:$dev_dir/TTDynamics/examples/holstein:$dev_dir/TTSolver:@stdlib"
export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_GC_THREADS=1
exec julia --threads=8 --startup-file=no --compiled-modules=existing "$benchmark_dir/$1" "${@:2}"
