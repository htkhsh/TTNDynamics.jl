#!/usr/bin/env bash
set -euo pipefail
benchmark_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
exec bash "$benchmark_dir/../holstein_longtime/run.sh" "../holstein_weighted_tree/$1" "${@:2}"
