# HEOMの必要ランクとBUG / tAMEnの速度

前回はTTランクが10までしか発達しなかったため、より大きなランクが必要な状態を調べる。TFDは対象外。
前回の対角な結合演算子とは別に、非可換なランダムHermitian結合の制御モデルを使う。
この系列内で結合強度と準備時間を変え、全ADOの相対誤差1e-4を共通基準にする。

1. 非可換なHermitian結合を持つHEOMの全状態を疎行列Krylov法で計算し、精度を再確認。
2. 発達した状態をTTとTTNに独立に分解し、圧縮誤差と必要ランクを実測。
3. その状態から共通の短区間をtAMEn / augmented BUG / parallel BUGで伝播。
4. 誤差条件を満たす設定だけをウォームアップ後に3回計時。入力準備・参照解・誤差評価は計時外。

TTNとTTの切断面が異なるため最大ランクだけでなく格納要素数も記録する。
基底を人工的に水増しせず、実際のHEOM状態に必要なランクを用いる。
結合強度によって時間スケールも変わるため、ランクだけの因果実験とは区別する。
勝てない条件・未達の設定も記録し、BUGが必ず優位になるとは仮定しない。
固定した階層打切りに対する数値誤差であり、物理的な階層収束の確認ではない。
`profile_*.csv` は圧縮1e-7、伝播入力は1e-5を用いる。`scan_*.csv` はコンパイルを含みうる
調整ログで、本計時とは区別する。高ランクの小系では格納量が密表現を上回る場合も明記する。

環境は `../tamen_comparison/README.md` と同じ。BLAS=1、Strided=1、Julia workers=8。

ワークスペースのルートから、公開した選定条件を保持して再計測する:

```bash
export JULIA_LOAD_PATH="$PWD/TTNDynamics:$PWD/TTSolver:@stdlib"
export JULIA_DEPOT_PATH=/tmp/ttndynamics-fixed-depot:$HOME/.julia
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1
julia --threads=8 --startup-file=no --compiled-modules=existing TTNDynamics/benchmarks/bug_rank_regimes/verify.jl
julia --threads=8 --startup-file=no TTNDynamics/benchmarks/bug_rank_regimes/probe.jl weak6 weak8 moderate8 strong6 strong6_short system8
julia --threads=8 --startup-file=no TTNDynamics/benchmarks/bug_rank_regimes/reproduce.jl
MPLCONFIGDIR=/tmp/ttndynamics-matplotlib python3 TTNDynamics/benchmarks/bug_rank_regimes/plot.py
```

`reproduce.jl` は保存済み `scan_*.csv` と `time_basis_*.csv` の選定条件を使い、
git管理外の入力キャッシュだけを再生成する。参照キャッシュは先に `probe.jl` で再生成する。
`scan.jl` や `run.jl` を改めて実行すると条件探索もやり直す。
現在の探索は保護方向のためランク上限を初期最大ランク+1から試すので、途中で既に採用した
W6のcap18やS6 augmentedのcap27とは別の有効設定を選ぶ場合がある。
スキャン秒数はコンパイルを含みうるため、本計測 `runs_*.csv` と混同しない。

[結果と解釈](../../docs/bug_rank_regimes.md)。`strong8` と `long8` のプロファイルは
ランク調査のみで、本計測・速度の図には使っていない。
