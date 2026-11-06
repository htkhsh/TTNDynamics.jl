# TT tAMEn と並列 TTN BUG の精度・速度比較

前回の `parallel_large_trees` と同じ M=4、局所次元3のモデルを使う。
今回は初期状態から t=0.25 までの全伝播を比較する（前回の最終1ステップの計時とは異なる）。
全状態の相対2ノルム誤差 ≤ 1e-4 を共通基準とする。HEOMのスケール済み全ADOを
比較し、密度行列誤差も記録する。TFDの追加比較は保留し、レポートはHEOMに限定する。

手順:

1. 小系の密行列解で、SoP→MPO、TT/TTN縮約、独立Taylor参照解を検証。
2. 大系ではTT Taylor解を時間刻みと丸め精度の両方で収束確認。
3. tAMEnの許容誤差、BUGの時間刻み・切り捨てを調整し、実測誤差で選択。
4. コンパイル・モデル構築・参照解・誤差評価を除き、全伝播を交互に3回計時。
   BUGは8 Juliaスレッド、BLASは両手法とも1スレッド。ノルムの事後補正はしない。

環境（ワークスペースのローカルパッケージを使用）:

```bash
export JULIA_LOAD_PATH="$PWD/TTNDynamics:$PWD/TTSolver:@stdlib"
export JULIA_DEPOT_PATH=/tmp/ttndynamics-fixed-depot:$HOME/.julia
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1
julia --threads=8 --startup-file=no TTNDynamics/benchmarks/tamen_comparison/verify.jl
julia --threads=8 --startup-file=no TTNDynamics/benchmarks/tamen_comparison/calibrate.jl heom16 heom32 heom64
julia --threads=8 --startup-file=no TTNDynamics/benchmarks/tamen_comparison/run.jl heom16 heom32 heom64
python3 TTNDynamics/benchmarks/tamen_comparison/plot.py
```

TTとTTNの物理軸順は同一。密展開のベクトル化順序だけが逆なので、比較は
木と列の直接縮約で行う。二乗ノルム差による誤差評価は約1e-7以下では丸め誤差の
影響を受けるため、本比較の1e-4判定に限定する。非連続な部分木は受け付けない。

参照解の `.jls` は再生成可能なローカルキャッシュでgit管理外。モデルや数値コードを
変更したときは `calibrate.jl` から再実行する。`run.jl` は各手法の採用済み設定だけを読む。
準備段階のログはコンパイルを含みうるので、本計時の `runs_*.csv` と区別する。
`pilot_rank64.*` はランク64での途中終了した試行で、本比較には含めない。
`calibration_tfd16.csv` と `measurements.log` はTFDを保留する前の調整・途中経過で、
HEOMの速度比較・図・集計には使わない。

結果と条件の詳細: [docs/tamen_comparison.md](../../docs/tamen_comparison.md)。
