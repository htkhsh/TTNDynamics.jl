# Holstein 10-site HEOM: split TT / TTN comparison

[1時間枠での途中計測結果](../../docs/holstein10_split.md)。100 fs全区間の比較は未完了。
同じモデル・参照でのaugmented BUG追試も同レポートに掲載している。

ユーザー指定: `TTDynamics/examples/holstein` の物理パラメータ、10サイト、
`system_layout=:split`、サイト1の局在初期状態から100 fsの平衡化。
元の例のconfigや出力は変更しない。このディレクトリに別の結果を保存する。

現在のconfigは hopping=4000、Brownian frequency=160、damping=300、
reorganization energy=800 cm^-1、300 K、PSD order=0、階層局所サイズ5。
10サイトでは複素共役の2成分×10浴=20階層添字。
系のket/braと合わせて22葉、43ノード、全空間の要素数は100×5^20。
初期条件はユーザー指定により例の現在のGibbs設定からlocalizedへ変更する。

`NoiseExp` が対にした `c1/c2` と `sqrt(abs_coeff)` をそのままTTNへ渡す。
単にc1を複素共役にする変換は使わない。小系の密行列解と生成子の一致を確認する。
`pilot.jl` は設定調整用の短い計算であり、100 fsの比較結果とは区別する。

実行環境は `run.sh` にまとめてある。Julia 8 workers、Strided 1 threadを使い、
BLAS threadsは各trajectoryの引数で指定する。

```bash
cd TTNDynamics/benchmarks/holstein10_split
bash run.sh verify.jl
bash run.sh pilot.jl
# 0–1 fsのBUG計算と、保存済みpilot参照との比較
bash run.sh bug_case.jl my_augmented augmented 0.05 32
# ラベル、手法、刻み幅、許容誤差、ランク上限、時間基底、点数、sweep、local iterations、BLAS、終了時刻
bash run.sh trajectory.jl my_run tamen 0.5 0.001 128 cheb 12 5 10 1 100
```

各trajectoryは伝播のみを計時する。コンパイル用warm-up、観測量、保存時間は除外する。
全ステップの時刻・時間・ランク・格納要素数・密度行列をCSVに、10 fsごとの状態と
最新状態をJulia serializationに保存する。既存のrunラベルには上書きしない。
`metadata.txt` に実際の設定、スレッド数、source hashを記録する。
TTの `truncation_met` と `discarded_bound` は該当診断がないため `NA`。
BUGの `atol` は空間切断の絶対許容誤差であり、時間積分誤差ではない。

`evaluate.jl` はTT参照状態との全階層相対誤差と密度行列誤差を縮約で求める。
既定では0–100 fsの全10 fsチェックポイントを要求し、不完全な軌道を100 fsの
比較結果として扱わない。`plot.py` は各軌道の到達時刻を凡例に表示する。
`pilot.jl` / `calibrate.jl` / `screen.jl` は設定調整用。
CNの外側の刻み幅とtAMEn内部の採用区間数は別で、後者もCSVに記録する。

`profile_step.jl` は保存したaugmentedの1 fs状態から1ステップを段階別に計測する。
`GroupedKernels.jl` と `grouped_probe.jl` は、局所演算子の完全一致する因子をまとめた
初期の診断実装。保存済みの `data/grouped_probe` は本体への反映前の結果である。
現在の本体には項の集約・環境縮約の再利用・作業配列の再利用が入っているため、
現在の変更前後比較には以下の `optimization_*` を使う。

```bash
bash run.sh profile_step.jl
bash run.sh grouped_probe.jl
```

`optimization_probe.jl` は同じ1 fsの保存状態から両BUG法を各3回計時し、
`optimization_trajectory.jl` は局在状態から0–1 fsの軌道を比較する。
入力には本作業で保存した `data/augmented_h005/latest.jls`、参照には
`data/pilot_reference.jls` が必要（大きな状態ファイルはgit対象外）。
`data/optimization_baseline/` は変更前の5つのソースを保存した参照実装で、
ベンチマークの別モジュール内だけで読み込む。時間刻み・許容誤差・ランク上限は同じ。
`optimization_unstructured.jl` は共通因子がない小さい局所問題の処理負担を調べる。

```bash
bash run.sh optimization_probe.jl optimized_steps_final_augmented augmented
bash run.sh optimization_probe.jl optimized_steps_final_parallel parallel
bash run.sh optimization_trajectory.jl optimized_trajectories
bash run.sh profile_step.jl profile_optimized
bash run.sh optimization_unstructured.jl
python3 plot_optimization.py
```

全軌道の比較は各1試行で、局所1ステップの3試行の中央値とは区別する。
同じ保存状態からの速度比を100 fsの速度比へ外挿しない。
