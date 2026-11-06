# BUGの割当削減と二次精度の試行

10サイトsplit Holstein、Brownian周波数800 / hopping400 cm⁻¹、局在初期状態、0–1 fs。
重み付き切断＋balanced treeを使う。[結果と制約](../../docs/bug_improvements.md)。

```bash
cd TTNDynamics/benchmarks/holstein_bug_improvements
bash run.sh allocations.jl before before
bash run.sh allocations.jl after current
bash run.sh verify_second_order.jl
bash run.sh order_comparison.jl order_clean
bash run.sh order_comparison.jl order_refined refine
bash run.sh compare_states.jl
MPLCONFIGDIR=/tmp/ttn-bug-mpl python3 plot.py
```

既存出力ラベルには上書きしない。時間計測は他のテストや計算と同時に走らせない。
`allocations.jl` の第2引数 `before` は、保存した旧直交化関数をそのJuliaプロセス内だけに
読み込む。パッケージのソースファイルは変更しない。`current` は現在の関数を使う。
計時はwarm-up後の3回、割当プロファイルは別実行の最終1ステップ（1%サンプル）。
サンプルによるカテゴリ推定は分散が大きく、削減率には `@timed` の累積割当量を使う。
累積割当量はピークメモリ使用量ではない。

二次精度版は `second_order.jl` に隔離した試行実装。自律線形SoP生成子に対し、
部分木基底の事前拡張、外側環境の拡張、augmentedまたはparallelの更新を行う。
全階層状態やその微分を密展開せず、局所縮約を使う。既定の時間積分器は変更していない。

```julia
include("second_order.jl")
using .SecondOrderBUG
y, info = second_order_bug_step(x, generator, h;
    method=:parallel, conserved=trace_functional,
    correct_invariant=false, truncation=:weighted, atol=1e-6, maxrank=64)
```

`order_comparison.jl` は同じモデルのTT参照2条件も計算し、各設定を2回計時する。
切断予算は `1e-5*h`。全体の切断予算をほぼ揃えた実用誤差の比較であり、
この曲線だけでは漸近的な時間精度の次数は判定できない。
`verify_second_order.jl` はランク多様体の接方向を保つ生成子で次数を分離して確認する。
任意の生成子では法線方向の近似誤差やランク上限による誤差も残る。

## tAMEnとの再比較

[5 fsの実測結果と計算時間上限による未完了区間](../../docs/tamen_recomparison.md)。

`recompare.jl` は現在の実装で長めの共通区間を測る追加実験。
各1 fsの全状態、観測値、全ステップの時間・割当・ランク・切断予算超過を保存する。
各手法を別プロセスで順番に起動する。以下は5 fsの例（既存ラベルには上書きしない）。

```bash
bash run.sh recompare.jl tamen_recompare5 reference_refined 5 1 600
bash run.sh recompare.jl tamen_recompare5 reference 5 1 400
bash run.sh recompare.jl tamen_recompare5 tamen 5 3 300
bash run.sh recompare.jl tamen_recompare5 augmented 5 1 600
bash run.sh recompare.jl tamen_recompare5 parallel 5 1 600
bash run.sh recompare.jl tamen_recompare5 parallel2 5 1 600
bash run.sh recompare.jl tamen_recompare5 augmented2 5 1 450
bash run.sh recompare_analysis.jl tamen_recompare5
MPLCONFIGDIR=/tmp/ttn-bug-mpl python3 plot_recompare.py tamen_recompare5
```

引数は出力ラベル、手法、終時刻 [fs]、反復数、起動ごとのソフト時間上限 [s]。
上限判定はステップ境界で行い、最後の状態と完了した1 fsごとの状態を残す。
完了した共通区間で比較できる。ソフト上限で中断した反復を同じ条件で最後まで延長する場合は、
元のプロセスが終了してから以下を使う。ソース・スレッド設定を検証し、
元のログと状態を `rep*/before_resume/` に保存してから追記する（各反復につき1回）。

```bash
bash run.sh recompare_resume.jl tamen_recompare5 augmented 900
```

伝播時間にはモデル構築、warm-up、観測、保存、誤差縮約を含まない。
tAMEnはBLAS8、BUGはJulia8スレッド・BLAS1。

| 手法名 | 時間刻み [fs] | 設定 |
|---|---:|---|
| tamen | 0.5 | tol=10⁻⁴ |
| reference | 0.25 | tol=10⁻⁶ |
| reference_refined | 0.125 | tol=10⁻⁷ |
| augmented | 0.05 | 一次BUG |
| parallel | 0.025 | 一次BUG |
| augmented2 | 0.1 | 二次BUG試作 |
| parallel2 | 0.125 | 二次BUG試作 |

BUGの切断予算は10⁻⁵×刻み幅、最大ランク64、重み付き切断、balanced tree。
`measurements.csv` は参照2条件の差を確認した各時刻での全状態・密度行列の差を持つ。
精度判定は、その時刻までの毎1 fsの最大差がともに10⁻³以内、かつ参照間の最大差が
10⁻⁴以内の場合だけ合格とする。途中点の確認は連続時間での誤差保証ではない。
`summary.csv` の通常の時間比と、精度判定を通った場合だけの時間比は別列に保存する。
TTとTTNのランクは異なる分割に対する値なので、ランク数だけでコストを比較しない。

参照：J. Kusch and D. Sulz, *Robust second-order time integration of tree tensor networks*,
[arXiv:2610.11747](https://arxiv.org/abs/2610.11747), §5.1–5.2。
