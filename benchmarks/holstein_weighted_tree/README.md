# Weighted truncation × grouped bath tree

10サイトHolstein、split、局在初期状態から0–1 fsの比較。
Brownian周波数800、hopping 400、再配置エネルギー800、減衰300 cm⁻¹、300 K。

| ケース | 切断 | 浴の木 |
|---|---|---|
| baseline | 従来の局所SVD | 全モードを二分 |
| weighted | 全状態の振幅を反映 | 全モードを二分 |
| grouped | 従来の局所SVD | 同一結合演算子のモードを先にまとめる |
| both | 全状態の振幅を反映 | 同一結合演算子のモードを先にまとめる |

augmented / parallel BUGの両方で比較する。h=0.1 fs、atol=10⁻⁶、最大ランク64、
Julia 8 workers、BLAS 1 thread、トレース保護あり・事後補正なし。
初期状態からwarm-upした後に各ケースを2回実行する。
厳しいtAMEn参照2条件を同じモデルで計算し、全階層状態と密度行列の差を保存する。
この時間比は設定固定の比較であり、同精度へ最適化した速度比ではない。

測定結果と精度の確認は [比較レポート](../../docs/weighted_tree.md)。
今回の1 fsでは、重み付き切断＋従来の木が両BUG手法で最速だった。

```bash
cd TTNDynamics/benchmarks/holstein_weighted_tree
bash run.sh experiment.jl trial 2
python3 plot.py output/trial
bash run.sh trees.jl trial
```

既存の段階別計時コードとの互換性は `bash run.sh verify_instrumentation.jl` で確認できる。

既存ラベルには上書きしない。実行は約40分のソフト上限を持ち、ステップ境界で確認する。
`trials.csv` に各実行の計時・誤差、`summary.csv` に時間の中央値と範囲を出力する。
`time_ratio_baseline` は同じBUG手法のbaseline時間÷対象時間。
累積割当量はピークメモリ使用量とは異なる。
`trees.jl` はTTNPlotsで最終状態の木と枝ランクをSVG・テキストに出力する。

直接ステップAPIでの指定例：

```julia
tree = heom_tree(system; bath_grouping=:coupling)
x = heom_initial_state(system, rho; tree)
c = trace_observable(system; tree)
y, info = augmented_bug_step(x, heom_generator(system), 0.1;
    truncation=:weighted, atol=1e-6, maxrank=64,
    conserved=c, correct_invariant=false, threaded=true)
```

`parallel_bug_step` も同じキーワードを受け取る。単独の圧縮は
`truncate!(x; weighting=:weighted, ...)`。現在は重み付き切断が既定値で、
`heom_evolve` / `tfd_evolve` の固定・adaptive APIと `adaptive_evolve` からも選べる。
従来方式は `truncation=:local`（単独の切断は `weighting=:local`）。
既定の木は `:balanced`。この比較コードでは切断方式と木を各ケースで明示する。
