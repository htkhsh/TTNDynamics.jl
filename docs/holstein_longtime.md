# Holsteinの長時間TT / TTN比較

[benchmarks/holstein_longtime](../benchmarks/holstein_longtime) に、TT＋tAMEnと
augmented / parallel BUGの比較コードを置いた。設定ファイルから同じ物理モデルを生成し、
共通の観測時刻・状態保存時刻で比較する。100 fsの本計算はまだ実行していない。

## 物理設定

[config.toml](../benchmarks/holstein_longtime/config.toml) の `[model]` と `[time]` が共通設定。

| 項目 | 既定値 |
|---|---:|
| サイト数・境界条件 | 10、周期境界 |
| 初期状態 | サイト1の局在状態、階層添字は真空 |
| 系の表現 | TT・TTNともket/braを分離 |
| 飛び移り積分 | 400 cm⁻¹ |
| Brownian浴の周波数 | 800 cm⁻¹ |
| 減衰 | 300 cm⁻¹ |
| 再配置エネルギー | 800 cm⁻¹ |
| 温度 | 300 K |
| 浴展開・階層 | PSD、Padé order 0、各階層添字のサイズ5 |
| 終了時刻 | 100 fs |
| 密度行列の観測間隔 | 1 fs |
| 全階層状態の保存間隔 | 10 fs（初期・最終状態を含む） |

元の `TTDynamics/examples/holstein/config.jl` は変更しない。
Brownian分解後の `NoiseExp.c1/c2` と `sqrt(abs_coeff)` を両表現で共有する。
TTのMPOとTTNの積和演算子の一致は小系で検証する。
Padé次数・階層サイズは物理モデルの収束も確認するための設定であり、既定値の収束は未確認。

さらに長く走らせる場合は、**開始前に** `end_fs` を1000などへ変更する。
再開時は物理設定・時間格子・選択した手法の設定を一致させる。
既存実行の終了時刻や刻み幅を変更する場合は、別の `output_directory` で開始する。
観測間隔は各手法の刻み幅の整数倍、状態保存間隔は観測間隔の整数倍とする。

## 実行するケース

| 名前 | 手法 | 主な設定 |
|---|---|---|
| `tamen` | TT＋tAMEn | 外側0.5 fs、Chebyshev12、tol=10⁻⁴ |
| `reference` | TT＋tAMEn | 外側0.25 fs、tol=10⁻⁶、反復数を増やす |
| `reference_refined` | TT＋tAMEn | 外側0.125 fs、tol=10⁻⁷、反復数をさらに増やす |
| `augmented` | augmented BUG | 0.1 fs、最大ランク64 |
| `parallel` | parallel BUG | 0.1 fs、最大ランク64 |
| `augmented_hhalf` / `parallel_hhalf` | BUGの時間刻み確認 | 0.05 fs |
| `augmented_rank` / `parallel_rank` | BUGのランク上限確認 | 最大ランク128 |

BUGの切断許容誤差は `truncation_rate × dt_fs`（既定rate=10⁻⁵）、局所Krylov許容誤差は10⁻¹²。
切断方式の既定値は `truncation = "weighted"`。`"local"` で従来の局所切断に戻せる。
木は従来のbalanced treeを使う。切断方式はrun identityにも記録する。
`config_50fs.toml` の出力先は `output/holstein50_weighted` に変更しており、
旧 `output/holstein50` のデータには書き込まない。
トレース方向を保護し、事後のトレース補正は使わない。tAMEnにもトレース観測量を渡す。
tAMEnは許容誤差によるランク適応を使い、このAPIにはBUGと同じ `maxrank` を渡さない。
ランクは全ケースで記録する。

`extends` により基準ケースを継承できる。刻み幅やランク上限だけを変更したケースを追加し、
`compare.jl` の引数に指定すれば比較対象へ加えられる。時間刻みとランク上限は別々に確認する。
参照確認用のケースは、参照より刻み幅・許容誤差を緩めず、時間基底を減らさずに、
少なくともいずれかを厳しくする必要がある。

## 計時と再開

```bash
cd TTNDynamics/benchmarks/holstein_longtime
bash run.sh run.jl config.toml tamen --wall-seconds 3600
bash run.sh run.jl config.toml augmented --wall-seconds 3600
bash run.sh run.jl config.toml parallel --wall-seconds 3600
bash run.sh run.jl config.toml reference --wall-seconds 3600
bash run.sh run.jl config.toml reference_refined --wall-seconds 3600
```

比較する計算は順番に実行する。既定ではJulia 8 workers、tAMEnはBLAS 8 threads、
BUGはBLAS 1 thread、Stridedは1 thread。Julia worker数は起動前の
`HOLSTEIN_JULIA_THREADS` で指定できる。実際のCPU・Julia・スレッド設定を保存する。
既存の開発版パッケージとHolstein例の環境を使い、必要なら `HOLSTEIN_DEPOT_PATH` を指定する。

`--wall-seconds` は**1回の起動**に対するソフト上限。準備とwarm-upも時間枠に含むが、
進行中の準備処理や1ステップを強制終了しないため、上限を超えることがある。
`PAUSED` は未完了を意味する。次回は同じ設定で再開する。

```bash
bash run.sh run.jl config.toml augmented --resume --wall-seconds 3600
```

伝播時間はステップの計算だけを加算し、コンパイル用warm-up、観測、保存を除く。
各セッションの実経過時間とwarm-up時間は `sessions.csv` に別記する。
再開した実行も、採用された軌道に対応する伝播時間を継続加算する。

最新状態は観測時刻と正常な中断時に一時ファイル経由で保存する。
異常終了した場合は最後の保存時点から再計算し、その状態より新しいCSV末尾を切り戻す。
tAMEnの乱数シードは `seed + step` とし、再開やwarm-upの有無に依存させない。
スクラッチ用workspaceは再構築する。状態・設定・依存コード・実行条件が合わない再開は拒否する。

同じrunへの二重書込みはロックディレクトリで防ぐ。強制killで `output/RUN.lock` が
残った場合は、書込みプロセスが停止したことを確認して `rmdir output/RUN.lock` で除去する。
Julia Serializationを使うため、チェックポイントの再開には同じJuliaとコードを用いる。

## 比較と図

```bash
bash run.sh compare.jl config.toml
python3 plot.py output

# 収束確認用ケースも比較する例
bash run.sh compare.jl config.toml tamen augmented augmented_hhalf augmented_rank parallel

# 未完了の共通区間だけを見る場合
bash run.sh compare.jl config.toml tamen augmented --partial
python3 plot.py output --partial
```

既定では参照と比較対象が終了時刻まで完了している必要がある。
`--partial` は共通の観測時刻までに制限し、`comparison_partial/` に明示して出力する。
異なる物理設定・時間格子・コード版・CPU/Julia/worker条件の結果を混ぜた比較は拒否する。
BLAS数はケースごとに設定した値との一致を確認する。

- `state_differences.csv`：保存時刻での全階層状態の相対差。密展開せずTT–TTN縮約で求める。
- `density_differences.csv`：観測時刻での密度行列Frobenius差とサイト占有数の最大差。
- `summary.csv`：共通区間の伝播時間、最大ランク、切断予算未達、最大差、精度判定。
- `comparison.svg/png`：占有数、ランク、累積伝播時間、参照との差。

TT–TTN縮約ではTTの右側ボンドを分割し、大きな中間配列を一度に作らない。
作業配列の予算は64 MiBを目安とするが、1スライスや保存中の部分木環境まで含めた
総メモリ上限ではない。全状態の相対差はノルムと内積から求めるため、
10⁻⁷程度以下では浮動小数点の桁落ちに注意が必要。

参照確認計算が未完了なら `reference_status=unchecked` とし、生の差と時間だけを示す。
参照と厳しい参照の最大差が目標値の1/10以下なら `checked`、超える場合は `failed`。
`speedup_vs_tamen` は全区間が揃い、参照確認と比較対象の精度目標が通った場合だけ出す。
満たさない場合は `NA`。目標は `[comparison]` のstate/density双方で、既定は10⁻³。
速度比は「tAMEnの伝播時間 ÷ 対象の伝播時間」で、1より大きければ対象が速い。
参照同士の一致は厳密解に対する保証ではなく、これらの判定も保存・観測時刻における比較である。

## 出力と動作確認

各runの `manifest.toml` に解決済み設定とコードのSHA256、`steps.csv` に毎ステップの
時間・GC・割当量・ランク・切断診断、`observables.csv` に密度行列等を保存する。
`latest.jls` が再開の基準で、`checkpoints/` は全階層状態の比較用。
大きな出力はgit対象外。`--max-steps N` で1回の起動をNステップまでに制限できる。

```bash
bash run.sh test_config.jl
bash run.sh test_integrity.jl
bash run.sh verify.jl
bash run.sh smoke.jl smoke
python3 plot.py output/smoke
```

`verify.jl` は2サイトの独立な密行列解、3手法の中断・再開一致、CSV末尾回復等を確認する。
`smoke.jl` は指定された10サイトの物理設定で0.2 fsだけ実行する動作確認。
長時間の精度や速度の結論には使わない。

## 利用したコード

- [TTDynamics Holstein例](../../TTDynamics/examples/holstein)：Brownian浴分解、HEOM-TT生成子、初期状態。
- [既存のsplit比較](../benchmarks/holstein10_split/common.jl)：同じNoiseExpからのTTN構築。
- [TT–TTN縮約](../benchmarks/tamen_comparison/common.jl)：全階層状態の比較。
- [tAMEn比較と文献](tamen_comparison.md)：tAMEnの実装と参照した文献。
- [実装済み積分法と文献](guide.md)：BUGの実装・文献への案内。
