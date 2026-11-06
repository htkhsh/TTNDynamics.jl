# 残差圧縮の最適化と計時

加算・QRの[バッファ再利用と部分木並列化](tamen_residual_threads.md)に続き、
大域残差の重み付き圧縮を最適化した。演算子の項順序、切り詰め予算、ランク上限、
切り捨て誤差の累積上界、採用判定は変更していない。

## 変更点

`info.timings.residual_global_parts` に、区間ごとの以下の実時間を記録する。

- `addition_seconds`: 項の加算と正準化。子タスクの待ち合わせを含む。
- `weight_qr_seconds`: 相補空間の重み付けとQR。
- `svd_seconds`: SVDと保持ランクの選択。
- `basis_update_seconds`: 相補因子、親・子コアの更新。
- `final_qr_seconds`: 残差ノルムを求める直前の最終QR。

各項は重ならず、`residual_global_seconds` の内訳になる。細かな管理処理は含まず、
合計値にこれらを再加算してはいけない。共有ワークスペースの累積値から区間差分を返す。

保存量の保護方向がない大域残差だけに専用の圧縮経路を設けた。単位行列の生成・
行列積を省き、振幅、QR入力・反射子・R、SVD入力、基底変換の一時領域を再利用する。
SVDは破壊してよい専用領域へ `svd!` を適用する。相補因子はノード別に保持し、
返却テンソルは作業領域と配列を共有しない。SVDの出力とLAPACK内部作業領域の確保は残る。
通常の `truncate!` と保存量を保護する圧縮には、従来の経路を使う。

## 測定結果

2026-11-06、Julia1.12.6、Threadripper PRO9985WX、Julia4スレッド・BLAS1。
ウォームアップ後5回の中央値。比較前後を別プロセスで順番に実行した。
コンパイル・初期化・明示GCは計時外、処理中のGCは計時内。

8浴モード・2状態split HEOM、各浴の局所次元3、Chebyshev点数5、区間幅0.2、
線形許容誤差1e-8、解のmaxrank32、Etter ALSを使用。
残差単体は同一の保存済み時空間状態、残差maxrank64で比較した。
時間は無次元であり、10サイトHolstein本計算とは別の検証用問題である。

| 対象 | 最適化前 | 最適化後 | 倍率 | 累積確保量 前→後 |
|---|---:|---:|---:|---:|
| 残差検証1回 | 0.2234秒 | 0.2130秒 | 1.05倍 | 395.81→167.10 MB |
| 5区間の計算全体 | 4.5249秒 | 4.0227秒 | 1.12倍 | 6.398→3.189 GB |

累積確保量はピークメモリ使用量ではない。5区間全体の確保量は50.2%減った。
一方、再利用のため保持する残差作業領域の数値配列は、単体検証4.83→7.33 MB、
5区間15.69→23.49 MBと増える。ベクトル容量の余裕や管理情報を含まない。

5区間の計時内訳（各項の中央値なので合計は全体中央値と厳密には一致しない）：

| 処理 | 最適化前（秒） | 最適化後（秒） |
|---|---:|---:|
| 加算・正準化 | 1.920 | 1.815 |
| 重み付けQR | 1.059 | 0.769 |
| SVD・ランク選択 | 0.347 | 0.296 |
| 基底更新 | 0.575 | 0.467 |
| 最終QR | 0.013 | 0.014 |
| 大域残差検証の全体 | 3.953 | 3.390 |

解のスイープ10回、大域検証6回、最終ランク26、最大相対残差上界7.17e-9は一致。
終状態の相対差は1.12e-15、トレース誤差は9.27e-13。
速度改善は穏やかで、残差単体の計時にはばらつきがある。
加算・正準化は引き続き大きなコストであり、今回の変更だけでTT版との速度差が
解消されたとは判断できない。

## 検証・再現

TTNSolver2533件、TTNDynamics595件の全テストが通過した。
追加テストは複素数、ランク制約、ゼロ・微小振幅、木やランクの変更、作業領域の
上書き後の出力所有権、時間内訳の区間差分を確認する。
QRブロック幅36の前後、縦長・横長行列を通常のテストにも含めた。
独立レビューの1302件でも境界形状、微小振幅、出力所有権を確認した。

共有パッケージディレクトリから次を順番に実行する。

```bash
export JULIA_DEPOT_PATH=/tmp/ttn-tamen-depot:/home/takahashi/.julia
export OPENBLAS_NUM_THREADS=1
previous_pkg=$(python3 TTNDynamics/benchmarks/tamen_residual_compression/reference_package.py)
TAMEN_CUT_LABEL=baseline julia --threads=4 --project="$previous_pkg" \
  TTNDynamics/benchmarks/tamen_residual_compression/run.jl
julia --threads=4 --project=TTNSolver TTNDynamics/benchmarks/tamen_residual_compression/run.jl
python3 TTNDynamics/benchmarks/tamen_residual_compression/summarize.py
```

[測定コード](../benchmarks/tamen_residual_compression/run.jl)、
[生データと要約](../benchmarks/tamen_residual_compression/data/summary.csv)。
参照パッケージは大域残差の圧縮呼び出しだけを従来の汎用経路に戻す。
