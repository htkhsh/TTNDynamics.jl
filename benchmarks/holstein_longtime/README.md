# Holstein HEOMの長時間比較

TT＋tAMEn、augmented BUG、parallel BUGを同じモデルで比較する。
設定は [config.toml](config.toml)、実行・収束確認・再開の詳細は
[docs/holstein_longtime.md](../../docs/holstein_longtime.md)。

既定値：10サイト、split ket/bra、サイト1局在状態、0–100 fs。
Brownian浴の周波数800、飛び移り積分400、減衰300、再配置エネルギー800 cm⁻¹、300 K。
BUGは重み付き切断（`truncation = "weighted"`）と従来のbalanced treeを使う。
従来の切断は `truncation = "local"` で選べる。
50 fs用の [config_50fs.toml](config_50fs.toml) は `output/holstein50_weighted` に出力し、
以前の `output/holstein50` と分ける。切断方式が異なるチェックポイントは再開できない。

```bash
cd TTNDynamics/benchmarks/holstein_longtime
bash run.sh run.jl config.toml tamen --wall-seconds 3600
bash run.sh run.jl config.toml augmented --wall-seconds 3600
bash run.sh run.jl config.toml parallel --wall-seconds 3600
bash run.sh run.jl config.toml reference --wall-seconds 3600
bash run.sh run.jl config.toml reference_refined --wall-seconds 3600
```

時間枠で中断した実行には `--resume` を追加して同じコマンドを再実行する。
時間枠は各起動に対するソフト上限で、実行中の1ステップは完了させる。

```bash
bash run.sh run.jl config.toml augmented --resume --wall-seconds 3600
bash run.sh compare.jl config.toml
python3 plot.py output
```

100 fsの本計算は準備時に実行していない。動作確認は次で行う。

```bash
bash run.sh test_config.jl
bash run.sh test_integrity.jl
bash run.sh verify.jl
bash run.sh smoke.jl smoke    # 実際の10サイト、0–0.2 fsのみ。既存ラベルには上書きしない
python3 plot.py output/smoke
```
