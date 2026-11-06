"""Render the accuracy-matched measurements; raw data are never modified."""
import csv
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = ("heom16", "heom32", "heom64")
LABELS = ("HEOM\n17 leaves", "HEOM\n33 leaves", "HEOM\n65 leaves")
METHODS = ("tamen", "parallel_bug")
NAMES = ("TT tAMEn (serial)", "TTN parallel BUG (8 threads)")
COLORS = ("#237c96", "#d78739")
data = {}
for case in CASES:
    with (ROOT / "data" / f"runs_{case}.csv").open() as f:
        rows = list(csv.DictReader(f))
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        assert len(group) == 3
        assert {int(r["repetition"]) for r in group} == {1, 2, 3}
        assert all(float(r["relative_error"]) <= 1e-4 for r in group)
        data[case, method] = group

def values(case, method, key):
    return [float(r[key]) for r in data[case, method]]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
x = np.arange(len(CASES))
for j, (method, label, color) in enumerate(zip(METHODS, NAMES, COLORS)):
    times = [values(case, method, "seconds") for case in CASES]
    centers = np.array([median(t) for t in times])
    errors = np.array([[median(t)-min(t) for t in times], [max(t)-median(t) for t in times]])
    axes[0].bar(x+(j-.5)*.34, centers, width=.32, color=color, label=label)
    axes[0].errorbar(x+(j-.5)*.34, centers, yerr=errors, fmt="none", ecolor="#333333", capsize=3)
    axes[1].scatter(x+(j-.5)*.12, [max(values(c, method, "relative_error")) for c in CASES],
                    color=color, s=60, label=label, zorder=3)
axes[0].set(ylabel="Propagation time [s] (median, min-max)", yscale="log",
            title="Same model, same error target")
axes[1].axhline(1e-4, ls="--", color="#666666", label="Target = 1e-4")
axes[1].set(ylabel="Final full-state relative error", yscale="log",
            title="Independent converged Taylor reference")
for ax in axes:
    ax.set_xticks(x, LABELS)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
axes[0].legend(fontsize=9)
axes[1].legend(fontsize=9)
fig.suptitle("Evolution from t = 0 to 0.25 | M = 4, bath dimension = 3 | BLAS = 1")
for extension in ("svg", "png"):
    fig.savefig(ROOT / "figures" / f"comparison.{extension}", dpi=180)

table = ["| 系 | TT tAMEn 秒 | parallel BUG 秒（8スレッド） | BUG / tAMEn | 誤差 TT / TTN |",
         "|---|---:|---:|---:|---:|"]
settings = ["| 系 / 手法 | 時間設定 | 最終最大ランク | 密度行列誤差 | 不変量ドリフト | 累積割当 GB |",
            "|---|---|---:|---:|---:|---:|"]
preparation = ["| 系 | MPO最大ランク | MPO構築 秒 | tAMEn伝播＋MPO構築 秒 |",
               "|---|---:|---:|---:|"]
for case, label in zip(CASES, LABELS):
    display = label.replace("\n", " / ").replace(" leaves", "葉")
    times = [values(case, method, "seconds") for method in METHODS]
    t, b = (median(a) for a in times)
    ranges = [f"{median(a):.3f} ({min(a):.3f}–{max(a):.3f})" for a in times]
    errs = [max(values(case, m, "relative_error")) for m in METHODS]
    table.append(f"| {display} | {ranges[0]} | {ranges[1]} | {b/t:.2f}× | {errs[0]:.2e} / {errs[1]:.2e} |")
    with (ROOT / "data" / f"mpo_setup_{case}.csv").open() as f:
        prep = list(csv.DictReader(f))
    assert len(prep) == 3
    seconds = median(float(r["seconds"]) for r in prep)
    preparation.append(f"| {display} | {prep[0]['rank']} | {seconds:.3f} | {seconds+t:.3f} |")
    for method in METHODS:
        r = data[case, method][0]
        parameter = (f"Chebyshev 8点, tol={float(r['tolerance']):.0e}" if method == "tamen" else
                     f"{r['steps']}ステップ, h={.25/int(r['steps']):.6g}, maxrank={r['maxrank']}, atol={float(r['atol']):.0e}")
        settings.append(f"| {display} / {method} | {parameter} | {r['rank']} | "
                        f"{max(values(case, method, 'density_error')):.2e} | "
                        f"{max(values(case, method, 'invariant_drift')):.2e} | "
                        f"{median(values(case, method, 'bytes'))/1e9:.3f} |")

report = """# TT tAMEn と並列 TTN BUG の比較

前回の大きな木のベンチマークと同じHEOMモデルについて、初期状態から t=0.25 までの伝播を比較した。
全状態の相対2ノルム誤差 **1e-4 以下**を両手法の共通採用条件とした。
スケール済み全ADOが対象で、密度行列だけの一致ではない。TFDの速度比較は今回の対象外。

![計算時間と実測誤差](../benchmarks/tamen_comparison/figures/comparison.svg)

## 実測結果

時間はウォームアップ後3回の中央値（最小–最大）。比が1より大きいとtAMEnが速い。

""" + "\n".join(table) + """

## 比較条件

- M=4、各浴モードの局所次元3。HEOMの系はfused表現。
- モデル・葉順・結合強度は[前回の大きな木の比較](parallel_large_trees.md)と同じ。
  TTは同じ葉順を列に並べ、同じSoPからMPOを構成した。
- 前回はt=0.20のスナップショットから1ステップだけを計時した。
  今回はt=0からの全伝播であり、前回の秒数とは直接比較できない。
- TTはTTSolverのtAMEn、時間方向は8点Chebyshev、内部の適応区間選択は有効。
  TTNはparallel BUG、固定刻み。どちらもノルムの事後補正なし。
  HEOMのトレース汎関数は両方の基底で保護する。
- BUGの刻みと最大ランク、tAMEnの許容誤差は実測誤差で選択。
  同じ数値の内部tolを使う比較ではなく、全状態の同じ誤差上限を満たす比較である。
  試したパラメータ格子内の設定であり、すべての設定に対する最速性は主張しない。
- BUGは8 Juliaワーカースレッド。tAMEnのコアスイープは逐次、Strided=1。
  両方BLAS=1、GC=1。同じプロセスで交互に3組実行。実行中のGC時間は含む。
- コンパイル、モデル生成、MPO生成、参照解、誤差評価は計時外。
  tAMEnのspace-time構築・終端解抽出・丸め、BUGの全ステップの準備・更新・切り捨ては計時内。
  共有ホスト上の測定であり、最小–最大は観測範囲であって信頼区間ではない。

""" + "\n".join(settings) + """

MPO構築もウォームアップ後3回の中央値を別に測定した。短い伝播では、この一度きりの
準備がtAMEnの伝播本体より長い場合がある。下表の合計はそれぞれの中央値の和で、
共通モデル生成やJulia起動は含まない。生成子が同じ反復計算ではMPOを再利用できる。

""" + "\n".join(preparation) + """

ランクは異なる分割に対する値なのでTTとTTNで直接同じ意味にはならない。
累積割当量はピークメモリではない。ランク上限によりBUGの局所切り捨て予算を超える場合も、
独立参照解に対する全状態誤差で採否を判定し、切り捨て超過数と切り捨て誤差上界の累積をCSVに残している。

## 精度検証と解釈

参照解はtAMEn/BUGから独立した、MPO作用・TT丸めによるTaylor伝播。
粗い設定（10ステップ、丸め1e-9）と細かい設定（20ステップ、丸め1e-10）を比較し、
差が目標誤差の1/100未満であることを確認した。
小さいHEOM/TFDでは密行列指数と参照解・tAMEnを照合している。
[小系の検証](../benchmarks/tamen_comparison/data/verification.txt)は19項目成功し、
[集計検証](../benchmarks/tamen_comparison/data/result_validation.txt)では18本の本計時と108個の記録済みソースハッシュを確認した。
TTNとTTの全状態誤差は密展開せず直接縮約する。二乗ノルムの差を用いるため約1e-7以下の
精度を主張する用途には使わない。各本計時でも1e-4以下を再検証する。

この比較には、木と列の形だけでなく、MPOとSoPの実装差、空間と時間をまとめて解くtAMEnと
小刻みに進めるparallel BUGの時間離散化の差も含まれる。
短時間で低ランクに保たれるこのモデルの結果を、長時間・強結合・別の木形へ一般化しない。
浴の打切り自体の物理的収束を比較したものでもない。

## 再現用ファイル・参考文献

[実行手順](../benchmarks/tamen_comparison/README.md)、
[生データ・環境・ソースハッシュ](../benchmarks/tamen_comparison/data/)、
[比較と独立参照解](../benchmarks/tamen_comparison/common.jl)、
[小系の検証](../benchmarks/tamen_comparison/verify.jl)。
参照解の収束量・準備時間は各 `reference_*.txt`、精度調整の全履歴は `calibration_*.csv` に保存。

- S. V. Dolgov, *A Tensor Decomposition Algorithm for Large ODEs with Conservation Laws*,
  [arXiv:1403.8085](https://arxiv.org/abs/1403.8085)。
  著者の実装: [dolgov/tamen](https://github.com/dolgov/tamen)。本測定はローカルのJulia実装を使用。
- G. Ceruti, J. Kusch, C. Lubich, D. Sulz,
  *A Parallel Basis Update and Galerkin Integrator for Tree Tensor Networks*,
  [doi:10.1137/24M1714502](https://doi.org/10.1137/24M1714502)。
"""
env = dict(line.split(": ", 1) for line in (ROOT / "data/environment_heom16.txt").read_text().splitlines() if ": " in line)
ratios = [median(values(c,"parallel_bug","seconds"))/median(values(c,"tamen","seconds")) for c in CASES]
report = report.replace("## 比較条件", f"この設定ではtAMEnの伝播が約{min(ratios):.0f}–{max(ratios):.0f}倍速い。"
                        "TTの最終ランクは10、MPOランクは8–9に収まり、時間方向も1区間で解けた。"
                        "BUGは全ケース20ステップを要した。木の並列化だけの優劣を表す数値ではない。\n\n"
                        f"実行環境: Julia {env['Julia']}、{env['CPU']}。\n\n## 比較条件", 1)
(ROOT / "../../docs/tamen_comparison.md").resolve().write_text(report)
print("\n".join(table))
