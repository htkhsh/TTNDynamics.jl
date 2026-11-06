"""Rebuild the HEOM rank-regime report from accuracy-checked timing CSVs."""
import csv
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = ("weak6", "weak8", "moderate8", "strong6", "strong6_short", "system8")
SHORT = ("W6", "W8", "M8", "S6", "S6-short", "Sys8")
LABELS = ("弱結合・6モード", "弱結合・8モード", "中結合・8モード", "強結合・6モード",
          "同じ状態・短区間", "M=8・4モード")
METHODS = ("tamen", "augmented", "parallel")
NAMES = ("TT tAMEn", "Augmented BUG (8 threads)", "Parallel BUG (8 threads)")
COLORS = ("#237c96", "#4b9567", "#d78739")
DATA, PROFILE = {}, {}
for case in CASES:
    with (ROOT / "data" / f"runs_{case}.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (ROOT / "data" / f"profile_{case}.csv").open() as f:
        PROFILE[case] = next(csv.DictReader(f))
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        assert len(group) == 3 and {int(r["repetition"]) for r in group} == {1, 2, 3}
        assert all(float(r["error"]) <= 1e-4 for r in group)
        DATA[case, method] = group

def vals(case, method, key):
    return [float(r[key]) for r in DATA[case, method]]

def time(case, method):
    return median(vals(case, method, "seconds"))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
x = np.arange(len(CASES))
for i, (method, label, color) in enumerate(zip(METHODS, NAMES, COLORS)):
    ts = [vals(c, method, "seconds") for c in CASES]
    mid = np.array([median(v) for v in ts])
    err = np.array([[median(v)-min(v) for v in ts], [max(v)-median(v) for v in ts]])
    axes[0].bar(x+(i-1)*.24, mid, width=.23, color=color, label=label)
    axes[0].errorbar(x+(i-1)*.24, mid, yerr=err, fmt="none", ecolor="#333333", capsize=2)
for method, label, color in zip(METHODS[1:], NAMES[1:], COLORS[1:]):
    axes[1].plot(x, [time(c,"tamen")/time(c,method) for c in CASES], "o-", color=color, label=label)
axes[0].set(yscale="log", ylabel="Propagation time [s] (median, min-max)", title="Actual full-state error <= 1e-4")
axes[1].set(yscale="log", ylabel="tAMEn time / BUG time", title="Above 1: BUG is faster")
axes[1].axhline(1, ls="--", color="#666666")
for ax in axes:
    ax.set_xticks(x, SHORT, rotation=20)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
axes[0].legend(fontsize=8)
axes[1].legend(fontsize=8)
fig.suptitle("HEOM continuation from developed states | BLAS = 1 | temporal basis tuned for tAMEn")
for ext in ("svg", "png"):
    fig.savefig(ROOT / "figures" / f"comparison.{ext}", dpi=180)

table = ["| ケース | 区間長 | 最終ランク TT / TTN | tAMEn 秒 | augmented BUG 秒 | parallel BUG 秒 |",
         "|---|---:|---:|---:|---:|---:|"]
settings = ["| ケース / 手法 | 設定 | 実測相対誤差 | 時間の最小–最大 秒 | 状態格納量 / 密表現 |",
            "|---|---|---:|---:|---:|"]
conditions = ["| 記号 | ケース | M | 浴モード数 | 結合係数倍率 | 準備時刻 | 区間長 | 入力ランク TT / TTN |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
for case, short, label in zip(CASES, SHORT, LABELS):
    p = PROFILE[case]
    ranks = [DATA[case,m][0]["output_rank"] for m in METHODS]
    table.append(f"| {short}: {label} | {float(p['interval']):g} | {ranks[0]} / {ranks[1]} | "
                 + " | ".join(f"{time(case,m):.3f}" for m in METHODS) + " |")
    inputs = [DATA[case,m][0]["input_rank"] for m in ("tamen","augmented")]
    conditions.append(f"| {short} | {label} | {p['m']} | {p['modes']} | {p['strength']} | {p['prep']} | {p['interval']} | {inputs[0]} / {inputs[1]} |")
    for method in METHODS:
        r = DATA[case,method][0]
        setting = (f"Chebyshev {r['points']}点, tol={float(r['tolerance']):.0e}" if method=="tamen" else
                   f"{r['steps']}ステップ, maxrank={r['maxrank']}, atol=1e-8")
        ts = vals(case,method,"seconds")
        ratio = int(r["output_storage"])/int(p["dimension"])
        settings.append(f"| {short} / {method} | {setting} | {max(vals(case,method,'error')):.2e} | {min(ts):.3f}–{max(ts):.3f} | {ratio:.3f} |")

strong_aug = time("strong6","tamen")/time("strong6","augmented")
short_parallel = time("strong6_short","tamen")/time("strong6_short","parallel")
report = f"""# HEOMの必要ランクとBUG系の速度

前回のTTランク10は上限制約ではなく、弱い相関のモデルで観測されたランクだった。
今回、より高いランクが必要な状態を用意し、tAMEnとaugmented / parallel BUGを比較した。
TFDは対象外。比較基準は引き続き、全スケール済みADOの終端相対2ノルム誤差1e-4以下。

**ランクが上がるとBUGが勝つ例はあるが、ランクだけで決まらない。**
S6ではaugmented BUGがtAMEnの{strong_aug:.2f}倍速い。
同じ入力状態・生成子で区間を0.05から0.005へ短くしたS6-shortでは、
tAMEn時間 / parallel BUG時間は{short_parallel:.2f}となった。
一方、より多いモードで圧縮が効くW8・M8では、tAMEnが速い。

![計算時間と速度比](../benchmarks/bug_rank_regimes/figures/comparison.svg)

## 本計測

3回の中央値。tAMEnは逐次スイープ、BUGは8 Juliaワーカー。
BLAS・Strided・GCは1。TTNランク欄はaugmented BUGの値。

""" + "\n".join(table) + """

## 何が効いているか

1. TTとTTNでは切断面が違う。S6ではTTランク81に対しTTNは27で、同じ状態でも
   必要な結合次元が異なる。高いTTランクを小さいTTNランクで表せる条件はBUG側に有利。
2. 同じTTNランクでも時間離散化誤差は異なる。S6のaugmented BUGは1ステップで基準を満たすが、
   parallel BUGは試した1・2・4…の刻みでは64ステップが最初の合格設定だった。
   全ノードを並列化できても、これで時間を失う。
3. S6-shortは初期状態も生成子もS6と同じで、区間長だけを変えた。
   短区間ならparallel BUGも1ステップで基準を満たすため、繰り返しの費用が減る。
   ただし、これを長時間伝播の速度比と読み替えてはいけない。
   一定生成子で長区間をまとめて解けるtAMEnを、出力点のためだけに毎回再起動する必要はない。
4. 高ランクであってもTTN内部コアの処理は重くなる。W8・M8は反例であり、
   「高ランクならBUGが必ず速い」という結果ではない。

S6やSys8は小さい全空間に対してほぼ全ランクとなり、テンソル形式の格納量が密表現を超える。
これは有限サイズでのアルゴリズム比較であり、巨大で十分に圧縮された高ランクHEOMにおける
BUGの優位性を実証したものではない。通常の疎行列伝播との優劣も本表からは結論しない。

## モデルと計時範囲

前回の対角結合モデルとは異なる制御モデル。乱数種119で生成したHermitian結合行列を
各モードで演算子ノルム1に規格化する。非可換な結合を採用し、相関を増やした。
Hは隣接ホッピング0.25±0.07iと対角成分-0.2…0.2、浴減衰率0.8…1.5。
係数は倍率×(0.08+0.01i)×8/K、逆向き係数は複素共役。全モードの局所次元3、系はfused。

""" + "\n".join(conditions) + """

全ADOの疎行列Krylov計算（tol=1e-12）で準備状態と終端参照解を求める。
準備状態をTT・TTNにそれぞれ圧縮（1e-5）し、圧縮誤差を含めた終端誤差で判定する。
両入力が数値的に完全に同じ配列という意味ではなく、同じ物理的準備状態の近似である。
入力誤差の上限チェックも行う。圧縮1e-7で調べた `profile_*.csv` のランクは
本計測の入力ランクと区別する。

tAMEnの時間方向4・8・12点を試し、誤差基準を満たす条件のウォーム済み計測から設定を選ぶ。
BUGの固定刻みも実測誤差で選択し、トレース保護に必要な追加基底をランク上限に許容する。
各手法の内部tolは同じ値にそろえない。試した有限の設定格子での比較であり、
各手法の絶対的最速設定を保証するものではない。

準備状態の生成・圧縮、MPO構築、参照解、誤差評価は計時外。
時間発展内の準備・局所解・切り捨て・tAMEnの終端解抽出は計時内。
各手法を1回ウォームアップし、実行順を回して3回測定する。
共有ホスト上の最小–最大は観測範囲であって信頼区間ではない。
局所切り捨て予算を超えた場合も、全状態の実測誤差で採否を決め、超過回数をCSVに残す。
階層打切りの物理的収束は調べていない。

## 設定・精度・格納量

格納量の比は、テンソルの複素数要素数 / 全ADOの複素数要素数。ワークスペースやピークメモリではない。

""" + "\n".join(settings) + """

## 再現と検証

[実行手順](../benchmarks/bug_rank_regimes/README.md)、
[生データとソースハッシュ](../benchmarks/bug_rank_regimes/data/)、
[モデル・分解・参照解](../benchmarks/bug_rank_regimes/common.jl)、
[検証](../benchmarks/bug_rank_regimes/verify.jl)。
疎行列生成子は直接HEOM方程式と照合し、低次元の密行列指数とも照合した。
TT/TTN分解の複素共役・軸順・積状態のランクを検証する。
[8項目のテストと対照条件の確認](../benchmarks/bug_rank_regimes/data/verification.txt)が成功。
S6/S6-shortの入力状態差はTT・TTNとも2e-15以下、MPO差は0で、伝播による入力の変化も0だった。
[集計検証](../benchmarks/bug_rank_regimes/data/result_validation.txt)で54本の計測と228件のソースハッシュ照合を確認した。

前回の結果・算法の参考文献は[TT tAMEn比較](tamen_comparison.md)を参照。
高ランクの状態から短区間を伝播する今回の秒数は、前回の初期状態からt=0.25までの秒数と直接比較しない。
"""
(ROOT / "../../docs/bug_rank_regimes.md").resolve().write_text(report)
print("\n".join(table))
