import csv
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent/"output"/os.environ.get("PS2_OUTPUT","holstein5_eager")
rows=list(csv.DictReader((root/"summary.csv").open()))
fig,axes=plt.subplots(1,3,figsize=(12,4.2))
fig.subplots_adjust(left=.065,right=.99,top=.97,bottom=.27,wspace=.32)
labels={"tt":"TT tAMEn","ps2_five_fs":"PS2 selected","ps2_h02":"PS2 h=0.2 fs, padding 2",
        "ps2_h01":"PS2 h=0.1 fs","ps2_h005":"PS2 h=0.05 fs",
        "ps2_h01_pad1":"PS2 h=0.1 fs, padding 1","ps2_h02_pad1":"PS2 h=0.2 fs, padding 1",
        "ps2_h005_pad1":"PS2 h=0.05 fs, padding 1","ps2_five_fs_refined":"PS2 refined"}
for name in dict.fromkeys(r["name"] for r in rows):
    group=[r for r in rows if r["name"]==name]
    t=[float(r["time_fs"]) for r in group]
    style="o-" if name=="tt" or name.startswith("ps2_five_fs") else "x"
    label=labels.get(name,name)
    if name.startswith("ps2_five_fs"):
        label+=f" (h={float(group[0]['dt_fs']):g} fs)"
    for ax,key in zip(axes,("seconds","max_state_error","peak_rank")):
        ax.plot(t,[float(r[key]) for r in group],style,label=label)
for ax in axes:
    ax.set_xlabel("Time (fs)");ax.grid(alpha=.2);ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Propagation time (s)");axes[0].set_yscale("log")
handles,names=axes[0].get_legend_handles_labels()
fig.legend(handles,names,loc="lower center",bbox_to_anchor=(.5,.015),ncol=3,frameon=False,fontsize=8)
axes[1].set_ylabel("Maximum state error so far");axes[1].set_yscale("log")
axes[1].axhline(1e-3,color=".4",ls="--",lw=1)
axes[1].text(.98,1e-3,"Target: 1e-3",transform=axes[1].get_yaxis_transform(),ha="right",va="bottom",fontsize=8)
axes[2].set_ylabel("Peak stored bond rank")
for ext in ("png","svg"):fig.savefig(root/f"comparison.{ext}",dpi=180)
