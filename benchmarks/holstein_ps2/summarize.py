import csv
import os
import statistics as st
import tomllib
from pathlib import Path

root=Path(__file__).resolve().parent/"output"/os.environ.get("PS2_OUTPUT","holstein5_eager")
rows=list(csv.DictReader((root/"comparison.csv").open()))
summary=[]
for name in dict.fromkeys(r["name"] for r in rows):
    spec=tomllib.loads((root/name/"metadata.toml").read_text())["spec"]
    group=[r for r in rows if r["name"]==name]
    for t in sorted({int(r["time_fs"]) for r in group}):
        samples=[r for r in group if int(r["time_fs"])==t]
        baseline=[r for r in rows if r["name"]=="tt" and int(r["time_fs"])==t]
        seconds=st.median(float(r["seconds"]) for r in samples)
        qualified=all(r["qualified"]=="true" for r in samples)
        ratio=seconds/st.median(float(r["seconds"]) for r in baseline) if qualified and baseline and all(r["qualified"]=="true" for r in baseline) else None
        summary.append(dict(name=name,time_fs=t,complete_5fs=t==5,repetitions=len(samples),
            dt_fs=spec["dt_fs"],truncation_rate=spec.get("truncation_rate"),maxrank_cap=spec.get("maxrank"),
            rank_padding=spec.get("rank_padding"),blas_threads=spec["blas_threads"],seconds=seconds,
            min_seconds=min(float(r["seconds"]) for r in samples),max_seconds=max(float(r["seconds"]) for r in samples),
            allocated_GB=st.median(int(r["bytes"]) for r in samples)/1e9,
            peak_rank=max(int(r["peak_rank"]) for r in samples),
            max_local_dimension=max(int(r["max_local_dimension"]) for r in samples),
            cut_budget_failures=max(int(r["cut_budget_failures"]) for r in samples),
            max_state_error=max(float(r["max_state_error"]) for r in samples),
            max_density_error=max(float(r["max_density_error"]) for r in samples),
            max_trace_error=max(float(r["max_trace_error"]) for r in samples),
            qualified=qualified,time_ratio_vs_tt=ratio))
if summary:
    with (root/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=summary[0].keys());writer.writeheader();writer.writerows(summary)
for name in dict.fromkeys(r["name"] for r in summary):
    print([r for r in summary if r["name"]==name][-1])

# Use only complete, accuracy-checked integer-fs prefixes. A partial next
# interval must not enter the timing breakdown of a completed checkpoint.
phases=[]
for row in rows:
    if row["name"]=="tt":
        continue
    steps=list(csv.DictReader((root/row["name"]/f"rep{row['repetition']}"/"steps.csv").open()))
    steps=[s for s in steps if float(s["time_fs"])<=float(row["time_fs"])+1e-9]
    elapsed=float(row["seconds"])
    totals={key:sum(float(s[key]) for s in steps)
            for key in ("solve_seconds","environment_seconds","split_seconds","gc_seconds")}
    # GC overlaps the phase timers; it is not an additional disjoint phase.
    phases.append(dict(name=row["name"],repetition=row["repetition"],time_fs=row["time_fs"],
        propagation_seconds=elapsed,**totals,
        other_seconds=elapsed-sum(totals[k] for k in ("solve_seconds","environment_seconds","split_seconds")),
        solve_fraction=totals["solve_seconds"]/elapsed,
        local_solves=sum(int(s["local_solves"]) for s in steps),
        matvecs=sum(int(s["matvecs"]) for s in steps)))
if phases:
    with (root/"phase_summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=phases[0].keys());writer.writeheader();writer.writerows(phases)
