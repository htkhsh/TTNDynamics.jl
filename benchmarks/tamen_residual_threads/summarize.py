"""Summarize warmed trials using only the Python standard library."""
import csv
import statistics as st
from pathlib import Path

folder = Path(__file__).resolve().parent / "data"
summary = []
for path in sorted(folder.glob("kernels_t*.csv")):
    rows = list(csv.DictReader(path.open()))
    for problem in dict.fromkeys(row["problem"] for row in rows):
        for method in dict.fromkeys(row["method"] for row in rows):
            group = [r for r in rows if r["problem"] == problem and r["method"] == method]
            med = lambda key: st.median(float(r[key]) for r in group)
            summary.append(dict(problem=problem, method=method, threads=int(group[0]["threads"]),
                seconds=med("seconds"), min_seconds=min(float(r["seconds"]) for r in group),
                max_seconds=max(float(r["seconds"]) for r in group), bytes=med("bytes"),
                gc_seconds=med("gc_seconds"), retained_bytes=med("retained_bytes"),
                max_difference=max(float(r["reference_difference"]) for r in group)))
for path in sorted(folder.glob("trajectory_*.csv")):
    rows = list(csv.DictReader(path.open()))
    med = lambda key: st.median(float(r[key]) for r in rows)
    summary.append(dict(problem="trajectory", method=rows[0]["label"], threads=4,
        seconds=med("seconds"), min_seconds=min(float(r["seconds"]) for r in rows),
        max_seconds=max(float(r["seconds"]) for r in rows), bytes=med("bytes"),
        gc_seconds=med("gc_seconds"), retained_bytes=med("retained_bytes"),
        max_difference=max(float(r["reference_difference"]) for r in rows)))
if not summary:
    raise SystemExit("No benchmark CSV files found; run the benchmark first.")
with (folder / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)
for row in summary:
    print(f'{row["problem"]:12} t{row["threads"]} {row["method"]:9}: '
          f'{row["seconds"]:.6f} s, {row["bytes"]/1e6:.2f} MB allocated, '
          f'{row["retained_bytes"]/1e6:.2f} MB retained, diff {row["max_difference"]:.3g}')
