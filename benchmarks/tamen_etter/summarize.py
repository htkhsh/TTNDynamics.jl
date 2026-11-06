"""Summarize completed run CSVs; standard library only."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

data = Path(__file__).resolve().parent / "data"
groups = defaultdict(list)
for path in sorted(data.glob("runs_threads*.csv")):
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = (row["case"], int(row["threads"]), row["als_sweep"], row["threaded"])
            groups[key].append(row)

fields = ("seconds", "gc_seconds", "bytes", "sweep_seconds", "als_seconds",
          "residual_seconds", "sweeps", "matvecs", "rank", "spawned_tasks")
medians = {key: {name: median(float(r[name]) for r in rows) for name in fields}
           for key, rows in groups.items()}
with (data / "summary.csv").open("w", newline="") as stream:
    names = ("case", "threads", "als_sweep", "threaded", "repeats", *fields,
             "same_algorithm_speedup", "efficiency", "max_reference_difference",
             "max_same_algorithm_difference")
    writer = csv.DictWriter(stream, fieldnames=names)
    writer.writeheader()
    for key, stats in sorted(medians.items()):
        case, threads, algorithm, threaded = key
        base = medians[(case, threads, algorithm, "false")]["seconds"]
        speedup = base / stats["seconds"]
        rows = groups[key]
        row = dict(case=case, threads=threads, als_sweep=algorithm, threaded=threaded,
                   repeats=len(rows), **stats, same_algorithm_speedup=speedup,
                   efficiency=speedup / (threads if threaded == "true" else 1),
                   max_reference_difference=max(float(r["reference_difference"]) for r in rows),
                   max_same_algorithm_difference=max(float(r["same_algorithm_delta"]) for r in rows))
        writer.writerow(row)
        print(f"{case:8} {threads:2} threads {algorithm:6} threaded={threaded:5} "
              f"{stats['seconds']:.6f} s, speedup={speedup:.3f}")
