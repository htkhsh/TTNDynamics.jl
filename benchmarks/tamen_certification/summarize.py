"""Summarize separately measured legacy and optimized runs."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

data = Path(__file__).resolve().parent / "data"
groups = defaultdict(list)
for label in ("reference", "optimized"):
    with (data / f"runs_{label}.csv").open() as stream:
        for row in csv.DictReader(stream):
            groups[(label, row["case"])].append(row)
fields = ("seconds", "bytes", "gc_seconds", "residual_seconds", "global_seconds",
          "als_seconds", "sweeps", "checks", "maxrank")
stats = {key: {field: median(float(r[field]) for r in rows) for field in fields}
         for key, rows in groups.items()}
with (data / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=("case", "label", "repeats", *fields,
        "speedup", "allocation_ratio", "max_reference_difference", "max_trace_error"))
    writer.writeheader()
    for case in ("certificate", "interval", "trajectory"):
        reference = stats[("reference", case)]
        for label in ("reference", "optimized"):
            values = stats[(label, case)]
            rows = groups[(label, case)]
            row = dict(case=case, label=label, repeats=len(rows), **values,
                speedup=reference["seconds"] / values["seconds"],
                allocation_ratio=values["bytes"] / reference["bytes"],
                max_reference_difference=max(float(r["reference_difference"]) for r in rows),
                max_trace_error=max(float(r["trace_error"]) for r in rows))
            writer.writerow(row)
            print(f"{case:12} {label:9} {values['seconds']:.6f} s, "
                  f"{values['bytes']/1e9:.4f} GB allocated, speedup {row['speedup']:.3f}")
