import csv
import statistics
from pathlib import Path

folder = Path(__file__).resolve().parent / "data"
summary = []
for label in ("baseline", "optimized"):
    rows = list(csv.DictReader((folder / f"runs_{label}.csv").open()))
    for case in ("certificate", "trajectory"):
        group = [r for r in rows if r["case"] == case]
        row = {"label": label, "case": case}
        for key in ("seconds", "bytes", "gc_seconds", "global_seconds", "addition_seconds",
                    "weight_qr_seconds", "svd_seconds", "basis_update_seconds", "final_qr_seconds",
                    "retained_bytes"):
            row[key] = statistics.median(float(r[key]) for r in group)
        row["min_seconds"] = min(float(r["seconds"]) for r in group)
        row["max_seconds"] = max(float(r["seconds"]) for r in group)
        row["max_difference"] = max(float(r["difference"]) for r in group)
        summary.append(row)
with (folder / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)
for row in summary:
    print(row)
