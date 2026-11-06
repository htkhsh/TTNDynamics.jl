"""Summarize alternating warmed serial/threaded trials; Python standard library only."""
import csv
from pathlib import Path
from statistics import median

DATA = Path(__file__).resolve().parent / "data"
summary = []
for kind in ("solves", "environments"):
    for threads in (1, 2, 4, 8):
        with (DATA / f"{kind}_threads{threads}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for case in dict.fromkeys(row["case"] for row in rows):
            serial = [r for r in rows if r["case"] == case and r["threaded"] == "false"]
            parallel = [r for r in rows if r["case"] == case and r["threaded"] == "true"]
            assert len(serial) == len(parallel) == 5, "Incomplete five-repeat measurement"
            t1 = median(float(r["seconds"]) for r in serial)
            tp = median(float(r["seconds"]) for r in parallel)
            result = dict(kind=kind, case=case, threads=threads,
                          serial_seconds=t1, threaded_seconds=tp,
                          speedup=t1/tp, efficiency=t1/tp/threads,
                          serial_min=min(float(r["seconds"]) for r in serial),
                          serial_max=max(float(r["seconds"]) for r in serial),
                          threaded_min=min(float(r["seconds"]) for r in parallel),
                          threaded_max=max(float(r["seconds"]) for r in parallel),
                          serial_environment_fraction="", serial_residual_fraction="",
                          serial_gc_seconds=median(float(r["gc_seconds"]) for r in serial),
                          threaded_gc_seconds=median(float(r["gc_seconds"]) for r in parallel))
            if kind == "solves":
                for stage in ("environment", "residual"):
                    result[f"serial_{stage}_fraction"] = median(
                        float(r[f"{stage}_seconds"])/float(r["seconds"]) for r in serial)
            summary.append(result)
# Also retain conventional strong scaling against the common one-thread run.
# Paired on/off comparisons above distinguish the task scheduler's effect from
# changes in runtime/GC configuration or drift across separate processes.
for row in summary:
    baseline = next(r["serial_seconds"] for r in summary
                    if r["case"] == row["case"] and r["threads"] == 1)
    row["one_thread_speedup"] = baseline / row["threaded_seconds"]
    row["one_thread_efficiency"] = row["one_thread_speedup"] / row["threads"]
with (DATA / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)
for row in summary:
    print(f'{row["case"]:16s} p={row["threads"]} '
          f'serial={row["serial_seconds"]:.6f}s threaded={row["threaded_seconds"]:.6f}s '
          f'S={row["speedup"]:.3f} E={100*row["efficiency"]:.1f}%')
