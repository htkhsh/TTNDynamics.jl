"""Summarize only observed common times; never extrapolate incomplete runs."""
import csv
import statistics as st
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parent / "output/five_fs"
rows = list(csv.DictReader((root / "comparison.csv").open()))
summary = []
phases = []
phase_keys = ("global_seconds", "addition_seconds", "weight_qr_seconds", "svd_seconds",
              "basis_update_seconds", "final_qr_seconds", "gc_seconds")
for name in dict.fromkeys(r["name"] for r in rows):
    metadata = tomllib.loads((root / name / "metadata.toml").read_text())
    spec = metadata["spec"]
    group = [r for r in rows if r["name"] == name]
    for t in sorted({int(r["time_fs"]) for r in group}):
        at_time = [r for r in group if int(r["time_fs"]) == t]
        times = [float(r["seconds"]) for r in at_time]
        baseline = [r for r in rows if r["name"] == "tt" and int(r["time_fs"]) == t]
        qualified = all(r["qualified"] == "true" for r in at_time)
        peak = []
        allocations = []
        global_time = []
        phase_samples = {key: [] for key in phase_keys}
        check_counts = []
        sweep_counts = []
        for row in at_time:
            with (root / name / f'rep{row["repetition"]}' / "steps.csv").open() as stream:
                steps = [r for r in csv.DictReader(stream) if float(r["time_fs"]) <= t + 1e-12]
            peak.append(max(int(r["maxrank"]) for r in steps))
            allocations.append(sum(int(r["bytes"]) for r in steps) / 1e9)
            global_time.append(sum(float(r["global_seconds"]) for r in steps))
            for key in phase_keys:
                phase_samples[key].append(sum(float(r[key]) for r in steps))
            check_counts.append(sum(int(r["global_checks"]) for r in steps))
            sweep_counts.append(sum(int(r["sweeps"]) for r in steps))
        ratio = (st.median(times) / st.median(float(r["seconds"]) for r in baseline)
                 if qualified and baseline and all(r["qualified"] == "true" for r in baseline) else None)
        summary.append(dict(name=name, time_fs=t, repetitions=len(times), complete_5fs=t == 5,
            dt_fs=spec["dt_fs"], time_points=spec["points"], tolerance=spec["tolerance"],
            julia_threads=metadata["runtime"]["julia_threads"], blas_threads=spec["blas_threads"],
            seconds=st.median(times), min_seconds=min(times), max_seconds=max(times),
            allocated_GB=st.median(allocations), peak_rank=max(peak),
            max_state_error=max(float(r["max_state_error"]) for r in at_time),
            max_density_error=max(float(r["max_density_error"]) for r in at_time),
            max_trace_error=max(float(r["max_trace_error"]) for r in at_time),
            qualified=qualified, time_ratio_vs_tt=ratio,
            global_fraction=(st.median(global_time) / st.median(times) if name.startswith("tree") else None)))
        if name.startswith("tree"):
            phases.append(dict(name=name, time_fs=t, repetitions=len(times),
                seconds=st.median(times), global_checks=st.median(check_counts),
                sweeps=st.median(sweep_counts),
                **{key: st.median(values) for key, values in phase_samples.items()}))
with (root / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)
if phases:
    with (root / "phase_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=phases[0].keys())
        writer.writeheader()
        writer.writerows(phases)
for name in dict.fromkeys(r["name"] for r in summary):
    print([r for r in summary if r["name"] == name][-1])
