"""Compare densities on the full common observation grid; no exact-solution claim."""
import argparse
import csv
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent / "data"


def read(label):
    with (BASE / label / "steps.csv").open() as stream:
        return {round(float(r["time_fs"]), 9): r for r in csv.DictReader(stream)}


def density(row):
    return np.array([[complex(float(row[f"rho_{a}_{b}_re"]), float(row[f"rho_{a}_{b}_im"]))
                      for b in range(1, 11)] for a in range(1, 11)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    reference = read(args.reference)
    fields = ["run", "reference", "time_fs", "density_absolute_difference",
              "density_relative_difference", "population_max_difference"]
    output = BASE / f"density_differences_vs_{args.reference}.csv"
    records = []
    for label in args.runs:
        run = read(label)
        common = sorted(reference.keys() & run.keys())
        if not args.partial and not set(np.arange(0., 100.25, .5)).issubset(common):
            raise ValueError(f"Incomplete 0–100 fs comparison for {label}")
        for time in common:
            rho, ref = density(run[time]), density(reference[time])
            difference = rho - ref
            absolute = np.linalg.norm(difference)
            records.append(dict(zip(fields, [label, args.reference, time, absolute,
                               absolute / np.linalg.norm(ref), np.max(np.abs(np.diag(difference)))])))
        subset = [row for row in records if row["run"] == label]
        worst = max(subset, key=lambda row: row["density_absolute_difference"])
        print(f"{label}: compared {common[0]:g}–{common[-1]:g} fs; "
              f"max density difference {worst['density_absolute_difference']:.6g} "
              f"at {worst['time_fs']:g} fs")
    with output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
