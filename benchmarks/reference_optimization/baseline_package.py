"""Create an isolated pre-optimization package; never modify the live checkout."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--eager", action="store_true", help="enable only the BUG eager checks")
args = parser.parse_args()

here = Path(__file__).resolve().parent
solver = here.parents[2] / "TTNSolver"
target = Path(tempfile.mkdtemp(prefix="ttn-reference-baseline-"))
shutil.copytree(solver / "src", target / "src")
for name in ("Project.toml", "Manifest.toml"):
    if (solver / name).exists():
        shutil.copy2(solver / name, target / name)
subprocess.run(["patch", "--batch", "--forward", "-p1", "-d", str(target),
                "-i", str(here / "data/pre_optimization.patch")], check=True,
               stdout=subprocess.DEVNULL)
if args.eager:
    for name in ("bug.jl", "parallel_bug.jl"):
        path = target / "src" / name
        source = path.read_text()
        assert source.count("verbosity=0)") == 1
        path.write_text(source.replace("verbosity=0)", "eager=true,verbosity=0)"))
print(target)
