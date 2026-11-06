"""Reproduce the pre-optimization cut while retaining phase instrumentation."""
import shutil
import tempfile
from pathlib import Path

solver = Path(__file__).resolve().parents[3] / "TTNSolver"
target = Path(tempfile.mkdtemp(prefix="ttn-before-residual-cut-"))
shutil.copytree(solver / "src", target / "src")
for name in ("Project.toml", "Manifest.toml"):
    if (solver / name).exists():
        shutil.copy2(solver / name, target / name)
path = target / "src/tamen_certification.jl"
source = path.read_text()
new = "_residual_truncate!(residual,work.cut;atol=perterm,maxrank,timings=work.timings)"
old = "_truncate_canonical!(residual;atol=perterm,maxrank,canonicalize=false,timings=work.timings)"
if source.count(new) != 1:
    raise RuntimeError("Expected exactly one residual cut call")
path.write_text(source.replace(new, old))
print(target)
