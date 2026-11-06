"""Create an isolated package with the original residual assembly for comparison."""
import shutil
import tempfile
from pathlib import Path

solver = Path(__file__).resolve().parents[3] / "TTNSolver"
target = Path(tempfile.mkdtemp(prefix="ttn-residual-reference-"))
shutil.copytree(solver / "src", target / "src")
for name in ("Project.toml", "Manifest.toml"):
    source = solver / name
    if source.exists():
        shutil.copy2(source, target / name)
reference = (solver / "test/fixtures/tamen_residual_reference.jl").read_text()
(target / "src/tamen_certification.jl").write_text(
    reference + '''
function _tamen_residual(A,x,b,budget,maxrank;workspace=nothing,threaded=false)
    reference_tamen_residual(A,x,b,budget,maxrank)
end
''')
print(target)
