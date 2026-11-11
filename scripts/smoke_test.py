"""Executes every notebook end-to-end on synthetic data (T2S_SMOKE=1). Usage: python scripts/smoke_test.py <workdir>"""
import os, shutil, sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
work = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".smoke"
if work.exists():
    shutil.rmtree(work)
for d in ["src", "app"]:
    shutil.copytree(ROOT / d, work / d)
(work / "reports").mkdir(parents=True)
import warnings
os.environ.update(FC_SMOKE="1", FC_PROJECT_DIR=str(work), MPLBACKEND="Agg")
only = sys.argv[2:] or sorted(p.name for p in (ROOT / "notebooks").glob("*.ipynb"))
for name in only:
    t = time.time()
    nb = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(work)}}).execute()
    nbformat.write(nb, work / name)
    print(f"OK  {name}  ({time.time()-t:.0f}s)", flush=True)
