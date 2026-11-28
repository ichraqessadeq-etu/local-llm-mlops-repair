"""Copy repair series from the working folder into results/repairs/, keeping only what the repository publishes:
results.json, run_XX/analysis.json and, for every round, pipeline.py, test_pipeline.py and the text files written
by the repair loop. MLflow runs, pickled models and copies of the data are left out (the scripts re-create them).

Usage: python export_repairs.py <source repairs folder> <destination repairs folder> <series> [<series> ...]
"""
import shutil, sys
from pathlib import Path

KEEP_SUFFIX = {".py", ".txt", ".json"}
SKIP_NAMES = {"_functional_check.py"}
SKIP_DIRS = {"mlruns", "data", "model", "__pycache__", ".pytest_cache"}


def export(src, dst):
    n = 0
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        # only results.json, run_XX/<file> and run_XX/round_K/<file>: deeper files are saved models or MLflow runs
        if (f.is_dir() or len(rel.parts) > 3 or SKIP_DIRS & set(rel.parts[:-1]) or f.name in SKIP_NAMES
                or f.suffix not in KEEP_SUFFIX):
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        n += 1
    return n


if __name__ == "__main__":
    src_root, dst_root = Path(sys.argv[1]), Path(sys.argv[2])
    for name in sys.argv[3:]:
        if not (src_root / name / "results.json").exists():
            sys.exit(f"{name}: results.json missing (series not finished?)")
        print(name, export(src_root / name, dst_root / name), "files")
