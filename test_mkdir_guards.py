from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
ML_DIR = REPO_ROOT / "ml"

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line)
    if not condition:
        failures.append(name)


def test_imports():
    sys.path.insert(0, str(REPO_ROOT))
    data_dir = ML_DIR / "data"
    artifact_dir = ML_DIR / "artifacts"
    try:
        from ml import dataset_builder
        check("dataset_builder imports", True)
        check("ml/data exists after dataset_builder import", data_dir.exists(), str(data_dir))
    except Exception as exc:
        check("dataset_builder imports", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return
    try:
        from ml import train
        check("train imports", True)
        check("ml/artifacts exists after train import", artifact_dir.exists(), str(artifact_dir))
    except Exception as exc:
        check("train imports", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return
    try:
        from ml import evaluate
        check("evaluate imports", True)
    except Exception as exc:
        check("evaluate imports", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return
    try:
        from ml import funded_sim
        check("funded_sim imports", True)
    except Exception as exc:
        check("funded_sim imports", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def test_mkdir_pattern():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "nested" / "deeply" / "data" / "features.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        df.to_parquet(target)
        check("mkdir + to_parquet into non-existent nested dir",
              target.exists() and target.stat().st_size > 0, str(target))


def test_output_path():
    sys.path.insert(0, str(REPO_ROOT))
    from ml.dataset_builder import _feature_matrix_output_path, ML_DATA_DIR
    path = _feature_matrix_output_path("mnq", "5min")
    check("parquet path under ML_DATA_DIR", ML_DATA_DIR in path.parents, str(path))
    check("ML_DATA_DIR exists", ML_DATA_DIR.exists(), str(ML_DATA_DIR))


def test_training_jobs():
    sys.path.insert(0, str(REPO_ROOT))
    from ml.train import build_training_jobs, ARTIFACT_DIR
    try:
        jobs = build_training_jobs()
    except Exception as exc:
        check("build_training_jobs runs", False, f"{type(exc).__name__}: {exc}")
        return
    check("build_training_jobs runs", True, f"{len(jobs)} jobs")
    if jobs:
        check("eval_path under ARTIFACT_DIR", str(ARTIFACT_DIR) in jobs[0].eval_path, jobs[0].eval_path)
    check("ARTIFACT_DIR exists", ARTIFACT_DIR.exists(), str(ARTIFACT_DIR))


def main():
    print("=" * 60)
    print("mkdir guard smoke test")
    print(f"repo root: {REPO_ROOT}")
    print("=" * 60)
    if not ML_DIR.exists():
        print(f"FATAL: {ML_DIR} missing")
        return 2
    test_imports()
    test_mkdir_pattern()
    test_output_path()
    test_training_jobs()
    print("=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("ALL CHECKS PASSED")
    print("Safe to run the full pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
