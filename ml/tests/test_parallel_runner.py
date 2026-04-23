"""Focused tests for grouped-model training orchestration."""

from __future__ import annotations

import pytest

from ml.train import (
    TrainingJobSpec,
    build_training_jobs,
    run_training_jobs_parallel,
    _validate_training_jobs,
)


def test_build_training_jobs_uses_expected_strategy_mapping() -> None:
    job_specs = build_training_jobs()
    manifest = {job.strategy_name: job.timeframe for job in job_specs}

    assert manifest == {
        "model_1": "5min",
        "model_2": "5min",
        "model_3": "5min",
        "model_4": "5min",
    }


def test_dry_run_lists_expected_jobs_and_paths() -> None:
    dry_run = run_training_jobs_parallel(strategy_names=["model_1", "model_4"], max_workers=2, dry_run=True)

    assert dry_run["mode"] == "dry_run"
    assert dry_run["max_workers"] == 2
    assert [job["strategy_name"] for job in dry_run["jobs"]] == ["model_1", "model_4"]
    assert dry_run["jobs"][0]["parquet_path"].endswith("features_mnq_5min.parquet")
    assert dry_run["jobs"][1]["parquet_path"].endswith("features_mnq_5min.parquet")
    assert dry_run["jobs"][0]["model_path"].endswith("best_model_model_1.pt")
    assert dry_run["jobs"][1]["eval_path"].endswith("eval_model_4.csv")


def test_duplicate_artifact_paths_are_rejected() -> None:
    duplicate_jobs = [
        TrainingJobSpec(
            strategy_name="ifvg",
            timeframe="1min",
            parquet_path="one.parquet",
            artifact_stem="shared",
        ),
        TrainingJobSpec(
            strategy_name="orb_wick",
            timeframe="5min",
            parquet_path="two.parquet",
            artifact_stem="shared",
        ),
    ]

    with pytest.raises(ValueError, match="Duplicate artifact stem detected"):
        _validate_training_jobs(duplicate_jobs)


def test_serial_and_parallel_dry_run_manifests_match() -> None:
    serial = run_training_jobs_parallel(strategy_names=["model_1", "model_2"], max_workers=1, dry_run=True)
    parallel = run_training_jobs_parallel(strategy_names=["model_1", "model_2"], max_workers=2, dry_run=True)

    assert [job["strategy_name"] for job in serial["jobs"]] == [job["strategy_name"] for job in parallel["jobs"]]
    assert [job["parquet_path"] for job in serial["jobs"]] == [job["parquet_path"] for job in parallel["jobs"]]
    assert [job["model_path"] for job in serial["jobs"]] == [job["model_path"] for job in parallel["jobs"]]


def test_dry_run_manifest_uses_unique_group_outputs() -> None:
    result = run_training_jobs_parallel(strategy_names=["model_3", "model_4"], max_workers=2, dry_run=True)

    assert result["mode"] == "dry_run"
    assert len(result["jobs"]) == 2
    assert len({item["model_path"] for item in result["jobs"]}) == 2
    assert len({item["eval_path"] for item in result["jobs"]}) == 2
