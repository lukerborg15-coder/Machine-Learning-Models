"""Staged hyperparameter search helpers for Agent 3A."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
import json
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.evaluate import _predict_with_model, evaluate_strategy
from ml.export_onnx import build_model_from_checkpoint, export_strategy, load_checkpoint, load_scaler_payload, verify_onnx_matches_pytorch
from ml.funded_sim import evaluate_deployment_gate, payout_adjusted_survival_score, simulate_funded_after_combine
from ml.train import (
    ARTIFACT_DIR,
    OBJECTIVE_META_LABEL,
    OBJECTIVE_THREE_CLASS,
    META_LABEL_MAX_BARS,
    SimpleStandardScaler,
    build_window_batch,
    combine_signal_columns,
    build_temporal_splits,
    build_training_jobs,
    feature_columns_from_frame,
    train_model,
    _load_strategy_frame,
    _train_one_fold,
)
from ml.dataset_builder import load_data


SEARCH_SPACE: dict[str, tuple[Any, ...]] = {
    "n_filters": (32, 64, 128),
    "kernel_size": (3, 5, 7),
    "n_layers": (2, 3, 4),
    "dropout": (0.2, 0.3, 0.5),
    "learning_rate": (1e-3, 3e-4, 1e-4),
    "seq_len": (20, 30, 60),
}

DEFAULT_HPO_CONFIG: dict[str, Any] = {
    "objective": OBJECTIVE_THREE_CLASS,
    "batch_size": 128,
    "max_epochs": 8,
    "patience": 3,
    "num_classes": 3,
    "device": "cpu",
    "dry_run_training": False,
}

HPO_CONFIG_COLUMNS: tuple[str, ...] = tuple(DEFAULT_HPO_CONFIG) + tuple(SEARCH_SPACE)
MAX_AGENT3_HPO_SECONDS = 24 * 60 * 60
MIN_DEPLOYMENT_TRADES = 20


def sample_search_configs(n_trials: int, seed: int = 42) -> list[dict[str, Any]]:
    """Return reproducible random configs from the Agent 3 search space."""
    all_configs = [
        dict(zip(SEARCH_SPACE.keys(), values, strict=True))
        for values in product(*(SEARCH_SPACE[key] for key in SEARCH_SPACE))
    ]
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(all_configs), size=min(n_trials, len(all_configs)), replace=False)
    return [all_configs[int(idx)] for idx in selected_indices]


def _resolve_hpo_config(config: Mapping[str, Any] | None = None, objective: str | None = None) -> dict[str, Any]:
    resolved = dict(DEFAULT_HPO_CONFIG)
    if config:
        resolved.update(dict(config))
    if objective is not None:
        resolved["objective"] = objective
    resolved["objective"] = str(resolved.get("objective", OBJECTIVE_THREE_CLASS)).strip().lower()
    if resolved["objective"] == OBJECTIVE_META_LABEL:
        resolved["num_classes"] = 2
    else:
        resolved["num_classes"] = int(resolved.get("num_classes", 3))
    return resolved


def _selection_metric_for_objective(objective: str) -> str:
    return "validation_auc" if objective == OBJECTIVE_META_LABEL else "validation_sharpe"


def build_hpo_manifest(
    strategy_names: Sequence[str] | None = None,
    n_trials: int = 3,
    seed: int = 42,
    objective: str = OBJECTIVE_THREE_CLASS,
) -> dict[str, Any]:
    """Build a no-training manifest that makes validation-only selection explicit."""
    jobs = build_training_jobs(strategy_names)
    configs = sample_search_configs(n_trials=n_trials, seed=seed)
    resolved_objective = str(objective).strip().lower()
    return {
        "mode": "dry_run",
        "objective": resolved_objective,
        "selection_metric": _selection_metric_for_objective(resolved_objective),
        "test_touched_during_search": False,
        "n_trials": len(configs),
        "strategies": [job.strategy_name for job in jobs],
        "trial_configs": configs,
    }


def _evaluate_on_validation(
    strategy_name: str,
    timeframe: str,
    trained: Mapping[str, Any],
    val_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    signal_column: str,
    signal_cols: Sequence[str],
    raw_session: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    scaler = trained["scaler"]
    val_scaled = scaler.transform_frame(val_frame)
    from ml.train import build_window_batch

    val_batch = build_window_batch(
        val_scaled,
        feature_columns=feature_columns,
        signal_column=signal_column,
        seq_len=int(config["seq_len"]),
        signal_values=combine_signal_columns(val_frame, signal_cols),
    )
    # val_frame already contains open/high/low/close/volume (added in 4A rebuild)
    # so we can use it directly as the eval frame without joining raw_session
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    if all(c in val_frame.columns for c in ohlcv_cols):
        val_eval_frame = val_frame
    else:
        raw_ohlcv = raw_session.loc[:, ohlcv_cols].reindex(val_frame.index)
        val_eval_frame = raw_ohlcv.join(
            val_frame.drop(columns=[c for c in ohlcv_cols if c in val_frame.columns]),
            how="right",
        )
    return evaluate_strategy(
        strategy_name,
        timeframe,
        trained["model"],
        val_batch,
        val_eval_frame,
        signal_column,
        device=str(config["device"]),
        objective=str(config.get("objective", OBJECTIVE_THREE_CLASS)),
        confidence_threshold=float(config.get("confidence_threshold", 0.60)),
    )


def run_strategy_hpo(
    strategy_name: str,
    n_trials: int = 3,
    seed: int = 42,
    fold_name: str = "fold_5",
    dry_run: bool = True,
    output_dir: Path | str | None = ARTIFACT_DIR,
    objective: str = OBJECTIVE_THREE_CLASS,
) -> pd.DataFrame:
    """Run or dry-run validation-Sharpe hyperparameter search for one strategy."""
    resolved_objective = str(objective).strip().lower()
    selection_metric = _selection_metric_for_objective(resolved_objective)
    trial_configs = sample_search_configs(n_trials=n_trials, seed=seed)
    job = build_training_jobs([strategy_name])[0]
    strategy_name = job.strategy_name
    if dry_run:
        rows = [
            {
                "strategy_name": strategy_name,
                "trial": trial_idx,
                "fold": fold_name,
                "objective": resolved_objective,
                "selection_metric": selection_metric,
                "test_touched_during_search": False,
                **config,
            }
            for trial_idx, config in enumerate(trial_configs, start=1)
        ]
        return pd.DataFrame(rows)

    frame = (
        _load_strategy_frame(job, objective=resolved_objective)
        if resolved_objective == OBJECTIVE_META_LABEL
        else _load_strategy_frame(job)
    )
    from ml.dataset_builder import DEFAULT_FORWARD_HORIZON_BARS

    splits = build_temporal_splits(
        frame,
        forward_horizon_bars=META_LABEL_MAX_BARS if resolved_objective == OBJECTIVE_META_LABEL else DEFAULT_FORWARD_HORIZON_BARS,
        timeframe=job.timeframe,
    )
    if fold_name not in splits:
        raise ValueError(f"Unknown fold for HPO: {fold_name}")
    split = splits[fold_name]
    feature_columns = feature_columns_from_frame(frame)
    signal_column = job.signal_column
    raw_session = load_data("mnq", job.timeframe, session_only=True)

    rows: list[dict[str, Any]] = []
    for trial_idx, config in enumerate(trial_configs, start=1):
        resolved_config = _resolve_hpo_config(config, objective=resolved_objective)
        start = perf_counter()
        trained = _train_one_fold(
            strategy_name=strategy_name,
            fold_name=fold_name,
            train_frame=split["train"],
            val_frame=split["val"],
            feature_columns=feature_columns,
            signal_column=signal_column,
            signal_cols=job.signal_cols,
            config=resolved_config,
        )
        val_eval = _evaluate_on_validation(
            strategy_name=strategy_name,
            timeframe=job.timeframe,
            trained=trained,
            val_frame=split["val"],
            feature_columns=feature_columns,
            signal_column=signal_column,
            signal_cols=job.signal_cols,
            raw_session=raw_session,
            config=resolved_config,
        )
        rows.append(
            {
                "strategy_name": strategy_name,
                "trial": trial_idx,
                "fold": fold_name,
                "objective": resolved_objective,
                "selection_metric": selection_metric,
                "test_touched_during_search": False,
                "validation_sharpe": val_eval["test_sharpe"],
                "validation_f1": val_eval["test_f1"],
                "validation_auc": val_eval.get("test_auc_roc", val_eval.get("test_roc_auc")),
                "validation_brier": val_eval.get("test_brier"),
                "validation_profit_factor": val_eval["test_profit_factor"],
                "validation_trade_count": val_eval["trade_count"],
                "runtime_seconds": perf_counter() - start,
                **resolved_config,
            }
        )

    sort_columns = ["validation_auc", "validation_sharpe", "validation_f1"] if resolved_objective == OBJECTIVE_META_LABEL else [
        "validation_sharpe",
        "validation_f1",
    ]
    result = pd.DataFrame(rows).sort_values(sort_columns, ascending=False)
    if output_dir is not None:
        output_path = Path(output_dir) / f"hyperparam_{strategy_name}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def hpo_trial_count_for_runtime(one_trial_seconds: float) -> int:
    """Scale Agent 3 HPO breadth from the one-strategy timing benchmark."""
    if one_trial_seconds < 10 * 60:
        return 20
    if one_trial_seconds < 30 * 60:
        return 10
    return 5


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _config_from_hpo_row(row: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(DEFAULT_HPO_CONFIG)
    for key in HPO_CONFIG_COLUMNS:
        if key in row and not pd.isna(row[key]):
            value = row[key]
            if isinstance(DEFAULT_HPO_CONFIG.get(key), bool):
                config[key] = bool(value)
            elif isinstance(DEFAULT_HPO_CONFIG.get(key), int) or key in {"n_filters", "kernel_size", "n_layers", "seq_len"}:
                config[key] = int(value)
            elif isinstance(DEFAULT_HPO_CONFIG.get(key), float) or key in {"dropout", "learning_rate"}:
                config[key] = float(value)
            else:
                config[key] = _json_safe(value)
    return config


def benchmark_hpo_runtime(
    strategy_name: str = "ttm",
    seed: int = 42,
    output_dir: Path | str | None = ARTIFACT_DIR,
    objective: str = OBJECTIVE_THREE_CLASS,
) -> pd.DataFrame:
    """Run one real HPO trial and persist a timing benchmark for Agent 3 scaling."""
    benchmark = run_strategy_hpo(
        strategy_name=strategy_name,
        n_trials=1,
        seed=seed,
        dry_run=False,
        output_dir=None,
        objective=objective,
    ).copy()
    benchmark["benchmark_strategy"] = strategy_name
    benchmark["benchmark_seed"] = seed
    if output_dir is not None:
        output_path = Path(output_dir) / "agent3_timing_benchmark.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark.to_csv(output_path, index=False)
    return benchmark


def run_all_strategy_hpo(
    strategy_names: Sequence[str] | None = None,
    seed: int = 42,
    benchmark_strategy: str = "ttm",
    output_dir: Path | str | None = ARTIFACT_DIR,
    max_estimated_seconds: float = MAX_AGENT3_HPO_SECONDS,
    objective: str = OBJECTIVE_THREE_CLASS,
) -> dict[str, Any]:
    """Benchmark then run validation-only HPO across the selected strategy universe."""
    jobs = build_training_jobs(strategy_names)
    benchmark = benchmark_hpo_runtime(benchmark_strategy, seed=seed, output_dir=output_dir, objective=objective)
    one_trial_seconds = float(benchmark["runtime_seconds"].sum())
    n_trials = hpo_trial_count_for_runtime(one_trial_seconds)
    estimated_seconds = one_trial_seconds * n_trials * len(jobs)
    if estimated_seconds > max_estimated_seconds:
        summary = pd.DataFrame(
            [
                {
                    "status": "aborted_estimate_over_limit",
                    "benchmark_strategy": benchmark_strategy,
                    "one_trial_seconds": one_trial_seconds,
                    "planned_trials_per_strategy": n_trials,
                    "strategy_count": len(jobs),
                    "estimated_seconds": estimated_seconds,
                    "max_estimated_seconds": max_estimated_seconds,
                }
            ]
        )
        if output_dir is not None:
            summary.to_csv(Path(output_dir) / "hpo_summary.csv", index=False)
        return {
            "status": "aborted_estimate_over_limit",
            "benchmark": benchmark,
            "summary": summary,
            "n_trials": n_trials,
            "estimated_seconds": estimated_seconds,
        }

    summary_rows: list[dict[str, Any]] = []
    hpo_results: dict[str, pd.DataFrame] = {}
    for strategy_idx, job in enumerate(jobs):
        result = run_strategy_hpo(
            strategy_name=job.strategy_name,
            n_trials=n_trials,
            seed=seed + strategy_idx,
            dry_run=False,
            output_dir=output_dir,
            objective=objective,
        )
        hpo_results[job.strategy_name] = result
        best = result.iloc[0].to_dict()
        summary_rows.append(
            {
                "status": "completed",
                "strategy_name": job.strategy_name,
                "timeframe": job.timeframe,
                "n_trials": len(result),
                "selection_metric": _selection_metric_for_objective(str(objective).strip().lower()),
                "best_validation_sharpe": best.get("validation_sharpe"),
                "best_validation_f1": best.get("validation_f1"),
                "best_validation_auc": best.get("validation_auc"),
                "best_validation_trade_count": best.get("validation_trade_count"),
                "runtime_seconds": float(result["runtime_seconds"].sum()),
                "best_config": json.dumps({key: _json_safe(_config_from_hpo_row(best).get(key)) for key in HPO_CONFIG_COLUMNS}),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["best_validation_sharpe", "best_validation_f1", "best_validation_trade_count"],
        ascending=False,
    )
    if output_dir is not None:
        summary.to_csv(Path(output_dir) / "hpo_summary.csv", index=False)
    return {
        "status": "completed",
        "benchmark": benchmark,
        "summary": summary,
        "hpo_results": hpo_results,
        "n_trials": n_trials,
        "estimated_seconds": estimated_seconds,
    }


def load_best_hpo_configs(
    strategy_names: Sequence[str] | None = None,
    artifact_dir: Path | str = ARTIFACT_DIR,
    objective: str = OBJECTIVE_THREE_CLASS,
) -> dict[str, dict[str, Any]]:
    """Load selected HPO configs from persisted validation-only search CSVs."""
    resolved_objective = str(objective).strip().lower()
    configs: dict[str, dict[str, Any]] = {}
    for job in build_training_jobs(strategy_names):
        hpo_path = Path(artifact_dir) / f"hyperparam_{job.strategy_name}.csv"
        if not hpo_path.exists():
            raise FileNotFoundError(f"Missing HPO result for {job.strategy_name}: {hpo_path}")
        hpo_frame = pd.read_csv(hpo_path)
        if hpo_frame.empty:
            raise ValueError(f"HPO result is empty for {job.strategy_name}: {hpo_path}")
        sort_columns = (
            ["validation_auc", "validation_sharpe", "validation_trade_count"]
            if resolved_objective == OBJECTIVE_META_LABEL and "validation_auc" in hpo_frame.columns
            else ["validation_sharpe", "validation_f1", "validation_trade_count"]
        )
        sorted_frame = hpo_frame.sort_values(sort_columns, ascending=False)
        config = _config_from_hpo_row(sorted_frame.iloc[0].to_dict())
        config["objective"] = resolved_objective
        if resolved_objective == OBJECTIVE_META_LABEL:
            config["num_classes"] = 2
        configs[job.strategy_name] = config
    return configs


def _saved_model_eval_frame(
    strategy_name: str,
    fold_name: str = "fold_5",
    artifact_dir: Path | str = ARTIFACT_DIR,
    device: str = "cpu",
    objective: str = OBJECTIVE_THREE_CLASS,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    job = build_training_jobs([strategy_name])[0]
    strategy_name = job.strategy_name
    checkpoint = load_checkpoint(strategy_name, artifact_dir=artifact_dir)
    scaler_payload = load_scaler_payload(strategy_name, artifact_dir=artifact_dir)
    scaler = SimpleStandardScaler(
        feature_columns=tuple(scaler_payload["feature_columns"]),
        mean_=np.asarray(scaler_payload["mean_"], dtype=np.float64),
        scale_=np.asarray(scaler_payload["scale_"], dtype=np.float64),
    )
    model = build_model_from_checkpoint(checkpoint)
    frame = _load_strategy_frame(job, objective=objective)
    from ml.dataset_builder import DEFAULT_FORWARD_HORIZON_BARS

    split = build_temporal_splits(
        frame,
        forward_horizon_bars=META_LABEL_MAX_BARS if objective == OBJECTIVE_META_LABEL else DEFAULT_FORWARD_HORIZON_BARS,
        timeframe=job.timeframe,
    )[fold_name]
    feature_columns = list(checkpoint["feature_columns"])
    signal_column = job.signal_column
    raw_session = load_data("mnq", job.timeframe, session_only=True)
    test_scaled = scaler.transform_frame(split["test"])
    test_batch = build_window_batch(
        test_scaled,
        feature_columns=feature_columns,
        signal_column=signal_column,
        seq_len=int(checkpoint["config"]["seq_len"]),
        signal_values=combine_signal_columns(split["test"], job.signal_cols),
    )
    # OHLCV is already in the feature matrix — use split["test"] directly.
    # Backfill from raw_session only if missing (matches train.py).
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    missing_ohlcv = [c for c in ohlcv_cols if c not in split["test"].columns]
    if missing_ohlcv:
        backfill = raw_session.loc[:, missing_ohlcv].reindex(split["test"].index)
        test_eval_frame = split["test"].join(backfill, how="left")
    else:
        test_eval_frame = split["test"]
    evaluation = evaluate_strategy(
        strategy_name,
        job.timeframe,
        model,
        test_batch,
        test_eval_frame,
        signal_column,
        device=device,
        objective=objective,
    )
    probabilities, predictions = _predict_with_model(model, test_batch.features, device=device)
    eval_frame = test_eval_frame.copy().sort_index().reindex(test_batch.timestamps)
    eval_frame["label"] = test_batch.labels[: len(eval_frame)]
    eval_frame["prediction"] = predictions[: len(eval_frame)]
    if probabilities is not None and probabilities.ndim == 2 and probabilities.shape[1] > 1:
        eval_frame["confidence"] = probabilities[: len(eval_frame), 1]
    eval_frame[signal_column] = test_batch.raw_signals[: len(eval_frame)]
    return eval_frame, evaluation, signal_column


def _summarize_funded_path(path_result: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_status": path_result.get("status"),
        f"{prefix}_active": path_result.get("active"),
        f"{prefix}_failure_reason": path_result.get("failure_reason"),
        f"{prefix}_ending_balance": path_result.get("ending_balance"),
        f"{prefix}_gross_payouts": path_result.get("gross_payouts"),
        f"{prefix}_trader_payouts": path_result.get("trader_payouts"),
        f"{prefix}_payout_count": path_result.get("payout_count"),
        f"{prefix}_trade_count": path_result.get("trade_count"),
        f"{prefix}_score": payout_adjusted_survival_score(path_result),
    }


def train_evaluate_agent3_strategy(
    strategy_name: str,
    config: Mapping[str, Any],
    fold_name: str = "fold_5",
    artifact_dir: Path | str = ARTIFACT_DIR,
    export_candidates: bool = True,
) -> dict[str, Any]:
    """Retrain one final strategy config, run Combine/XFA evaluation, and export eligible artifacts."""
    final_config = dict(config)
    final_config["save_fold_name"] = fold_name
    training_result = train_model(strategy_name, config=final_config)
    eval_frame, evaluation, signal_column = _saved_model_eval_frame(
        strategy_name=strategy_name,
        fold_name=fold_name,
        artifact_dir=artifact_dir,
        device=str(final_config.get("device", "cpu")),
        objective=str(final_config.get("objective", OBJECTIVE_THREE_CLASS)),
    )
    combine = evaluation["trading"]
    funded = simulate_funded_after_combine(
        eval_frame,
        strategy_name=strategy_name,
        combine_result=combine,
        signal_column=signal_column,
    )

    paths = funded.get("paths", {})
    for path_name, path_result in paths.items():
        ledger = path_result.get("ledger")
        payouts = path_result.get("payouts")
        trades = path_result.get("trades")
        if isinstance(ledger, pd.DataFrame):
            ledger.to_csv(Path(artifact_dir) / f"funded_ledger_{strategy_name}_{path_name}.csv", index=False)
        if isinstance(payouts, pd.DataFrame):
            payouts.to_csv(Path(artifact_dir) / f"funded_payouts_{strategy_name}_{path_name}.csv", index=False)
        if isinstance(trades, pd.DataFrame):
            trades.to_csv(Path(artifact_dir) / f"funded_trades_{strategy_name}_{path_name}.csv", index=False)

    best_path = funded.get("best_path")
    best_result = paths.get(best_path, {}) if best_path else {}
    best_score = float(funded.get("best_score", float("-inf")))
    best_payout_count = int(best_result.get("payout_count", 0)) if best_result else 0
    best_trade_count = int(best_result.get("trade_count", 0)) if best_result else 0
    combine_passed = bool(combine.get("combine_passed", False))
    survived_funded = bool(best_result.get("active", False)) if best_result else False
    deployment_decision = evaluate_deployment_gate(training_result.get("fold_results", []))
    pre_audit_deployment_eligible = bool(deployment_decision.approved)
    research_candidate = bool(combine_passed or best_score > 0 or best_trade_count >= MIN_DEPLOYMENT_TRADES)

    export_status = "not_exported"
    parity_max_diff = float("nan")
    if export_candidates and (research_candidate or pre_audit_deployment_eligible):
        export_strategy(strategy_name, output_dir=artifact_dir, deployment_candidate=False)
        parity = verify_onnx_matches_pytorch(strategy_name, artifact_dir=artifact_dir)
        parity_max_diff = float(parity["max_diff"])
        export_status = "exported_research_only"

    row = {
        "strategy_name": strategy_name,
        "fold": fold_name,
        "research_candidate": research_candidate,
        "deployment_candidate": False,
        "pre_audit_deployment_eligible": pre_audit_deployment_eligible,
        "deployment_gate_reason": deployment_decision.reason,
        "combine_passed": combine_passed,
        "pass_day": combine.get("pass_day"),
        "combine_trade_count": combine.get("trade_count"),
        "combine_sharpe": combine.get("sharpe"),
        "combine_profit_factor": combine.get("profit_factor"),
        "combine_days_to_pass": combine.get("days_to_pass"),
        "funded_best_path": best_path,
        "funded_best_score": best_score,
        "funded_best_payout_count": best_payout_count,
        "funded_best_trader_payouts": best_result.get("trader_payouts", 0.0) if best_result else 0.0,
        "funded_best_ending_balance": best_result.get("ending_balance", 0.0) if best_result else 0.0,
        "export_status": export_status,
        "onnx_parity_max_diff": parity_max_diff,
        "artifact_model_path": training_result["model_path"],
        "artifact_eval_path": training_result["eval_path"],
        "config": json.dumps({key: _json_safe(final_config.get(key)) for key in final_config}),
    }
    for path_name in ("standard", "consistency"):
        if path_name in paths:
            row.update(_summarize_funded_path(paths[path_name], path_name))
    return row


def run_fixed_trial_hpo(
    strategy_names: Sequence[str] | None = None,
    n_trials: int = 10,
    seed: int = 42,
    output_dir: Path | str | None = ARTIFACT_DIR,
    objective: str = OBJECTIVE_THREE_CLASS,
) -> dict[str, Any]:
    """Run a fixed-width validation-only HPO pass for Agent 3C."""
    jobs = build_training_jobs(strategy_names)
    summary_rows: list[dict[str, Any]] = []
    hpo_results: dict[str, pd.DataFrame] = {}
    for strategy_idx, job in enumerate(jobs):
        result = run_strategy_hpo(
            strategy_name=job.strategy_name,
            n_trials=n_trials,
            seed=seed + strategy_idx,
            dry_run=False,
            output_dir=output_dir,
            objective=objective,
        )
        hpo_results[job.strategy_name] = result
        best = result.iloc[0].to_dict()
        summary_rows.append(
            {
                "status": "completed",
                "strategy_name": job.strategy_name,
                "timeframe": job.timeframe,
                "n_trials": len(result),
                "selection_metric": _selection_metric_for_objective(str(objective).strip().lower()),
                "best_validation_sharpe": best.get("validation_sharpe"),
                "best_validation_f1": best.get("validation_f1"),
                "best_validation_auc": best.get("validation_auc"),
                "best_validation_trade_count": best.get("validation_trade_count"),
                "runtime_seconds": float(result["runtime_seconds"].sum()),
                "best_config": json.dumps({key: _json_safe(_config_from_hpo_row(best).get(key)) for key in HPO_CONFIG_COLUMNS}),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["best_validation_sharpe", "best_validation_f1", "best_validation_trade_count"],
        ascending=False,
    )
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary.to_csv(Path(output_dir) / "hpo_summary.csv", index=False)
    return {
        "status": "completed",
        "summary": summary,
        "hpo_results": hpo_results,
        "n_trials": n_trials,
    }


def write_agent3_final_report(rankings: pd.DataFrame, output_dir: Path | str = ARTIFACT_DIR) -> Path:
    output_path = Path(output_dir) / "FINAL_EVAL_REPORT.md"
    deployment_count = int(rankings["deployment_candidate"].sum()) if "deployment_candidate" in rankings else 0
    research_count = int(rankings["research_candidate"].sum()) if "research_candidate" in rankings else 0
    lines = [
        "# Final Evaluation Report - Agent 3",
        "",
        "**Deployment status:** No deployment candidates approved until Agent 3B audit clears them.",
        "",
        "## Summary",
        "",
        f"- Strategies evaluated: {len(rankings)}",
        f"- Research candidates: {research_count}",
        f"- Deployment candidates: {deployment_count}",
        "- Ranking priority: Combine pass, Express Funded survival, simulated payouts, trade count, then Sharpe/F1 diagnostics.",
        "",
        "## Rankings",
        "",
        "| Strategy | Research | Deployment | Combine | Best XFA Path | Payouts | Trader Payouts | Score | Export |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rankings.to_dict(orient="records"):
        lines.append(
            "| {strategy_name} | {research_candidate} | {deployment_candidate} | {combine_passed} | {funded_best_path} | {funded_best_payout_count} | {funded_best_trader_payouts:.2f} | {funded_best_score:.2f} | {export_status} |".format(
                **{key: ("" if value is None else value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Gate Notes",
            "",
            "- `deployment_candidate` remains false until Agent 3B audits the real HPO outputs, funded ledgers, ONNX sidecars, and report.",
            "- `research_candidate` can be true for strategies that merit paper/research inspection but miss the strict payout/audit deployment gate.",
            "- Current funded simulation uses conservative intratrade DLL/MLL liquidation when a bar can hit a risk threshold.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_agent3_completion(
    strategy_names: Sequence[str] | None = None,
    seed: int = 42,
    artifact_dir: Path | str = ARTIFACT_DIR,
    export_candidates: bool = True,
) -> dict[str, Any]:
    """Run the full Agent 3A benchmark, HPO, final retrain, funded ranking, and report."""
    hpo = run_all_strategy_hpo(strategy_names=strategy_names, seed=seed, output_dir=artifact_dir)
    if hpo["status"] != "completed":
        return hpo

    configs = load_best_hpo_configs(strategy_names=strategy_names, artifact_dir=artifact_dir)
    rows = [
        train_evaluate_agent3_strategy(
            strategy_name=strategy_name,
            config=config,
            artifact_dir=artifact_dir,
            export_candidates=export_candidates,
        )
        for strategy_name, config in configs.items()
    ]
    rankings = pd.DataFrame(rows).sort_values(
        [
            "pre_audit_deployment_eligible",
            "research_candidate",
            "funded_best_score",
            "funded_best_trader_payouts",
            "combine_sharpe",
        ],
        ascending=False,
    )
    rankings_path = Path(artifact_dir) / "agent3_rankings.csv"
    rankings.to_csv(rankings_path, index=False)
    report_path = write_agent3_final_report(rankings, output_dir=artifact_dir)
    return {
        "status": "completed",
        "hpo": hpo,
        "rankings_path": str(rankings_path),
        "report_path": str(report_path),
        "rankings": rankings,
    }


__all__ = [
    "DEFAULT_HPO_CONFIG",
    "HPO_CONFIG_COLUMNS",
    "MAX_AGENT3_HPO_SECONDS",
    "MIN_DEPLOYMENT_TRADES",
    "SEARCH_SPACE",
    "benchmark_hpo_runtime",
    "build_hpo_manifest",
    "hpo_trial_count_for_runtime",
    "load_best_hpo_configs",
    "run_agent3_completion",
    "run_all_strategy_hpo",
    "run_fixed_trial_hpo",
    "run_strategy_hpo",
    "sample_search_configs",
    "train_evaluate_agent3_strategy",
    "write_agent3_final_report",
]


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run validation-only hyperparameter search.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strategy", action="append", dest="strategies", help="Strategy name to search; repeatable.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    parser.add_argument(
        "--objective",
        choices=(OBJECTIVE_THREE_CLASS, OBJECTIVE_META_LABEL),
        default=OBJECTIVE_THREE_CLASS,
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        manifest = build_hpo_manifest(
            strategy_names=args.strategies,
            n_trials=args.trials,
            seed=args.seed,
            objective=args.objective,
        )
        print(json.dumps(manifest, default=str, indent=2))
        return 0

    result = run_fixed_trial_hpo(
        strategy_names=args.strategies,
        n_trials=args.trials,
        seed=args.seed,
        output_dir=args.artifact_dir,
        objective=args.objective,
    )
    print(result["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
