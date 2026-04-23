"""Training orchestration and walk-forward helpers for Agent 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import pickle
import subprocess
import sys
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

TORCH_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    DataLoader = None

    class Dataset:  # type: ignore[no-redef]
        """Fallback base so the module stays importable without torch."""

    TORCH_IMPORT_ERROR = exc


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
ML_DIR = Path(__file__).resolve().parent
FEATURE_DIR = ML_DIR / "data"
ARTIFACT_DIR = ML_DIR / "artifacts"
TARGET_COLUMNS = ("future_return", "label")
OBJECTIVE_THREE_CLASS = "three_class"
OBJECTIVE_META_LABEL = "meta_label"
META_LABEL_MAX_BARS = 60

STRATEGY_TIMEFRAME_MAP: dict[str, str] = {
    "ifvg": "5min",
    "ifvg_open": "5min",
    "orb_ib": "5min",
    "orb_vol": "5min",
    "orb_wick": "5min",
    "ttm": "5min",
    "connors": "5min",
    "session_pivot": "5min",
    "session_pivot_break": "5min",
}

STRATEGY_SIGNAL_COLUMN_MAP: dict[str, str] = {
    "ifvg": "ifvg_signal",
    "ifvg_open": "ifvg_open_signal",
    "orb_ib": "orb_ib_signal",
    "orb_vol": "orb_vol_signal",
    "orb_wick": "orb_wick_signal",
    "ttm": "ttm_signal",
    "connors": "connors_signal",
    "session_pivot": "session_pivot_signal",
    "session_pivot_break": "session_pivot_break_signal",
}

MODEL_GROUPS: list[dict[str, Any]] = [
    {
        "model_name": "model_1",
        "strategies": ["ifvg", "connors"],
        "signal_cols": ["ifvg_signal", "connors_signal"],
        "timeframe": "5min",
        "parquet": "ml/data/features_mnq_5min.parquet",
        "description": "IFVG (trend sweep) + ConnorsRSI2 (mean reversion) - opposite logic",
    },
    {
        "model_name": "model_2",
        "strategies": ["ifvg_open", "ttm"],
        "signal_cols": ["ifvg_open_signal", "ttm_signal"],
        "timeframe": "5min",
        "parquet": "ml/data/features_mnq_5min.parquet",
        "description": "IFVG Open (9:30 sweep) + TTMSqueeze (vol compression) - different character",
    },
    {
        "model_name": "model_3",
        "strategies": ["orb_vol", "session_pivot", "session_pivot_break"],
        "signal_cols": ["orb_vol_signal", "session_pivot_signal", "session_pivot_break_signal"],
        "timeframe": "5min",
        "parquet": "ml/data/features_mnq_5min.parquet",
        "description": "ORB Vol + Camarilla Rejection + Camarilla Continuation",
    },
    {
        "model_name": "model_4",
        "strategies": ["orb_ib", "orb_wick"],
        "signal_cols": ["orb_ib_signal", "orb_wick_signal"],
        "timeframe": "5min",
        "parquet": "ml/data/features_mnq_5min.parquet",
        "description": "ORB IB (initial balance break) + ORBWick (clean candle filter)",
    },
]

MODEL_GROUP_MAP: dict[str, dict[str, Any]] = {str(group["model_name"]): group for group in MODEL_GROUPS}
STRATEGY_TO_MODEL_GROUP: dict[str, str] = {
    strategy: str(group["model_name"])
    for group in MODEL_GROUPS
    for strategy in group["strategies"]
}


@dataclass(frozen=True)
class FoldSpec:
    """Anchored walk-forward split specification."""

    name: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


WALK_FORWARD_FOLDS: tuple[FoldSpec, ...] = (
    FoldSpec(
        name="fold_1",
        train_start="2021-03-19",
        train_end="2023-06-30",
        val_start="2023-07-01",
        val_end="2023-12-31",
        test_start="2024-01-01",
        test_end="2024-06-30",
    ),
    FoldSpec(
        name="fold_2",
        train_start="2021-03-19",
        train_end="2023-12-31",
        val_start="2024-01-01",
        val_end="2024-06-30",
        test_start="2024-07-01",
        test_end="2024-12-31",
    ),
    FoldSpec(
        name="fold_3",
        train_start="2021-03-19",
        train_end="2024-06-30",
        val_start="2024-07-01",
        val_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-06-30",
    ),
    FoldSpec(
        name="fold_4",
        train_start="2021-03-19",
        train_end="2024-12-31",
        val_start="2025-01-01",
        val_end="2025-06-30",
        test_start="2025-07-01",
        test_end="2025-12-31",
    ),
    FoldSpec(
        name="fold_5",
        train_start="2021-03-19",
        train_end="2025-06-30",
        val_start="2025-07-01",
        val_end="2025-12-31",
        test_start="2026-01-01",
        test_end="2026-03-18",
    ),
)

DEFAULT_CONFIG: dict[str, Any] = {
    "objective": OBJECTIVE_THREE_CLASS,
    "seq_len": 30,
    "batch_size": 128,
    "max_epochs": 25,
    "patience": 10,
    "learning_rate": 3e-4,
    "dropout": 0.3,
    "n_filters": 64,
    "n_layers": 2,
    "kernel_size": 3,
    "num_classes": 3,
    "max_workers": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dry_run_training": False,
}


def _tz_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="America/New_York")


@dataclass(frozen=True)
class TrainingJobSpec:
    """Minimal orchestration contract for one grouped-model training job."""

    strategy_name: str
    timeframe: str
    parquet_path: str
    artifact_stem: str
    strategies: tuple[str, ...] = ()
    signal_cols: tuple[str, ...] = ()
    description: str = ""

    @property
    def model_name(self) -> str:
        return self.artifact_stem

    @property
    def signal_column(self) -> str:
        return f"{self.artifact_stem}_signal"

    @property
    def scaler_path(self) -> str:
        return str(ARTIFACT_DIR / f"scaler_{self.artifact_stem}.pkl")

    @property
    def model_path(self) -> str:
        return str(ARTIFACT_DIR / f"best_model_{self.artifact_stem}.pt")

    @property
    def eval_path(self) -> str:
        return str(ARTIFACT_DIR / f"eval_{self.artifact_stem}.csv")

    @property
    def log_path(self) -> str:
        return str(ARTIFACT_DIR / f"run_{self.artifact_stem}.log")

    def to_manifest(self) -> dict[str, str]:
        return {
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
            "parquet_path": self.parquet_path,
            "artifact_stem": self.artifact_stem,
            "scaler_path": self.scaler_path,
            "model_path": self.model_path,
            "eval_path": self.eval_path,
            "log_path": self.log_path,
            "model_name": self.model_name,
            "strategies": list(self.strategies),
            "signal_cols": list(self.signal_cols),
            "signal_column": self.signal_column,
            "description": self.description,
        }


@dataclass
class WindowBatch:
    """Prepared windowed inputs for one split."""

    features: np.ndarray
    labels: np.ndarray
    timestamps: pd.DatetimeIndex
    raw_signals: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def is_empty(self) -> bool:
        return len(self.labels) == 0


@dataclass
class SimpleStandardScaler:
    """Small replacement for sklearn's StandardScaler."""

    feature_columns: tuple[str, ...]
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_columns: Sequence[str]) -> "SimpleStandardScaler":
        values = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(tuple(feature_columns), mean, scale)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.feature_columns].to_numpy(dtype=np.float32)
        return ((values - self.mean_) / self.scale_).astype(np.float32)

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        scaled = pd.DataFrame(
            self.transform(frame),
            index=frame.index,
            columns=self.feature_columns,
        )
        for column in self.feature_columns:
            transformed[column] = scaled[column]
        return transformed

    def dump(self, path: str | Path) -> None:
        payload = {
            "feature_columns": self.feature_columns,
            "mean_": self.mean_,
            "scale_": self.scale_,
        }
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)


class TradingDataset(Dataset):
    """Sequence dataset over all eligible session bars."""

    def __init__(self, window_batch: WindowBatch):
        self.X = window_batch.features.astype(np.float32)
        self.y = window_batch.labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        if torch is None:
            return self.X[idx], self.y[idx]
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def _require_torch() -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "PyTorch is required for Agent 2 training. Install 'torch' into the active Python environment."
        ) from TORCH_IMPORT_ERROR


def _normalize_strategy_names(strategy_names: Sequence[str] | None) -> list[str]:
    if strategy_names is None:
        return [str(group["model_name"]) for group in MODEL_GROUPS]

    normalized: list[str] = []
    unknown: list[str] = []
    for name in strategy_names:
        normalized_name = name.strip().lower()
        if normalized_name in STRATEGY_TO_MODEL_GROUP:
            normalized_name = STRATEGY_TO_MODEL_GROUP[normalized_name]
        if normalized_name not in MODEL_GROUP_MAP:
            unknown.append(name)
            continue
        if normalized_name not in normalized:
            normalized.append(normalized_name)

    if unknown:
        raise ValueError(f"Unknown model groups requested: {', '.join(unknown)}")

    return normalized


def _resolve_group_parquet(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def _build_job_spec(strategy_name: str) -> TrainingJobSpec:
    normalized_name = strategy_name.strip().lower()
    if normalized_name in STRATEGY_TO_MODEL_GROUP:
        normalized_name = STRATEGY_TO_MODEL_GROUP[normalized_name]
    if normalized_name not in MODEL_GROUP_MAP:
        raise ValueError(f"Unknown model group requested: {strategy_name}")

    group = MODEL_GROUP_MAP[normalized_name]
    timeframe = str(group["timeframe"])
    parquet_path = _resolve_group_parquet(str(group["parquet"]))
    return TrainingJobSpec(
        strategy_name=normalized_name,
        timeframe=timeframe,
        parquet_path=str(parquet_path),
        artifact_stem=normalized_name,
        strategies=tuple(str(item) for item in group["strategies"]),
        signal_cols=tuple(str(item) for item in group["signal_cols"]),
        description=str(group["description"]),
    )


def _validate_training_jobs(job_specs: Iterable[TrainingJobSpec]) -> None:
    seen_strategies: set[str] = set()
    seen_artifact_stems: set[str] = set()
    seen_output_paths: set[str] = set()

    for job_spec in job_specs:
        if job_spec.strategy_name in seen_strategies:
            raise ValueError(f"Duplicate strategy job detected: {job_spec.strategy_name}")
        if job_spec.artifact_stem in seen_artifact_stems:
            raise ValueError(f"Duplicate artifact stem detected: {job_spec.artifact_stem}")

        seen_strategies.add(job_spec.strategy_name)
        seen_artifact_stems.add(job_spec.artifact_stem)

        for output_path in (
            job_spec.scaler_path,
            job_spec.model_path,
            job_spec.eval_path,
            job_spec.log_path,
        ):
            if output_path in seen_output_paths:
                raise ValueError(f"Duplicate artifact path detected: {output_path}")
            seen_output_paths.add(output_path)


def build_training_jobs(strategy_names: Sequence[str] | None = None) -> list[TrainingJobSpec]:
    """Build validated training job specs for the current grouped-model universe."""
    selected_names = _normalize_strategy_names(strategy_names)
    job_specs = [_build_job_spec(strategy_name) for strategy_name in selected_names]
    _validate_training_jobs(job_specs)
    return job_specs


def _normalize_objective(objective: str | None) -> str:
    normalized = (objective or OBJECTIVE_THREE_CLASS).strip().lower()
    if normalized not in {OBJECTIVE_THREE_CLASS, OBJECTIVE_META_LABEL}:
        raise ValueError(f"Unsupported training objective: {objective}")
    return normalized


def _is_meta_label(config: Mapping[str, Any]) -> bool:
    return _normalize_objective(str(config.get("objective", OBJECTIVE_THREE_CLASS))) == OBJECTIVE_META_LABEL


def _meta_label_column(strategy_name: str) -> str:
    return f"label_{strategy_name}"


def _forward_horizon_for_objective(objective: str) -> int:
    return META_LABEL_MAX_BARS if objective == OBJECTIVE_META_LABEL else 5


def _metadata_like_column(column: str) -> bool:
    if column in TARGET_COLUMNS or column == "atr_14":
        return True
    if column.startswith("model_") and column.endswith("_signal"):
        return True
    prefixes = ("label_", "exit_bar_", "exit_time_", "exit_price_", "r_multiple_", "barrier_hit_")
    return any(column.startswith(prefix) for prefix in prefixes)


def feature_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    attr_columns = frame.attrs.get("feature_columns")
    if attr_columns:
        return [str(column) for column in attr_columns if str(column) in frame.columns]

    return [
        str(column)
        for column in frame.columns
        if not _metadata_like_column(str(column)) and pd.api.types.is_numeric_dtype(frame[column])
    ]


def combine_signal_columns(df: pd.DataFrame, signal_cols: Sequence[str]) -> pd.Series:
    """Combine a model group's signal columns into one trade-direction series."""
    missing = [column for column in signal_cols if column not in df.columns]
    if missing:
        raise KeyError(f"Missing signal columns: {missing}")

    long_signal = pd.Series(False, index=df.index)
    short_signal = pd.Series(False, index=df.index)
    for column in signal_cols:
        values = pd.to_numeric(df[column], errors="coerce").fillna(0)
        long_signal |= values.eq(1)
        short_signal |= values.eq(-1)

    combined = pd.Series(0, index=df.index, dtype=int)
    combined.loc[long_signal & ~short_signal] = 1
    combined.loc[short_signal & ~long_signal] = -1
    return combined


def assign_labels(
    df: pd.DataFrame,
    signal_cols: Sequence[str],
    forward_bars: int = 5,
) -> pd.DataFrame:
    """Assign 3-class labels to all bars for one grouped model.

    Non-signal bars and losing signal bars are class 2 (NoTrade). Long winners
    are class 0, short winners are class 1, and conflicting signal directions
    are forced to class 2.
    """
    missing = [column for column in signal_cols if column not in df.columns]
    if missing:
        raise KeyError(f"Missing signal columns: {missing}")
    if "close" not in df.columns:
        raise KeyError("assign_labels requires a 'close' column")

    labeled = df.copy()
    labeled["future_return"] = labeled["close"].shift(-forward_bars).div(labeled["close"]).sub(1.0)
    labeled["label"] = 2

    long_signal = pd.Series(False, index=labeled.index)
    short_signal = pd.Series(False, index=labeled.index)
    for column in signal_cols:
        values = pd.to_numeric(labeled[column], errors="coerce").fillna(0)
        long_signal |= values.eq(1)
        short_signal |= values.eq(-1)

    labeled.loc[long_signal & labeled["future_return"].gt(0), "label"] = 0
    labeled.loc[short_signal & labeled["future_return"].lt(0), "label"] = 1
    labeled.loc[long_signal & short_signal, "label"] = 2
    labeled["label"] = labeled["label"].fillna(2).astype(int)
    return labeled


def compute_class_weights(labels: np.ndarray, n_classes: int = 3) -> Any:
    """Inverse-frequency class weights computed from train labels only."""
    counts = np.bincount(labels.astype(int), minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    weights = weights / weights.sum()
    if torch is not None:
        return torch.tensor(weights, dtype=torch.float32)
    return weights.astype(np.float32)


def compute_binary_pos_weight(labels: np.ndarray) -> float:
    """Return BCE positive-class weight computed from train labels only."""
    label_values = labels.astype(int)
    wins = int((label_values == 1).sum())
    losses = int((label_values == 0).sum())
    if wins <= 0:
        return 1.0
    return float(losses / wins)


def build_temporal_splits(
    df: pd.DataFrame,
    forward_horizon_bars: int | None = None,
    timeframe: str = "5min",
) -> dict[str, dict[str, pd.DataFrame]]:
    """Return purged rolling walk-forward splits keyed by fold name."""
    from ml.dataset_builder import DEFAULT_FORWARD_HORIZON_BARS, apply_purge_embargo, embargo_bars_for_timeframe

    horizon = DEFAULT_FORWARD_HORIZON_BARS if forward_horizon_bars is None else int(forward_horizon_bars)
    embargo_bars = embargo_bars_for_timeframe(timeframe)
    splits: dict[str, dict[str, pd.DataFrame]] = {}
    for fold in WALK_FORWARD_FOLDS:
        splits[fold.name] = apply_purge_embargo(
            df=df,
            fold_spec=fold,
            forward_horizon_bars=horizon,
            embargo_bars=embargo_bars,
        )
    return splits


def build_window_batch(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    signal_column: str,
    seq_len: int = 30,
    signal_values: pd.Series | np.ndarray | None = None,
) -> WindowBatch:
    """Build contiguous bar windows whose endpoints are all eligible bars."""
    features = frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
    labels = pd.to_numeric(frame.loc[:, "label"], errors="coerce").to_numpy(dtype=np.float64)
    if signal_values is None:
        raw_signals = frame.loc[:, signal_column].to_numpy(dtype=np.int64)
    else:
        raw_signals = np.asarray(signal_values, dtype=np.int64)

    X: list[np.ndarray] = []
    y: list[int] = []
    timestamps: list[pd.Timestamp] = []
    setup_signals: list[int] = []

    for idx in range(seq_len - 1, len(frame)):
        label_value = labels[idx]
        if not np.isfinite(label_value):
            continue
        window = features[idx - seq_len + 1 : idx + 1]
        if not np.isfinite(window).all():
            continue
        X.append(window)
        y.append(int(label_value))
        timestamps.append(frame.index[idx])
        setup_signals.append(int(raw_signals[idx]))

    feature_array = np.asarray(X, dtype=np.float32)
    if feature_array.size == 0:
        feature_array = np.empty((0, seq_len, len(feature_columns)), dtype=np.float32)

    label_array = np.asarray(y, dtype=np.int64)
    signal_array = np.asarray(setup_signals, dtype=np.int64)

    return WindowBatch(
        features=feature_array,
        labels=label_array,
        timestamps=pd.DatetimeIndex(timestamps),
        raw_signals=signal_array,
    )


def _resolve_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved = dict(DEFAULT_CONFIG)
    if config:
        resolved.update(dict(config))
    objective = _normalize_objective(str(resolved.get("objective", OBJECTIVE_THREE_CLASS)))
    resolved["objective"] = objective
    if objective == OBJECTIVE_META_LABEL:
        resolved["num_classes"] = 2
    else:
        resolved["num_classes"] = int(resolved.get("num_classes", 3))
    return resolved


def _load_strategy_frame(
    job_spec: TrainingJobSpec,
    objective: str = OBJECTIVE_THREE_CLASS,
) -> pd.DataFrame:
    frame = pd.read_parquet(job_spec.parquet_path)
    missing_signals = [column for column in job_spec.signal_cols if column not in frame.columns]
    if missing_signals:
        raise KeyError(f"Missing signal columns {missing_signals} in {job_spec.parquet_path}")

    normalized_objective = _normalize_objective(objective)
    if normalized_objective == OBJECTIVE_META_LABEL:
        raise ValueError("Grouped-model training supports the three_class all-bars objective only.")

    prepared = frame.sort_index().copy()
    combined_label = pd.Series(np.nan, index=prepared.index, dtype=float)
    for strategy_name in job_spec.strategies:
        label_column = f"label_{strategy_name}"
        if label_column not in prepared.columns:
            raise KeyError(f"Missing triple-barrier label column '{label_column}' in {job_spec.parquet_path}")

        strategy_label = pd.to_numeric(prepared[label_column], errors="coerce")
        usable_label = strategy_label.notna() & strategy_label.ne(2)
        combined_label = combined_label.where(combined_label.notna(), strategy_label.where(usable_label))

    prepared["label"] = combined_label.fillna(2).astype(int)
    prepared.attrs.update(frame.attrs)
    prepared.attrs["target_column"] = "label"
    prepared.attrs["objective"] = OBJECTIVE_THREE_CLASS
    prepared.attrs["model_name"] = job_spec.model_name
    prepared.attrs["strategies"] = list(job_spec.strategies)
    prepared.attrs["signal_cols"] = list(job_spec.signal_cols)
    return prepared.sort_index()


def _dataloader_from_batch(window_batch: WindowBatch, batch_size: int, shuffle: bool) -> Any:
    _require_torch()
    dataset = TradingDataset(window_batch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _loss_from_logits(logits: Any, labels: Any, loss_fn: Any, objective: str) -> Any:
    if objective == OBJECTIVE_META_LABEL:
        binary_logits = logits[:, 1] - logits[:, 0]
        return loss_fn(binary_logits, labels.float())
    return loss_fn(logits, labels.long())


def _run_epoch(model: Any, dataloader: Any, loss_fn: Any, optimizer: Any, device: str, objective: str) -> float:
    total_loss = 0.0
    total_items = 0

    for batch_features, batch_labels in dataloader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        logits = model(batch_features)
        loss = _loss_from_logits(logits, batch_labels, loss_fn, objective)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(batch_labels)
        total_items += len(batch_labels)

    return total_loss / max(total_items, 1)


def _predict_window_batch(model: Any, window_batch: WindowBatch, device: str) -> tuple[np.ndarray, np.ndarray]:
    _require_torch()
    if window_batch.is_empty:
        n_classes = int(getattr(model, "n_classes", 3))
        return np.empty((0,), dtype=np.int64), np.empty((0, n_classes), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(window_batch.features).to(device)
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        predictions = probabilities.argmax(axis=1).astype(np.int64)
    return predictions, probabilities


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    actual = y_true.astype(int)
    scores = y_score.astype(float)
    positives = int((actual == 1).sum())
    negatives = int((actual == 0).sum())
    if positives == 0 or negatives == 0 or len(actual) != len(scores):
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = actual[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + end + 1) / 2.0
        start = end
    rank_sum_positive = ranks[sorted_labels == 1].sum()
    return float((rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives))


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    if len(y_true) == 0:
        return 0.0

    f1_scores: list[float] = []
    for label in range(n_classes):
        true_positive = int(((y_true == label) & (y_pred == label)).sum())
        false_positive = int(((y_true != label) & (y_pred == label)).sum())
        false_negative = int(((y_true == label) & (y_pred != label)).sum())

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        if precision + recall == 0.0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))

    return float(np.mean(f1_scores))


def _evaluate_loss(
    model: Any,
    window_batch: WindowBatch,
    loss_fn: Any,
    device: str,
    objective: str,
    n_classes: int,
) -> tuple[float, float, float]:
    predictions, probabilities = _predict_window_batch(model, window_batch, device)
    if window_batch.is_empty:
        return 0.0, 0.0, float("nan")

    _require_torch()
    with torch.no_grad():
        inputs = torch.from_numpy(window_batch.features).to(device)
        labels = torch.from_numpy(window_batch.labels).to(device)
        logits = model(inputs)
        loss = float(_loss_from_logits(logits, labels, loss_fn, objective).item())
    auc = _binary_auc(window_batch.labels, probabilities[:, 1]) if objective == OBJECTIVE_META_LABEL else float("nan")
    return loss, _macro_f1(window_batch.labels, predictions, n_classes=n_classes), auc


def _train_one_fold(
    strategy_name: str,
    fold_name: str,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    signal_column: str,
    signal_cols: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _require_torch()

    from ml.model import TradingCNN

    objective = _normalize_objective(str(config.get("objective", OBJECTIVE_THREE_CLASS)))
    n_classes = int(config["num_classes"])
    scaler = SimpleStandardScaler.fit(train_frame, feature_columns)
    train_scaled = scaler.transform_frame(train_frame)
    val_scaled = scaler.transform_frame(val_frame)

    train_batch = build_window_batch(
        train_scaled,
        feature_columns,
        signal_column,
        seq_len=int(config["seq_len"]),
        signal_values=combine_signal_columns(train_frame, signal_cols),
    )
    val_batch = build_window_batch(
        val_scaled,
        feature_columns,
        signal_column,
        seq_len=int(config["seq_len"]),
        signal_values=combine_signal_columns(val_frame, signal_cols),
    )

    if train_batch.is_empty:
        raise ValueError(f"{strategy_name} {fold_name} produced no training windows.")
    if val_batch.is_empty:
        raise ValueError(f"{strategy_name} {fold_name} produced no validation windows.")

    model = TradingCNN(
        n_features=len(feature_columns),
        seq_len=int(config["seq_len"]),
        n_filters=int(config["n_filters"]),
        kernel_size=int(config["kernel_size"]),
        n_layers=int(config["n_layers"]),
        dropout=float(config["dropout"]),
        n_classes=n_classes,
    )
    device = str(config["device"])
    model = model.to(device)

    train_split_labels = pd.to_numeric(train_frame["label"], errors="coerce").fillna(2).to_numpy(dtype=int)
    class_counts = np.bincount(train_split_labels, minlength=n_classes)
    pos_weight = float("nan")
    if objective == OBJECTIVE_META_LABEL:
        pos_weight = compute_binary_pos_weight(train_split_labels)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
        )
        class_weights = np.asarray([1.0, pos_weight], dtype=np.float32)
    else:
        class_weights = compute_class_weights(train_split_labels, n_classes=n_classes)
        if isinstance(class_weights, torch.Tensor):
            loss_weight = class_weights.to(device=device, dtype=torch.float32)
        else:
            loss_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    train_loader = _dataloader_from_batch(train_batch, batch_size=int(config["batch_size"]), shuffle=False)

    best_state: dict[str, Any] | None = None
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_val_auc = float("nan")
    best_train_auc = float("nan")
    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer, device, objective)
        train_eval_loss, train_f1, train_auc = _evaluate_loss(
            model,
            train_batch,
            loss_fn,
            device,
            objective=objective,
            n_classes=n_classes,
        )
        val_loss, val_f1, val_auc = _evaluate_loss(
            model,
            val_batch,
            loss_fn,
            device,
            objective=objective,
            n_classes=n_classes,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_eval_loss": float(train_eval_loss),
                "train_f1": float(train_f1),
                "train_auc": float(train_auc),
                "val_loss": float(val_loss),
                "val_f1": float(val_f1),
                "val_auc": float(val_auc),
            }
        )

        if objective == OBJECTIVE_META_LABEL:
            score = val_auc if np.isfinite(val_auc) else -val_loss
            improved = score > best_score + 1e-6
        else:
            score = -val_loss
            improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_score = float(score)
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_val_auc = val_auc
            best_train_auc = train_auc
            best_epoch = epoch
            best_state = {
                "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
                "config": dict(config),
                "feature_columns": list(feature_columns),
                "signal_column": signal_column,
                "signal_cols": list(signal_cols),
                "strategy_name": strategy_name,
                "fold_name": fold_name,
                "objective": objective,
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(config["patience"]):
            break

    if best_state is None:
        raise RuntimeError(f"{strategy_name} {fold_name} did not produce a best checkpoint.")

    model.load_state_dict(best_state["model_state"])

    return {
        "model": model,
        "scaler": scaler,
        "state": best_state,
        "train_batch": train_batch,
        "val_batch": val_batch,
        "history": history,
        "best_val_loss": float(best_val_loss),
        "best_val_f1": float(best_val_f1),
        "best_val_auc": float(best_val_auc),
        "best_train_auc": float(best_train_auc),
        "best_epoch": int(best_epoch),
        "class_weights": class_weights.tolist(),
        "class_counts": class_counts.astype(int).tolist(),
        "pos_weight": float(pos_weight),
    }


def train_model(strategy_name: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Train one grouped model across anchored walk-forward folds."""
    _require_torch()

    from ml.evaluate import aggregate_across_folds, evaluate_strategy
    from ml.dataset_builder import DEFAULT_FORWARD_HORIZON_BARS

    resolved_config = _resolve_config(config)
    objective = _normalize_objective(str(resolved_config["objective"]))
    job_spec = _build_job_spec(strategy_name)
    strategy_name = job_spec.strategy_name
    dry_run_training = bool(resolved_config.get("dry_run_training", False))
    if not dry_run_training:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    frame = _load_strategy_frame(job_spec, objective=objective)
    feature_columns = feature_columns_from_frame(frame)
    signal_column = job_spec.signal_column
    from ml.dataset_builder import load_data

    raw_session = load_data("mnq", job_spec.timeframe, session_only=True)

    splits = build_temporal_splits(
        frame,
        forward_horizon_bars=_forward_horizon_for_objective(objective),
        timeframe=job_spec.timeframe,
    )
    fold_results: list[dict[str, Any]] = []
    best_global: dict[str, Any] | None = None
    save_fold_name = resolved_config.get("save_fold_name")
    saved_requested_fold = False

    for fold_name, split in splits.items():
        if split["train"].empty or split["val"].empty or split["test"].empty:
            continue

        fold_started_at = perf_counter()
        trained = _train_one_fold(
            strategy_name=strategy_name,
            fold_name=fold_name,
            train_frame=split["train"],
            val_frame=split["val"],
            feature_columns=feature_columns,
            signal_column=signal_column,
            signal_cols=job_spec.signal_cols,
            config=resolved_config,
        )

        scaler = trained["scaler"]
        test_scaled = scaler.transform_frame(split["test"])
        test_batch = build_window_batch(
            test_scaled,
            feature_columns=feature_columns,
            signal_column=signal_column,
            seq_len=int(resolved_config["seq_len"]),
            signal_values=combine_signal_columns(split["test"], job_spec.signal_cols),
        )
        # OHLCV columns are already included in the feature matrix (see
        # dataset_builder.build_feature_matrix: output_columns = RAW_OHLCV_COLUMNS + ...).
        # Use split["test"] directly; only backfill from raw_session if OHLCV is somehow
        # missing (future-proofs against feature matrix schema changes).
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        missing_ohlcv = [c for c in ohlcv_cols if c not in split["test"].columns]
        if missing_ohlcv:
            backfill = raw_session.loc[:, missing_ohlcv].reindex(split["test"].index)
            test_eval_frame = split["test"].join(backfill, how="left")
        else:
            test_eval_frame = split["test"]

        evaluation = evaluate_strategy(
            strategy_name,
            job_spec.timeframe,
            trained["model"],
            test_batch,
            test_eval_frame,
            signal_column,
            device=str(resolved_config["device"]),
            objective=objective,
            confidence_threshold=float(resolved_config.get("confidence_threshold", 0.60)),
        )

        class_weights = list(trained["class_weights"])
        class_counts = list(trained["class_counts"])
        threshold_sweep = evaluation.get("threshold_sweep", {})
        fold_summary = {
            "fold": fold_name,
            "best_epoch": trained["best_epoch"],
            "val_loss": trained["best_val_loss"],
            "val_f1": trained["best_val_f1"],
            "val_auc": trained["best_val_auc"],
            "train_auc": trained["best_train_auc"],
            "train_rows": len(split["train"]),
            "val_rows": len(split["val"]),
            "test_rows": len(split["test"]),
            "train_windows": len(trained["train_batch"]),
            "val_windows": len(trained["val_batch"]),
            "test_windows": len(test_batch),
            "objective": objective,
            "class_count_0": class_counts[0] if len(class_counts) > 0 else 0,
            "class_count_1": class_counts[1] if len(class_counts) > 1 else 0,
            "class_count_2": class_counts[2] if len(class_counts) > 2 else 0,
            "pos_weight": trained["pos_weight"],
            "class_weight_long": class_weights[0] if len(class_weights) > 0 else float("nan"),
            "class_weight_short": class_weights[1] if len(class_weights) > 1 else float("nan"),
            "class_weight_no_trade": class_weights[2] if len(class_weights) > 2 else float("nan"),
            "test_f1": evaluation["test_f1"],
            "test_roc_auc": evaluation["test_roc_auc"],
            "test_auc_roc": evaluation.get("test_auc_roc", evaluation["test_roc_auc"]),
            "test_brier": evaluation.get("test_brier", float("nan")),
            "precision_top_50": evaluation.get("precision_top_50", float("nan")),
            "precision_top_20": evaluation.get("precision_top_20", float("nan")),
            "test_accuracy": evaluation["test_accuracy"],
            "test_sharpe": evaluation["test_sharpe"],
            "test_profit_factor": evaluation["test_profit_factor"],
            "test_win_rate": evaluation["test_win_rate"],
            "test_avg_r": evaluation["test_avg_r"],
            "test_max_drawdown": evaluation["test_max_drawdown"],
            "combine_pass_rate": evaluation["combine_pass_rate"],
            "avg_days_to_pass": evaluation["avg_days_to_pass"],
            "trade_count": evaluation["trade_count"],
            "low_sample": bool(evaluation["trade_count"] < 30),
            "sample_flag": "low sample" if evaluation["trade_count"] < 30 else "",
            "runtime_seconds": perf_counter() - fold_started_at,
            "confusion_matrix": json.dumps(np.asarray(evaluation["confusion_matrix"]).tolist()),
        }
        for threshold, metrics in threshold_sweep.items():
            threshold_key = str(threshold).replace(".", "_")
            fold_summary[f"test_sharpe_thr_{threshold_key}"] = metrics.get("sharpe")
            fold_summary[f"trade_count_thr_{threshold_key}"] = metrics.get("trade_count")
            fold_summary[f"win_rate_thr_{threshold_key}"] = metrics.get("win_rate")
        fold_results.append(fold_summary)
        fold_eval_path = ARTIFACT_DIR / f"eval_{strategy_name}_{fold_name}.csv"
        if not dry_run_training:
            pd.DataFrame([fold_summary]).to_csv(fold_eval_path, index=False)

        # Agent 3E: persist per-fold trade P&L + daily P&L series for the bootstrap
        # pipeline. These are additive artifacts and never modify other outputs.
        trading_detail = evaluation.get("trading", {}) if isinstance(evaluation.get("trading"), dict) else {}
        fold_trades_df = trading_detail.get("trades")
        if isinstance(fold_trades_df, pd.DataFrame) and not fold_trades_df.empty:
            trade_cols = [
                col
                for col in ("entry_time", "exit_time", "direction", "pnl", "r_multiple", "confidence")
                if col in fold_trades_df.columns
            ]
            trade_frame = fold_trades_df.loc[:, trade_cols].copy() if trade_cols else fold_trades_df.copy()
            if not dry_run_training:
                trade_frame.to_csv(
                    ARTIFACT_DIR / f"fold_trades_{strategy_name}_{fold_name}.csv",
                    index=False,
                )
        fold_daily = trading_detail.get("daily_pnls")
        if isinstance(fold_daily, pd.Series) and not fold_daily.empty:
            daily_frame = fold_daily.rename("pnl").reset_index().rename(columns={"index": "date"})
            if not dry_run_training:
                daily_frame.to_csv(
                    ARTIFACT_DIR / f"fold_daily_pnls_{strategy_name}_{fold_name}.csv",
                    index=False,
                )

        if save_fold_name:
            candidate_best = fold_name == save_fold_name
        else:
            selection_key = "best_val_auc" if objective == OBJECTIVE_META_LABEL else "best_val_f1"
            candidate_value = trained[selection_key]
            if objective == OBJECTIVE_META_LABEL and not np.isfinite(candidate_value):
                candidate_value = -trained["best_val_loss"]
            best_value = best_global.get(selection_key, float("-inf")) if best_global is not None else float("-inf")
            if objective == OBJECTIVE_META_LABEL and best_global is not None and not np.isfinite(best_value):
                best_value = -best_global.get("best_val_loss", float("inf"))
            candidate_best = best_global is None or candidate_value > best_value
        if candidate_best:
            saved_requested_fold = bool(save_fold_name and fold_name == save_fold_name)
            best_global = {
                "best_val_f1": trained["best_val_f1"],
                "best_val_auc": trained["best_val_auc"],
                "best_val_loss": trained["best_val_loss"],
                "state": trained["state"],
                "scaler": scaler,
            }

    if save_fold_name and not saved_requested_fold:
        raise RuntimeError(f"Requested save_fold_name='{save_fold_name}' was not available for {strategy_name}.")
    if not fold_results or best_global is None:
        raise RuntimeError(f"No valid walk-forward folds were available for {strategy_name}.")

    eval_frame = pd.DataFrame(fold_results)
    summary_row = aggregate_across_folds(eval_frame)
    eval_frame = pd.concat([eval_frame, pd.DataFrame([summary_row])], ignore_index=True)
    if not dry_run_training:
        eval_frame.to_csv(job_spec.eval_path, index=False)

    if not dry_run_training:
        best_global["scaler"].dump(job_spec.scaler_path)
        torch.save(best_global["state"], job_spec.model_path)

    return {
        "strategy_name": strategy_name,
        "model_name": job_spec.model_name,
        "strategies": list(job_spec.strategies),
        "timeframe": job_spec.timeframe,
        "parquet_path": job_spec.parquet_path,
        "artifact_stem": job_spec.artifact_stem,
        "scaler_path": job_spec.scaler_path,
        "model_path": job_spec.model_path,
        "eval_path": job_spec.eval_path,
        "log_path": job_spec.log_path,
        "status": "trained",
        "pid": os.getpid(),
        "config": resolved_config,
        "feature_count": len(feature_columns),
        "signal_column": signal_column,
        "signal_cols": list(job_spec.signal_cols),
        "fold_count": len(fold_results),
        "objective": objective,
        "val_auc_mean": float(eval_frame.loc[eval_frame["fold"] != "summary", "val_auc"].mean())
        if "val_auc" in eval_frame.columns
        else float("nan"),
        "val_f1_mean": float(eval_frame.loc[eval_frame["fold"] != "summary", "val_f1"].mean()),
        "test_f1_mean": float(eval_frame.loc[eval_frame["fold"] != "summary", "test_f1"].mean()),
        "test_sharpe_mean": float(eval_frame.loc[eval_frame["fold"] != "summary", "test_sharpe"].mean()),
        "combine_pass_rate_mean": float(eval_frame.loc[eval_frame["fold"] != "summary", "combine_pass_rate"].mean()),
        "fold_results": fold_results,
    }


def run_strategy_job(job_spec: TrainingJobSpec, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Prepare one isolated training job or run it end-to-end."""
    config_dict = _resolve_config(config)
    parquet_exists = Path(job_spec.parquet_path).exists()

    if not parquet_exists:
        raise FileNotFoundError(f"Parquet file not found for {job_spec.strategy_name}: {job_spec.parquet_path}")

    return train_model(job_spec.strategy_name, config=config_dict)


def _dry_run_manifest(job_specs: Sequence[TrainingJobSpec], max_workers: int) -> dict[str, Any]:
    return {
        "mode": "dry_run",
        "max_workers": max_workers,
        "jobs": [job_spec.to_manifest() for job_spec in job_specs],
    }


def _run_strategy_job_subprocess(job_spec: TrainingJobSpec, config: Mapping[str, Any]) -> dict[str, Any]:
    command = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "from ml.train import TrainingJobSpec, run_strategy_job; "
            "job = TrainingJobSpec(**json.loads(sys.argv[1])); "
            "config = json.loads(sys.argv[2]); "
            "result = run_strategy_job(job, config); "
            "print(json.dumps(result))"
        ),
        json.dumps(asdict(job_spec)),
        json.dumps(dict(config)),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Strategy job failed for {job_spec.strategy_name}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return json.loads(completed.stdout)


def run_training_jobs_parallel(
    strategy_names: Sequence[str] | None = None,
    max_workers: int = 2,
    dry_run: bool = False,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run isolated strategy jobs serially or in separate processes."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    job_specs = build_training_jobs(strategy_names)
    if dry_run:
        return _dry_run_manifest(job_specs, max_workers=max_workers)

    config_dict = _resolve_config(config)
    if max_workers == 1 or len(job_specs) == 1:
        results = [run_strategy_job(job_spec, config_dict) for job_spec in job_specs]
        return {
            "mode": "serial",
            "max_workers": max_workers,
            "results": results,
        }

    results: list[dict[str, Any]] = []
    for start in range(0, len(job_specs), max_workers):
        batch = job_specs[start : start + max_workers]
        processes: list[tuple[TrainingJobSpec, subprocess.Popen[str]]] = []

        for job_spec in batch:
            command = [
                sys.executable,
                "-c",
                (
                    "import json, sys; "
                    "from ml.train import TrainingJobSpec, run_strategy_job; "
                    "job = TrainingJobSpec(**json.loads(sys.argv[1])); "
                    "config = json.loads(sys.argv[2]); "
                    "result = run_strategy_job(job, config); "
                    "print(json.dumps(result))"
                ),
                json.dumps(asdict(job_spec)),
                json.dumps(config_dict),
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append((job_spec, process))

        for job_spec, process in processes:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(
                    f"Strategy job failed for {job_spec.strategy_name}: {stderr.strip() or stdout.strip()}"
                )
            results.append(json.loads(stdout))

    return {
        "mode": "parallel",
        "max_workers": max_workers,
        "results": results,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train strategy models across rolling walk-forward folds.")
    parser.add_argument("--strategy", action="append", dest="strategies", help="Model group name to train; repeatable.")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--retrain-from-hpo", action="store_true")
    parser.add_argument("--save-fold-name", default="fold_5")
    parser.add_argument(
        "--objective",
        choices=(OBJECTIVE_THREE_CLASS, OBJECTIVE_META_LABEL),
        default=OBJECTIVE_THREE_CLASS,
    )
    args = parser.parse_args(argv)

    if args.retrain_from_hpo:
        from ml.hyperparam_search import load_best_hpo_configs

        configs = load_best_hpo_configs(
            strategy_names=args.strategies,
            artifact_dir=ARTIFACT_DIR,
            objective=args.objective,
        )
        results = []
        for strategy_name, config in configs.items():
            final_config = dict(config)
            final_config["objective"] = args.objective
            final_config["save_fold_name"] = args.save_fold_name
            if args.max_epochs is not None:
                final_config["max_epochs"] = args.max_epochs
            results.append(train_model(strategy_name, config=final_config))
        print(json.dumps({"status": "completed", "results": results}, default=_json_default, indent=2))
        return 0

    run_config: dict[str, Any] = {"objective": args.objective}
    if args.max_epochs is not None:
        run_config["max_epochs"] = args.max_epochs
    if args.dry_run:
        run_config["dry_run_training"] = True
        run_config["max_epochs"] = int(args.max_epochs or 2)

    result = run_training_jobs_parallel(
        strategy_names=args.strategies,
        max_workers=args.max_workers,
        dry_run=False,
        config=run_config,
    )
    print(json.dumps(result, default=_json_default, indent=2))
    return 0


__all__ = [
    "FoldSpec",
    "SimpleStandardScaler",
    "TradingDataset",
    "TrainingJobSpec",
    "WALK_FORWARD_FOLDS",
    "MODEL_GROUPS",
    "STRATEGY_SIGNAL_COLUMN_MAP",
    "STRATEGY_TIMEFRAME_MAP",
    "OBJECTIVE_META_LABEL",
    "OBJECTIVE_THREE_CLASS",
    "WindowBatch",
    "assign_labels",
    "build_temporal_splits",
    "build_training_jobs",
    "build_window_batch",
    "combine_signal_columns",
    "compute_binary_pos_weight",
    "compute_class_weights",
    "feature_columns_from_frame",
    "run_strategy_job",
    "run_training_jobs_parallel",
    "train_model",
    "_validate_training_jobs",
]


if __name__ == "__main__":
    raise SystemExit(main())
