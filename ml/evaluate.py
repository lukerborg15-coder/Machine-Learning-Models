"""Evaluation helpers for classification and TopStep-style trade simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import math
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.topstep_risk import TopStepRiskManager

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch is expected in the training environment.
    torch = None
    nn = None

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
LABELS = (0, 1, 2)
BINARY_LABELS = (0, 1)
LABEL_NAMES = {0: "long", 1: "short", 2: "no_trade"}
BINARY_LABEL_NAMES = {0: "loss", 1: "win"}
DEFAULT_SEQ_LEN = 30
DEFAULT_COMMISSION_PER_RT = 1.40
DEFAULT_STOP_ATR_MULT = 1.5
DEFAULT_TARGET_R = 1.0
DEFAULT_MAX_BARS = 60
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
CONFIDENCE_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)
OBJECTIVE_THREE_CLASS = "three_class"
OBJECTIVE_META_LABEL = "meta_label"
AGGREGATE_SUFFIXES = ("_mean", "_median", "_std", "_min")

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

NON_FEATURE_COLUMNS = {
    "label",
    "future_return",
    "prediction",
    "predicted_label",
    "predicted_class",
    "open",
    "high",
    "low",
    "close",
    "atr",
    "atr_14",
    "datetime",
}


def _ensure_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, Mapping) and "frame" in data:
        frame = pd.DataFrame(data["frame"]).copy()
    else:
        raise TypeError("test_dataset must be a DataFrame or a mapping containing 'frame'")

    if not isinstance(frame.index, pd.DatetimeIndex):
        if "datetime" not in frame.columns:
            raise ValueError("Evaluation data must be indexed by datetime or include a 'datetime' column")
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame.set_index("datetime")

    return frame.sort_index()


def _resolve_feature_columns(frame: pd.DataFrame, feature_columns: Sequence[str] | None) -> list[str]:
    if feature_columns is not None:
        columns = [str(column) for column in feature_columns]
    else:
        attr_columns = frame.attrs.get("feature_columns")
        if attr_columns:
            columns = [str(column) for column in attr_columns]
        else:
            columns = [
                str(column)
                for column in frame.columns
                if pd.api.types.is_numeric_dtype(frame[column]) and str(column) not in NON_FEATURE_COLUMNS
            ]
    if not columns:
        raise ValueError("Could not resolve feature columns for evaluation")
    return columns


def _resolve_signal_column(strategy_name: str, signal_column: str | None) -> str:
    if signal_column is not None:
        return signal_column
    strategy_key = strategy_name.strip().lower()
    if strategy_key not in STRATEGY_SIGNAL_COLUMN_MAP:
        raise ValueError(f"Unsupported strategy name: {strategy_name}")
    return STRATEGY_SIGNAL_COLUMN_MAP[strategy_key]


def _compute_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def _resolve_atr(frame: pd.DataFrame) -> pd.Series | None:
    for column in ("atr", "atr_14"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    required = {"high", "low", "close"}
    if required.issubset(frame.columns):
        return _compute_atr(frame)
    return None


def build_sequence_array(frame: pd.DataFrame, feature_columns: Sequence[str], seq_len: int) -> tuple[np.ndarray, pd.Index]:
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")

    feature_frame = frame.loc[:, list(feature_columns)].astype(np.float32)
    if len(feature_frame) < seq_len:
        empty = np.empty((0, seq_len, len(feature_columns)), dtype=np.float32)
        return empty, feature_frame.index[:0]

    windows = [
        feature_frame.iloc[start - seq_len + 1 : start + 1].to_numpy(dtype=np.float32, copy=True)
        for start in range(seq_len - 1, len(feature_frame))
    ]
    return np.asarray(windows, dtype=np.float32), feature_frame.index[seq_len - 1 :]


def _to_numpy_scores(model_output: Any) -> tuple[np.ndarray | None, np.ndarray]:
    if torch is not None and isinstance(model_output, torch.Tensor):
        array = model_output.detach().cpu().numpy()
    else:
        array = np.asarray(model_output)

    if array.ndim == 1:
        predictions = array.astype(int, copy=False)
        return None, predictions

    if array.ndim != 2:
        raise ValueError("Model output must be a 1D prediction array or a 2D score array")

    if array.shape[1] in (len(LABELS), len(BINARY_LABELS)):
        predictions = array.argmax(axis=1).astype(int, copy=False)
        return array, predictions

    predictions = array.reshape(-1).astype(int, copy=False)
    return None, predictions


def _predict_with_model(
    model: Any,
    features: np.ndarray,
    batch_size: int = 512,
    device: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    if features.size == 0:
        return np.empty((0, len(LABELS)), dtype=np.float32), np.empty(0, dtype=int)

    if model is None:
        raise ValueError("model must not be None when no precomputed predictions are supplied")

    if hasattr(model, "predict_proba") and hasattr(model, "predict"):
        probabilities = np.asarray(model.predict_proba(features))
        predictions = np.asarray(model.predict(features)).astype(int, copy=False)
        return probabilities, predictions

    if torch is not None and nn is not None and isinstance(model, nn.Module):
        model.eval()
        param = next(model.parameters(), None)
        model_device = torch.device(device) if device is not None else (param.device if param is not None else torch.device("cpu"))

        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), batch_size):
                batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=model_device)
                logits = model(batch)
                logits_np = logits.detach().cpu().numpy()
                batches.append(logits_np)

        score_array = np.concatenate(batches, axis=0)
        if score_array.ndim == 2 and score_array.shape[1] == 1:
            positive = 1.0 / (1.0 + np.exp(-score_array[:, 0]))
            probabilities = np.column_stack([1.0 - positive, positive]).astype(np.float32)
        else:
            score_array = score_array - score_array.max(axis=1, keepdims=True)
            exp_scores = np.exp(score_array)
            probabilities = exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)
        predictions = probabilities.argmax(axis=1).astype(int, copy=False)
        return probabilities, predictions

    if hasattr(model, "predict"):
        return _to_numpy_scores(model.predict(features))

    if callable(model):
        return _to_numpy_scores(model(features))

    raise TypeError("Unsupported model type for evaluation")


def confusion_matrix_safe(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] = LABELS) -> np.ndarray:
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_idx = {int(label): idx for idx, label in enumerate(labels)}
    for actual, predicted in zip(np.asarray(y_true), np.asarray(y_pred), strict=False):
        if int(actual) in label_to_idx and int(predicted) in label_to_idx:
            matrix[label_to_idx[int(actual)], label_to_idx[int(predicted)]] += 1
    return matrix


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: np.ndarray | None = None,
    labels: Sequence[int] = LABELS,
) -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    common_length = min(len(actual), len(predicted))
    if common_length != len(actual) or common_length != len(predicted):
        actual = actual[:common_length]
        predicted = predicted[:common_length]
        if y_score is not None:
            y_score = np.asarray(y_score)[:common_length]
    matrix = confusion_matrix_safe(actual, predicted, labels=labels)

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for idx, label in enumerate(labels):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        precision_den = tp + fp
        recall_den = tp + fn
        precision = tp / precision_den if precision_den > 0 else 0.0
        recall = tp / recall_den if recall_den > 0 else 0.0
        f1_den = precision + recall
        f1 = (2.0 * precision * recall / f1_den) if f1_den > 0 else 0.0
        f1_values.append(f1)
        name_map = BINARY_LABEL_NAMES if tuple(labels) == BINARY_LABELS else LABEL_NAMES
        per_class[name_map.get(int(label), str(label))] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(matrix[idx, :].sum()),
        }

    accuracy = float((actual == predicted).mean()) if len(actual) else float("nan")
    macro_f1 = float(np.mean(f1_values)) if f1_values else float("nan")
    if y_score is not None and tuple(labels) == BINARY_LABELS:
        scores = np.asarray(y_score)
        positive_scores = scores[:, 1] if scores.ndim == 2 and scores.shape[1] > 1 else scores.reshape(-1)
        roc_auc = _binary_roc_auc((actual == 1).astype(int), positive_scores)
    else:
        roc_auc = multiclass_roc_auc_ovr(actual, y_score, labels=labels) if y_score is not None else float("nan")

    return {
        "count": int(len(actual)),
        "accuracy": accuracy,
        "f1_macro": macro_f1,
        "roc_auc_ovr": roc_auc,
        "roc_auc": roc_auc,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_labels = y_true[order]

    ranks = np.empty(len(y_score), dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + end + 1) / 2.0
        ranks[start:end] = average_rank
        start = end

    rank_sum_positive = ranks[sorted_labels == 1].sum()
    auc = (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def brier_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    actual = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    if len(actual) == 0 or len(actual) != len(scores):
        return float("nan")
    return float(np.mean((scores - actual) ** 2))


def precision_at_top_fraction(y_true: Sequence[int], y_score: Sequence[float], fraction: float) -> float:
    actual = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if len(actual) == 0 or len(actual) != len(scores):
        return float("nan")
    take = max(int(math.ceil(len(scores) * fraction)), 1)
    order = np.argsort(scores)[::-1][:take]
    return float((actual[order] == 1).mean())


def calibration_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
    bins: int = 10,
) -> pd.DataFrame:
    actual = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if len(actual) == 0 or len(actual) != len(scores):
        return pd.DataFrame(columns=["bin", "count", "mean_confidence", "win_rate"])

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float | int]] = []
    for bin_idx in range(bins):
        left = edges[bin_idx]
        right = edges[bin_idx + 1]
        if bin_idx == bins - 1:
            mask = (scores >= left) & (scores <= right)
        else:
            mask = (scores >= left) & (scores < right)
        if not mask.any():
            rows.append({"bin": bin_idx, "count": 0, "mean_confidence": float("nan"), "win_rate": float("nan")})
            continue
        rows.append(
            {
                "bin": bin_idx,
                "count": int(mask.sum()),
                "mean_confidence": float(scores[mask].mean()),
                "win_rate": float((actual[mask] == 1).mean()),
            }
        )
    return pd.DataFrame(rows)


def multiclass_roc_auc_ovr(y_true: Sequence[int], y_score: np.ndarray, labels: Sequence[int] = LABELS) -> float:
    scores = np.asarray(y_score, dtype=np.float64)
    actual = np.asarray(y_true, dtype=int)
    if scores.ndim != 2 or len(actual) != len(scores):
        return float("nan")

    aucs: list[float] = []
    for idx, label in enumerate(labels):
        binary_true = (actual == int(label)).astype(int)
        auc = _binary_roc_auc(binary_true, scores[:, idx])
        if not np.isnan(auc):
            aucs.append(auc)
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def _get_exit_price_and_index(
    frame: pd.DataFrame,
    entry_pos: int,
    direction: int,
    stop: float,
    target: float,
    max_bars: int = DEFAULT_MAX_BARS,
) -> tuple[float, int]:
    entry_day = frame.index[entry_pos].normalize()
    last_same_day_pos = entry_pos
    max_pos = min(entry_pos + int(max_bars), len(frame) - 1)
    session_end = entry_day + pd.Timedelta(hours=15)

    for cursor in range(entry_pos + 1, max_pos + 1):
        if frame.index[cursor].normalize() != entry_day:
            break
        if frame.index[cursor] > session_end:
            break

        last_same_day_pos = cursor
        row = frame.iloc[cursor]
        high = float(row["high"])
        low = float(row["low"])

        if direction > 0:
            stop_hit = low <= stop
            target_hit = high >= target
            if stop_hit and target_hit:
                return stop, cursor
            if stop_hit:
                return stop, cursor
            if target_hit:
                return target, cursor
        else:
            stop_hit = high >= stop
            target_hit = low <= target
            if stop_hit and target_hit:
                return stop, cursor
            if stop_hit:
                return stop, cursor
            if target_hit:
                return target, cursor

    exit_row = frame.iloc[last_same_day_pos]
    return float(exit_row["close"]), last_same_day_pos


def _max_drawdown_pct(equity_curve: Sequence[float]) -> float:
    if not equity_curve:
        return float("nan")
    curve = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(curve)
    drawdowns = np.divide(peaks - curve, peaks, out=np.zeros_like(curve), where=peaks != 0)
    return float(drawdowns.max(initial=0.0))


def _annualized_return(initial_equity: float, final_equity: float, periods: int) -> float:
    if initial_equity <= 0 or final_equity <= 0 or periods <= 0:
        return float("nan")
    return float((final_equity / initial_equity) ** (252.0 / periods) - 1.0)


def simulate_trading(
    frame: pd.DataFrame,
    strategy_name: str,
    signal_column: str | None = None,
    commission_per_rt: float = DEFAULT_COMMISSION_PER_RT,
    stop_atr_mult: float = DEFAULT_STOP_ATR_MULT,
    target_r: float = DEFAULT_TARGET_R,
    objective: str = OBJECTIVE_THREE_CLASS,
    confidence_threshold: float | None = None,
    max_bars: int = DEFAULT_MAX_BARS,
) -> dict[str, Any]:
    trading_frame = frame.copy().sort_index()
    resolved_signal_column = _resolve_signal_column(strategy_name, signal_column)
    if resolved_signal_column not in trading_frame.columns:
        raise ValueError(f"Signal column '{resolved_signal_column}' was not found in the evaluation frame")

    required_price_columns = {"open", "high", "low", "close"}
    if not required_price_columns.issubset(trading_frame.columns):
        return {
            "trades": pd.DataFrame(),
            "trade_count": 0,
            "win_rate": float("nan"),
            "profit_factor": float("nan"),
            "avg_r": float("nan"),
            "max_drawdown_pct": float("nan"),
            "sharpe": float("nan"),
            "calmar": float("nan"),
            "combine_passed": False,
            "combine_pass_rate": 0.0,
            "days_to_pass": float("nan"),
            "consistency_ok": False,
            "active": True,
            "warning": "Trading metrics unavailable because raw OHLC prices were not provided.",
        }

    atr = _resolve_atr(trading_frame)
    if atr is None:
        return {
            "trades": pd.DataFrame(),
            "trade_count": 0,
            "win_rate": float("nan"),
            "profit_factor": float("nan"),
            "avg_r": float("nan"),
            "max_drawdown_pct": float("nan"),
            "sharpe": float("nan"),
            "calmar": float("nan"),
            "combine_passed": False,
            "combine_pass_rate": 0.0,
            "days_to_pass": float("nan"),
            "consistency_ok": False,
            "active": True,
            "warning": "Trading metrics unavailable because ATR could not be resolved.",
        }

    trading_frame["atr_eval"] = atr
    if "prediction" not in trading_frame.columns:
        raise ValueError("Trading simulation requires a 'prediction' column")

    risk_manager = TopStepRiskManager(commission_per_rt=commission_per_rt)
    trade_records: list[dict[str, Any]] = []
    daily_pnl_by_day: dict[pd.Timestamp, float] = {}
    eod_equity: list[float] = []
    eod_days: list[pd.Timestamp] = []
    pass_day: pd.Timestamp | None = None
    normalized_objective = objective.strip().lower()
    if normalized_objective not in {OBJECTIVE_THREE_CLASS, OBJECTIVE_META_LABEL}:
        raise ValueError(f"Unsupported evaluation objective: {objective}")
    active_confidence_threshold = (
        DEFAULT_CONFIDENCE_THRESHOLD if confidence_threshold is None and normalized_objective == OBJECTIVE_META_LABEL else confidence_threshold
    )

    for day, day_frame in trading_frame.groupby(trading_frame.index.normalize(), sort=True):
        blocked_day = False
        day_indices = list(day_frame.index)
        if len(day_indices) < 2:
            risk_manager.update_eod(risk_manager.account, daily_pnl_by_day.get(day, 0.0))
            eod_days.append(pd.Timestamp(day))
            eod_equity.append(risk_manager.account)
            if pass_day is None and risk_manager.is_passed():
                pass_day = pd.Timestamp(day)
            continue

        cursor = 0
        while cursor < len(day_frame) - 1:
            if blocked_day or not risk_manager.active:
                break

            row = day_frame.iloc[cursor]
            setup_signal = int(row[resolved_signal_column])
            prediction = int(row["prediction"])
            atr_value = float(row["atr_eval"]) if pd.notna(row["atr_eval"]) else float("nan")
            if setup_signal == 0 or prediction not in (0, 1) or not np.isfinite(atr_value) or atr_value <= 0:
                cursor += 1
                continue
            if normalized_objective == OBJECTIVE_META_LABEL:
                confidence = float(row["confidence"]) if "confidence" in row.index and pd.notna(row["confidence"]) else float("nan")
                if prediction != 1:
                    cursor += 1
                    continue
                if active_confidence_threshold is not None and "confidence" in row.index:
                    if not np.isfinite(confidence) or confidence < float(active_confidence_threshold):
                        cursor += 1
                        continue
                direction = 1 if setup_signal > 0 else -1
            else:
                if (setup_signal > 0 and prediction != 0) or (setup_signal < 0 and prediction != 1):
                    cursor += 1
                    continue
                direction = 1 if prediction == 0 else -1

            entry = float(row["close"])
            stop_distance = atr_value * stop_atr_mult
            if stop_distance <= 0:
                cursor += 1
                continue
            sizing_confidence = (
                float(row["confidence"])
                if "confidence" in row.index and pd.notna(row["confidence"]) and np.isfinite(float(row["confidence"]))
                else DEFAULT_CONFIDENCE_THRESHOLD
            )
            try:
                contracts = risk_manager.position_size(stop_distance, sizing_confidence)
            except ValueError:
                cursor += 1
                continue

            stop = entry - stop_distance if direction > 0 else entry + stop_distance
            target = entry + (stop_distance * target_r) if direction > 0 else entry - (stop_distance * target_r)
            exit_price, exit_pos = _get_exit_price_and_index(
                day_frame,
                entry_pos=cursor,
                direction=direction,
                stop=stop,
                target=target,
                max_bars=max_bars,
            )

            pnl = float(
                risk_manager.simulate_trade(
                    entry=entry,
                    stop=stop,
                    target=target,
                    exit_price=exit_price,
                    contracts=contracts,
                )
            )
            risk_amount = stop_distance * contracts * risk_manager.point_value
            r_multiple = pnl / risk_amount if risk_amount > 0 else float("nan")

            risk_manager.account += pnl
            day_key = pd.Timestamp(day)
            daily_pnl_by_day[day_key] = daily_pnl_by_day.get(day_key, 0.0) + pnl

            trailing_floor = risk_manager.max_equity_eod - risk_manager.max_trailing_dd
            if risk_manager.account < trailing_floor:
                risk_manager.active = False

            if not risk_manager.check_intraday(risk_manager.account):
                blocked_day = True

            trade_records.append(
                {
                    "entry_time": day_indices[cursor],
                    "exit_time": day_indices[exit_pos],
                    "setup_signal": setup_signal,
                    "prediction": prediction,
                    "confidence": float(row["confidence"]) if "confidence" in row.index and pd.notna(row["confidence"]) else np.nan,
                    "direction": direction,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "exit_price": exit_price,
                    "contracts": contracts,
                    "pnl": pnl,
                    "r_multiple": r_multiple,
                    "account_after_trade": risk_manager.account,
                }
            )

            cursor = max(exit_pos + 1, cursor + 1)

        day_key = pd.Timestamp(day)
        risk_manager.update_eod(risk_manager.account, daily_pnl_by_day.get(day_key, 0.0))
        eod_days.append(day_key)
        eod_equity.append(risk_manager.account)
        if pass_day is None and risk_manager.is_passed():
            pass_day = day_key

    trades = pd.DataFrame(trade_records)
    gross_wins = float(trades.loc[trades["pnl"] > 0, "pnl"].sum()) if not trades.empty else 0.0
    gross_losses = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum()) if not trades.empty else 0.0
    profit_factor = float(gross_wins / gross_losses) if gross_losses > 0 else (float("inf") if gross_wins > 0 else float("nan"))
    win_rate = float((trades["pnl"] > 0).mean()) if not trades.empty else float("nan")
    avg_r = float(trades["r_multiple"].mean()) if not trades.empty else float("nan")

    daily_pnls = pd.Series(daily_pnl_by_day).sort_index()
    daily_returns = pd.Series(dtype=float)
    if eod_days:
        equity_by_day = pd.Series(eod_equity, index=pd.Index(eod_days), dtype=float)
        previous_equity = equity_by_day.shift(1).fillna(risk_manager.initial_account)
        daily_returns = (equity_by_day - previous_equity) / previous_equity.replace(0.0, np.nan)
    else:
        equity_by_day = pd.Series(dtype=float)

    sharpe = float("nan")
    if not daily_returns.empty:
        returns_std = float(daily_returns.std(ddof=0))
        if returns_std > 0:
            sharpe = float(math.sqrt(252.0) * daily_returns.mean() / returns_std)

    max_drawdown_pct = _max_drawdown_pct(equity_by_day.tolist() if not equity_by_day.empty else [risk_manager.initial_account])
    annualized_return = _annualized_return(
        initial_equity=risk_manager.initial_account,
        final_equity=risk_manager.account,
        periods=max(len(equity_by_day), 1),
    )
    calmar = float(annualized_return / max_drawdown_pct) if np.isfinite(annualized_return) and max_drawdown_pct > 0 else float("nan")

    days_to_pass = float("nan")
    if pass_day is not None and len(equity_by_day.index) > 0:
        first_day = pd.Timestamp(equity_by_day.index[0]).normalize()
        days_to_pass = float((pass_day - first_day).days + 1)

    return {
        "trades": trades,
        "trade_count": int(len(trades)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_r": avg_r,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "calmar": calmar,
        "combine_passed": pass_day is not None,
        "combine_pass_rate": 1.0 if pass_day is not None else 0.0,
        "days_to_pass": days_to_pass,
        "pass_day": pass_day,
        "consistency_ok": risk_manager.check_consistency(),
        "active": risk_manager.active,
        "ending_account": float(risk_manager.account),
        "daily_pnls": daily_pnls,
        "eod_equity": equity_by_day,
    }


def confidence_threshold_sweep(
    frame: pd.DataFrame,
    strategy_name: str,
    signal_column: str | None = None,
    thresholds: Sequence[float] = CONFIDENCE_THRESHOLDS,
    objective: str = OBJECTIVE_META_LABEL,
) -> dict[str, dict[str, float]]:
    """Run trading metrics across confidence gates for meta-label evaluation."""
    sweep: dict[str, dict[str, float]] = {}
    resolved_signal_column = _resolve_signal_column(strategy_name, signal_column)
    for threshold in thresholds:
        result = simulate_trading(
            frame,
            strategy_name=strategy_name,
            signal_column=signal_column,
            objective=objective,
            confidence_threshold=float(threshold),
        )
        eligible_count = result["trade_count"]
        if "confidence" in frame.columns and resolved_signal_column in frame.columns:
            confidence = pd.to_numeric(frame["confidence"], errors="coerce")
            predictions = pd.to_numeric(frame.get("prediction", pd.Series(index=frame.index)), errors="coerce")
            signals = pd.to_numeric(frame[resolved_signal_column], errors="coerce").fillna(0)
            eligible_count = int(((signals != 0) & (predictions == 1) & (confidence >= float(threshold))).sum())
        sweep[f"{threshold:.2f}"] = {
            "threshold": float(threshold),
            "sharpe": result["sharpe"],
            "trade_count": eligible_count,
            "executed_trade_count": result["trade_count"],
            "win_rate": result["win_rate"],
            "profit_factor": result["profit_factor"],
            "avg_r": result["avg_r"],
            "combine_pass_rate": result["combine_pass_rate"],
        }
    return sweep


def _evaluate_single_frame(
    strategy_name: str,
    frame: pd.DataFrame,
    model: Any,
    feature_columns: Sequence[str] | None,
    seq_len: int,
    signal_column: str | None,
    device: str | None = None,
    objective: str = OBJECTIVE_THREE_CLASS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    working_frame = frame.copy().sort_index()
    features, prediction_index = build_sequence_array(working_frame, _resolve_feature_columns(working_frame, feature_columns), seq_len)

    if features.size == 0:
        empty_metrics = classification_metrics([], [])
        return {
            "strategy_name": strategy_name,
            "frame_start": working_frame.index.min() if not working_frame.empty else pd.NaT,
            "frame_end": working_frame.index.max() if not working_frame.empty else pd.NaT,
            "classification": empty_metrics,
            "trading": {
                "trades": pd.DataFrame(),
                "trade_count": 0,
                "win_rate": float("nan"),
                "profit_factor": float("nan"),
                "avg_r": float("nan"),
                "max_drawdown_pct": float("nan"),
                "sharpe": float("nan"),
                "calmar": float("nan"),
                "combine_passed": False,
                "combine_pass_rate": 0.0,
                "days_to_pass": float("nan"),
                "consistency_ok": False,
                "active": True,
            },
        }

    y_score, predictions = _predict_with_model(model, features, device=device)
    prediction_series = pd.Series(predictions, index=prediction_index, dtype=int, name="prediction")
    eval_frame = working_frame.loc[prediction_index].copy()
    eval_frame["prediction"] = prediction_series
    if y_score is not None and y_score.ndim == 2 and y_score.shape[1] > 1:
        eval_frame["confidence"] = y_score[:, 1]

    y_true = eval_frame["label"].astype(int).to_numpy() if "label" in eval_frame.columns else np.empty(0, dtype=int)
    labels = BINARY_LABELS if objective == OBJECTIVE_META_LABEL else LABELS
    class_metrics = classification_metrics(y_true, predictions, y_score=y_score, labels=labels)
    trading_metrics = simulate_trading(
        eval_frame,
        strategy_name=strategy_name,
        signal_column=signal_column,
        objective=objective,
        confidence_threshold=confidence_threshold,
    )
    sweep = (
        confidence_threshold_sweep(eval_frame, strategy_name=strategy_name, signal_column=signal_column)
        if objective == OBJECTIVE_META_LABEL
        else {}
    )

    return {
        "strategy_name": strategy_name,
        "frame_start": eval_frame.index.min(),
        "frame_end": eval_frame.index.max(),
        "classification": class_metrics,
        "trading": trading_metrics,
        "threshold_sweep": sweep,
    }


def _coerce_frames(test_dataset: Any) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    if isinstance(test_dataset, pd.DataFrame):
        return [_ensure_dataframe(test_dataset)], {}

    if isinstance(test_dataset, Mapping):
        metadata = {key: value for key, value in test_dataset.items() if key != "frame"}
        if "frames" in test_dataset:
            return [_ensure_dataframe(frame) for frame in test_dataset["frames"]], metadata
        return [_ensure_dataframe(test_dataset)], metadata

    if isinstance(test_dataset, Sequence) and not isinstance(test_dataset, (str, bytes)):
        return [_ensure_dataframe(frame) for frame in test_dataset], {}

    raise TypeError("test_dataset must be a DataFrame, mapping, or sequence of DataFrames")


def _flatten_results(strategy_name: str, window_results: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, result in enumerate(window_results):
        classification = result["classification"]
        trading = result["trading"]
        rows.append(
            {
                "strategy_name": strategy_name,
                "window": idx,
                "frame_start": result["frame_start"],
                "frame_end": result["frame_end"],
                "samples": classification["count"],
                "accuracy": classification["accuracy"],
                "f1_macro": classification["f1_macro"],
                "roc_auc_ovr": classification["roc_auc_ovr"],
                "trade_count": trading["trade_count"],
                "win_rate": trading["win_rate"],
                "profit_factor": trading["profit_factor"],
                "avg_r": trading["avg_r"],
                "sharpe": trading["sharpe"],
                "calmar": trading["calmar"],
                "max_drawdown_pct": trading["max_drawdown_pct"],
                "combine_passed": trading["combine_passed"],
                "combine_pass_rate": trading["combine_pass_rate"],
                "days_to_pass": trading["days_to_pass"],
                "consistency_ok": trading["consistency_ok"],
                "ending_account": trading.get("ending_account"),
            }
        )
    return pd.DataFrame(rows)


def aggregate_across_folds(per_fold_results: Sequence[Mapping[str, Any]] | pd.DataFrame) -> dict[str, Any]:
    """Return a summary row with mean/median/std/min for numeric fold metrics."""
    frame = per_fold_results.copy() if isinstance(per_fold_results, pd.DataFrame) else pd.DataFrame(list(per_fold_results))
    if "fold" in frame.columns:
        frame = frame.loc[frame["fold"].astype(str) != "summary"].copy()

    summary: dict[str, Any] = {"fold": "summary"}
    for column in frame.columns:
        if column == "fold" or any(str(column).endswith(suffix) for suffix in AGGREGATE_SUFFIXES):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            series = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                summary[column] = float("nan")
                summary[f"{column}_mean"] = float("nan")
                summary[f"{column}_median"] = float("nan")
                summary[f"{column}_std"] = float("nan")
                summary[f"{column}_min"] = float("nan")
                continue
            summary[column] = float(series.mean())
            summary[f"{column}_mean"] = float(series.mean())
            summary[f"{column}_median"] = float(series.median())
            summary[f"{column}_std"] = float(series.std(ddof=0))
            summary[f"{column}_min"] = float(series.min())
        else:
            summary[column] = ""
    return summary


def _as_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_timestamp_index(timestamps: Any) -> pd.DatetimeIndex:
    if isinstance(timestamps, pd.DatetimeIndex):
        return timestamps
    return pd.DatetimeIndex(pd.to_datetime(list(timestamps)))


def _evaluate_window_batch(
    strategy_name: str,
    timeframe: str,
    model: Any,
    window_batch: Any,
    feature_frame: pd.DataFrame,
    signal_column: str,
    device: str = "cpu",
    objective: str = OBJECTIVE_THREE_CLASS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    features = _as_numpy(window_batch.features).astype(np.float32, copy=False)
    labels = _as_numpy(window_batch.labels).astype(int, copy=False)
    timestamps = _as_timestamp_index(window_batch.timestamps)
    raw_signals = _as_numpy(window_batch.raw_signals).astype(int, copy=False)

    y_score, predictions = _predict_with_model(model, features, device=device)
    metric_labels = BINARY_LABELS if objective == OBJECTIVE_META_LABEL else LABELS
    class_metrics = classification_metrics(labels, predictions, y_score=y_score, labels=metric_labels)

    eval_frame = feature_frame.copy().sort_index().reindex(timestamps)
    eval_frame["label"] = labels[: len(eval_frame)]
    eval_frame["prediction"] = predictions[: len(eval_frame)]
    positive_scores = np.full(len(eval_frame), np.nan, dtype=float)
    if y_score is not None and y_score.ndim == 2 and y_score.shape[1] > 1:
        positive_scores = y_score[: len(eval_frame), 1].astype(float, copy=False)
        eval_frame["confidence"] = positive_scores
    eval_frame[signal_column] = raw_signals[: len(eval_frame)]

    trading_metrics = simulate_trading(
        eval_frame,
        strategy_name=strategy_name,
        signal_column=signal_column,
        objective=objective,
        confidence_threshold=confidence_threshold,
    )
    threshold_sweep = (
        confidence_threshold_sweep(eval_frame, strategy_name=strategy_name, signal_column=signal_column)
        if objective == OBJECTIVE_META_LABEL
        else {}
    )
    brier = brier_score(labels, positive_scores) if objective == OBJECTIVE_META_LABEL else float("nan")
    precision_top_50 = (
        precision_at_top_fraction(labels, positive_scores, 0.50) if objective == OBJECTIVE_META_LABEL else float("nan")
    )
    precision_top_20 = (
        precision_at_top_fraction(labels, positive_scores, 0.20) if objective == OBJECTIVE_META_LABEL else float("nan")
    )

    return {
        "strategy_name": strategy_name,
        "timeframe": timeframe,
        "test_f1": class_metrics["f1_macro"],
        "test_roc_auc": class_metrics["roc_auc_ovr"],
        "test_auc_roc": class_metrics["roc_auc"],
        "test_brier": brier,
        "precision_top_50": precision_top_50,
        "precision_top_20": precision_top_20,
        "test_accuracy": class_metrics["accuracy"],
        "test_sharpe": trading_metrics["sharpe"],
        "test_profit_factor": trading_metrics["profit_factor"],
        "test_win_rate": trading_metrics["win_rate"],
        "test_avg_r": trading_metrics["avg_r"],
        "test_max_drawdown": trading_metrics["max_drawdown_pct"],
        "combine_pass_rate": trading_metrics["combine_pass_rate"],
        "avg_days_to_pass": trading_metrics["days_to_pass"],
        "trade_count": trading_metrics["trade_count"],
        "confusion_matrix": class_metrics["confusion_matrix"],
        "classification": class_metrics,
        "trading": trading_metrics,
        "threshold_sweep": threshold_sweep,
    }


def evaluate_strategy(strategy_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    if args and isinstance(args[0], str):
        timeframe = args[0]
        model = args[1] if len(args) > 1 else kwargs["model"]
        window_batch = args[2] if len(args) > 2 else kwargs["window_batch"]
        feature_frame = args[3] if len(args) > 3 else kwargs["feature_frame"]
        signal_column = args[4] if len(args) > 4 else kwargs["signal_column"]
        device = kwargs.get("device", "cpu")
        objective = kwargs.get("objective", OBJECTIVE_THREE_CLASS)
        confidence_threshold = float(kwargs.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))
        return _evaluate_window_batch(
            strategy_name=strategy_name,
            timeframe=timeframe,
            model=model,
            window_batch=window_batch,
            feature_frame=feature_frame,
            signal_column=signal_column,
            device=device,
            objective=objective,
            confidence_threshold=confidence_threshold,
        )

    model = args[0] if len(args) > 0 else kwargs["model"]
    test_dataset = args[1] if len(args) > 1 else kwargs["test_dataset"]
    feature_columns = kwargs.get("feature_columns")
    seq_len = kwargs.get("seq_len")
    signal_column = kwargs.get("signal_column")
    save = kwargs.get("save", True)
    device = kwargs.get("device")
    confidence_threshold = float(kwargs.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD))

    frames, metadata = _coerce_frames(test_dataset)
    objective = kwargs.get("objective", metadata.get("objective", OBJECTIVE_THREE_CLASS))
    resolved_seq_len = int(seq_len or metadata.get("seq_len") or getattr(model, "seq_len", DEFAULT_SEQ_LEN))
    resolved_feature_columns = feature_columns or metadata.get("feature_columns")
    resolved_signal_column = signal_column or metadata.get("signal_column")

    window_results = [
        _evaluate_single_frame(
            strategy_name=strategy_name,
            frame=frame,
            model=model,
            feature_columns=resolved_feature_columns,
            seq_len=resolved_seq_len,
            signal_column=resolved_signal_column,
            device=device,
            objective=objective,
            confidence_threshold=confidence_threshold,
        )
        for frame in frames
    ]

    summary = _flatten_results(strategy_name, window_results)
    aggregate = {
        "strategy_name": strategy_name,
        "windows": window_results,
        "summary": summary,
        "combine_pass_rate": float(summary["combine_passed"].mean()) if not summary.empty else 0.0,
        "avg_days_to_pass": float(summary["days_to_pass"].dropna().mean()) if not summary.empty else float("nan"),
    }

    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = ARTIFACT_DIR / f"eval_{strategy_name}.csv"
        summary_row = aggregate_across_folds(summary)
        summary_with_aggregate = pd.concat([summary, pd.DataFrame([summary_row])], ignore_index=True)
        summary_with_aggregate.to_csv(output_path, index=False)
        aggregate["output_path"] = str(output_path)

    return aggregate


def refresh_eval_artifact(strategy_name: str, artifact_dir: Path | str = ARTIFACT_DIR) -> pd.DataFrame:
    eval_path = Path(artifact_dir) / f"eval_{strategy_name}.csv"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluation artifact: {eval_path}")
    frame = pd.read_csv(eval_path)
    per_fold = frame.loc[frame["fold"].astype(str) != "summary"].copy() if "fold" in frame.columns else frame.copy()
    summary_row = aggregate_across_folds(per_fold)
    refreshed = pd.concat([per_fold, pd.DataFrame([summary_row])], ignore_index=True)
    refreshed.to_csv(eval_path, index=False)
    return refreshed


def refresh_all_eval_artifacts(
    strategy_names: Sequence[str] | None = None,
    artifact_dir: Path | str = ARTIFACT_DIR,
) -> dict[str, pd.DataFrame]:
    from ml.train import build_training_jobs

    refreshed: dict[str, pd.DataFrame] = {}
    for job in build_training_jobs(strategy_names):
        refreshed[job.strategy_name] = refresh_eval_artifact(job.strategy_name, artifact_dir=artifact_dir)
    return refreshed


def _format_report_float(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def _load_old_three_class_sharpe(artifact_dir: Path) -> dict[str, float]:
    path = artifact_dir / "agent3d_old_three_class_sharpe.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "strategy_name" not in frame.columns:
        return {}
    value_column = "old_three_class_test_sharpe_median"
    if value_column not in frame.columns:
        numeric_columns = [column for column in frame.columns if column != "strategy_name"]
        if not numeric_columns:
            return {}
        value_column = numeric_columns[0]
    return {
        str(row["strategy_name"]): float(row[value_column])
        for row in frame.to_dict(orient="records")
        if pd.notna(row.get(value_column))
    }


def write_meta_label_final_report(
    strategy_names: Sequence[str] | None = None,
    artifact_dir: Path | str = ARTIFACT_DIR,
) -> Path:
    """Write Agent 3D evaluation report from per-strategy eval CSVs."""
    from ml.train import build_training_jobs

    artifact_path = Path(artifact_dir)
    old_sharpe = _load_old_three_class_sharpe(artifact_path)
    rows: list[dict[str, Any]] = []
    for job in build_training_jobs(strategy_names):
        eval_path = artifact_path / f"eval_{job.strategy_name}.csv"
        if not eval_path.exists():
            continue
        frame = pd.read_csv(eval_path)
        per_fold = frame.loc[frame["fold"].astype(str) != "summary"].copy() if "fold" in frame.columns else frame
        row: dict[str, Any] = {
            "strategy_name": job.strategy_name,
            "old_three_class_sharpe": old_sharpe.get(job.strategy_name, float("nan")),
            "test_auc_roc": pd.to_numeric(per_fold.get("test_auc_roc", pd.Series(dtype=float)), errors="coerce").median(),
            "test_brier": pd.to_numeric(per_fold.get("test_brier", pd.Series(dtype=float)), errors="coerce").median(),
        }
        for threshold in CONFIDENCE_THRESHOLDS:
            key = f"{threshold:.2f}".replace(".", "_")
            row[f"sharpe_{key}"] = pd.to_numeric(
                per_fold.get(f"test_sharpe_thr_{key}", pd.Series(dtype=float)),
                errors="coerce",
            ).median()
            row[f"trades_{key}"] = pd.to_numeric(
                per_fold.get(f"trade_count_thr_{key}", pd.Series(dtype=float)),
                errors="coerce",
            ).median()
        rows.append(row)

    output_path = artifact_path / "FINAL_EVAL_REPORT.md"
    lines = [
        "# Final Evaluation Report - Agent 3D",
        "",
        "**Objective:** meta-label binary win/loss on strategy signal bars.",
        "",
        "## Sharpe By Confidence Threshold",
        "",
        "| Strategy | Old 3-class Sharpe | AUC-ROC | Brier | Sharpe @0.50 | Trades @0.50 | Sharpe @0.55 | Trades @0.55 | Sharpe @0.60 | Trades @0.60 | Sharpe @0.65 | Trades @0.65 | Sharpe @0.70 | Trades @0.70 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {strategy} | {old} | {auc} | {brier} | {s050} | {t050} | {s055} | {t055} | {s060} | {t060} | {s065} | {t065} | {s070} | {t070} |".format(
                strategy=row["strategy_name"],
                old=_format_report_float(row["old_three_class_sharpe"]),
                auc=_format_report_float(row["test_auc_roc"]),
                brier=_format_report_float(row["test_brier"]),
                s050=_format_report_float(row["sharpe_0_50"]),
                t050=_format_report_float(row["trades_0_50"], digits=0),
                s055=_format_report_float(row["sharpe_0_55"]),
                t055=_format_report_float(row["trades_0_55"], digits=0),
                s060=_format_report_float(row["sharpe_0_60"]),
                t060=_format_report_float(row["trades_0_60"], digits=0),
                s065=_format_report_float(row["sharpe_0_65"]),
                t065=_format_report_float(row["trades_0_65"], digits=0),
                s070=_format_report_float(row["sharpe_0_70"]),
                t070=_format_report_float(row["trades_0_70"], digits=0),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Sharpe and trade counts are medians across rolling OOS folds.",
            "- Confidence thresholds gate meta-label win probability before trade simulation.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def compute_bootstrap_cis(
    strategy_name: str,
    artifact_dir: Path | str = ARTIFACT_DIR,
    n_resamples: int = 1000,
    expected_block_len: float = 5.0,
    random_state: int | None = 42,
) -> dict[str, Any]:
    """Agent 3E: compute bootstrap CIs per fold and aggregated across folds.

    Reads per-fold trade + daily-pnl CSVs produced by ``train._train_one_fold``
    (``fold_trades_{strategy}_{fold}.csv`` and
    ``fold_daily_pnls_{strategy}_{fold}.csv``) and runs
    ``ml.bootstrap.bootstrap_trade_metrics`` on each plus on the concatenated
    aggregate. The aggregated pass-rate bootstrap resamples day sequences and
    re-runs ``TopStepRiskManager`` rather than resampling trades.

    Writes ``eval_{strategy}_bootstrap.json`` to ``artifact_dir``.
    """
    import json

    from ml.bootstrap import (
        bootstrap_pass_rate,
        bootstrap_trade_metrics,
        load_per_fold_series,
    )

    root = Path(artifact_dir)
    per_fold_data = load_per_fold_series(strategy_name, root)
    per_fold_ci: list[dict[str, Any]] = []
    trade_streams: list[np.ndarray] = []
    daily_streams: list[np.ndarray] = []
    for fold_name in sorted(per_fold_data):
        trades = per_fold_data[fold_name]["trade_pnl"]
        daily = per_fold_data[fold_name]["daily_pnl"]
        fold_entry: dict[str, Any] = {
            "fold": fold_name,
            "n_trades": int(len(trades)),
            "n_days": int(len(daily)),
        }
        if len(trades) > 0:
            fold_entry.update(
                bootstrap_trade_metrics(
                    trades,
                    n_resamples=n_resamples,
                    expected_block_len=expected_block_len,
                    random_state=random_state,
                )
            )
            fold_entry["pass_rate"] = bootstrap_pass_rate(
                daily,
                n_resamples=n_resamples,
                expected_block_len=expected_block_len,
                random_state=random_state,
            )
            trade_streams.append(trades)
            daily_streams.append(daily)
        per_fold_ci.append(fold_entry)

    aggregated: dict[str, Any] = {"n_trades": 0, "n_days": 0}
    if trade_streams:
        all_trades = np.concatenate(trade_streams)
        all_daily = np.concatenate(daily_streams)
        aggregated["n_trades"] = int(len(all_trades))
        aggregated["n_days"] = int(len(all_daily))
        aggregated.update(
            bootstrap_trade_metrics(
                all_trades,
                n_resamples=n_resamples,
                expected_block_len=expected_block_len,
                random_state=random_state,
            )
        )
        aggregated["pass_rate"] = bootstrap_pass_rate(
            all_daily,
            n_resamples=n_resamples,
            expected_block_len=expected_block_len,
            random_state=random_state,
        )

    result = {
        "strategy_name": strategy_name,
        "per_fold": per_fold_ci,
        "aggregated": aggregated,
        "config": {
            "n_resamples": int(n_resamples),
            "expected_block_len": float(expected_block_len),
            "random_state": random_state,
        },
    }

    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"eval_{strategy_name}_bootstrap.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["output_path"] = str(output_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Refresh aggregate evaluation rows for strategy eval CSVs.")
    parser.add_argument("--strategy", action="append", dest="strategies", help="Strategy name to refresh; repeatable.")
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    parser.add_argument(
        "--objective",
        choices=(OBJECTIVE_THREE_CLASS, OBJECTIVE_META_LABEL),
        default=OBJECTIVE_THREE_CLASS,
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Agent 3E: compute per-fold + aggregated bootstrap CIs and write eval_{strategy}_bootstrap.json.",
    )
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--block-len", type=float, default=5.0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    if args.bootstrap:
        from ml.train import build_training_jobs

        jobs = build_training_jobs(args.strategies)
        written: list[str] = []
        for job in jobs:
            result = compute_bootstrap_cis(
                job.strategy_name,
                artifact_dir=args.artifact_dir,
                n_resamples=args.n_resamples,
                expected_block_len=args.block_len,
                random_state=args.random_state,
            )
            agg = result.get("aggregated", {})
            n_trades = int(agg.get("n_trades", 0))
            sharpe_p05 = agg.get("sharpe", {}).get("p05") if n_trades > 0 else None
            print(
                f"{job.strategy_name}: n_trades={n_trades} "
                f"sharpe_p05={sharpe_p05!r} -> {result['output_path']}"
            )
            written.append(result["output_path"])
        print(f"wrote {len(written)} bootstrap artifacts")
        return 0

    refreshed = refresh_all_eval_artifacts(strategy_names=args.strategies, artifact_dir=args.artifact_dir)
    for strategy_name, frame in refreshed.items():
        fold_count = int((frame["fold"].astype(str) != "summary").sum()) if "fold" in frame.columns else 0
        print(f"{strategy_name}: refreshed {fold_count} fold rows plus summary")
    if args.objective == OBJECTIVE_META_LABEL:
        report_path = write_meta_label_final_report(strategy_names=args.strategies, artifact_dir=args.artifact_dir)
        print(f"report_path={report_path}")
    return 0


__all__ = [
    "ARTIFACT_DIR",
    "DEFAULT_COMMISSION_PER_RT",
    "DEFAULT_STOP_ATR_MULT",
    "DEFAULT_TARGET_R",
    "CONFIDENCE_THRESHOLDS",
    "OBJECTIVE_META_LABEL",
    "OBJECTIVE_THREE_CLASS",
    "STRATEGY_SIGNAL_COLUMN_MAP",
    "aggregate_across_folds",
    "brier_score",
    "build_sequence_array",
    "calibration_curve",
    "classification_metrics",
    "compute_bootstrap_cis",
    "confidence_threshold_sweep",
    "confusion_matrix_safe",
    "evaluate_strategy",
    "precision_at_top_fraction",
    "simulate_trading",
    "write_meta_label_final_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
