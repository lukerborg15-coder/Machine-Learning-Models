"""ONNX export and verification helpers for Agent 3A."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json
import pickle

import numpy as np

from ml.model import TradingCNN
from ml.train import ARTIFACT_DIR, STRATEGY_TIMEFRAME_MAP, build_training_jobs


OPSET_VERSION = 12
CLASS_MAPPING = {"0": "Long", "1": "Short", "2": "NoTrade"}


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise ModuleNotFoundError("PyTorch is required for ONNX export") from exc
    return torch


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise ModuleNotFoundError(
            "onnxruntime is required for ONNX parity verification. Use Python 3.13 in this workspace."
        ) from exc
    return ort


def load_checkpoint(strategy_name: str, artifact_dir: Path | str = ARTIFACT_DIR) -> dict[str, Any]:
    torch = _require_torch()
    checkpoint_path = Path(artifact_dir) / f"best_model_{strategy_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def load_scaler_payload(strategy_name: str, artifact_dir: Path | str = ARTIFACT_DIR) -> dict[str, Any]:
    scaler_path = Path(artifact_dir) / f"scaler_{strategy_name}.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    with scaler_path.open("rb") as handle:
        return pickle.load(handle)


def build_model_from_checkpoint(checkpoint: dict[str, Any]) -> TradingCNN:
    config = checkpoint["config"]
    feature_columns = checkpoint["feature_columns"]
    model = TradingCNN(
        n_features=len(feature_columns),
        seq_len=int(config["seq_len"]),
        n_filters=int(config["n_filters"]),
        kernel_size=int(config["kernel_size"]),
        n_layers=int(config["n_layers"]),
        dropout=float(config["dropout"]),
        n_classes=int(config["num_classes"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def build_model_config(
    strategy_name: str,
    checkpoint: dict[str, Any],
    deployment_candidate: bool,
    confidence_threshold: float = 0.60,
) -> dict[str, Any]:
    config = checkpoint["config"]
    feature_columns = checkpoint["feature_columns"]
    timeframe_map = {job.strategy_name: job.timeframe for job in build_training_jobs()}
    return {
        "strategy": strategy_name,
        "timeframe": timeframe_map.get(strategy_name, STRATEGY_TIMEFRAME_MAP.get(strategy_name)),
        "seq_len": int(config["seq_len"]),
        "n_features": len(feature_columns),
        "n_classes": int(config["num_classes"]),
        "class_mapping": CLASS_MAPPING,
        "confidence_threshold": confidence_threshold,
        "deployment_candidate": bool(deployment_candidate),
        "opset_version": OPSET_VERSION,
    }


def export_strategy(
    strategy_name: str,
    output_dir: Path | str = ARTIFACT_DIR,
    deployment_candidate: bool = False,
    confidence_threshold: float = 0.60,
) -> dict[str, str]:
    torch = _require_torch()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(strategy_name, artifact_dir=output_path)
    scaler = load_scaler_payload(strategy_name, artifact_dir=output_path)
    model = build_model_from_checkpoint(checkpoint)
    feature_columns = list(checkpoint["feature_columns"])
    model_config = build_model_config(
        strategy_name=strategy_name,
        checkpoint=checkpoint,
        deployment_candidate=deployment_candidate,
        confidence_threshold=confidence_threshold,
    )

    onnx_path = output_path / f"model_{strategy_name}.onnx"
    dummy = torch.zeros(1, model_config["seq_len"], model_config["n_features"])
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}},
        opset_version=OPSET_VERSION,
        dynamo=False,
    )

    feature_order_path = output_path / f"feature_order_{strategy_name}.json"
    feature_order_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    scaler_params_path = output_path / f"scaler_params_{strategy_name}.json"
    scaler_params = {
        "mean": np.asarray(scaler["mean_"], dtype=float).tolist(),
        "std": np.asarray(scaler["scale_"], dtype=float).tolist(),
        "feature_order": feature_columns,
    }
    scaler_params_path.write_text(json.dumps(scaler_params, indent=2), encoding="utf-8")

    model_config_path = output_path / f"model_config_{strategy_name}.json"
    model_config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")

    return {
        "onnx_path": str(onnx_path),
        "feature_order_path": str(feature_order_path),
        "scaler_params_path": str(scaler_params_path),
        "model_config_path": str(model_config_path),
    }


def verify_onnx_matches_pytorch(
    strategy_name: str,
    artifact_dir: Path | str = ARTIFACT_DIR,
    atol: float = 1e-5,
) -> dict[str, float]:
    torch = _require_torch()
    ort = _require_onnxruntime()
    artifact_path = Path(artifact_dir)
    checkpoint = load_checkpoint(strategy_name, artifact_dir=artifact_path)
    model = build_model_from_checkpoint(checkpoint)
    config = build_model_config(strategy_name, checkpoint, deployment_candidate=False)
    rng = np.random.default_rng(20260412)
    random_inputs = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(2, config["seq_len"], config["n_features"]),
    ).astype(np.float32)
    parity_input = np.concatenate(
        [
            np.zeros((1, config["seq_len"], config["n_features"]), dtype=np.float32),
            random_inputs,
        ],
        axis=0,
    )
    parity_tensor = torch.from_numpy(parity_input)
    with torch.no_grad():
        pytorch_out = model(parity_tensor).detach().cpu().numpy()

    session = ort.InferenceSession(str(artifact_path / f"model_{strategy_name}.onnx"))
    onnx_out = session.run(None, {"input": parity_input})[0]
    max_diff = float(np.abs(pytorch_out - onnx_out).max())
    if max_diff >= atol:
        raise AssertionError(f"{strategy_name}: ONNX mismatch, max diff={max_diff}")
    return {"max_diff": max_diff, "atol": float(atol), "samples": float(parity_input.shape[0])}


def export_strategies(
    strategy_names: Sequence[str] | None = None,
    deployment_candidates: set[str] | None = None,
    output_dir: Path | str = ARTIFACT_DIR,
) -> list[dict[str, str]]:
    selected_names = [job.strategy_name for job in build_training_jobs(strategy_names)]
    candidates = deployment_candidates or set()
    return [
        export_strategy(
            strategy_name=strategy_name,
            output_dir=output_dir,
            deployment_candidate=strategy_name in candidates,
        )
        for strategy_name in selected_names
    ]


__all__ = [
    "CLASS_MAPPING",
    "OPSET_VERSION",
    "build_model_config",
    "build_model_from_checkpoint",
    "export_strategies",
    "export_strategy",
    "load_checkpoint",
    "load_scaler_payload",
    "verify_onnx_matches_pytorch",
]
