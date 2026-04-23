"""Conv1D trading model with causal padding."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _normalize_kernel_sizes(kernel_size: int | Sequence[int], n_layers: int) -> list[int]:
    if isinstance(kernel_size, Sequence) and not isinstance(kernel_size, (str, bytes)):
        kernel_sizes = [int(size) for size in kernel_size]
    else:
        kernel_sizes = [int(kernel_size)] * int(n_layers)

    if not kernel_sizes:
        raise ValueError("kernel_size must define at least one convolution")
    if any(size < 1 for size in kernel_sizes):
        raise ValueError("kernel_size values must be positive")
    return kernel_sizes


def _normalize_channels(
    n_filters: int,
    n_layers: int,
    conv_channels: Sequence[int] | None,
) -> list[int]:
    if conv_channels is not None:
        channels = [int(channel) for channel in conv_channels]
    else:
        channels = [int(n_filters) * (2 ** layer_idx) for layer_idx in range(int(n_layers))]

    if not channels:
        raise ValueError("conv_channels must define at least one convolution")
    if any(channel < 1 for channel in channels):
        raise ValueError("conv_channels values must be positive")
    return channels


class TradingCNN(nn.Module):
    """Conv1D classifier for bar sequences with left-only causal padding."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        n_filters: int = 64,
        kernel_size: int | Sequence[int] = 3,
        n_layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 3,
        conv_channels: Sequence[int] | None = None,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")

        channel_stack = _normalize_channels(n_filters=n_filters, n_layers=n_layers, conv_channels=conv_channels)
        kernel_stack = _normalize_kernel_sizes(kernel_size=kernel_size, n_layers=len(channel_stack))
        if len(kernel_stack) != len(channel_stack):
            raise ValueError("kernel_size and conv stack length must match")

        self.n_features = int(n_features)
        self.seq_len = int(seq_len)
        self.n_classes = int(n_classes)

        blocks: list[nn.Module] = []
        in_channels = self.n_features
        for out_channels, kernel in zip(channel_stack, kernel_stack, strict=True):
            blocks.extend(
                [
                    nn.ConstantPad1d((kernel - 1, 0), 0.0),
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=0),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels),
                ]
            )
            in_channels = out_channels

        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.n_classes),
        )

    def _coerce_input(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        squeeze_batch = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.ndim != 3:
            raise ValueError("TradingCNN expects input of shape (batch, seq_len, n_features)")

        if x.shape[-1] == self.n_features:
            return x, squeeze_batch
        if x.shape[1] == self.n_features:
            return x.transpose(1, 2), squeeze_batch

        raise ValueError(
            f"Expected feature dimension {self.n_features}, received shape {tuple(int(dim) for dim in x.shape)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, squeeze_batch = self._coerce_input(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        if squeeze_batch:
            return logits.squeeze(0)
        return logits


__all__ = ["TradingCNN"]
