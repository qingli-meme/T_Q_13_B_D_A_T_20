"""Deterministic fixed spatial triggers for quantization-backdoor experiments."""

from __future__ import annotations

import torch


SUPPORTED_FIXED_TRIGGERS = {
    "badnet_white",
    "badnet_grid",
}


def build_fixed_spatial_trigger(
    trigger_mode: str,
    trigger_size: int,
    channels: int = 3,
) -> torch.Tensor:
    """Return a fixed CHW trigger tensor in raw pixel space [0, 1]."""
    mode = str(trigger_mode).strip().lower()
    size = int(trigger_size)
    channels = int(channels)

    if mode not in SUPPORTED_FIXED_TRIGGERS:
        raise ValueError(
            f"Unsupported fixed trigger mode: {trigger_mode}. "
            f"Expected one of {sorted(SUPPORTED_FIXED_TRIGGERS)}."
        )
    if size <= 0:
        raise ValueError(f"trigger_size must be positive, got {size}")
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")

    if mode == "badnet_white":
        return torch.ones((channels, size, size), dtype=torch.float32)

    trigger = torch.empty((channels, size, size), dtype=torch.float32)
    dark = 1.0 / 255.0
    for row in range(size):
        for col in range(size):
            value = 1.0 if (row + col) % 2 == 0 else dark
            trigger[:, row, col] = value
    return trigger
