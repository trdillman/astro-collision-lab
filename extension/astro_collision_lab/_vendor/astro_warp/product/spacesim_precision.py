"""Host admission for the SpaceSim engine's float32 device channels."""

from __future__ import annotations

import numpy as np


def finite_float32(values, name: str, *, positive: bool = False):
    """Stage an owned destination buffer and refuse overflow or positive underflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        staged = np.array(values, dtype=np.float32, copy=True)
    if not np.all(np.isfinite(staged)):
        raise ValueError(f"{name} must be finite in float32 storage")
    if positive and np.any(staged <= 0.0):
        raise ValueError(f"{name} must remain positive in float32 storage")
    return staged
