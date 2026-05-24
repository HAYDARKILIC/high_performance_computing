"""
Memory-accounting helpers used throughout the course to reason about
the GPU footprint of training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def format_bytes(n: int | float) -> str:
    """Render a byte count with a binary-prefix suffix (KiB, MiB, GiB)."""
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:6.2f} {unit}"
        n /= 1024.0
    return f"{n:6.2f} PiB"


@dataclass
class MemoryBreakdown:
    """Per-component memory estimate (in bytes) for mixed-precision training.

    Convention follows the ZeRO paper: a 16-bit master copy of parameters and
    gradients (2 bytes each), plus FP32 optimizer state for Adam — that is,
    momentum (4 bytes), variance (4 bytes), and an FP32 parameter copy
    (4 bytes). Activations are excluded because they depend on the model
    architecture, batch size, and whether checkpointing is used.
    """

    params_fp16: int
    grads_fp16: int
    optimizer_state: int

    @property
    def total(self) -> int:
        return self.params_fp16 + self.grads_fp16 + self.optimizer_state

    def as_dict(self) -> dict[str, str]:
        return {
            "params (fp16)": format_bytes(self.params_fp16),
            "grads (fp16)": format_bytes(self.grads_fp16),
            "optimizer state (fp32)": format_bytes(self.optimizer_state),
            "total": format_bytes(self.total),
        }


def memory_breakdown(model: nn.Module) -> MemoryBreakdown:
    """Estimate the static memory footprint of mixed-precision Adam training."""
    n_params = sum(p.numel() for p in model.parameters())
    return MemoryBreakdown(
        params_fp16=n_params * 2,
        grads_fp16=n_params * 2,
        optimizer_state=n_params * 12,  # m, v, and master fp32 copy
    )


def cuda_memory_summary() -> dict[str, str]:
    """Snapshot of allocated / reserved CUDA memory on the current device."""
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    return {
        "allocated": format_bytes(torch.cuda.memory_allocated()),
        "reserved": format_bytes(torch.cuda.memory_reserved()),
        "max_allocated": format_bytes(torch.cuda.max_memory_allocated()),
    }
