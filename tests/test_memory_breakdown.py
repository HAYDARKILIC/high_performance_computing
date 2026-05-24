"""Tests for the memory accounting helpers from Week 4."""

from __future__ import annotations

import torch.nn as nn

from src.utils.memory import format_bytes, memory_breakdown


def test_memory_breakdown_known_size():
    """For a 1M-parameter model with Adam mixed precision the total is 16 MB."""
    model = nn.Linear(1000, 1000, bias=False)  # exactly 1,000,000 parameters
    bd = memory_breakdown(model)
    n_params = 1_000_000
    assert bd.params_fp16 == n_params * 2
    assert bd.grads_fp16 == n_params * 2
    assert bd.optimizer_state == n_params * 12
    assert bd.total == n_params * 16  # 16 bytes/param under the ZeRO accounting


def test_format_bytes_units():
    assert "KiB" in format_bytes(2048)
    assert "MiB" in format_bytes(5 * 1024 * 1024)
    assert "GiB" in format_bytes(3 * 1024**3)


def test_memory_breakdown_dict_keys():
    bd = memory_breakdown(nn.Linear(10, 10))
    keys = set(bd.as_dict().keys())
    assert {"params (fp16)", "grads (fp16)", "optimizer state (fp32)", "total"} <= keys
