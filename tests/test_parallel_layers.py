"""Tests for the Megatron-style parallel layers (Week 5).

These tests run on a single device, where the world size is 1, so the
parallel layers must reduce to plain ``nn.Linear`` behaviour. Genuine
multi-rank testing requires a torchrun launch and is left to integration
tests outside the CPU CI matrix.
"""

from __future__ import annotations

import torch

from src.distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear


def test_column_parallel_single_rank_matches_linear():
    torch.manual_seed(0)
    layer = ColumnParallelLinear(64, 32, bias=True, gather_output=True)
    x = torch.randn(4, 64)
    out = layer(x)
    assert out.shape == (4, 32)
    # With world_size=1, ColumnParallelLinear == nn.Linear in arithmetic.
    expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
    assert torch.allclose(out, expected, atol=1e-6)


def test_row_parallel_single_rank_matches_linear():
    torch.manual_seed(0)
    layer = RowParallelLinear(64, 32, bias=True, input_is_parallel=True)
    x = torch.randn(4, 64)
    out = layer(x)
    assert out.shape == (4, 32)
    expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
    assert torch.allclose(out, expected, atol=1e-6)


def test_column_then_row_composes_correctly():
    """Stack column-parallel then row-parallel and check end-to-end shape."""
    torch.manual_seed(0)
    col = ColumnParallelLinear(64, 128, bias=False, gather_output=False)
    row = RowParallelLinear(128, 64, bias=False, input_is_parallel=True)
    x = torch.randn(2, 64)
    out = row(col(x))
    assert out.shape == (2, 64)
