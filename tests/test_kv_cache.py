"""Tests for the static KV cache from Week 3."""

from __future__ import annotations

import pytest
import torch

from src.inference.kv_cache import StaticKVCache


def test_kv_cache_update_returns_full_prefix():
    cache = StaticKVCache(
        batch_size=2, num_heads=4, max_seq_len=16, head_dim=8, device="cpu", dtype=torch.float32
    )
    k1 = torch.randn(2, 4, 5, 8)
    v1 = torch.randn(2, 4, 5, 8)
    k_view, v_view = cache.update(k1, v1)
    assert k_view.shape == (2, 4, 5, 8)
    assert torch.equal(k_view, k1)
    assert torch.equal(v_view, v1)
    assert cache.cur_len == 5

    # Append a single new step and confirm the view grows by one.
    k2 = torch.randn(2, 4, 1, 8)
    v2 = torch.randn(2, 4, 1, 8)
    k_view, v_view = cache.update(k2, v2)
    assert k_view.shape == (2, 4, 6, 8)
    assert torch.equal(k_view[:, :, :5, :], k1)
    assert torch.equal(k_view[:, :, 5:, :], k2)


def test_kv_cache_overflow_raises():
    cache = StaticKVCache(
        batch_size=1, num_heads=1, max_seq_len=4, head_dim=2, device="cpu", dtype=torch.float32
    )
    with pytest.raises(RuntimeError):
        cache.update(torch.zeros(1, 1, 5, 2), torch.zeros(1, 1, 5, 2))


def test_kv_cache_num_bytes():
    cache = StaticKVCache(
        batch_size=1, num_heads=2, max_seq_len=8, head_dim=4, device="cpu", dtype=torch.float32
    )
    # 1 * 2 * 8 * 4 floats * 4 bytes = 256 bytes per tensor, 512 in total.
    assert cache.num_bytes == 2 * 1 * 2 * 8 * 4 * 4
