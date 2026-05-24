"""Shared pytest configuration: GPU skip markers and helper fixtures."""

from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Skip GPU and multi-GPU tests when the required hardware is absent."""
    skip_gpu = pytest.mark.skip(reason="requires CUDA")
    skip_multi = pytest.mark.skip(reason="requires multiple CUDA devices")
    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    for item in items:
        if "gpu" in item.keywords and n_devices < 1:
            item.add_marker(skip_gpu)
        if "multi_gpu" in item.keywords and n_devices < 2:
            item.add_marker(skip_multi)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
