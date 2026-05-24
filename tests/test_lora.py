"""Unit tests for the LoRA implementation from Week 6."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Inline copy of the Week 6 LoRA module so the test is self-contained."""

    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / r
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


def test_lora_zero_at_init_matches_base():
    """B is zero-initialized, so the LoRA module must equal its base at init."""
    torch.manual_seed(0)
    layer = LoRALinear(32, 32, r=4, alpha=8)
    x = torch.randn(8, 32)
    out_lora = layer(x)
    out_base = layer.base(x)
    assert torch.allclose(out_lora, out_base, atol=1e-6)


def test_lora_merge_equivalence():
    """After updating lora_B, the merged weight must reproduce forward()."""
    torch.manual_seed(0)
    layer = LoRALinear(16, 16, r=4, alpha=8)
    layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1
    x = torch.randn(4, 16)
    expected = layer(x)
    merged = layer.merged_weight()
    actual = x @ merged.T
    assert torch.allclose(expected, actual, atol=1e-5)


def test_lora_parameter_count():
    """Only the two LoRA matrices should be trainable."""
    layer = LoRALinear(128, 128, r=8, alpha=16)
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert trainable == 8 * 128 + 128 * 8
