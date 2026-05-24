"""
Reference implementations of Megatron-LM-style tensor-parallel layers.

These follow the design of Shoeybi et al. (2019): a *column-parallel* linear
shards the weight matrix along its output dimension, and a *row-parallel*
linear shards along the input dimension. When the two are chained — first
column-parallel, then row-parallel — they bracket the bias-free portion of
the layer so that only a single all-reduce is required per
forward/backward pass.

The implementations here use ``torch.distributed`` collectives and degrade
gracefully to a plain ``nn.Linear`` when no process group is initialised,
which keeps unit tests and notebook demonstrations runnable on a single
device.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class _AllReduce(torch.autograd.Function):
    """All-reduce in forward; identity in backward (g(x) = x)."""

    @staticmethod
    def forward(ctx, x):
        if _world_size() > 1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _Identity(torch.autograd.Function):
    """Identity in forward; all-reduce in backward (f(x) = x)."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if _world_size() > 1:
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output


class ColumnParallelLinear(nn.Module):
    """Linear layer with its weight sharded along the output dimension.

    Each rank holds ``out_features // tp_size`` columns and computes a
    partial output for the full input. When ``gather_output`` is ``True``
    the partials are concatenated back into the full output; otherwise the
    sharded output is returned as-is, which is the usual setting when the
    next layer is a :class:`RowParallelLinear`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 gather_output: bool = False):
        super().__init__()
        tp = _world_size()
        assert out_features % tp == 0, "out_features must be divisible by world_size"
        self.in_features = in_features
        self.out_features = out_features
        self.out_per_partition = out_features // tp
        self.gather_output = gather_output

        self.weight = nn.Parameter(torch.empty(self.out_per_partition, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias = nn.Parameter(torch.zeros(self.out_per_partition)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _Identity.apply(x)
        out = F.linear(x, self.weight, self.bias)
        if self.gather_output and _world_size() > 1:
            gathered = [torch.empty_like(out) for _ in range(_world_size())]
            dist.all_gather(gathered, out)
            out = torch.cat(gathered, dim=-1)
        return out


class RowParallelLinear(nn.Module):
    """Linear layer with its weight sharded along the input dimension.

    Each rank receives a shard of the input (typically produced by a
    preceding :class:`ColumnParallelLinear` with ``gather_output=False``),
    multiplies by its slice of the weight, and the partial outputs are
    summed across ranks with an all-reduce.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 input_is_parallel: bool = True):
        super().__init__()
        tp = _world_size()
        assert in_features % tp == 0, "in_features must be divisible by world_size"
        self.in_features = in_features
        self.out_features = out_features
        self.in_per_partition = in_features // tp
        self.input_is_parallel = input_is_parallel

        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_partition))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel and _world_size() > 1:
            # Split the last dim across ranks.
            x = x.chunk(_world_size(), dim=-1)[_rank()].contiguous()
        partial = F.linear(x, self.weight)
        out = _AllReduce.apply(partial)
        if self.bias is not None:
            out = out + self.bias
        return out
