"""
Helpers for FSDP wrapping, mixed precision, and gradient checkpointing.
These mirror the patterns covered in the Week 4 notebook.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Type

import torch
import torch.nn as nn


def get_mixed_precision_policy(compute_dtype: torch.dtype = torch.bfloat16):
    """Return a sensible ``MixedPrecision`` policy for FSDP."""
    from torch.distributed.fsdp import MixedPrecision

    return MixedPrecision(
        param_dtype=compute_dtype,
        reduce_dtype=compute_dtype,
        buffer_dtype=compute_dtype,
    )


def get_size_based_auto_wrap_policy(min_num_params: int = 1_000_000):
    """Wrap any submodule whose parameter count exceeds ``min_num_params``."""
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    return partial(size_based_auto_wrap_policy, min_num_params=min_num_params)


def get_transformer_auto_wrap_policy(layer_cls: Type[nn.Module]):
    """Wrap each instance of a given transformer block class."""
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    return partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})


def wrap_with_fsdp(
    model: nn.Module,
    transformer_layer_cls: Optional[Type[nn.Module]] = None,
    min_num_params: int = 1_000_000,
    compute_dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = False,
):
    """Wrap ``model`` in FSDP with reasonable defaults.

    If ``transformer_layer_cls`` is given the canonical transformer wrap
    policy is used; otherwise a size-based policy is applied.
    """
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP

    policy = (
        get_transformer_auto_wrap_policy(transformer_layer_cls)
        if transformer_layer_cls is not None
        else get_size_based_auto_wrap_policy(min_num_params)
    )
    return FSDP(
        model,
        auto_wrap_policy=policy,
        mixed_precision=get_mixed_precision_policy(compute_dtype),
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
    )


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable activation recomputation on a Hugging Face model if supported."""
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
