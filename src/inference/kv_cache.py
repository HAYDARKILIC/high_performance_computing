"""
A minimal static key/value cache illustrating the memory model behind
PagedAttention. Real serving systems (vLLM) replace the contiguous tensor
allocation here with a page table so that distinct sequences can share
the same physical block — see Kwon et al. (2023).
"""

from __future__ import annotations

import torch


class StaticKVCache:
    """Pre-allocated KV cache with shape (batch, heads, max_seq_len, head_dim).

    The two tensors are filled in-place as decoding proceeds. After each
    step, :meth:`update` writes the freshly computed K/V tensors at the
    current position and returns the cumulative view up to that position.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len
        self.cur_len = 0

    @property
    def num_bytes(self) -> int:
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append ``new_k`` and ``new_v`` to the cache and return the prefix views."""
        step = new_k.size(-2)
        if self.cur_len + step > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: cur_len={self.cur_len}, step={step}, "
                f"max_seq_len={self.max_seq_len}"
            )
        end = self.cur_len + step
        self.k[:, :, self.cur_len:end, :] = new_k
        self.v[:, :, self.cur_len:end, :] = new_v
        self.cur_len = end
        return self.k[:, :, :end, :], self.v[:, :, :end, :]

    def reset(self) -> None:
        self.cur_len = 0
        self.k.zero_()
        self.v.zero_()
