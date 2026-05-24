"""
Lightweight profiling helpers: NVTX ranges for Nsight Systems and a tiny
microbenchmark function with proper CUDA synchronisation. Both are
deliberately framework-free so they can be dropped into any training or
inference script.
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Callable, Iterator

import torch


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Tag a block of code with an NVTX range visible in Nsight Systems.

    The context degrades to a no-op if CUDA is not available, which makes
    the same code path usable on CPU-only CI runners.
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def bench(
    fn: Callable[[], object],
    *,
    warmup: int = 5,
    iters: int = 20,
    sync: bool = True,
) -> float:
    """Return the mean wall-clock time per call to ``fn`` in milliseconds.

    The function is called ``warmup`` times to amortise JIT compilation
    and allocator warm-up, then timed across ``iters`` calls. CUDA streams
    are synchronised before and after each timing window so the measured
    interval covers actual device work.
    """
    for _ in range(warmup):
        fn()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = perf_counter()
    for _ in range(iters):
        fn()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    return (perf_counter() - t0) * 1000.0 / iters
