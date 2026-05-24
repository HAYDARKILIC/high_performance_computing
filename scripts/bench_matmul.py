"""
Benchmark naive vs tiled CUDA matmul against cuBLAS (Week 2).

Run with::

    python scripts/bench_matmul.py --sizes 512 1024 2048
"""

from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for matmul benchmark.")

    from src.cuda.matmul import naive_matmul, tiled_matmul
    from src.utils.profiling import bench

    print(f"{'N':>6} | {'naive ms':>10} | {'tiled ms':>10} | {'cuBLAS ms':>10} | {'naive GF/s':>11} | {'tiled GF/s':>11} | {'cuBLAS GF/s':>12}")
    print("-" * 95)

    for n in args.sizes:
        a = torch.randn(n, n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, n, device="cuda", dtype=torch.float32)
        flops = 2.0 * n * n * n  # 2*N^3 for matmul

        t_naive = bench(lambda: naive_matmul(a, b), warmup=args.warmup, iters=args.iters)
        t_tiled = bench(lambda: tiled_matmul(a, b), warmup=args.warmup, iters=args.iters)
        t_cublas = bench(lambda: torch.matmul(a, b), warmup=args.warmup, iters=args.iters)

        gflops = lambda ms: flops / (ms * 1e6)  # noqa: E731
        print(
            f"{n:>6} | {t_naive:>10.3f} | {t_tiled:>10.3f} | {t_cublas:>10.3f} | "
            f"{gflops(t_naive):>11.2f} | {gflops(t_tiled):>11.2f} | {gflops(t_cublas):>12.2f}"
        )


if __name__ == "__main__":
    main()
