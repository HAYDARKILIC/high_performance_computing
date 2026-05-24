"""
Python wrappers for the custom CUDA matrix-multiplication kernels developed
in Week 2. The kernels are loaded just-in-time via
``torch.utils.cpp_extension.load_inline`` so the package has no compiled
artifacts to ship.

Both kernels accept row-major float32 tensors on CUDA and return the product.
The tiled implementation uses shared-memory blocking for a higher arithmetic
intensity; on an A100 it typically reaches 2-3x the throughput of the naive
version, while still trailing cuBLAS by an order of magnitude (cuBLAS uses
tensor-core MMAs and far more sophisticated scheduling).
"""

from __future__ import annotations

import functools

import torch


_NAIVE_SOURCE = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void naive_matmul_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

torch::Tensor naive_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Only float32 supported");
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    naive_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                          C.data_ptr<float>(), M, N, K);
    return C;
}
"""

_TILED_SOURCE = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

#define TILE 32

__global__ void tiled_matmul_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

torch::Tensor tiled_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Only float32 supported");
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    tiled_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                          C.data_ptr<float>(), M, N, K);
    return C;
}
"""


@functools.lru_cache(maxsize=1)
def _load_extension():
    """JIT-compile both kernels into a single extension on first use."""
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="hpc_matmul",
        cpp_sources=[
            "torch::Tensor naive_matmul_cuda(torch::Tensor A, torch::Tensor B);",
            "torch::Tensor tiled_matmul_cuda(torch::Tensor A, torch::Tensor B);",
        ],
        cuda_sources=[_NAIVE_SOURCE, _TILED_SOURCE],
        functions=["naive_matmul_cuda", "tiled_matmul_cuda"],
        verbose=False,
    )


def naive_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ``a @ b`` with the naive one-thread-per-output kernel."""
    return _load_extension().naive_matmul_cuda(a.contiguous(), b.contiguous())


def tiled_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ``a @ b`` with the shared-memory tiled kernel."""
    return _load_extension().tiled_matmul_cuda(a.contiguous(), b.contiguous())
