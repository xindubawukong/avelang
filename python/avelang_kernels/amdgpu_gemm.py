#!/usr/bin/env python3

import torch

import avelang
import avelang.language as al


TILE_M = 16
TILE_N = 16


@avelang.jit
def _gemm_pipeline_transposed_b_kernel(
    A_ptr: al.Pointer(al.bf16),
    B_ptr: al.Pointer(al.bf16),
    C_ptr: al.Pointer(al.bf16),
    m: al.i32,
    n: al.i32,
    k: al.i32,
):
    row = al.block_id(1) * TILE_M + al.thread_id(1)
    col = al.block_id(0) * TILE_N + al.thread_id(0)

    A = al.make_tensor(A_ptr, al.bf16, al.make_layout((m * k,), (1,)))
    B = al.make_tensor(B_ptr, al.bf16, al.make_layout((n * k,), (1,)))
    C = al.make_tensor(C_ptr, al.bf16, al.make_layout((m * n,), (1,)))

    if row < m and col < n:
        acc = al.convert(0.0, al.f32)
        for kk in al.range(k):
            a = al.convert(A[row * k + kk], al.f32)
            b = al.convert(B[col * k + kk], al.f32)
            acc += a * b
        C[row * n + col] = al.convert(acc, al.bf16)


def gemm_pipeline_transposed_b(A, B, out=None):
    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[0]

    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)

    grid_x = (n + TILE_N - 1) // TILE_N
    grid_y = (m + TILE_M - 1) // TILE_M

    _gemm_pipeline_transposed_b_kernel[
        lambda: ((grid_x, grid_y, 1), (TILE_N, TILE_M, 1))
    ](A, B, out, m, n, k, num_warps=4)
    return out
