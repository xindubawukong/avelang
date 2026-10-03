"""Cross-warp K reduction for AMDGPU FP4 GEMM."""

from __future__ import annotations

import avelang
import avelang.language as al

from .config import (
    ACCUM_VALUES,
    WARP_SIZE,
)


def make_reduction(config):
    shm_words = config.n_tiles * (config.num_warps // 2) * WARP_SIZE * ACCUM_VALUES
    M_TILES = al.constexpr(config.m_tiles)
    N_TILES = al.constexpr(config.n_tiles)
    NUM_WARPS = al.constexpr(config.num_warps)
    WARP_M = al.constexpr(config.warp_m)
    WARP_N = al.constexpr(config.warp_n)
    WARP_K = al.constexpr(config.warp_k)
    SHM_U32 = al.constexpr(shm_words)

    @avelang.jit
    def reduce_k(
        shm: al.Tensor((SHM_U32,), al.u32),
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        # Keep annotation-only constants in the Python closure.
        _ = SHM_U32
        reduction = al.view(
            shm,
            al.f32,
            al.make_layout(
                (
                    N_TILES,
                    NUM_WARPS // 2,
                    WARP_SIZE,
                    ACCUM_VALUES,
                ),
                (
                    NUM_WARPS // 2 * WARP_SIZE * ACCUM_VALUES,
                    WARP_SIZE * ACCUM_VALUES,
                    ACCUM_VALUES,
                    1,
                ),
            ),
        )
        acc_col = warp_m * WARP_N + warp_n
        for tile_m in al.range(M_TILES):
            red_offset = WARP_K // 2
            while red_offset:
                if red_offset <= warp_k:
                    if warp_k < 2 * red_offset:
                        write_row = (
                            (warp_k - red_offset) * WARP_M * WARP_N
                            + acc_col
                        )
                        for tile_n in al.range(N_TILES):
                            if red_offset < WARP_K // 2:
                                read_row = warp_k * WARP_M * WARP_N + acc_col
                                acc[tile_m, tile_n] = (
                                    acc[tile_m, tile_n]
                                    + (
                                        reduction[tile_n, read_row, wtid]
                                        + reduction[tile_n, write_row, wtid]
                                    )
                                )
                            reduction[tile_n, write_row, wtid] = acc[
                                tile_m, tile_n
                            ]
                al.syncthreads()
                red_offset = red_offset // 2

            if warp_k == 0:
                for tile_n in al.range(N_TILES):
                    acc[tile_m, tile_n] = (
                        acc[tile_m, tile_n]
                        + reduction[tile_n, acc_col, wtid]
                    )
            al.syncthreads()

    return shm_words if config.warp_k > 1 else 0, reduce_k
