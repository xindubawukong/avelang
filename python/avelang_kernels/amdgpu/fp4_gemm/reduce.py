"""Cross-warp K reduction for AMDGPU FP4 GEMM."""

from __future__ import annotations

import avelang
import avelang.language as al

from .config import FP4GemmConfig


def fp4_reduction_shm_u32(config: FP4GemmConfig) -> int:
    if config.warp_k <= 1:
        return 0
    return (
        config.n_tiles
        * (config.num_warps // 2)
        * config.warp_size
        * config.accumulator_shape[2]
    )


@avelang.jit
def reduce_k(
    config: al.constexpr,
    shm: al.Tensor((config.shm_u32,), al.u32),
    warp_m: al.u32,
    warp_n: al.u32,
    warp_k: al.u32,
    wtid: al.u32,
    acc: al.Tensor((config.gemm.m_tiles, config.gemm.n_tiles, 4), al.f32),
):
    reduction = al.view(
        shm,
        al.f32,
        al.make_layout(
            (
                config.gemm.n_tiles,
                config.gemm.num_warps // 2,
                config.gemm.warp_size,
                config.gemm.accum_values,
            ),
            (
                config.gemm.num_warps // 2 * config.gemm.warp_size * config.gemm.accum_values,
                config.gemm.warp_size * config.gemm.accum_values,
                config.gemm.accum_values,
                1,
            ),
        ),
    )
    acc_col = warp_m * config.gemm.warp_n + warp_n
    for tile_m in al.range(config.gemm.m_tiles):
        red_offset = config.gemm.warp_k // 2
        while red_offset:
            if red_offset <= warp_k:
                if warp_k < 2 * red_offset:
                    write_row = (
                        (warp_k - red_offset) * config.gemm.warp_m * config.gemm.warp_n
                        + acc_col
                    )
                    for tile_n in al.range(config.gemm.n_tiles):
                        if red_offset < config.gemm.warp_k // 2:
                            read_row = warp_k * config.gemm.warp_m * config.gemm.warp_n + acc_col
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
            for tile_n in al.range(config.gemm.n_tiles):
                acc[tile_m, tile_n] = (
                    acc[tile_m, tile_n]
                    + reduction[tile_n, acc_col, wtid]
                )
        al.syncthreads()
