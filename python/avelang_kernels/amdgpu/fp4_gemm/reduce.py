"""Cross-warp K reduction for AMDGPU FP4 GEMM."""

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


def make_fp4_reduce(config: FP4GemmConfig, shm_u32: int):
    WARP_M = config.warp_m
    WARP_N = config.warp_n
    WARP_K = config.warp_k
    WARP_SIZE = config.warp_size
    NUM_WARPS = config.num_warps
    M_TILES = config.m_tiles
    N_TILES = config.n_tiles
    ACCUM_VALUES = config.accumulator_shape[2]
    ACC_ROW_STRIDE = WARP_M * WARP_N
    REDUCTION_WARPS = NUM_WARPS // 2

    @avelang.jit
    def _reduce_k(
        shm: al.Tensor((shm_u32,), al.u32),
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        reduction = al.view(
            shm,
            al.f32,
            al.make_layout(
                (
                    N_TILES,
                    REDUCTION_WARPS,
                    WARP_SIZE,
                    ACCUM_VALUES,
                ),
                (
                    REDUCTION_WARPS * WARP_SIZE * ACCUM_VALUES,
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
                            (warp_k - red_offset) * ACC_ROW_STRIDE
                            + acc_col
                        )
                        for tile_n in al.range(N_TILES):
                            if red_offset < WARP_K // 2:
                                read_row = warp_k * ACC_ROW_STRIDE + acc_col
                                acc[tile_m, tile_n] = (
                                    acc[tile_m, tile_n]
                                    + reduction[tile_n, read_row, wtid]
                                    + reduction[tile_n, write_row, wtid]
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

    return _reduce_k
