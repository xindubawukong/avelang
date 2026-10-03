"""FP4 GEMM result writeback for AMDGPU."""

from __future__ import annotations

import avelang
import avelang.language as al

from .config import (
    ACCUM_VALUES,
    MAX_SHM_BYTES,
    QUANT_VEC_SIZE,
    TILE,
    U16_BYTES,
    U32_BYTES,
    UINT4_BYTES,
    WARP_SIZE,
)
from .solution import MatmulMfmaType
from .utils import _ceildiv

RESULT_VEC_SIZE = UINT4_BYTES // U16_BYTES
SHM_VEC_SIZE = 2 * U32_BYTES // U16_BYTES


def make_writeback_ops(config):
    max_result_tiles = min(config.m_tiles, MAX_SHM_BYTES // (config.group_n * U16_BYTES * TILE * config.warp_m))
    partition_tiles = config.m_tiles // _ceildiv(config.m_tiles, max_result_tiles)
    shm_result_u32 = partition_tiles * config.warp_m * TILE * config.group_n * U16_BYTES // U32_BYTES
    M_TILES = al.constexpr(config.m_tiles)
    N_TILES = al.constexpr(config.n_tiles)
    WARP_M = al.constexpr(config.warp_m)
    WARP_N = al.constexpr(config.warp_n)
    WARP_K = al.constexpr(config.warp_k)
    WARP_TILE_M = al.constexpr(config.warp_tile_m)
    WARP_TILE_N = al.constexpr(config.warp_tile_n)
    GROUP_N = al.constexpr(config.group_n)
    THREADS = al.constexpr(config.threads)
    MFMA_TYPE = al.constexpr(config.mfma_type)
    FP16 = MatmulMfmaType.FP16
    RESULT_M_TILES = al.constexpr(partition_tiles)
    UNEVEN_RESULT_PARTITIONS = al.constexpr(config.m_tiles % partition_tiles != 0)
    SHM_U32 = al.constexpr(shm_result_u32)

    @avelang.jit
    def pack_result_partition(
        result_u64: al.Tensor((SHM_U32 // 2, 2), al.u32),
        result_partition: al.u32,
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        alpha: al.f32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        # Keep annotation-only constants in the Python closure.
        _ = SHM_U32
        tile_m_stride = (
            WARP_M if UNEVEN_RESULT_PARTITIONS else 1
        ) * TILE * GROUP_N // SHM_VEC_SIZE
        warp_m_stride = (
            1 if UNEVEN_RESULT_PARTITIONS else RESULT_M_TILES
        ) * TILE * GROUP_N // SHM_VEC_SIZE
        lane_row = wtid // TILE
        lane_col = wtid % TILE
        alpha_pair = al.full((1, 2), alpha, al.f32)
        acc_pairs = al.view(
            acc, al.f32,
            al.make_layout((M_TILES, N_TILES, 2, 2), (N_TILES * 4, 4, 2, 1)),
        )
        packed = al.make_local((1, 2), al.u32)
        result = al.view(
            result_u64,
            al.u32,
            al.make_layout(
                (
                    RESULT_M_TILES,
                    N_TILES,
                    WARP_M,
                    WARP_N,
                    TILE,
                    WARP_SIZE // TILE,
                    2,
                ),
                (
                    tile_m_stride * 2,
                    ACCUM_VALUES * 2,
                    warp_m_stride * 2,
                    (WARP_TILE_N // SHM_VEC_SIZE) * 2,
                    (GROUP_N // SHM_VEC_SIZE) * 2,
                    2,
                    1,
                ),
            ),
        )

        if WARP_K == 1 or warp_k == 0:
            for tile_m in al.range(RESULT_M_TILES):
                acc_m = result_partition * RESULT_M_TILES + tile_m
                if acc_m < M_TILES:
                    for tile_n in al.range(N_TILES):
                        for i in al.range(2):
                            pair = acc_pairs[acc_m, tile_n, i] * alpha_pair[0]
                            if MFMA_TYPE == FP16:
                                packed[0, i] = al.amdgpu.cvt_pk_f16_f32(pair[0], pair[1])
                            else:
                                packed[0, i] = al.amdgpu.cvt_pk_bf16_f32(pair[0], pair[1])
                        result[
                            tile_m,
                            tile_n,
                            warp_m,
                            warp_n,
                            lane_col,
                            lane_row,
                        ] = packed[0]

    @avelang.jit
    def store_result_vector(
        c_rsrc: al.Tensor((4,), al.u32),
        result_u128: al.Tensor((SHM_U32 // 4, 4), al.u32),
        n: al.u32,
        result_partition: al.u32,
        idx: al.u32,
    ):
        _ = SHM_U32
        m_tile_start = result_partition * RESULT_M_TILES
        partition_idx = al.convert(0, al.u32)
        row_base = al.convert(0, al.u32)
        if UNEVEN_RESULT_PARTITIONS:
            stripe_vectors = WARP_M * TILE * GROUP_N // RESULT_VEC_SIZE
            stripe = idx // stripe_vectors
            stripe_idx = idx % stripe_vectors
            warp_vectors = TILE * GROUP_N // RESULT_VEC_SIZE
            output_warp_m = stripe_idx // warp_vectors
            partition_idx = stripe_idx % warp_vectors
            row_base = output_warp_m * WARP_TILE_M + (m_tile_start + stripe) * TILE
        else:
            warp_vectors = RESULT_M_TILES * TILE * GROUP_N // RESULT_VEC_SIZE
            output_warp_m = idx // warp_vectors
            partition_idx = idx % warp_vectors
            row_base = output_warp_m * WARP_TILE_M + m_tile_start * TILE
        row = partition_idx // (GROUP_N // RESULT_VEC_SIZE)
        col = partition_idx % (GROUP_N // RESULT_VEC_SIZE)
        output_row = row_base + row
        al.amdgpu.raw_buffer_store_x4(
            result_u128[idx],
            c_rsrc,
            (output_row * n // RESULT_VEC_SIZE + col) * UINT4_BYTES,
            0,
            0,
        )

    @avelang.jit
    def write_results(
        c_rsrc: al.Tensor((4,), al.u32),
        shm: al.Tensor((SHM_U32,), al.u32),
        n: al.u32,
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        tid: al.u32,
        alpha: al.f32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        # Keep annotation-only constants in the Python closure.
        _ = N_TILES
        result_partitions = (M_TILES + RESULT_M_TILES - 1) // RESULT_M_TILES
        result_vectors = RESULT_M_TILES * WARP_M * TILE * GROUP_N // RESULT_VEC_SIZE
        result_loads = (result_vectors + THREADS - 1) // THREADS
        result_u64 = al.view(
            shm,
            al.u32,
            al.make_layout(
                (SHM_U32 // 2, 2),
                (2, 1),
            ),
        )
        result_u128 = al.view(
            shm,
            al.u32,
            al.make_layout(
                (SHM_U32 // QUANT_VEC_SIZE, QUANT_VEC_SIZE),
                (QUANT_VEC_SIZE, 1),
            ),
        )
        for result_partition in al.range(result_partitions):
            pack_result_partition(
                result_u64, result_partition,
                warp_m, warp_n, warp_k, wtid, alpha, acc,
            )
            al.syncthreads()
            pending_items = (
                al.min(RESULT_M_TILES, M_TILES - result_partition * RESULT_M_TILES)
                * WARP_M * TILE * GROUP_N // RESULT_VEC_SIZE
            )
            for i in al.range(result_loads):
                idx = tid + i * THREADS
                if idx < pending_items:
                    store_result_vector(
                        c_rsrc, result_u128, n, result_partition, idx,
                    )
            if result_partition + 1 < result_partitions:
                al.syncthreads()

    return shm_result_u32, write_results
