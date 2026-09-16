"""FP4 GEMM result writeback for AMDGPU."""

from dataclasses import dataclass

import avelang
import avelang.language as al

from .solution import MatmulMfmaType

from .config import (
    FP4GemmConfig,
    U32_BYTES,
    UINT4_BYTES,
)


def _ceildiv(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


@dataclass(frozen=True, slots=True)
class FP4WritebackConfig:
    gemm: FP4GemmConfig
    result_m_tiles: int
    result_partitions: int
    uneven_result_partitions: bool
    result_tile_m_stride: int
    result_warp_m_stride: int
    result_minor_count: int
    shm_result_u32: int
    result_full_loads: int
    result_remainder: int
    last_result_full_loads: int
    last_result_remainder: int

    @classmethod
    def from_config(cls, gemm: FP4GemmConfig) -> "FP4WritebackConfig":
        group_n = gemm.group_n
        warp_m = gemm.warp_m
        m_tiles = gemm.m_tiles
        threads = gemm.threads
        max_result_tiles = min(
            m_tiles,
            gemm.max_shm_bytes
            // (group_n * gemm.output_bytes * gemm.tile * warp_m),
        )
        result_m_tiles = m_tiles // _ceildiv(m_tiles, max_result_tiles)
        result_partitions = _ceildiv(m_tiles, result_m_tiles)
        uneven = m_tiles % result_m_tiles != 0
        shm_result_u32 = (
            result_m_tiles
            * warp_m
            * gemm.tile
            * group_n
            * gemm.output_bytes
            // U32_BYTES
        )
        result_vector_count = shm_result_u32 // gemm.quant_vec_size
        last_result_m_tiles = (
            m_tiles - (result_partitions - 1) * result_m_tiles
        )
        last_result_vector_count = (
            last_result_m_tiles
            * warp_m
            * gemm.tile
            * group_n
            // gemm.result_vec_size
        )
        return cls(
            gemm=gemm,
            result_m_tiles=result_m_tiles,
            result_partitions=result_partitions,
            uneven_result_partitions=uneven,
            result_tile_m_stride=(warp_m if uneven else 1)
            * gemm.tile
            * group_n
            // gemm.shm_vec_size,
            result_warp_m_stride=(1 if uneven else result_m_tiles)
            * gemm.tile
            * group_n
            // gemm.shm_vec_size,
            result_minor_count=warp_m if uneven else result_m_tiles,
            shm_result_u32=shm_result_u32,
            result_full_loads=result_vector_count // threads,
            result_remainder=result_vector_count % threads,
            last_result_full_loads=last_result_vector_count // threads,
            last_result_remainder=last_result_vector_count % threads,
        )


def make_fp4_writeback(writeback: FP4WritebackConfig, shm_u32: int):
    gemm = writeback.gemm
    IS_FP16 = gemm.mfma_type == MatmulMfmaType.FP16
    GROUP_N = gemm.group_n
    WARP_M = gemm.warp_m
    WARP_N = gemm.warp_n
    WARP_K = gemm.warp_k
    WARP_SIZE = gemm.warp_size
    WARP_TILE_M = gemm.warp_tile_m
    WARP_TILE_N = gemm.warp_tile_n
    M_TILES = gemm.m_tiles
    N_TILES = gemm.n_tiles
    THREADS = gemm.threads
    TILE = gemm.tile
    ACCUM_VALUES = gemm.accumulator_shape[2]
    RESULT_VEC_SIZE = gemm.result_vec_size
    SHM_VEC_SIZE = gemm.shm_vec_size
    QUANT_VEC_SIZE = gemm.quant_vec_size
    RESULT_M_TILES = writeback.result_m_tiles
    RESULT_PARTITIONS = writeback.result_partitions
    UNEVEN_RESULT_PARTITIONS = writeback.uneven_result_partitions
    RESULT_TILE_M_STRIDE = writeback.result_tile_m_stride
    RESULT_WARP_M_STRIDE = writeback.result_warp_m_stride
    RESULT_MINOR_COUNT = writeback.result_minor_count
    RESULT_FULL_LOADS = writeback.result_full_loads
    RESULT_REMAINDER = writeback.result_remainder
    LAST_RESULT_FULL_LOADS = writeback.last_result_full_loads
    LAST_RESULT_REMAINDER = writeback.last_result_remainder

    @avelang.jit
    def _pack_result_partition(
        result_u64: al.Tensor((shm_u32 // 2, 2), al.u32),
        result_partition: al.u32,
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        alpha: al.f32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        lane_row = wtid // TILE
        lane_col = wtid % TILE
        alpha_vec = al.full((1, 4), alpha, al.f32)
        scaled = al.make_local((1, 4), al.f32)
        packed = al.make_local((1, 2), al.u32)
        converted = al.view(
            packed, al.u16, al.make_layout((1, 4), (4, 1)),
        )
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
                    RESULT_TILE_M_STRIDE * 2,
                    ACCUM_VALUES * 2,
                    RESULT_WARP_M_STRIDE * 2,
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
                        scaled[0] = acc[acc_m, tile_n] * alpha_vec[0]
                        for i in al.range(ACCUM_VALUES):
                            if IS_FP16:
                                converted[0, i] = al.bitcast(al.convert(scaled[0, i], al.f16), al.u16)
                            else:
                                converted[0, i] = al.bitcast(al.convert(scaled[0, i], al.bf16), al.u16)
                        result[
                            tile_m,
                            tile_n,
                            warp_m,
                            warp_n,
                            lane_col,
                            lane_row,
                        ] = packed[0]

    @avelang.jit
    def _store_result_vector(
        c_rsrc: al.Tensor((4,), al.u32),
        result_u128: al.Tensor((shm_u32 // 4, 4), al.u32),
        n: al.u32,
        result_partition: al.u32,
        idx: al.u32,
    ):
        major_vectors = RESULT_MINOR_COUNT * TILE * GROUP_N // RESULT_VEC_SIZE
        major = idx // major_vectors
        major_idx = idx % major_vectors
        minor_vectors = TILE * GROUP_N // RESULT_VEC_SIZE
        minor = major_idx // minor_vectors
        partition_idx = major_idx % minor_vectors
        output_warp_m = al.select(
            UNEVEN_RESULT_PARTITIONS, minor, major
        )
        stripe = al.select(
            UNEVEN_RESULT_PARTITIONS, major, minor
        )
        row = partition_idx // (GROUP_N // RESULT_VEC_SIZE)
        col = partition_idx % (GROUP_N // RESULT_VEC_SIZE)
        output_row = (
            output_warp_m * WARP_TILE_M
            + result_partition * RESULT_M_TILES * TILE
            + stripe * TILE
            + row
        )
        al.amdgpu.raw_buffer_store_x4(
            result_u128[idx],
            c_rsrc,
            (output_row * n // RESULT_VEC_SIZE + col) * UINT4_BYTES,
            0,
            0,
        )

    @avelang.jit
    def _write_results(
        c_rsrc: al.Tensor((4,), al.u32),
        shm: al.Tensor((shm_u32,), al.u32),
        n: al.u32,
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        tid: al.u32,
        alpha: al.f32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        result_u64 = al.view(
            shm,
            al.u32,
            al.make_layout(
                (shm_u32 // 2, 2),
                (2, 1),
            ),
        )
        result_u128 = al.view(
            shm,
            al.u32,
            al.make_layout(
                (shm_u32 // QUANT_VEC_SIZE, QUANT_VEC_SIZE),
                (QUANT_VEC_SIZE, 1),
            ),
        )
        for result_partition in al.range(RESULT_PARTITIONS - 1):
            _pack_result_partition(
                result_u64, result_partition,
                warp_m, warp_n, warp_k, wtid, alpha, acc,
            )
            al.syncthreads()
            for i in al.range(RESULT_FULL_LOADS):
                idx = tid + i * THREADS
                _store_result_vector(
                    c_rsrc, result_u128, n, result_partition, idx,
                )
            if RESULT_REMAINDER:
                if tid < RESULT_REMAINDER:
                    idx = tid + RESULT_FULL_LOADS * THREADS
                    _store_result_vector(
                        c_rsrc, result_u128, n, result_partition, idx,
                    )
            al.syncthreads()

        result_partition = RESULT_PARTITIONS - 1
        _pack_result_partition(
            result_u64, result_partition,
            warp_m, warp_n, warp_k, wtid, alpha, acc,
        )
        al.syncthreads()
        for i in al.range(LAST_RESULT_FULL_LOADS):
            idx = tid + i * THREADS
            _store_result_vector(
                c_rsrc, result_u128, n, result_partition, idx,
            )
        if LAST_RESULT_REMAINDER:
            if tid < LAST_RESULT_REMAINDER:
                idx = tid + LAST_RESULT_FULL_LOADS * THREADS
                _store_result_vector(
                    c_rsrc, result_u128, n, result_partition, idx,
                )

    return _write_results
