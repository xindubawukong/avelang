"""FP4 GEMM result writeback for AMDGPU."""

from __future__ import annotations

from dataclasses import dataclass

import avelang
import avelang.language as al

from .config import (
    FP4GemmConfig,
    U32_BYTES,
    UINT4_BYTES,
)
from .utils import _ceildiv


@dataclass(frozen=True, slots=True)
class FP4WritebackConfig:
    result_m_tiles: int
    result_partitions: int
    uneven_result_partitions: bool
    result_tile_m_stride: int
    result_warp_m_stride: int
    shm_result_u32: int
    result_loads: int

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
        return cls(
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
            shm_result_u32=shm_result_u32,
            result_loads=_ceildiv(result_vector_count, threads),
        )


@avelang.jit
def pack_result_partition(
    config: al.constexpr,
    result_u64: al.Tensor((config.shm_u32 // 2, 2), al.u32),
    result_partition: al.u32,
    warp_m: al.u32,
    warp_n: al.u32,
    warp_k: al.u32,
    wtid: al.u32,
    alpha: al.f32,
    acc: al.Tensor((config.gemm.m_tiles, config.gemm.n_tiles, 4), al.f32),
):
    lane_row = wtid // config.gemm.tile
    lane_col = wtid % config.gemm.tile
    alpha_pair = al.full((1, 2), alpha, al.f32)
    acc_pairs = al.view(
        acc, al.f32,
        al.make_layout((config.gemm.m_tiles, config.gemm.n_tiles, 2, 2), (config.gemm.n_tiles * 4, 4, 2, 1)),
    )
    packed = al.make_local((1, 2), al.u32)
    result = al.view(
        result_u64,
        al.u32,
        al.make_layout(
            (
                config.writeback.result_m_tiles,
                config.gemm.n_tiles,
                config.gemm.warp_m,
                config.gemm.warp_n,
                config.gemm.tile,
                config.gemm.warp_size // config.gemm.tile,
                2,
            ),
            (
                config.writeback.result_tile_m_stride * 2,
                config.gemm.accum_values * 2,
                config.writeback.result_warp_m_stride * 2,
                (config.gemm.warp_tile_n // config.gemm.shm_vec_size) * 2,
                (config.gemm.group_n // config.gemm.shm_vec_size) * 2,
                2,
                1,
            ),
        ),
    )

    if config.gemm.warp_k == 1 or warp_k == 0:
        for tile_m in al.range(config.writeback.result_m_tiles):
            acc_m = result_partition * config.writeback.result_m_tiles + tile_m
            if acc_m < config.gemm.m_tiles:
                for tile_n in al.range(config.gemm.n_tiles):
                    for i in al.range(2):
                        pair = acc_pairs[acc_m, tile_n, i] * alpha_pair[0]
                        if config.gemm.is_fp16:
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
    config: al.constexpr,
    c_rsrc: al.Tensor((4,), al.u32),
    result_u128: al.Tensor((config.shm_u32 // 4, 4), al.u32),
    n: al.u32,
    result_partition: al.u32,
    idx: al.u32,
):
    m_tile_start = result_partition * config.writeback.result_m_tiles
    partition_idx = al.convert(0, al.u32)
    row_base = al.convert(0, al.u32)
    if config.writeback.uneven_result_partitions:
        stripe_vectors = config.gemm.warp_m * config.gemm.tile * config.gemm.group_n // config.gemm.result_vec_size
        stripe = idx // stripe_vectors
        stripe_idx = idx % stripe_vectors
        warp_vectors = config.gemm.tile * config.gemm.group_n // config.gemm.result_vec_size
        output_warp_m = stripe_idx // warp_vectors
        partition_idx = stripe_idx % warp_vectors
        row_base = output_warp_m * config.gemm.warp_tile_m + (m_tile_start + stripe) * config.gemm.tile
    else:
        warp_vectors = config.writeback.result_m_tiles * config.gemm.tile * config.gemm.group_n // config.gemm.result_vec_size
        output_warp_m = idx // warp_vectors
        partition_idx = idx % warp_vectors
        row_base = output_warp_m * config.gemm.warp_tile_m + m_tile_start * config.gemm.tile
    row = partition_idx // (config.gemm.group_n // config.gemm.result_vec_size)
    col = partition_idx % (config.gemm.group_n // config.gemm.result_vec_size)
    output_row = row_base + row
    al.amdgpu.raw_buffer_store_x4(
        result_u128[idx],
        c_rsrc,
        (output_row * n // config.gemm.result_vec_size + col) * UINT4_BYTES,
        0,
        0,
    )


@avelang.jit
def write_results(
    config: al.constexpr,
    c_rsrc: al.Tensor((4,), al.u32),
    shm: al.Tensor((config.shm_u32,), al.u32),
    n: al.u32,
    warp_m: al.u32,
    warp_n: al.u32,
    warp_k: al.u32,
    wtid: al.u32,
    tid: al.u32,
    alpha: al.f32,
    acc: al.Tensor((config.gemm.m_tiles, config.gemm.n_tiles, 4), al.f32),
):
    result_u64 = al.view(
        shm,
        al.u32,
        al.make_layout(
            (config.shm_u32 // 2, 2),
            (2, 1),
        ),
    )
    result_u128 = al.view(
        shm,
        al.u32,
        al.make_layout(
            (config.shm_u32 // config.gemm.quant_vec_size, config.gemm.quant_vec_size),
            (config.gemm.quant_vec_size, 1),
        ),
    )
    for result_partition in al.range(config.writeback.result_partitions):
        pack_result_partition(
            config, result_u64, result_partition,
            warp_m, warp_n, warp_k, wtid, alpha, acc,
        )
        al.syncthreads()
        pending_items = (
            al.min(config.writeback.result_m_tiles, config.gemm.m_tiles - result_partition * config.writeback.result_m_tiles)
            * config.gemm.warp_m * config.gemm.tile * config.gemm.group_n // config.gemm.result_vec_size
        )
        for i in al.range(config.writeback.result_loads):
            idx = tid + i * config.gemm.threads
            if idx < pending_items:
                store_result_vector(
                    config, c_rsrc, result_u128, n, result_partition, idx,
                )
        if result_partition + 1 < config.writeback.result_partitions:
            al.syncthreads()
