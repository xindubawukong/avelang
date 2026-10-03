"""Global and shared-memory operations for AMDGPU FP4 GEMM."""

from __future__ import annotations

from dataclasses import dataclass

import avelang
import avelang.language as al

from .config import (
    FP4GemmConfig,
    U16_BYTES,
    U32_BYTES,
    UINT4_BYTES,
)
from .utils import _ceildiv

SHM_DISCARD_VEC = (160 * 1024) // UINT4_BYTES


@dataclass(frozen=True, slots=True)
class FP4MemoryConfig:
    a_vector_count: int
    b_vector_count: int
    scale_vector_count: int
    a_loads: int
    b_loads: int
    scale_loads: int
    shm_data_u32: int
    shm_b_vec_offset: int
    shm_scale_vec_offset: int
    shm_scale_u16_offset: int
    pipeline_shm_u32: int
    single_buffer: bool
    second_shm_stage: int

    @classmethod
    def from_config(cls, gemm: FP4GemmConfig) -> "FP4MemoryConfig":
        group_m = gemm.group_m
        group_n = gemm.group_n
        group_k = gemm.group_k
        threads = gemm.threads

        a_vector_count = group_m * group_k // gemm.vec_size
        b_vector_count = group_k * group_n // gemm.packed_values_per_vector
        scale_vector_count = (
            group_k * group_n // (gemm.scale_group_size * gemm.scale_vec_size)
        )
        shm_a_u32 = group_m * group_k * gemm.element_a_bytes // U32_BYTES
        shm_b_u32 = group_k * group_n // gemm.pack_factor
        shm_scale_u32 = group_k * group_n // gemm.scale_group_size // U32_BYTES
        shm_data_u32 = shm_a_u32 + shm_b_u32 + shm_scale_u32
        shm_stages = min(gemm.pipeline_stages, gemm.max_shm_bytes // (shm_data_u32 * U32_BYTES))
        return cls(
            a_vector_count=a_vector_count,
            b_vector_count=b_vector_count,
            scale_vector_count=scale_vector_count,
            a_loads=_ceildiv(a_vector_count, threads),
            b_loads=_ceildiv(b_vector_count, threads),
            scale_loads=_ceildiv(scale_vector_count, threads),
            shm_data_u32=shm_data_u32,
            shm_b_vec_offset=shm_a_u32 // gemm.quant_vec_size,
            shm_scale_vec_offset=(shm_a_u32 + shm_b_u32)
            // gemm.quant_vec_size,
            shm_scale_u16_offset=(shm_a_u32 + shm_b_u32)
            * (U32_BYTES // U16_BYTES),
            pipeline_shm_u32=shm_stages * shm_data_u32,
            single_buffer=shm_stages == 1,
            second_shm_stage=0 if shm_stages == 1 else 1,
        )


@avelang.jit
def advance_global_ptr(
    config: al.constexpr,
    a_rsrc: al.Tensor((4,), al.u32),
    b_rsrc: al.Tensor((4,), al.u32),
    scale_rsrc: al.Tensor((4,), al.u32),
    n: al.u32,
):
    # Descriptor words 0/1 contain the 64-bit base pointer. Preserve
    # words 2/3 (range/config), exactly as Petit AdvanceGlobalPtr does.
    a_ptr = al.view(a_rsrc, al.u64, al.make_layout((2,), (1,)))
    b_ptr = al.view(b_rsrc, al.u64, al.make_layout((2,), (1,)))
    scale_ptr = al.view(scale_rsrc, al.u64, al.make_layout((2,), (1,)))
    a_ptr[0] = a_ptr[0] + al.convert(config.gemm.group_k * config.gemm.element_a_bytes, al.u64)
    b_ptr[0] = b_ptr[0] + al.convert(n * config.gemm.group_k // config.gemm.pack_factor * U32_BYTES, al.u64)
    scale_ptr[0] = scale_ptr[0] + al.convert(n * config.gemm.group_k // config.gemm.scale_group_size, al.u64)


@avelang.jit
def load_global(
    config: al.constexpr,
    a_rsrc: al.Tensor((4,), al.u32),
    b_rsrc: al.Tensor((4,), al.u32),
    scale_rsrc: al.Tensor((4,), al.u32),
    n: al.u32,
    k: al.u32,
    tid: al.u32,
    reg_a: al.Tensor((config.memory.a_loads, 4), al.u32),
    reg_b: al.Tensor((config.memory.b_loads, 4), al.u32),
    reg_scale: al.Tensor((config.memory.scale_loads, 4), al.u32),
):
    for i in al.range(config.memory.a_loads):
        idx = tid + i * config.gemm.threads
        row = idx // (config.gemm.group_k // config.gemm.vec_size)
        col = (idx % (config.gemm.group_k // config.gemm.vec_size)) * config.gemm.vec_size
        offset = (row * k + col) * config.gemm.element_a_bytes
        reg_a[i] = al.amdgpu.raw_buffer_load_x4(a_rsrc, offset, 0, 0)

    b_row_size = config.gemm.layout_k * config.gemm.group_n // config.gemm.packed_values_per_vector
    for i in al.range(config.memory.b_loads):
        idx = tid + i * config.gemm.threads
        row = idx // b_row_size
        col = idx % b_row_size
        offset = (row * (n * config.gemm.layout_k // config.gemm.packed_values_per_vector) + col) * UINT4_BYTES
        reg_b[i] = al.amdgpu.raw_buffer_load_x4(b_rsrc, offset, 0, 0)

    scale_row_size = config.gemm.layout_k * config.gemm.group_n // (config.gemm.scale_group_size * config.gemm.scale_vec_size)
    for i in al.range(config.memory.scale_loads):
        idx = tid + i * config.gemm.threads
        row = idx // scale_row_size
        col = idx % scale_row_size
        offset = (row * (n * config.gemm.layout_k // (config.gemm.scale_group_size * config.gemm.scale_vec_size)) + col) * UINT4_BYTES
        reg_scale[i] = al.amdgpu.raw_buffer_load_x4(
            scale_rsrc, offset, 0, 0
        )


@avelang.jit
def store_shm(
    config: al.constexpr,
    shm: al.Tensor((config.shm_u32,), al.u32),
    stage: al.u32,
    reg_a: al.Tensor((config.memory.a_loads, 4), al.u32),
    reg_b: al.Tensor((config.memory.b_loads, 4), al.u32),
    reg_scale: al.Tensor((config.memory.scale_loads, 4), al.u32),
    tid: al.u32,
):
    shm_vec = al.view(
        shm,
        al.u32,
        al.make_layout(
            (config.shm_u32 // config.gemm.quant_vec_size, config.gemm.quant_vec_size),
            (config.gemm.quant_vec_size, 1),
        ),
    )
    stage_vec_offset = stage * (config.memory.shm_data_u32 // config.gemm.quant_vec_size)
    discard_vec = al.convert(SHM_DISCARD_VEC, al.u32)
    for i in al.range(config.memory.a_loads):
        idx = tid + i * config.gemm.threads
        row = idx // (config.gemm.group_k // config.gemm.vec_size)
        col = idx % (config.gemm.group_k // config.gemm.vec_size)
        tile_m = row // config.gemm.tile
        tile_k = col // (config.gemm.layout_k // config.gemm.vec_size)
        row_in_tile = row % config.gemm.tile
        col_in_tile = col % (config.gemm.layout_k // config.gemm.vec_size)
        batch = col_in_tile % config.gemm.layout_elements_k
        inner_col = col_in_tile // config.gemm.layout_elements_k
        coord = (
            tile_m * (config.gemm.tile * config.gemm.group_k // config.gemm.vec_size)
            + tile_k * (config.gemm.tile * config.gemm.layout_k // config.gemm.vec_size)
            + batch * config.gemm.warp_size
            + inner_col * config.gemm.tile
            + row_in_tile
        ) ^ (inner_col * config.gemm.layout_elements_k + batch)
        store_idx = stage_vec_offset + coord
        if config.memory.a_vector_count % config.gemm.threads != 0:
            store_idx = al.select(
                idx < config.memory.a_vector_count, store_idx, discard_vec
            )
        shm_vec[store_idx] = reg_a[i]

    for i in al.range(config.memory.b_loads):
        idx = tid + i * config.gemm.threads
        store_idx = stage_vec_offset + config.memory.shm_b_vec_offset + idx
        if config.memory.b_vector_count % config.gemm.threads != 0:
            store_idx = al.select(
                idx < config.memory.b_vector_count, store_idx, discard_vec
            )
        shm_vec[store_idx] = reg_b[i]

    for i in al.range(config.memory.scale_loads):
        idx = tid + i * config.gemm.threads
        store_idx = stage_vec_offset + config.memory.shm_scale_vec_offset + idx
        if config.memory.scale_vector_count % config.gemm.threads != 0:
            store_idx = al.select(
                idx < config.memory.scale_vector_count, store_idx, discard_vec
            )
        shm_vec[store_idx] = reg_scale[i]


@avelang.jit
def read_shm_a(
    config: al.constexpr,
    shm: al.Tensor((config.shm_u32,), al.u32),
    stage: al.u32,
    warp_m: al.u32,
    tile_idx_k: al.u32,
    wtid: al.u32,
    data_a: al.Tensor((config.gemm.m_tiles, config.gemm.read_batch_a, 4), al.u32),
):
    shm_vec = al.view(
        shm,
        al.u32,
        al.make_layout(
            (config.shm_u32 // config.gemm.quant_vec_size, config.gemm.quant_vec_size),
            (config.gemm.quant_vec_size, 1),
        ),
    )
    row = wtid % config.gemm.tile
    inner_col = wtid // config.gemm.tile
    stage_vec_offset = stage * (config.memory.shm_data_u32 // config.gemm.quant_vec_size)
    for batch in al.range(config.gemm.read_batch_a):
        for tile_m in al.range(config.gemm.m_tiles):
            global_tile_m = warp_m * config.gemm.m_tiles + tile_m
            coord = (
                global_tile_m * (config.gemm.tile * config.gemm.group_k // config.gemm.vec_size)
                + tile_idx_k * (config.gemm.tile * config.gemm.layout_k // config.gemm.vec_size)
                + batch * config.gemm.warp_size
                + inner_col * config.gemm.tile
                + row
            ) ^ (inner_col * config.gemm.layout_elements_k + batch)
            data_a[tile_m, batch] = shm_vec[stage_vec_offset + coord]


@avelang.jit
def read_shm_b(
    config: al.constexpr,
    shm: al.Tensor((config.shm_u32,), al.u32),
    stage: al.u32,
    qword: al.Tensor((1, 4), al.u32),
    packed_scale: al.Tensor((1, 1), al.u16),
    warp_n: al.u32,
    tile_idx_k: al.u32,
    n_atom: al.u32,
    wtid: al.u32,
):
    b_storage = al.subview(
        shm,
        (
            stage * config.memory.shm_data_u32
            + config.memory.shm_b_vec_offset * config.gemm.quant_vec_size,
        ),
        (config.memory.b_vector_count * config.gemm.quant_vec_size,),
        (1,),
    )
    b_tiles = al.view(
        b_storage,
        al.u32,
        al.make_layout(
            (
                config.gemm.group_k // config.gemm.layout_k,
                config.gemm.group_n // config.gemm.layout_n,
                config.gemm.warp_size,
                config.gemm.quant_vec_size,
            ),
            (
                config.gemm.layout_k * config.gemm.group_n // config.gemm.packed_values_per_vector
                * config.gemm.quant_vec_size,
                config.gemm.layout_k * config.gemm.layout_n // config.gemm.packed_values_per_vector
                * config.gemm.quant_vec_size,
                config.gemm.quant_vec_size,
                1,
            ),
        ),
    )
    shm_u16 = al.view(
        shm,
        al.u16,
        al.make_layout((config.shm_u32 * (U32_BYTES // U16_BYTES),), (1,)),
    )
    scale_storage = al.subview(
        shm_u16,
        (
            stage * config.memory.shm_data_u32 * (U32_BYTES // U16_BYTES)
            + config.memory.shm_scale_u16_offset,
        ),
        (config.memory.scale_vector_count * (UINT4_BYTES // U16_BYTES),),
        (1,),
    )
    scale_tiles = al.view(
        scale_storage,
        al.u16,
        al.make_layout(
            (
                config.gemm.group_k // config.gemm.layout_k,
                config.gemm.group_n // config.gemm.layout_n,
                config.gemm.layout_k * config.gemm.layout_n // config.gemm.scale_group_size // U16_BYTES,
            ),
            (
                config.gemm.group_n * (config.gemm.layout_k // config.gemm.scale_group_size // U16_BYTES),
                config.gemm.layout_k * config.gemm.layout_n // config.gemm.scale_group_size // U16_BYTES,
                1,
            ),
        ),
    )
    tile_idx_n = warp_n * config.gemm.warp_atom_n + n_atom
    qword[0] = b_tiles[tile_idx_k, tile_idx_n, wtid]
    scale_lane = wtid
    if config.gemm.scale_group_size == 32:
        scale_lane = (wtid // 32) * (config.gemm.layout_n // 2) + wtid % (config.gemm.layout_n // 2)
    packed_scale[0, 0] = scale_tiles[tile_idx_k, tile_idx_n, scale_lane]
