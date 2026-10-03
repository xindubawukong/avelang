"""Global and shared-memory operations for AMDGPU FP4 GEMM."""

from __future__ import annotations

from dataclasses import dataclass

import avelang
import avelang.language as al

from .config import (
    LAYOUT_ELEMENTS_K,
    LAYOUT_K,
    LAYOUT_N,
    MAX_SHM_BYTES,
    PACKED_VALUES_PER_VECTOR,
    PACK_FACTOR,
    PIPELINE_STAGES,
    QUANT_VEC_SIZE,
    SCALE_VEC_SIZE,
    TILE,
    U16_BYTES,
    U32_BYTES,
    UINT4_BYTES,
    WARP_SIZE,
)
from .utils import _ceildiv

SHM_DISCARD_VEC = (160 * 1024) // UINT4_BYTES


@dataclass(frozen=True, slots=True)
class MemoryLayout:
    a_loads: int
    b_loads: int
    scale_loads: int
    shm_u32: int
    shm_stages: int


def make_memory_ops(config):
    a_vector_count = config.group_m * config.group_k // config.vec_size
    b_vector_count = config.group_k * config.group_n // PACKED_VALUES_PER_VECTOR
    scale_vector_count = config.group_k * config.group_n // (config.scale_group_size * SCALE_VEC_SIZE)
    shm_data_u32 = (
        config.group_m * config.group_k * config.element_a_bytes // U32_BYTES
        + config.group_k * config.group_n // PACK_FACTOR
        + config.group_k * config.group_n // config.scale_group_size // U32_BYTES
    )
    shm_stages = min(PIPELINE_STAGES, MAX_SHM_BYTES // (shm_data_u32 * U32_BYTES))
    memory = MemoryLayout(
        a_loads=_ceildiv(a_vector_count, config.threads),
        b_loads=_ceildiv(b_vector_count, config.threads),
        scale_loads=_ceildiv(scale_vector_count, config.threads),
        shm_u32=shm_stages * shm_data_u32,
        shm_stages=shm_stages,
    )
    GROUP_K = al.constexpr(config.group_k)
    GROUP_N = al.constexpr(config.group_n)
    ELEMENT_A_BYTES = al.constexpr(config.element_a_bytes)
    VEC_SIZE = al.constexpr(config.vec_size)
    SCALE_GROUP_SIZE = al.constexpr(config.scale_group_size)
    THREADS = al.constexpr(config.threads)
    M_TILES = al.constexpr(config.m_tiles)
    READ_BATCH_A = al.constexpr(config.read_batch_a)
    WARP_ATOM_N = al.constexpr(config.warp_atom_n)
    A_LOADS = al.constexpr(memory.a_loads)
    B_LOADS = al.constexpr(memory.b_loads)
    SCALE_LOADS = al.constexpr(memory.scale_loads)
    A_VECTOR_COUNT = al.constexpr(a_vector_count)
    B_VECTOR_COUNT = al.constexpr(b_vector_count)
    SCALE_VECTOR_COUNT = al.constexpr(scale_vector_count)
    SHM_DATA_U32 = al.constexpr(shm_data_u32)
    SHM_U32 = al.constexpr(memory.shm_u32)

    @avelang.jit
    def advance_global_ptr(
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
        a_ptr[0] = a_ptr[0] + al.convert(GROUP_K * ELEMENT_A_BYTES, al.u64)
        b_ptr[0] = b_ptr[0] + al.convert(n * GROUP_K // PACK_FACTOR * U32_BYTES, al.u64)
        scale_ptr[0] = scale_ptr[0] + al.convert(n * GROUP_K // SCALE_GROUP_SIZE, al.u64)

    @avelang.jit
    def load_global(
        a_rsrc: al.Tensor((4,), al.u32),
        b_rsrc: al.Tensor((4,), al.u32),
        scale_rsrc: al.Tensor((4,), al.u32),
        n: al.u32,
        k: al.u32,
        tid: al.u32,
        reg_a: al.Tensor((A_LOADS, 4), al.u32),
        reg_b: al.Tensor((B_LOADS, 4), al.u32),
        reg_scale: al.Tensor((SCALE_LOADS, 4), al.u32),
    ):
        for i in al.range(A_LOADS):
            idx = tid + i * THREADS
            row = idx // (GROUP_K // VEC_SIZE)
            col = (idx % (GROUP_K // VEC_SIZE)) * VEC_SIZE
            offset = (row * k + col) * ELEMENT_A_BYTES
            reg_a[i] = al.amdgpu.raw_buffer_load_x4(a_rsrc, offset, 0, 0)

        b_row_size = LAYOUT_K * GROUP_N // PACKED_VALUES_PER_VECTOR
        for i in al.range(B_LOADS):
            idx = tid + i * THREADS
            row = idx // b_row_size
            col = idx % b_row_size
            offset = (row * (n * LAYOUT_K // PACKED_VALUES_PER_VECTOR) + col) * UINT4_BYTES
            reg_b[i] = al.amdgpu.raw_buffer_load_x4(b_rsrc, offset, 0, 0)

        scale_row_size = LAYOUT_K * GROUP_N // (SCALE_GROUP_SIZE * SCALE_VEC_SIZE)
        for i in al.range(SCALE_LOADS):
            idx = tid + i * THREADS
            row = idx // scale_row_size
            col = idx % scale_row_size
            offset = (row * (n * LAYOUT_K // (SCALE_GROUP_SIZE * SCALE_VEC_SIZE)) + col) * UINT4_BYTES
            reg_scale[i] = al.amdgpu.raw_buffer_load_x4(
                scale_rsrc, offset, 0, 0
            )

    @avelang.jit
    def store_shm(
        shm: al.Tensor((SHM_U32,), al.u32),
        stage: al.u32,
        reg_a: al.Tensor((A_LOADS, 4), al.u32),
        reg_b: al.Tensor((B_LOADS, 4), al.u32),
        reg_scale: al.Tensor((SCALE_LOADS, 4), al.u32),
        tid: al.u32,
    ):
        shm_vec = al.view(
            shm,
            al.u32,
            al.make_layout(
                (SHM_U32 // QUANT_VEC_SIZE, QUANT_VEC_SIZE),
                (QUANT_VEC_SIZE, 1),
            ),
        )
        stage_vec_offset = stage * (SHM_DATA_U32 // QUANT_VEC_SIZE)
        discard_vec = al.convert(SHM_DISCARD_VEC, al.u32)
        for i in al.range(A_LOADS):
            idx = tid + i * THREADS
            row = idx // (GROUP_K // VEC_SIZE)
            col = idx % (GROUP_K // VEC_SIZE)
            tile_m = row // TILE
            tile_k = col // (LAYOUT_K // VEC_SIZE)
            row_in_tile = row % TILE
            col_in_tile = col % (LAYOUT_K // VEC_SIZE)
            batch = col_in_tile % LAYOUT_ELEMENTS_K
            inner_col = col_in_tile // LAYOUT_ELEMENTS_K
            coord = (
                tile_m * (TILE * GROUP_K // VEC_SIZE)
                + tile_k * (TILE * LAYOUT_K // VEC_SIZE)
                + batch * WARP_SIZE
                + inner_col * TILE
                + row_in_tile
            ) ^ (inner_col * LAYOUT_ELEMENTS_K + batch)
            store_idx = stage_vec_offset + coord
            if A_VECTOR_COUNT % THREADS != 0:
                store_idx = al.select(
                    idx < A_VECTOR_COUNT, store_idx, discard_vec
                )
            shm_vec[store_idx] = reg_a[i]

        for i in al.range(B_LOADS):
            idx = tid + i * THREADS
            store_idx = stage_vec_offset + A_VECTOR_COUNT + idx
            if B_VECTOR_COUNT % THREADS != 0:
                store_idx = al.select(
                    idx < B_VECTOR_COUNT, store_idx, discard_vec
                )
            shm_vec[store_idx] = reg_b[i]

        for i in al.range(SCALE_LOADS):
            idx = tid + i * THREADS
            store_idx = stage_vec_offset + A_VECTOR_COUNT + B_VECTOR_COUNT + idx
            if SCALE_VECTOR_COUNT % THREADS != 0:
                store_idx = al.select(
                    idx < SCALE_VECTOR_COUNT, store_idx, discard_vec
                )
            shm_vec[store_idx] = reg_scale[i]

    @avelang.jit
    def read_shm_a(
        shm: al.Tensor((SHM_U32,), al.u32),
        stage: al.u32,
        warp_m: al.u32,
        tile_idx_k: al.u32,
        wtid: al.u32,
        data_a: al.Tensor((M_TILES, READ_BATCH_A, 4), al.u32),
    ):
        shm_vec = al.view(
            shm,
            al.u32,
            al.make_layout(
                (SHM_U32 // QUANT_VEC_SIZE, QUANT_VEC_SIZE),
                (QUANT_VEC_SIZE, 1),
            ),
        )
        row = wtid % TILE
        inner_col = wtid // TILE
        stage_vec_offset = stage * (SHM_DATA_U32 // QUANT_VEC_SIZE)
        for batch in al.range(READ_BATCH_A):
            for tile_m in al.range(M_TILES):
                global_tile_m = warp_m * M_TILES + tile_m
                coord = (
                    global_tile_m * (TILE * GROUP_K // VEC_SIZE)
                    + tile_idx_k * (TILE * LAYOUT_K // VEC_SIZE)
                    + batch * WARP_SIZE
                    + inner_col * TILE
                    + row
                ) ^ (inner_col * LAYOUT_ELEMENTS_K + batch)
                data_a[tile_m, batch] = shm_vec[stage_vec_offset + coord]

    @avelang.jit
    def read_shm_b(
        shm: al.Tensor((SHM_U32,), al.u32),
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
                stage * SHM_DATA_U32
                + A_VECTOR_COUNT * QUANT_VEC_SIZE,
            ),
            (B_VECTOR_COUNT * QUANT_VEC_SIZE,),
            (1,),
        )
        b_tiles = al.view(
            b_storage,
            al.u32,
            al.make_layout(
                (
                    GROUP_K // LAYOUT_K,
                    GROUP_N // LAYOUT_N,
                    WARP_SIZE,
                    QUANT_VEC_SIZE,
                ),
                (
                    LAYOUT_K * GROUP_N // PACKED_VALUES_PER_VECTOR
                    * QUANT_VEC_SIZE,
                    LAYOUT_K * LAYOUT_N // PACKED_VALUES_PER_VECTOR
                    * QUANT_VEC_SIZE,
                    QUANT_VEC_SIZE,
                    1,
                ),
            ),
        )
        shm_u16 = al.view(
            shm,
            al.u16,
            al.make_layout((SHM_U32 * (U32_BYTES // U16_BYTES),), (1,)),
        )
        scale_storage = al.subview(
            shm_u16,
            (
                stage * SHM_DATA_U32 * (U32_BYTES // U16_BYTES)
                + (A_VECTOR_COUNT + B_VECTOR_COUNT) * (UINT4_BYTES // U16_BYTES),
            ),
            (SCALE_VECTOR_COUNT * (UINT4_BYTES // U16_BYTES),),
            (1,),
        )
        scale_tiles = al.view(
            scale_storage,
            al.u16,
            al.make_layout(
                (
                    GROUP_K // LAYOUT_K,
                    GROUP_N // LAYOUT_N,
                    LAYOUT_K * LAYOUT_N // SCALE_GROUP_SIZE // U16_BYTES,
                ),
                (
                    GROUP_N * (LAYOUT_K // SCALE_GROUP_SIZE // U16_BYTES),
                    LAYOUT_K * LAYOUT_N // SCALE_GROUP_SIZE // U16_BYTES,
                    1,
                ),
            ),
        )
        tile_idx_n = warp_n * WARP_ATOM_N + n_atom
        qword[0] = b_tiles[tile_idx_k, tile_idx_n, wtid]
        scale_lane = wtid
        if SCALE_GROUP_SIZE == 32:
            scale_lane = (wtid // 32) * (LAYOUT_N // 2) + wtid % (LAYOUT_N // 2)
        packed_scale[0, 0] = scale_tiles[tile_idx_k, tile_idx_n, scale_lane]

    return memory, advance_global_ptr, load_global, store_shm, read_shm_a, read_shm_b
