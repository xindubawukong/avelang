"""Global and shared-memory operations for AMDGPU FP4 GEMM."""

from dataclasses import dataclass

import avelang
import avelang.language as al

from .config import (
    FP4GemmConfig,
    U16_BYTES,
    U32_BYTES,
    UINT4_BYTES,
)

SHM_DISCARD_VEC = (160 * 1024) // UINT4_BYTES


def _ceildiv(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


@dataclass(frozen=True, slots=True)
class FP4MemoryConfig:
    gemm: FP4GemmConfig
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
    shm_stages: int

    @classmethod
    def from_config(cls, gemm: FP4GemmConfig) -> "FP4MemoryConfig":
        group_m = gemm.group_m
        group_n = gemm.group_n
        group_k = gemm.group_k
        threads = gemm.threads

        a_vector_count = group_m * group_k // gemm.vec_size
        b_vector_count = group_k * group_n // gemm.packed_values_per_vector
        scale_vector_count = (
            group_k * group_n // (gemm.group_size * gemm.scale_vec_size)
        )
        shm_a_u32 = group_m * group_k * gemm.element_a_bytes // U32_BYTES
        shm_b_u32 = group_k * group_n // gemm.pack_factor
        shm_scale_u32 = group_k * group_n // gemm.group_size // U32_BYTES
        shm_data_u32 = shm_a_u32 + shm_b_u32 + shm_scale_u32
        return cls(
            gemm=gemm,
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
            shm_stages=min(
                gemm.pipeline_stages,
                gemm.max_shm_bytes // (shm_data_u32 * U32_BYTES),
            ),
        )

    @property
    def pipeline_shm_u32(self) -> int:
        return self.shm_stages * self.shm_data_u32

    @property
    def single_buffer(self) -> bool:
        return self.shm_stages == 1

    @property
    def second_shm_stage(self) -> int:
        return 0 if self.single_buffer else 1


def make_fp4_memory_ops(memory: FP4MemoryConfig, shm_u32: int):
    gemm = memory.gemm
    GROUP_N = gemm.group_n
    GROUP_K = gemm.group_k
    THREADS = gemm.threads
    WARP_SIZE = gemm.warp_size
    TILE = gemm.tile
    LAYOUT_K = gemm.layout_k
    LAYOUT_N = gemm.layout_n
    LAYOUT_ELEMENTS_K = gemm.layout_elements_k
    VEC_SIZE = gemm.vec_size
    ELEMENT_A_BYTES = gemm.element_a_bytes
    GROUP_SIZE = gemm.group_size
    PACK_FACTOR = gemm.pack_factor
    SCALE_VEC_SIZE = gemm.scale_vec_size
    PACKED_VALUES_PER_VECTOR = gemm.packed_values_per_vector
    QUANT_VEC_SIZE = gemm.quant_vec_size
    A_VECTOR_COUNT = memory.a_vector_count
    B_VECTOR_COUNT = memory.b_vector_count
    SCALE_VECTOR_COUNT = memory.scale_vector_count
    A_LOADS = memory.a_loads
    B_LOADS = memory.b_loads
    SCALE_LOADS = memory.scale_loads
    SHM_DATA_U32 = memory.shm_data_u32
    SHM_B_VEC_OFFSET = memory.shm_b_vec_offset
    SHM_SCALE_VEC_OFFSET = memory.shm_scale_vec_offset
    SHM_SCALE_U16_OFFSET = memory.shm_scale_u16_offset
    WARP_ATOM_N = gemm.warp_atom_n
    M_TILES = gemm.m_tiles
    READ_BATCH_A = gemm.read_batch_a
    SCALE_LANES = LAYOUT_K * LAYOUT_N // GROUP_SIZE // U16_BYTES

    @avelang.jit
    def _advance_global_offsets(
        a_offset: al.u32,
        b_offset: al.u32,
        scale_offset: al.u32,
        n: al.u32,
    ) -> (al.u32, al.u32, al.u32):
        return (
            a_offset + GROUP_K * ELEMENT_A_BYTES,
            b_offset + n * GROUP_K // PACK_FACTOR * U32_BYTES,
            scale_offset + n * GROUP_K // GROUP_SIZE,
        )

    @avelang.jit
    def _load_global(
        a_rsrc: al.Tensor((4,), al.u32),
        b_rsrc: al.Tensor((4,), al.u32),
        scale_rsrc: al.Tensor((4,), al.u32),
        n: al.u32,
        k: al.u32,
        a_offset: al.u32,
        b_offset: al.u32,
        scale_offset: al.u32,
        tid: al.u32,
        reg_a: al.Tensor((A_LOADS, 4), al.u32),
        reg_b: al.Tensor((B_LOADS, 4), al.u32),
        reg_scale: al.Tensor((SCALE_LOADS, 4), al.u32),
    ):
        for i in al.range(A_LOADS):
            idx = tid + i * THREADS
            row = idx // (GROUP_K // VEC_SIZE)
            col = (idx % (GROUP_K // VEC_SIZE)) * VEC_SIZE
            offset = a_offset + (row * k + col) * ELEMENT_A_BYTES
            reg_a[i] = al.amdgpu.raw_buffer_load_x4(a_rsrc, offset, 0, 0)

        b_row_size = LAYOUT_K * GROUP_N // PACKED_VALUES_PER_VECTOR
        for i in al.range(B_LOADS):
            idx = tid + i * THREADS
            row = idx // b_row_size
            col = idx % b_row_size
            offset = (
                b_offset
                + (
                    row * (n * LAYOUT_K // PACKED_VALUES_PER_VECTOR)
                    + col
                )
                * UINT4_BYTES
            )
            reg_b[i] = al.amdgpu.raw_buffer_load_x4(b_rsrc, offset, 0, 0)

        scale_row_size = LAYOUT_K * GROUP_N // (GROUP_SIZE * SCALE_VEC_SIZE)
        for i in al.range(SCALE_LOADS):
            idx = tid + i * THREADS
            row = idx // scale_row_size
            col = idx % scale_row_size
            offset = (
                scale_offset
                + (
                    row
                    * (n * LAYOUT_K // (GROUP_SIZE * SCALE_VEC_SIZE))
                    + col
                )
                * UINT4_BYTES
            )
            reg_scale[i] = al.amdgpu.raw_buffer_load_x4(
                scale_rsrc, offset, 0, 0
            )

    @avelang.jit
    def _store_shm(
        shm: al.Tensor((shm_u32,), al.u32),
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
                (shm_u32 // QUANT_VEC_SIZE, QUANT_VEC_SIZE),
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
            store_idx = stage_vec_offset + SHM_B_VEC_OFFSET + idx
            if B_VECTOR_COUNT % THREADS != 0:
                store_idx = al.select(
                    idx < B_VECTOR_COUNT, store_idx, discard_vec
                )
            shm_vec[store_idx] = reg_b[i]

        for i in al.range(SCALE_LOADS):
            idx = tid + i * THREADS
            store_idx = stage_vec_offset + SHM_SCALE_VEC_OFFSET + idx
            if SCALE_VECTOR_COUNT % THREADS != 0:
                store_idx = al.select(
                    idx < SCALE_VECTOR_COUNT, store_idx, discard_vec
                )
            shm_vec[store_idx] = reg_scale[i]

    @avelang.jit
    def _read_shm_a(
        shm: al.Tensor((shm_u32,), al.u32),
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
                (shm_u32 // QUANT_VEC_SIZE, QUANT_VEC_SIZE),
                (QUANT_VEC_SIZE, 1),
            ),
        )
        row = wtid % TILE
        inner_col = wtid // TILE
        stage_vec_offset = stage * (SHM_DATA_U32 // QUANT_VEC_SIZE)
        for tile_m in al.range(M_TILES):
            global_tile_m = warp_m * M_TILES + tile_m
            for batch in al.range(READ_BATCH_A):
                coord = (
                    global_tile_m * (TILE * GROUP_K // VEC_SIZE)
                    + tile_idx_k * (TILE * LAYOUT_K // VEC_SIZE)
                    + batch * WARP_SIZE
                    + inner_col * TILE
                    + row
                ) ^ (inner_col * LAYOUT_ELEMENTS_K + batch)
                data_a[tile_m, batch] = shm_vec[stage_vec_offset + coord]

    @avelang.jit
    def _read_shm_b(
        shm: al.Tensor((shm_u32,), al.u32),
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
                + SHM_B_VEC_OFFSET * QUANT_VEC_SIZE,
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
            al.make_layout((shm_u32 * (U32_BYTES // U16_BYTES),), (1,)),
        )
        scale_storage = al.subview(
            shm_u16,
            (
                stage * SHM_DATA_U32 * (U32_BYTES // U16_BYTES)
                + SHM_SCALE_U16_OFFSET,
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
                    SCALE_LANES,
                ),
                (
                    GROUP_N * (LAYOUT_K // GROUP_SIZE // U16_BYTES),
                    SCALE_LANES,
                    1,
                ),
            ),
        )
        tile_idx_n = warp_n * WARP_ATOM_N + n_atom
        qword[0] = b_tiles[tile_idx_k, tile_idx_n, wtid]
        scale_lane = wtid
        if GROUP_SIZE == 32:
            scale_lane = (wtid // 32) * (LAYOUT_N // 2) + wtid % (LAYOUT_N // 2)
        packed_scale[0, 0] = scale_tiles[tile_idx_k, tile_idx_n, scale_lane]

    return (
        _advance_global_offsets,
        _load_global,
        _store_shm,
        _read_shm_a,
        _read_shm_b,
    )
