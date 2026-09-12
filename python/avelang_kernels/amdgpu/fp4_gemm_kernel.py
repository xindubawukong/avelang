"""AveLang-packed FP4 GEMM for gfx942."""

import struct
from functools import cache

import avelang
import avelang.language as al
import torch
from avelang_kernels.amdgpu.fp4_dequant import fp4_dequant
from avelang_kernels.amdgpu.fp4_gemm_utils import (
    SCALE_TILE_N,
    WEIGHT_TILE_K,
    packed_scale_shape,
    packed_weight_shape,
)
from avelang_kernels.amdgpu.fp4_gemm_solution import (
    LAYOUT_K,
    LAYOUT_N,
    SOLUTION_LIST,
    MatmulElementB,
    MatmulFeatures,
    MatmulMfmaType,
    MatmulWarpPartition,
    SolutionId,
    choose_default_solution,
)

WARP_SIZE = 64
VEC_BF16 = 8
BF16_BYTES = 2
GLOBAL_SCALE_FACTOR = 256.0
MAX_SHM_BYTES = 64 * 1024
SHM_DISCARD_VEC = (160 * 1024) // 16
PIPELINE_STAGES = 2
SCHED_MASK_MFMA = 0x8
SCHED_MASK_BUFFER_LOAD = 0x20
SCHED_MASK_DS_READ = 0x100
SCHED_MASK_DS_WRITE = 0x200

SUPPORTED_SOLUTIONS = frozenset(
    solution
    for solution in SOLUTION_LIST
    if solution.features == MatmulFeatures.GRID
    and solution.element_b == MatmulElementB.NVFP4
    and solution.mfma_type == MatmulMfmaType.BF16
    and solution.warp_partition == MatmulWarpPartition.NK
)


def _ceildiv(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _make_fp4_gemm_kernel(solution: SolutionId):
    shape = solution.shape
    GROUP_M = shape.group_m
    GROUP_N = shape.group_n
    GROUP_K = shape.group_k
    WARP_M = shape.warp_partition_m
    WARP_N = shape.warp_partition_n
    WARP_K = shape.warp_partition_k
    NUM_WARPS = shape.num_warps
    THREADS = NUM_WARPS * WARP_SIZE

    WARP_TILE_M = GROUP_M // WARP_M
    WARP_TILE_N = GROUP_N // WARP_N
    WARP_ATOM_K = GROUP_K // LAYOUT_K // WARP_K
    WARP_ATOM_N = GROUP_N // LAYOUT_N // WARP_N
    M_TILES = WARP_TILE_M // 16
    N_TILES = WARP_TILE_N // 16

    A_VECTOR_COUNT = GROUP_M * GROUP_K // VEC_BF16
    B_VECTOR_COUNT = GROUP_K * GROUP_N // 32
    SCALE_VECTOR_COUNT = GROUP_K * GROUP_N // 256
    A_LOADS = _ceildiv(A_VECTOR_COUNT, THREADS)
    B_LOADS = _ceildiv(B_VECTOR_COUNT, THREADS)
    SCALE_LOADS = _ceildiv(SCALE_VECTOR_COUNT, THREADS)
    PIPELINE_GLOBAL_LOADS = A_LOADS + B_LOADS + SCALE_LOADS
    PIPELINE_MFMAS = M_TILES * N_TILES * (GROUP_K // 16 // WARP_K)
    PIPELINE_MFMAS_PER_LOAD = (
        4
        if PIPELINE_MFMAS // PIPELINE_GLOBAL_LOADS > 12
        else 2
        if PIPELINE_MFMAS // PIPELINE_GLOBAL_LOADS > 6
        else 1
    )

    SHM_A_U32 = GROUP_M * GROUP_K // 2
    SHM_B_U32 = GROUP_K * GROUP_N // 8
    SHM_SCALE_U32 = GROUP_K * GROUP_N // 64
    SHM_DATA_U32 = SHM_A_U32 + SHM_B_U32 + SHM_SCALE_U32
    SHM_B_VEC_OFFSET = SHM_A_U32 // 4
    SHM_SCALE_VEC_OFFSET = (SHM_A_U32 + SHM_B_U32) // 4
    SHM_SCALE_U16_OFFSET = (SHM_A_U32 + SHM_B_U32) * 2
    SHM_STAGES = min(2, MAX_SHM_BYTES // (SHM_DATA_U32 * 4))
    SINGLE_BUFFER = SHM_STAGES == 1
    SECOND_SHM_STAGE = 0 if SINGLE_BUFFER else 1

    SHM_REDUCTION_U32 = (
        (WARP_K - 1) * WARP_M * WARP_N * M_TILES * N_TILES * WARP_SIZE * 4
    )
    max_result_tiles = min(
        M_TILES,
        MAX_SHM_BYTES // (GROUP_N * BF16_BYTES * 16 * WARP_M),
    )
    RESULT_M_TILES = M_TILES // _ceildiv(M_TILES, max_result_tiles)
    RESULT_PARTITIONS = _ceildiv(M_TILES, RESULT_M_TILES)
    UNEVEN_RESULT_PARTITIONS = M_TILES % RESULT_M_TILES != 0
    RESULT_TILE_M_STRIDE = (
        (WARP_M if UNEVEN_RESULT_PARTITIONS else 1) * 16 * GROUP_N // 4
    )
    RESULT_WARP_M_STRIDE = (
        (1 if UNEVEN_RESULT_PARTITIONS else RESULT_M_TILES)
        * 16
        * GROUP_N
        // 4
    )
    RESULT_MINOR_COUNT = (
        WARP_M if UNEVEN_RESULT_PARTITIONS else RESULT_M_TILES
    )
    SHM_RESULT_U32 = RESULT_M_TILES * WARP_M * 16 * GROUP_N // 2
    RESULT_VECTOR_COUNT = SHM_RESULT_U32 // 4
    RESULT_LOADS = _ceildiv(RESULT_VECTOR_COUNT, THREADS)
    SHM_U32 = max(
        SHM_STAGES * SHM_DATA_U32,
        SHM_REDUCTION_U32,
        SHM_RESULT_U32,
    )
    if SHM_U32 * 4 > MAX_SHM_BYTES:
        raise ValueError(
            f"solution {int(solution):#x} requires {SHM_U32 * 4} bytes of LDS"
        )

    @avelang.jit
    def _load_global(
        a_rsrc: al.Tensor((4,), al.u32),
        b_rsrc: al.Tensor((4,), al.u32),
        scale_rsrc: al.Tensor((4,), al.u32),
        n: al.u32,
        k: al.u32,
        group_m: al.u32,
        group_n: al.u32,
        k_idx: al.u32,
        tid: al.u32,
        reg_a: al.Tensor((A_LOADS, 4), al.u32),
        reg_b: al.Tensor((B_LOADS, 4), al.u32),
        reg_scale: al.Tensor((SCALE_LOADS, 4), al.u32),
    ):
        for i in al.range(A_LOADS):
            idx = tid + i * THREADS
            row = idx // (GROUP_K // VEC_BF16)
            col = (idx % (GROUP_K // VEC_BF16)) * VEC_BF16
            offset = (
                (group_m * GROUP_M * k + k_idx * GROUP_K + row * k + col)
                * BF16_BYTES
            )
            reg_a[i] = al.amdgpu.raw_buffer_load_x4(a_rsrc, offset, 0, 0)

        b_row_size = LAYOUT_K * GROUP_N // 32
        for i in al.range(B_LOADS):
            idx = tid + i * THREADS
            row = idx // b_row_size
            col = idx % b_row_size
            offset = (
                group_n * b_row_size
                + k_idx * (n * GROUP_K // 32)
                + row * (n * LAYOUT_K // 32)
                + col
            ) * 16
            reg_b[i] = al.amdgpu.raw_buffer_load_x4(b_rsrc, offset, 0, 0)

        scale_row_size = LAYOUT_K * GROUP_N // 256
        for i in al.range(SCALE_LOADS):
            idx = tid + i * THREADS
            row = idx // scale_row_size
            col = idx % scale_row_size
            offset = (
                group_n * scale_row_size
                + k_idx * (n * GROUP_K // 256)
                + row * (n * LAYOUT_K // 256)
                + col
            ) * 16
            reg_scale[i] = al.amdgpu.raw_buffer_load_x4(
                scale_rsrc, offset, 0, 0
            )

    @avelang.jit
    def _store_shm(
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
            al.make_layout((SHM_U32 // 4, 4), (4, 1)),
        )
        stage_vec_offset = stage * (SHM_DATA_U32 // 4)
        discard_vec = al.convert(SHM_DISCARD_VEC, al.u32)
        for i in al.range(A_LOADS):
            idx = tid + i * THREADS
            row = idx // (GROUP_K // VEC_BF16)
            col = idx % (GROUP_K // VEC_BF16)
            tile_m = row // 16
            tile_k = col // (LAYOUT_K // VEC_BF16)
            row_in_tile = row % 16
            col_in_tile = col % (LAYOUT_K // VEC_BF16)
            batch = col_in_tile % 2
            inner_col = col_in_tile // 2
            coord = (
                tile_m * (16 * GROUP_K // VEC_BF16)
                + tile_k * (16 * LAYOUT_K // VEC_BF16)
                + batch * WARP_SIZE
                + inner_col * 16
                + row_in_tile
            ) ^ (inner_col * 2 + batch)
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
        shm: al.Tensor((SHM_U32,), al.u32),
        stage: al.u32,
        warp_m: al.u32,
        tile_idx_k: al.u32,
        wtid: al.u32,
        data_a0: al.Tensor((M_TILES, 4), al.u32),
        data_a1: al.Tensor((M_TILES, 4), al.u32),
    ):
        shm_vec = al.view(
            shm,
            al.u32,
            al.make_layout((SHM_U32 // 4, 4), (4, 1)),
        )
        row = wtid % 16
        inner_col = wtid // 16
        stage_vec_offset = stage * (SHM_DATA_U32 // 4)
        for tile_m in al.range(M_TILES):
            global_tile_m = warp_m * M_TILES + tile_m
            coord0 = (
                global_tile_m * (16 * GROUP_K // VEC_BF16)
                + tile_idx_k * (16 * LAYOUT_K // VEC_BF16)
                + inner_col * 16
                + row
            ) ^ (inner_col * 2)
            coord1 = (
                global_tile_m * (16 * GROUP_K // VEC_BF16)
                + tile_idx_k * (16 * LAYOUT_K // VEC_BF16)
                + WARP_SIZE
                + inner_col * 16
                + row
            ) ^ (inner_col * 2 + 1)
            data_a0[tile_m] = shm_vec[stage_vec_offset + coord0]
            data_a1[tile_m] = shm_vec[stage_vec_offset + coord1]

    @avelang.jit
    def _mfma(
        data_a: al.Tensor((M_TILES, 4), al.u32),
        data_b: al.Tensor((1, 4), al.u32),
        tile_n: al.u32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        for tile_m in al.range(M_TILES):
            frag_a = al.view(data_a[tile_m], al.Tensor((2, 2, 1), al.u32))
            frag_b = al.view(data_b[0], al.Tensor((2, 2, 1), al.u32))
            acc[tile_m, tile_n] = al.amdgpu.mfma_16x16x16_bf16_f32(
                frag_b[0], frag_a[0], acc[tile_m, tile_n]
            )
            acc[tile_m, tile_n] = al.amdgpu.mfma_16x16x16_bf16_f32(
                frag_b[1], frag_a[1], acc[tile_m, tile_n]
            )

    @avelang.jit
    def _read_shm_b(
        shm: al.Tensor((SHM_U32,), al.u32),
        stage: al.u32,
        qword: al.Tensor((1, 4), al.u32),
        packed_scale: al.Tensor((1, 1), al.u16),
        warp_n: al.u32,
        tile_idx_k: al.u32,
        n_atom: al.u32,
        wtid: al.u32,
    ):
        shm_vec = al.view(
            shm,
            al.u32,
            al.make_layout((SHM_U32 // 4, 4), (4, 1)),
        )
        scale_u16 = al.view(
            shm,
            al.u16,
            al.make_layout((SHM_U32 * 2,), (1,)),
        )
        stage_vec_offset = stage * (SHM_DATA_U32 // 4)
        stage_u16_offset = stage * SHM_DATA_U32 * 2
        tile_idx_n = warp_n * WARP_ATOM_N + n_atom
        qword[0] = shm_vec[
            stage_vec_offset
            + SHM_B_VEC_OFFSET
            + tile_idx_k * (LAYOUT_K * GROUP_N // 32)
            + tile_idx_n * (LAYOUT_K * LAYOUT_N // 32)
            + wtid
        ]
        packed_scale[0, 0] = scale_u16[
            stage_u16_offset
            + SHM_SCALE_U16_OFFSET
            + tile_idx_k * GROUP_N * 2
            + tile_idx_n * WARP_SIZE
            + wtid
        ]

    @avelang.jit
    def _prefetch(
        shm: al.Tensor((SHM_U32,), al.u32),
        stage: al.u32,
        data_a0: al.Tensor((M_TILES, 4), al.u32),
        data_a1: al.Tensor((M_TILES, 4), al.u32),
        qword: al.Tensor((1, 4), al.u32),
        packed_scale: al.Tensor((1, 1), al.u16),
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
    ):
        tile_idx_k = warp_k * WARP_ATOM_K
        _read_shm_a(
            shm, stage, warp_m, tile_idx_k, wtid, data_a0, data_a1
        )
        _read_shm_b(
            shm, stage, qword, packed_scale, warp_n, tile_idx_k, 0, wtid
        )

    @avelang.jit
    def _matmul(
        data_a0: al.Tensor((M_TILES, 4), al.u32),
        data_a1: al.Tensor((M_TILES, 4), al.u32),
        qword: al.Tensor((1, 4), al.u32),
        packed_scale: al.Tensor((1, 1), al.u16),
        data_b: al.Tensor((1, 4), al.u32),
        n_atom: al.u32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        _ = N_TILES
        scale_f32 = al.make_local((1, 2), al.f32)
        for i in al.range(2):
            scale_byte = (
                al.convert(packed_scale[0, 0], al.u32) >> (i * 8)
            ) & 0xFF
            scale_bits = al.convert(scale_byte << 7, al.u16)
            scale_f32[0, i] = al.convert(
                al.bitcast(scale_bits, al.f16), al.f32
            )

        tile_n = n_atom * 2
        fp4_dequant(qword[0, 0], scale_f32[0, 0], data_b)
        _mfma(data_a0, data_b, tile_n, acc)
        fp4_dequant(qword[0, 1], scale_f32[0, 0], data_b)
        _mfma(data_a1, data_b, tile_n, acc)
        fp4_dequant(qword[0, 2], scale_f32[0, 1], data_b)
        _mfma(data_a0, data_b, tile_n + 1, acc)
        fp4_dequant(qword[0, 3], scale_f32[0, 1], data_b)
        _mfma(data_a1, data_b, tile_n + 1, acc)

    @avelang.jit
    def _pipeline_compute(
        shm: al.Tensor((SHM_U32,), al.u32),
        stage: al.u32,
        data_a0: al.Tensor((M_TILES, 4), al.u32),
        data_a1: al.Tensor((M_TILES, 4), al.u32),
        qword: al.Tensor((1, 4), al.u32),
        packed_scale: al.Tensor((1, 1), al.u16),
        data_b: al.Tensor((1, 4), al.u32),
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        for k_atom in al.range(WARP_ATOM_K):
            tile_idx_k = warp_k * WARP_ATOM_K + k_atom
            for n_atom in al.range(WARP_ATOM_N):
                _matmul(
                    data_a0, data_a1, qword, packed_scale,
                    data_b, n_atom, acc,
                )
                if n_atom + 1 < WARP_ATOM_N:
                    _read_shm_b(
                        shm, stage, qword, packed_scale,
                        warp_n, tile_idx_k, n_atom + 1, wtid,
                    )
            if k_atom + 1 < WARP_ATOM_K:
                tile_idx_k = tile_idx_k + 1
                _read_shm_a(
                    shm, stage, warp_m, tile_idx_k, wtid,
                    data_a0, data_a1,
                )
                _read_shm_b(
                    shm, stage, qword, packed_scale,
                    warp_n, tile_idx_k, 0, wtid,
                )

    @avelang.jit
    def _hot_loop_scheduler():
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, A_LOADS + 2, 0)
        for _ in al.range(PIPELINE_GLOBAL_LOADS):
            al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
            al.amdgpu.sched_group_barrier(
                SCHED_MASK_MFMA, PIPELINE_MFMAS_PER_LOAD, 0
            )
        for _ in al.range(PIPELINE_GLOBAL_LOADS):
            al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
            al.amdgpu.sched_group_barrier(
                SCHED_MASK_MFMA, PIPELINE_MFMAS_PER_LOAD * 2, 0
            )
        al.amdgpu.sched_barrier(0)

    @avelang.jit
    def _reduce_k(
        shm: al.Tensor((SHM_U32,), al.u32),
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        reduction = al.view(
            shm,
            al.f32,
            al.make_layout((SHM_U32,), (1,)),
        )
        if warp_k > 0:
            slot = (warp_k - 1) * WARP_M * WARP_N + warp_m * WARP_N + warp_n
            for tile_m in al.range(M_TILES):
                for tile_n in al.range(N_TILES):
                    for i in al.range(4):
                        idx = (
                            (((slot * M_TILES + tile_m) * N_TILES + tile_n)
                             * WARP_SIZE + wtid)
                            * 4 + i
                        )
                        reduction[idx] = acc[tile_m, tile_n, i]

        al.syncthreads()
        if warp_k == 0:
            for other_k in al.range(WARP_K - 1):
                slot = other_k * WARP_M * WARP_N + warp_m * WARP_N + warp_n
                for tile_m in al.range(M_TILES):
                    for tile_n in al.range(N_TILES):
                        for i in al.range(4):
                            idx = (
                                (((slot * M_TILES + tile_m) * N_TILES + tile_n)
                                 * WARP_SIZE + wtid)
                                * 4 + i
                            )
                            acc[tile_m, tile_n, i] = (
                                acc[tile_m, tile_n, i] + reduction[idx]
                            )
        al.syncthreads()

    @avelang.jit
    def _write_results(
        c_rsrc: al.Tensor((4,), al.u32),
        shm: al.Tensor((SHM_U32,), al.u32),
        n: al.u32,
        group_m: al.u32,
        group_n: al.u32,
        warp_m: al.u32,
        warp_n: al.u32,
        warp_k: al.u32,
        wtid: al.u32,
        tid: al.u32,
        alpha_bits: al.u32,
        acc: al.Tensor((M_TILES, N_TILES, 4), al.f32),
    ):
        _ = M_TILES
        result_u64 = al.view(
            shm,
            al.u32,
            al.make_layout((SHM_U32 // 2, 2), (2, 1)),
        )
        result_u128 = al.view(
            shm,
            al.u32,
            al.make_layout((SHM_U32 // 4, 4), (4, 1)),
        )
        alpha = al.bitcast(alpha_bits, al.f32)
        lane_row = wtid // 16
        lane_col = wtid % 16
        packed = al.make_local((1, 2), al.u32)

        for result_partition in al.range(RESULT_PARTITIONS):
            if warp_k == 0:
                for tile_m in al.range(RESULT_M_TILES):
                    acc_m = result_partition * RESULT_M_TILES + tile_m
                    if acc_m < M_TILES:
                        for tile_n in al.range(N_TILES):
                            value0 = al.bitcast(
                                acc[acc_m, tile_n, 0] * alpha, al.u32
                            )
                            value1 = al.bitcast(
                                acc[acc_m, tile_n, 1] * alpha, al.u32
                            )
                            value2 = al.bitcast(
                                acc[acc_m, tile_n, 2] * alpha, al.u32
                            )
                            value3 = al.bitcast(
                                acc[acc_m, tile_n, 3] * alpha, al.u32
                            )
                            packed[0, 0] = al.amdgpu.perm(
                                value1, value0, 0x07060302
                            )
                            packed[0, 1] = al.amdgpu.perm(
                                value3, value2, 0x07060302
                            )
                            coord = (
                                tile_m * RESULT_TILE_M_STRIDE
                                + tile_n * 4
                                + warp_m * RESULT_WARP_M_STRIDE
                                + warp_n * (WARP_TILE_N // 4)
                                + lane_col * (GROUP_N // 4)
                                + lane_row
                            )
                            result_u64[coord] = packed[0]

            al.syncthreads()
            tile_offset = (
                group_m * GROUP_M * n + group_n * GROUP_N
            ) * BF16_BYTES
            remaining_tiles = M_TILES - result_partition * RESULT_M_TILES
            pending_tiles = al.select(
                remaining_tiles < RESULT_M_TILES,
                remaining_tiles,
                RESULT_M_TILES,
            )
            pending_vectors = pending_tiles * WARP_M * 16 * GROUP_N // 8
            for i in al.range(RESULT_LOADS):
                idx = tid + i * THREADS
                if idx < pending_vectors:
                    major_vectors = RESULT_MINOR_COUNT * 16 * GROUP_N // 8
                    major = idx // major_vectors
                    major_idx = idx % major_vectors
                    minor_vectors = 16 * GROUP_N // 8
                    minor = major_idx // minor_vectors
                    partition_idx = major_idx % minor_vectors
                    output_warp_m = al.select(
                        UNEVEN_RESULT_PARTITIONS, minor, major
                    )
                    stripe = al.select(
                        UNEVEN_RESULT_PARTITIONS, major, minor
                    )
                    row = partition_idx // (GROUP_N // 8)
                    col = partition_idx % (GROUP_N // 8)
                    output_row = (
                        output_warp_m * WARP_TILE_M
                        + result_partition * RESULT_M_TILES * 16
                        + stripe * 16
                        + row
                    )
                    al.amdgpu.raw_buffer_store_x4(
                        result_u128[idx],
                        c_rsrc,
                        (output_row * n // 8 + col) * 16,
                        tile_offset,
                        0,
                    )
            al.amdgpu.s_waitcnt(0, -1, -1)
            al.syncthreads()

    @avelang.jit
    def _kernel(
        A: al.Pointer(al.bf16),
        B: al.Pointer(al.u32),
        scales: al.Pointer(al.u8),
        alpha_bits: al.u32,
        C: al.Pointer(al.bf16),
        m: al.u32,
        n: al.u32,
        k: al.u32,
    ):
        tid = al.thread_id(0)
        wid = tid // WARP_SIZE
        wtid = tid % WARP_SIZE
        warp_k = wid // WARP_N // WARP_M
        warp_n = wid % WARP_N
        warp_m = (wid // WARP_N) % WARP_M

        group_m = al.block_id(0)
        group_n = al.block_id(1)

        a_tensor = al.make_tensor(A, al.bf16, al.make_layout((m * k,), (1,)))
        b_tensor = al.make_tensor(B, al.u32, al.make_layout((n * k // 8,), (1,)))
        s_tensor = al.make_tensor(
            scales, al.u8, al.make_layout((n * k // 16,), (1,))
        )
        c_tensor = al.make_tensor(C, al.bf16, al.make_layout((m * n,), (1,)))
        a_rsrc = al.amdgpu.make_rsrc(a_tensor, m * k * BF16_BYTES)
        b_rsrc = al.amdgpu.make_rsrc(b_tensor, n * k // 2)
        scale_rsrc = al.amdgpu.make_rsrc(s_tensor, n * k // 16)
        c_rsrc = al.amdgpu.make_rsrc(c_tensor, m * n * BF16_BYTES)

        shm = al.make_shared((SHM_U32,), al.u32)
        reg_a = al.make_local((PIPELINE_STAGES, A_LOADS, 4), al.u32)
        reg_b = al.make_local((PIPELINE_STAGES, B_LOADS, 4), al.u32)
        reg_scale = al.make_local((PIPELINE_STAGES, SCALE_LOADS, 4), al.u32)
        data_a0 = al.make_local((M_TILES, 4), al.u32)
        data_a1 = al.make_local((M_TILES, 4), al.u32)
        qword = al.make_local((1, 4), al.u32)
        packed_scale = al.make_local((1, 1), al.u16)
        data_b = al.make_local((1, 4), al.u32)
        acc = al.make_local((M_TILES, N_TILES, 4), al.f32)

        for tile_m in al.range(M_TILES):
            for tile_n in al.range(N_TILES):
                for acc_idx in al.range(4):
                    acc[tile_m, tile_n, acc_idx] = al.convert(0.0, al.f32)

        k_total = k // GROUP_K
        _load_global(
            a_rsrc, b_rsrc, scale_rsrc,
            n, k, group_m, group_n, 0, tid,
            reg_a[0], reg_b[0], reg_scale[0],
        )
        al.amdgpu.sched_barrier(0x7DF)
        _store_shm(shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid)

        if k_total > 1:
            al.syncthreads()
            _load_global(
                a_rsrc, b_rsrc, scale_rsrc,
                n, k, group_m, group_n, 1, tid,
                reg_a[1], reg_b[1], reg_scale[1],
            )

        k_idx = al.convert(0, al.u32)
        while k_idx + 3 < k_total:
            al.amdgpu.sched_barrier(0)
            al.syncthreads()

            _prefetch(
                shm, 0, data_a0, data_a1,
                qword, packed_scale, warp_m, warp_n, warp_k, wtid,
            )
            _load_global(
                a_rsrc, b_rsrc, scale_rsrc,
                n, k, group_m, group_n, k_idx + 2, tid,
                reg_a[0], reg_b[0], reg_scale[0],
            )
            if SINGLE_BUFFER:
                _pipeline_compute(
                    shm, 0, data_a0, data_a1,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )
                al.syncthreads()
                _store_shm(
                    shm, 0, reg_a[1], reg_b[1], reg_scale[1], tid
                )
            else:
                _store_shm(
                    shm, 1, reg_a[1], reg_b[1], reg_scale[1], tid
                )
                _pipeline_compute(
                    shm, 0, data_a0, data_a1,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )
            _hot_loop_scheduler()

            al.syncthreads()
            _prefetch(
                shm, SECOND_SHM_STAGE, data_a0, data_a1,
                qword, packed_scale, warp_m, warp_n, warp_k, wtid,
            )
            _load_global(
                a_rsrc, b_rsrc, scale_rsrc,
                n, k, group_m, group_n, k_idx + 3, tid,
                reg_a[1], reg_b[1], reg_scale[1],
            )
            if SINGLE_BUFFER:
                _pipeline_compute(
                    shm, 0, data_a0, data_a1,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )
                al.syncthreads()
                _store_shm(
                    shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid
                )
            else:
                _store_shm(
                    shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid
                )
                _pipeline_compute(
                    shm, 1, data_a0, data_a1,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )
            _hot_loop_scheduler()
            k_idx = k_idx + 2

        # Epilogue: statically unroll up to three remaining K tiles.
        al.syncthreads()
        _prefetch(
            shm, 0, data_a0, data_a1,
            qword, packed_scale, warp_m, warp_n, warp_k, wtid,
        )
        if k_idx + 2 < k_total:
            _load_global(
                a_rsrc, b_rsrc, scale_rsrc,
                n, k, group_m, group_n, k_idx + 2, tid,
                reg_a[0], reg_b[0], reg_scale[0],
            )
        if SINGLE_BUFFER:
            _pipeline_compute(
                shm, 0, data_a0, data_a1,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )
            if k_idx + 1 < k_total:
                al.syncthreads()
                _store_shm(
                    shm, 0, reg_a[1], reg_b[1], reg_scale[1], tid
                )
        else:
            if k_idx + 1 < k_total:
                _store_shm(
                    shm, 1, reg_a[1], reg_b[1], reg_scale[1], tid
                )
            _pipeline_compute(
                shm, 0, data_a0, data_a1,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )

        if k_idx + 1 < k_total:
            al.syncthreads()
            _prefetch(
                shm, SECOND_SHM_STAGE, data_a0, data_a1,
                qword, packed_scale, warp_m, warp_n, warp_k, wtid,
            )
            if SINGLE_BUFFER:
                _pipeline_compute(
                    shm, 0, data_a0, data_a1,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )
                if k_idx + 2 < k_total:
                    al.syncthreads()
                    _store_shm(
                        shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid
                    )
            else:
                if k_idx + 2 < k_total:
                    _store_shm(
                        shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid
                    )
                _pipeline_compute(
                    shm, 1, data_a0, data_a1,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )

        if k_idx + 2 < k_total:
            al.syncthreads()
            _prefetch(
                shm, 0, data_a0, data_a1,
                qword, packed_scale, warp_m, warp_n, warp_k, wtid,
            )
            _pipeline_compute(
                shm, 0, data_a0, data_a1,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )

        if WARP_K > 1:
            _reduce_k(shm, warp_m, warp_n, warp_k, wtid, acc)
        _write_results(
            c_rsrc, shm, n, group_m, group_n,
            warp_m, warp_n, warp_k, wtid, tid,
            alpha_bits, acc,
        )

    _kernel.__name__ = f"fp4_gemm_{int(solution):012x}"
    return _kernel


def resolve_solution(
    m: int,
    n: int,
    k: int,
    solution_id: SolutionId | int | None = None,
) -> SolutionId:
    if solution_id is None or solution_id == -1:
        solution = choose_default_solution(m, n, k)
    elif isinstance(solution_id, SolutionId):
        solution = solution_id
    else:
        solution = SolutionId.from_int(solution_id)
    if solution not in SUPPORTED_SOLUTIONS:
        raise ValueError(f"unsupported AveLang FP4 BF16 solution {int(solution):#x}")
    if n % solution.group_n or k % solution.group_k:
        raise ValueError(
            f"solution {int(solution):#x} is incompatible with N={n}, K={k}"
        )
    return solution


@cache
def get_fp4_gemm_kernel(solution_id: int):
    solution = SolutionId.from_int(solution_id)
    if solution not in SUPPORTED_SOLUTIONS:
        raise ValueError(f"unsupported AveLang FP4 BF16 solution {solution_id:#x}")
    return _make_fp4_gemm_kernel(solution)


REFERENCE_SOLUTION_ID = 0x122111020408
_fp4_gemm_kernel = get_fp4_gemm_kernel(REFERENCE_SOLUTION_ID)


def fp4_gemm(
    A: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_scales: torch.Tensor,
    global_scale: torch.Tensor | float,
    out: torch.Tensor | None = None,
    solution_id: SolutionId | int | None = None,
) -> torch.Tensor:
    """Multiply BF16 ``A`` by an AveLang-packed FP4 matrix.

    ``packed_weight`` and ``packed_scales`` must come from
    :func:`avelang_kernels.amdgpu.fp4_gemm_utils.repack_fp4` and
    :func:`avelang_kernels.amdgpu.fp4_gemm_utils.process_fp4_scales`,
    respectively.
    """
    if (
        A.ndim != 2
        or A.dtype != torch.bfloat16
        or not A.is_contiguous()
        or A.device.type != "cuda"
    ):
        raise ValueError("A must be a contiguous 2-D CUDA torch.bfloat16 tensor.")
    m, k = A.shape
    if k % WEIGHT_TILE_K:
        raise ValueError(f"A's K dimension must be divisible by {WEIGHT_TILE_K}.")
    if (
        packed_weight.ndim != 5
        or packed_weight.dtype != torch.int32
        or not packed_weight.is_contiguous()
        or packed_weight.shape[1] != 4
        or packed_weight.shape[3:] != (64, 4)
    ):
        raise ValueError("packed_weight is not an AveLang packed FP4 weight tensor.")
    n = packed_weight.shape[2] * 32
    if n % SCALE_TILE_N or packed_weight.shape != packed_weight_shape(n, k):
        raise ValueError(
            f"packed_weight must have AveLang packed shape {packed_weight_shape(n, k)}."
        )
    if (
        packed_scales.dtype != torch.uint8
        or packed_scales.shape != packed_scale_shape(n, k)
        or not packed_scales.is_contiguous()
    ):
        raise ValueError(
            f"packed_scales must have AveLang packed uint8 shape "
            f"{packed_scale_shape(n, k)}."
        )
    if not (A.device == packed_weight.device == packed_scales.device):
        raise ValueError("All inputs must be on the same device.")

    if isinstance(global_scale, torch.Tensor):
        if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
            raise ValueError("global_scale must contain one float32 value.")
        scale_value = float(global_scale.item())
    else:
        scale_value = float(global_scale)

    solution = resolve_solution(m, n, k, solution_id)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
    elif (
        out.shape != (m, n)
        or out.dtype != torch.bfloat16
        or out.device != A.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"out must be contiguous BF16 with shape {(m, n)} on {A.device}."
        )

    grid_m = _ceildiv(m, solution.group_m)
    grid_n = n // solution.group_n
    threads = solution.shape.num_warps * WARP_SIZE
    kernel = get_fp4_gemm_kernel(int(solution))
    kernel[lambda: ((grid_m, grid_n, 1), (threads, 1, 1))](
        A,
        packed_weight,
        packed_scales,
        struct.unpack("I", struct.pack("f", scale_value * GLOBAL_SCALE_FACTOR))[0],
        out,
        m,
        n,
        k,
        num_warps=solution.shape.num_warps,
    )
    return out
