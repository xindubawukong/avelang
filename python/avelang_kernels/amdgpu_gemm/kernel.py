from __future__ import annotations

from functools import lru_cache

import torch

import avelang
import avelang.language as al

from .config import (
    LOAD_MODE_BASE_OFFSET,
    STAGGER_BY_N,
    WGM_ROW_MAJOR,
    WGM_XCC,
    WGM_XCC_MAPPING8,
    WGM_XCC_MAPPING32,
    GemmConfig,
)
from .registry import CONFIG_BY_KEY, default_config, get_config


WARP_SIZE = 64
BF16_BYTES = 2
MI300_CU_COUNT = 38 * 8
WGM_XCC_WIDTH = 8

SCHED_MASK_MFMA = 0x8
SCHED_MASK_BUFFER_LOAD = 0x20
SCHED_MASK_DS_READ = 0x100
SCHED_MASK_DS_WRITE = 0x200

B4_GROUP_M = 224
B4_GROUP_N = 256
B4_GROUP_K = 64
B4_PARTITION_M = 2
B4_PARTITION_N = 2
B4_NUM_WARPS = 4
B4_THREADS = WARP_SIZE * B4_NUM_WARPS
B4_WARP_MAT_M = B4_GROUP_M // B4_PARTITION_M
B4_WARP_MAT_N = B4_GROUP_N // B4_PARTITION_N
B4_M_TILES_PER_WARP = B4_WARP_MAT_M // 16
B4_N_TILES_PER_WARP = B4_WARP_MAT_N // 16
B4_GLOBAL_WORDS_PER_ROW = B4_GROUP_K * BF16_BYTES // 4
B4_GLOBAL_ROWS_PER_ROUND = B4_THREADS // B4_GLOBAL_WORDS_PER_ROW
B4_REG_WORDS_A = B4_GROUP_M // B4_GLOBAL_ROWS_PER_ROUND
B4_REG_WORDS_B = B4_GROUP_N // B4_GLOBAL_ROWS_PER_ROUND
B4_READ_ROWS_A = 1
B4_READ_ROWS_B = 8
B4_SHM_ROW_WORDS = B4_GROUP_K * BF16_BYTES // 4
B4_SHM_GROUP_WORDS_A = B4_READ_ROWS_A * B4_SHM_ROW_WORDS + 2
B4_SHM_GROUP_WORDS_B = B4_READ_ROWS_B * B4_SHM_ROW_WORDS + 2
B4_SHM_GROUPS_A = B4_GROUP_M // B4_READ_ROWS_A
B4_SHM_GROUPS_B = B4_GROUP_N // B4_READ_ROWS_B
B4_SHM_TOTAL_WORDS_A = B4_SHM_GROUPS_A * B4_SHM_GROUP_WORDS_A
B4_SHM_TOTAL_WORDS_B = B4_SHM_GROUPS_B * B4_SHM_GROUP_WORDS_B


@avelang.jit
def _batch4_store_shm_a(
    shm: al.Tensor((B4_SHM_TOTAL_WORDS_A,), al.u32),
    reg: al.Tensor((B4_REG_WORDS_A,), al.u32),
    tid: al.u32,
):
    row = tid // B4_GLOBAL_WORDS_PER_ROW
    col_word = tid - row * B4_GLOBAL_WORDS_PER_ROW
    shm_word = row * B4_SHM_GROUP_WORDS_A + col_word
    shm_word_stride = B4_GLOBAL_ROWS_PER_ROUND * B4_SHM_GROUP_WORDS_A
    for i in al.range(B4_REG_WORDS_A):
        shm[shm_word + i * shm_word_stride] = reg[i]


@avelang.jit
def _batch4_store_shm_b(
    shm: al.Tensor((B4_SHM_TOTAL_WORDS_B,), al.u32),
    reg: al.Tensor((B4_REG_WORDS_B,), al.u32),
    tid: al.u32,
):
    row = tid // B4_GLOBAL_WORDS_PER_ROW
    col_word = tid - row * B4_GLOBAL_WORDS_PER_ROW
    row_group = row // B4_READ_ROWS_B
    row_in_group = row - row_group * B4_READ_ROWS_B
    shm_word = (
        row_group * B4_SHM_GROUP_WORDS_B
        + row_in_group * B4_SHM_ROW_WORDS
        + col_word
    )
    shm_word_stride = (
        B4_GLOBAL_ROWS_PER_ROUND // B4_READ_ROWS_B
    ) * B4_SHM_GROUP_WORDS_B
    for i in al.range(B4_REG_WORDS_B):
        shm[shm_word + i * shm_word_stride] = reg[i]


@avelang.jit
def _batch4_store_shm_ba(
    shm_a: al.Tensor((B4_SHM_TOTAL_WORDS_A,), al.u32),
    shm_b: al.Tensor((B4_SHM_TOTAL_WORDS_B,), al.u32),
    reg_a: al.Tensor((B4_REG_WORDS_A,), al.u32),
    reg_b: al.Tensor((B4_REG_WORDS_B,), al.u32),
    tid: al.u32,
):
    _batch4_store_shm_b(shm_b, reg_b, tid)
    _batch4_store_shm_a(shm_a, reg_a, tid)


@avelang.jit
def _batch4_load_shm_to_regs_a(
    shm: al.Tensor((B4_SHM_TOTAL_WORDS_A,), al.u32),
    wid: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
    data: al.Tensor((B4_M_TILES_PER_WARP, 2), al.u32),
):
    lane = wtid % 16
    quad = wtid // 16
    warp_row = wid // B4_PARTITION_N
    start_row = warp_row * 16 + lane
    col_uint2 = quad + batch_id * 4
    for tile in al.range(B4_M_TILES_PER_WARP):
        uint2_index = (
            start_row * (B4_SHM_GROUP_WORDS_A // 2)
            + col_uint2
            + tile * 32 * (B4_SHM_GROUP_WORDS_A // 2)
        )
        word_index = uint2_index * 2
        data[tile, 0] = shm[word_index]
        data[tile, 1] = shm[word_index + 1]


@avelang.jit
def _batch4_load_shm_to_regs_b(
    shm: al.Tensor((B4_SHM_TOTAL_WORDS_B,), al.u32),
    wid: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
    data: al.Tensor((B4_N_TILES_PER_WARP, 2), al.u32),
):
    lane = wtid % 16
    quad = wtid // 16
    warp_col = wid % B4_PARTITION_N
    start_row = warp_col * B4_WARP_MAT_N + lane * B4_READ_ROWS_B
    col_uint2 = quad + batch_id * 4
    start_uint2 = (start_row // B4_READ_ROWS_B) * (B4_SHM_GROUP_WORDS_B // 2)
    for tile in al.range(B4_N_TILES_PER_WARP):
        uint2_index = start_uint2 + col_uint2 + tile * 16
        word_index = uint2_index * 2
        data[tile, 0] = shm[word_index]
        data[tile, 1] = shm[word_index + 1]


@avelang.jit
def _batch4_read_shm_ba(
    shm_a: al.Tensor((B4_SHM_TOTAL_WORDS_A,), al.u32),
    shm_b: al.Tensor((B4_SHM_TOTAL_WORDS_B,), al.u32),
    wid: al.u32,
    batch_id: al.u32,
    wtid: al.u32,
    data_a: al.Tensor((B4_M_TILES_PER_WARP, 2), al.u32),
    data_b: al.Tensor((B4_N_TILES_PER_WARP, 2), al.u32),
):
    _batch4_load_shm_to_regs_b(shm_b, wid, batch_id, wtid, data_b)
    _batch4_load_shm_to_regs_a(shm_a, wid, batch_id, wtid, data_a)


def _make_batch2_kernel(config: GemmConfig):
    GROUP_M = config.group_m
    GROUP_N = config.group_n
    GROUP_K = config.group_k
    WARP_PER_ROW = config.partition_m
    WARP_PER_COL = config.partition_n
    NUM_WARPS = config.partition_m * config.partition_n * config.partition_k
    THREADS = WARP_SIZE * NUM_WARPS
    WARP_MAT_M = GROUP_M // WARP_PER_ROW
    WARP_MAT_N = GROUP_N // WARP_PER_COL
    M_TILES_PER_WARP = WARP_MAT_M // 16
    N_TILES_PER_WARP = WARP_MAT_N // 16
    VEC_SIZE = 8
    REG_ROWS_A = GROUP_M * GROUP_K // VEC_SIZE // THREADS
    REG_ROWS_B = GROUP_N * GROUP_K // VEC_SIZE // THREADS
    READ_ROWS_A = config.read_rows_a
    READ_ROWS_B = config.read_rows_b
    SHM_PAD_BF16_A = config.pad_a_bytes // BF16_BYTES
    SHM_PAD_BF16_B = config.pad_b_bytes // BF16_BYTES
    SHM_GROUPS_A = GROUP_M // READ_ROWS_A
    SHM_GROUPS_B = GROUP_N // READ_ROWS_B
    SHM_GROUP_BF16_A = READ_ROWS_A * GROUP_K + SHM_PAD_BF16_A
    SHM_GROUP_BF16_B = READ_ROWS_B * GROUP_K + SHM_PAD_BF16_B
    SHM_GROUP_WORDS_A = SHM_GROUP_BF16_A // 2
    SHM_GROUP_WORDS_B = SHM_GROUP_BF16_B // 2
    SHM_TOTAL_BF16_A = SHM_GROUPS_A * SHM_GROUP_BF16_A
    SHM_TOTAL_BF16_B = SHM_GROUPS_B * SHM_GROUP_BF16_B
    SHM_CHUNKS_PER_ROW = GROUP_K // VEC_SIZE
    WGM_MODE = config.wgm_mode
    STAGGER_MASK = config.stagger_mask
    STAGGER_STRIDE = config.stagger_stride
    STAGGER_BY = config.stagger_by
    LOOP_SCHEDULER = config.loop_scheduler
    LOAD_MODE = config.load_mode
    STORE_VEC = config.store_vec
    PREFETCH_BEFORE_ZERO = config.prefetch_before_zero
    PREFETCH1_BEFORE_READ = config.prefetch1_before_read
    PIPELINE_INTERLEAVE = config.pipeline_interleave

    @avelang.jit
    def _wgm_mapping(m: al.u32, n: al.u32) -> (al.u32, al.u32):
        linear_group_id = al.block_id(0)
        m_groups = m // GROUP_M
        n_groups = n // GROUP_N

        if WGM_MODE == WGM_ROW_MAJOR:
            group_m = linear_group_id // n_groups
            group_n = linear_group_id - group_m * n_groups
        else:
            total_groups = m_groups * n_groups
            cu_count = al.convert(MI300_CU_COUNT, al.u32)
            wgm_xcc = al.convert(WGM_XCC_WIDTH, al.u32)

            linear_group_limit = (total_groups // wgm_xcc) * wgm_xcc
            cu_base = (linear_group_id // cu_count) * cu_count
            cu_xcc = (linear_group_id % cu_count) // wgm_xcc
            cu_base = cu_base + cu_xcc

            cu_tail_limit = (total_groups // cu_count) * cu_count
            active_cu = (total_groups % cu_count) if (linear_group_id > cu_tail_limit) else cu_count
            cu_xcc_stride = (active_cu // wgm_xcc) * (linear_group_id % wgm_xcc)
            linear_group_mapped = cu_base + cu_xcc_stride
            linear_group_id = linear_group_mapped if (linear_group_id < linear_group_limit) else linear_group_id

            group_m = linear_group_id // n_groups
            group_n = linear_group_id - group_m * n_groups

            if WGM_MODE == WGM_XCC_MAPPING8 or WGM_MODE == WGM_XCC_MAPPING32:
                workgroup_mapping = al.convert(8, al.u32)
                if WGM_MODE == WGM_XCC_MAPPING32:
                    workgroup_mapping = al.convert(32, al.u32)
                mapping_block = group_m // workgroup_mapping
                mapping_linear = group_n + (group_m % workgroup_mapping) * n_groups
                mapping_groups = m_groups // workgroup_mapping
                mapping_tail = m_groups % workgroup_mapping
                mapping_tail = workgroup_mapping if (mapping_tail == 0) else mapping_tail
                mapping_span = mapping_tail if (mapping_block >= mapping_groups) else workgroup_mapping
                group_n = mapping_linear // mapping_span
                group_m = mapping_linear % mapping_span
                group_m = group_m + mapping_block * workgroup_mapping

        return group_m, group_n

    @avelang.jit
    def _k_tile(group_m: al.u32, group_n: al.u32, k_total: al.u32, offset: al.u32) -> al.u32:
        k_start = al.convert(0, al.u32)
        if STAGGER_MASK > 0:
            stagger_data = group_n if (STAGGER_BY == STAGGER_BY_N) else group_m
            k_start = (stagger_data & STAGGER_MASK) * STAGGER_STRIDE
            k_start = al.convert(0, al.u32) if (k_start >= k_total) else k_start
        return (k_start + offset) % k_total

    @avelang.jit
    def _load_global_a(src_rsrc: al.Tensor((4,), al.u32), k: al.u32, group_row: al.u32, k_idx: al.u32, tid: al.u32, reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16)):
        row = tid // SHM_CHUNKS_PER_ROW
        col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
        tile_offset = (group_row * GROUP_M * k + k_idx * GROUP_K) * BF16_BYTES
        thread_offset = (row * k + col) * BF16_BYTES
        if LOAD_MODE == LOAD_MODE_BASE_OFFSET:
            thread_offset = thread_offset + tile_offset
            tile_offset = al.convert(0, al.u32)
        thread_offset_stride = (THREADS * VEC_SIZE // GROUP_K) * k * BF16_BYTES
        for i in al.range(REG_ROWS_A):
            packed = al.amdgpu.raw_buffer_load_x4(src_rsrc, thread_offset, tile_offset + i * thread_offset_stride, 0)
            frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))
            for v in al.range(VEC_SIZE):
                reg[i, v] = frag[v]

    @avelang.jit
    def _load_global_b(src_rsrc: al.Tensor((4,), al.u32), k: al.u32, group_row: al.u32, k_idx: al.u32, tid: al.u32, reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16)):
        row = tid // SHM_CHUNKS_PER_ROW
        col = (tid - row * SHM_CHUNKS_PER_ROW) * VEC_SIZE
        tile_offset = (group_row * GROUP_N * k + k_idx * GROUP_K) * BF16_BYTES
        thread_offset = (row * k + col) * BF16_BYTES
        if LOAD_MODE == LOAD_MODE_BASE_OFFSET:
            thread_offset = thread_offset + tile_offset
            tile_offset = al.convert(0, al.u32)
        thread_offset_stride = (THREADS * VEC_SIZE // GROUP_K) * k * BF16_BYTES
        for i in al.range(REG_ROWS_B):
            packed = al.amdgpu.raw_buffer_load_x4(src_rsrc, thread_offset, tile_offset + i * thread_offset_stride, 0)
            frag = al.view(packed, al.Tensor((VEC_SIZE,), al.bf16))
            for v in al.range(VEC_SIZE):
                reg[i, v] = frag[v]

    @avelang.jit
    def _load_global_ab(a_rsrc: al.Tensor((4,), al.u32), b_rsrc: al.Tensor((4,), al.u32), k: al.u32, group_m: al.u32, group_n: al.u32, k_idx: al.u32, tid: al.u32, reg_a: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16), reg_b: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16)):
        _load_global_b(b_rsrc, k, group_n, k_idx, tid, reg_b)
        _load_global_a(a_rsrc, k, group_m, k_idx, tid, reg_a)

    @avelang.jit
    def _load_global_next(a_rsrc: al.Tensor((4,), al.u32), b_rsrc: al.Tensor((4,), al.u32), k: al.u32, group_m: al.u32, group_n: al.u32, k_idx: al.u32, tid: al.u32, reg_a: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16), reg_b: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16)):
        _load_global_b(b_rsrc, k, group_n, k_idx, tid, reg_b)
        _load_global_a(a_rsrc, k, group_m, k_idx, tid, reg_a)

    @avelang.jit
    def _store_shm_a(shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16), reg: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16), tid: al.u32):
        shm_vec = al.view(shm, al.u32, al.make_layout((SHM_TOTAL_BF16_A // VEC_SIZE, 4), (VEC_SIZE // 2, 1)))
        row = tid // SHM_CHUNKS_PER_ROW
        row_group = row // READ_ROWS_A
        row_in_group = row - row_group * READ_ROWS_A
        chunk = tid - row * SHM_CHUNKS_PER_ROW
        shm_chunk = row_group * (SHM_GROUP_WORDS_A // (VEC_SIZE // 2)) + row_in_group * SHM_CHUNKS_PER_ROW + chunk
        shm_chunk_stride = (THREADS * VEC_SIZE // GROUP_K // READ_ROWS_A) * (SHM_GROUP_WORDS_A // (VEC_SIZE // 2))
        for i in al.range(REG_ROWS_A):
            packed = al.view(reg[i], al.Tensor((4,), al.u32))
            shm_vec[shm_chunk + i * shm_chunk_stride] = packed

    @avelang.jit
    def _store_shm_b(shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16), reg: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16), tid: al.u32):
        shm_vec = al.view(shm, al.u32, al.make_layout((SHM_TOTAL_BF16_B // VEC_SIZE, 4), (VEC_SIZE // 2, 1)))
        row = tid // SHM_CHUNKS_PER_ROW
        row_group = row // READ_ROWS_B
        row_in_group = row - row_group * READ_ROWS_B
        chunk = tid - row * SHM_CHUNKS_PER_ROW
        shm_chunk = row_group * (SHM_GROUP_WORDS_B // (VEC_SIZE // 2)) + row_in_group * SHM_CHUNKS_PER_ROW + chunk
        shm_chunk_stride = (THREADS * VEC_SIZE // GROUP_K // READ_ROWS_B) * (SHM_GROUP_WORDS_B // (VEC_SIZE // 2))
        for i in al.range(REG_ROWS_B):
            packed = al.view(reg[i], al.Tensor((4,), al.u32))
            shm_vec[shm_chunk + i * shm_chunk_stride] = packed

    @avelang.jit
    def _store_shm_ab(shm_a: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16), shm_b: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16), reg_a: al.Tensor((REG_ROWS_A, VEC_SIZE), al.bf16), reg_b: al.Tensor((REG_ROWS_B, VEC_SIZE), al.bf16), tid: al.u32):
        _store_shm_b(shm_b, reg_b, tid)
        _store_shm_a(shm_a, reg_a, tid)

    @avelang.jit
    def _load_shm_to_regs_a(shm: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16), row_base: al.u32, batch_id: al.u32, wtid: al.u32, data: al.Tensor((M_TILES_PER_WARP, 4), al.u32)):
        shm_vec = al.view(shm, al.u32, al.make_layout((SHM_GROUPS_A, READ_ROWS_A, SHM_CHUNKS_PER_ROW, 4), (SHM_GROUP_WORDS_A, GROUP_K // 2, 4, 1)))
        row_start = row_base + (wtid % 16) * M_TILES_PER_WARP
        chunk_base = (wtid // 16) + batch_id * (32 // VEC_SIZE)
        for tile in al.range(M_TILES_PER_WARP):
            row = row_start + tile
            row_group = row // READ_ROWS_A
            row_in_group = row - row_group * READ_ROWS_A
            data[tile] = shm_vec[row_group, row_in_group, chunk_base]

    @avelang.jit
    def _load_shm_to_regs_b(shm: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16), row_base: al.u32, batch_id: al.u32, wtid: al.u32, data: al.Tensor((N_TILES_PER_WARP, 4), al.u32)):
        shm_vec = al.view(shm, al.u32, al.make_layout((SHM_GROUPS_B, READ_ROWS_B, SHM_CHUNKS_PER_ROW, 4), (SHM_GROUP_WORDS_B, GROUP_K // 2, 4, 1)))
        row_start = row_base + (wtid % 16) * N_TILES_PER_WARP
        chunk_base = (wtid // 16) + batch_id * (32 // VEC_SIZE)
        for tile in al.range(N_TILES_PER_WARP):
            row = row_start + tile
            row_group = row // READ_ROWS_B
            row_in_group = row - row_group * READ_ROWS_B
            data[tile] = shm_vec[row_group, row_in_group, chunk_base]

    @avelang.jit
    def _read_shm_ab(shm_a: al.Tensor((SHM_TOTAL_BF16_A,), al.bf16), shm_b: al.Tensor((SHM_TOTAL_BF16_B,), al.bf16), wid: al.u32, batch_id: al.u32, wtid: al.u32, data_a: al.Tensor((M_TILES_PER_WARP, 4), al.u32), data_b: al.Tensor((N_TILES_PER_WARP, 4), al.u32)):
        warp_row = wid // WARP_PER_COL
        warp_col = wid % WARP_PER_COL
        _load_shm_to_regs_b(shm_b, warp_col * WARP_MAT_N, batch_id, wtid, data_b)
        _load_shm_to_regs_a(shm_a, warp_row * WARP_MAT_M, batch_id, wtid, data_a)

    @avelang.jit
    def _matmul(data_a: al.Tensor((M_TILES_PER_WARP, 4), al.u32), data_b: al.Tensor((N_TILES_PER_WARP, 4), al.u32), acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32)):
        for tile_m in al.range(M_TILES_PER_WARP):
            for tile_n in al.range(N_TILES_PER_WARP):
                frag_a = al.view(data_a[tile_m], al.Tensor((2, 2, 1), al.u32))
                frag_b = al.view(data_b[tile_n], al.Tensor((2, 2, 1), al.u32))
                acc[tile_m, tile_n] = al.amdgpu.mfma_16x16x16_bf16_f32(frag_a[0], frag_b[0], acc[tile_m, tile_n])
                acc[tile_m, tile_n] = al.amdgpu.mfma_16x16x16_bf16_f32(frag_a[1], frag_b[1], acc[tile_m, tile_n])

    @avelang.jit
    def _hot_loop_scheduler():
        if LOOP_SCHEDULER == 0:
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
            for _ in al.range(8):
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)
            al.amdgpu.sched_group_barrier(0x0800, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
            for _ in al.range(2):
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)
            al.amdgpu.sched_group_barrier(0x0800, 1, 0)
            for _ in al.range(8):
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)
        else:
            for _ in al.range(45):
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 8, 0)
            al.amdgpu.sched_group_barrier(0x0800, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
            for _ in al.range(30):
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)
            al.amdgpu.sched_group_barrier(0x0800, 1, 0)
            for _ in al.range(15):
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 5, 0)

    @avelang.jit
    def _write_results(dst_rsrc: al.Tensor((4,), al.u32), n: al.u32, group_m: al.u32, group_n: al.u32, wtid: al.u32, wid: al.u32, acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32)):
        warp_row = wid // WARP_PER_COL
        warp_col = wid % WARP_PER_COL
        lane_row_group = wtid // 16
        lane_col = wtid % 16
        warp_offset = ((group_m * GROUP_M + warp_row * WARP_MAT_M) * n + group_n * GROUP_N + warp_col * WARP_MAT_N) * BF16_BYTES
        for tile_m in al.range(M_TILES_PER_WARP):
            for acc_idx in al.range(4):
                row_offset = (lane_row_group * (4 * M_TILES_PER_WARP) + acc_idx * M_TILES_PER_WARP + tile_m) * n
                col_offset = lane_col * N_TILES_PER_WARP
                thread_offset = (row_offset + col_offset) * BF16_BYTES
                if STORE_VEC == 1:
                    lo0 = al.bitcast(acc[tile_m, 0, acc_idx], al.u32)
                    hi0 = al.bitcast(acc[tile_m, 1, acc_idx], al.u32)
                    packed = al.amdgpu.perm(hi0, lo0, 0x07060302)
                    al.amdgpu.raw_buffer_store_x1(packed, dst_rsrc, thread_offset, warp_offset, 0)
                else:
                    lo0 = al.bitcast(acc[tile_m, 0, acc_idx], al.u32)
                    hi0 = al.bitcast(acc[tile_m, 1, acc_idx], al.u32)
                    lo1 = al.bitcast(acc[tile_m, 2, acc_idx], al.u32)
                    hi1 = al.bitcast(acc[tile_m, 3, acc_idx], al.u32)
                    packed = al.full((2,), 0, al.u32)
                    packed[0] = al.amdgpu.perm(hi0, lo0, 0x07060302)
                    packed[1] = al.amdgpu.perm(hi1, lo1, 0x07060302)
                    al.amdgpu.raw_buffer_store_x2(packed, dst_rsrc, thread_offset, warp_offset, 0)

    @avelang.jit
    def kernel(A: al.Pointer(al.bf16), B: al.Pointer(al.bf16), C: al.Pointer(al.bf16), m: al.u32, n: al.u32, k: al.u32):
        tid = al.thread_id(0)
        wid = tid // WARP_SIZE
        wtid = tid % WARP_SIZE
        warp_row = wid // WARP_PER_COL
        warp_col = wid % WARP_PER_COL

        linear_group_id = al.block_id(0)
        m_groups = m // GROUP_M
        n_groups = n // GROUP_N
        group_m = al.convert(0, al.u32)
        group_n = al.convert(0, al.u32)
        if WGM_MODE == WGM_ROW_MAJOR:
            group_m = linear_group_id // n_groups
            group_n = linear_group_id - group_m * n_groups
        else:
            total_groups = m_groups * n_groups
            cu_count = al.convert(MI300_CU_COUNT, al.u32)
            wgm_xcc = al.convert(WGM_XCC_WIDTH, al.u32)
            linear_group_limit = (total_groups // wgm_xcc) * wgm_xcc
            cu_base = (linear_group_id // cu_count) * cu_count
            cu_xcc = (linear_group_id % cu_count) // wgm_xcc
            cu_base = cu_base + cu_xcc
            cu_tail_limit = (total_groups // cu_count) * cu_count
            active_cu = (total_groups % cu_count) if (linear_group_id > cu_tail_limit) else cu_count
            cu_xcc_stride = (active_cu // wgm_xcc) * (linear_group_id % wgm_xcc)
            linear_group_mapped = cu_base + cu_xcc_stride
            linear_group_id = linear_group_mapped if (linear_group_id < linear_group_limit) else linear_group_id
            group_m = linear_group_id // n_groups
            group_n = linear_group_id - group_m * n_groups
            if WGM_MODE == WGM_XCC_MAPPING8 or WGM_MODE == WGM_XCC_MAPPING32:
                workgroup_mapping = al.convert(8, al.u32)
                if WGM_MODE == WGM_XCC_MAPPING32:
                    workgroup_mapping = al.convert(32, al.u32)
                mapping_block = group_m // workgroup_mapping
                mapping_linear = group_n + (group_m % workgroup_mapping) * n_groups
                mapping_groups = m_groups // workgroup_mapping
                mapping_tail = m_groups % workgroup_mapping
                mapping_tail = workgroup_mapping if (mapping_tail == 0) else mapping_tail
                mapping_span = mapping_tail if (mapping_block >= mapping_groups) else workgroup_mapping
                group_n = mapping_linear // mapping_span
                group_m = mapping_linear % mapping_span
                group_m = group_m + mapping_block * workgroup_mapping

        a_tensor = al.make_tensor(A, al.bf16, al.make_layout((m, k), (k, 1)))
        b_tensor = al.make_tensor(B, al.bf16, al.make_layout((n, k), (k, 1)))
        c_tensor = al.make_tensor(C, al.bf16, al.make_layout((m * n,), (1,)))
        a_rsrc = al.amdgpu.make_rsrc(a_tensor, m * k * BF16_BYTES)
        b_rsrc = al.amdgpu.make_rsrc(b_tensor, n * k * BF16_BYTES)
        c_rsrc = al.amdgpu.make_rsrc(c_tensor, m * n * BF16_BYTES)

        shm_a = al.make_shared((SHM_TOTAL_BF16_A,), al.bf16)
        shm_b = al.make_shared((SHM_TOTAL_BF16_B,), al.bf16)
        reg_a = al.make_local((REG_ROWS_A, VEC_SIZE), al.bf16)
        reg_b = al.make_local((REG_ROWS_B, VEC_SIZE), al.bf16)
        data_a0 = al.make_local((M_TILES_PER_WARP, 4), al.u32)
        data_a1 = al.make_local((M_TILES_PER_WARP, 4), al.u32)
        data_b0 = al.make_local((N_TILES_PER_WARP, 4), al.u32)
        data_b1 = al.make_local((N_TILES_PER_WARP, 4), al.u32)
        acc = al.make_local((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32)
        k_total = k // GROUP_K

        if PREFETCH_BEFORE_ZERO:
            _load_global_ab(a_rsrc, b_rsrc, k, group_m, group_n, _k_tile(group_m, group_n, k_total, 0), tid, reg_a, reg_b)

        for tile_m in al.range(M_TILES_PER_WARP):
            for tile_n in al.range(N_TILES_PER_WARP):
                for acc_idx in al.range(4):
                    acc[tile_m, tile_n, acc_idx] = al.convert(0.0, al.f32)

        if not PREFETCH_BEFORE_ZERO:
            _load_global_ab(a_rsrc, b_rsrc, k, group_m, group_n, _k_tile(group_m, group_n, k_total, 0), tid, reg_a, reg_b)

        _store_shm_ab(shm_a, shm_b, reg_a, reg_b, tid)
        al.syncthreads()

        if PREFETCH1_BEFORE_READ:
            _load_global_next(a_rsrc, b_rsrc, k, group_m, group_n, _k_tile(group_m, group_n, k_total, 1), tid, reg_a, reg_b)
            _read_shm_ab(shm_a, shm_b, wid, 0, wtid, data_a0, data_b0)
        else:
            _read_shm_ab(shm_a, shm_b, wid, 0, wtid, data_a0, data_b0)
            _load_global_next(a_rsrc, b_rsrc, k, group_m, group_n, _k_tile(group_m, group_n, k_total, 1), tid, reg_a, reg_b)

        for k_idx in al.range(0, k_total - 3, 2):
            _read_shm_ab(shm_a, shm_b, wid, 1, wtid, data_a1, data_b1)
            _matmul(data_a0, data_b0, acc)
            al.syncthreads()
            if PIPELINE_INTERLEAVE:
                _store_shm_b(shm_b, reg_b, tid)
                _load_global_b(b_rsrc, k, group_n, _k_tile(group_m, group_n, k_total, k_idx + 2), tid, reg_b)
                _store_shm_a(shm_a, reg_a, tid)
                _load_global_a(a_rsrc, k, group_m, _k_tile(group_m, group_n, k_total, k_idx + 2), tid, reg_a)
            else:
                _store_shm_ab(shm_a, shm_b, reg_a, reg_b, tid)
                _load_global_ab(a_rsrc, b_rsrc, k, group_m, group_n, _k_tile(group_m, group_n, k_total, k_idx + 2), tid, reg_a, reg_b)
            al.syncthreads()

            _read_shm_ab(shm_a, shm_b, wid, 0, wtid, data_a0, data_b0)
            _matmul(data_a1, data_b1, acc)

            _read_shm_ab(shm_a, shm_b, wid, 1, wtid, data_a1, data_b1)
            _matmul(data_a0, data_b0, acc)
            al.syncthreads()
            if PIPELINE_INTERLEAVE:
                _store_shm_b(shm_b, reg_b, tid)
                _load_global_b(b_rsrc, k, group_n, _k_tile(group_m, group_n, k_total, k_idx + 3), tid, reg_b)
                _store_shm_a(shm_a, reg_a, tid)
                _load_global_a(a_rsrc, k, group_m, _k_tile(group_m, group_n, k_total, k_idx + 3), tid, reg_a)
            else:
                _store_shm_ab(shm_a, shm_b, reg_a, reg_b, tid)
                _load_global_ab(a_rsrc, b_rsrc, k, group_m, group_n, _k_tile(group_m, group_n, k_total, k_idx + 3), tid, reg_a, reg_b)
            al.syncthreads()

            _read_shm_ab(shm_a, shm_b, wid, 0, wtid, data_a0, data_b0)
            _matmul(data_a1, data_b1, acc)
            _hot_loop_scheduler()
            _hot_loop_scheduler()

        _read_shm_ab(shm_a, shm_b, wid, 1, wtid, data_a1, data_b1)
        _matmul(data_a0, data_b0, acc)
        al.syncthreads()
        _store_shm_ab(shm_a, shm_b, reg_a, reg_b, tid)
        al.syncthreads()
        _read_shm_ab(shm_a, shm_b, wid, 0, wtid, data_a0, data_b0)
        _matmul(data_a1, data_b1, acc)
        _read_shm_ab(shm_a, shm_b, wid, 1, wtid, data_a1, data_b1)
        _matmul(data_a0, data_b0, acc)
        _matmul(data_a1, data_b1, acc)
        _write_results(c_rsrc, n, group_m, group_n, wtid, wid, acc)

    return kernel



def _make_batch4_kernel(config: GemmConfig):
    GROUP_M = config.group_m
    GROUP_N = config.group_n
    GROUP_K = config.group_k
    WARP_PER_ROW = config.partition_m
    WARP_PER_COL = config.partition_n
    NUM_WARPS = config.partition_m * config.partition_n * config.partition_k
    THREADS = WARP_SIZE * NUM_WARPS
    WARP_MAT_M = GROUP_M // WARP_PER_ROW
    WARP_MAT_N = GROUP_N // WARP_PER_COL
    M_TILES_PER_WARP = WARP_MAT_M // 16
    N_TILES_PER_WARP = WARP_MAT_N // 16
    GLOBAL_WORDS_PER_ROW = GROUP_K * BF16_BYTES // 4
    GLOBAL_ROWS_PER_ROUND = THREADS // GLOBAL_WORDS_PER_ROW
    REG_WORDS_A = GROUP_M // GLOBAL_ROWS_PER_ROUND
    REG_WORDS_B = GROUP_N // GLOBAL_ROWS_PER_ROUND
    READ_ROWS_A = config.read_rows_a
    READ_ROWS_B = config.read_rows_b
    SHM_PAD_WORDS_A = config.pad_a_bytes // 4
    SHM_PAD_WORDS_B = config.pad_b_bytes // 4
    SHM_ROW_WORDS = GROUP_K * BF16_BYTES // 4
    SHM_GROUP_WORDS_A = READ_ROWS_A * SHM_ROW_WORDS + SHM_PAD_WORDS_A
    SHM_GROUP_WORDS_B = READ_ROWS_B * SHM_ROW_WORDS + SHM_PAD_WORDS_B
    SHM_GROUPS_A = GROUP_M // READ_ROWS_A
    SHM_GROUPS_B = GROUP_N // READ_ROWS_B
    SHM_TOTAL_WORDS_A = SHM_GROUPS_A * SHM_GROUP_WORDS_A
    SHM_TOTAL_WORDS_B = SHM_GROUPS_B * SHM_GROUP_WORDS_B
    WGM_MODE = config.wgm_mode
    LOAD_MODE = config.load_mode
    STORE_VEC = config.store_vec
    PREFETCH_BEFORE_ZERO = config.prefetch_before_zero
    PIPELINE_INTERLEAVE = config.pipeline_interleave
    LOOP_SCHEDULER = config.loop_scheduler

    if WGM_MODE == WGM_ROW_MAJOR:
        @avelang.jit
        def _group_m(m: al.u32, n: al.u32) -> al.u32:
            linear_group_id = al.block_id(0)
            n_groups = (n + GROUP_N - 1) // GROUP_N
            return linear_group_id // n_groups

        @avelang.jit
        def _group_n(m: al.u32, n: al.u32) -> al.u32:
            linear_group_id = al.block_id(0)
            n_groups = (n + GROUP_N - 1) // GROUP_N
            group_m = linear_group_id // n_groups
            return linear_group_id - group_m * n_groups
    else:
        @avelang.jit
        def _linear_group_id(m: al.u32, n: al.u32) -> al.u32:
            linear_group_id = al.block_id(0)
            m_groups = (m + GROUP_M - 1) // GROUP_M
            n_groups = (n + GROUP_N - 1) // GROUP_N
            total_groups = m_groups * n_groups
            cu_count = al.convert(MI300_CU_COUNT, al.u32)
            wgm_xcc = al.convert(WGM_XCC_WIDTH, al.u32)
            linear_group_limit = (total_groups // wgm_xcc) * wgm_xcc
            cu_base = (linear_group_id // cu_count) * cu_count
            cu_xcc = (linear_group_id % cu_count) // wgm_xcc
            cu_base = cu_base + cu_xcc
            cu_tail_limit = (total_groups // cu_count) * cu_count
            active_cu = (total_groups % cu_count) if (linear_group_id > cu_tail_limit) else cu_count
            cu_xcc_stride = (active_cu // wgm_xcc) * (linear_group_id % wgm_xcc)
            mapped = cu_base + cu_xcc_stride
            return mapped if (linear_group_id < linear_group_limit) else linear_group_id

        @avelang.jit
        def _group_m(m: al.u32, n: al.u32) -> al.u32:
            linear_group_id = _linear_group_id(m, n)
            m_groups = (m + GROUP_M - 1) // GROUP_M
            n_groups = (n + GROUP_N - 1) // GROUP_N
            group_m = linear_group_id // n_groups
            group_n = linear_group_id - group_m * n_groups
            workgroup_mapping = al.convert(8, al.u32)
            if WGM_MODE == WGM_XCC_MAPPING32:
                workgroup_mapping = al.convert(32, al.u32)
            mapping_block = group_m // workgroup_mapping
            mapping_linear = group_n + (group_m % workgroup_mapping) * n_groups
            mapping_groups = m_groups // workgroup_mapping
            mapping_tail = m_groups % workgroup_mapping
            mapping_tail = workgroup_mapping if (mapping_tail == 0) else mapping_tail
            mapping_span = mapping_tail if (mapping_block >= mapping_groups) else workgroup_mapping
            return mapping_linear % mapping_span + mapping_block * workgroup_mapping

        @avelang.jit
        def _group_n(m: al.u32, n: al.u32) -> al.u32:
            linear_group_id = _linear_group_id(m, n)
            m_groups = (m + GROUP_M - 1) // GROUP_M
            n_groups = (n + GROUP_N - 1) // GROUP_N
            group_m = linear_group_id // n_groups
            group_n = linear_group_id - group_m * n_groups
            workgroup_mapping = al.convert(8, al.u32)
            if WGM_MODE == WGM_XCC_MAPPING32:
                workgroup_mapping = al.convert(32, al.u32)
            mapping_block = group_m // workgroup_mapping
            mapping_linear = group_n + (group_m % workgroup_mapping) * n_groups
            mapping_groups = m_groups // workgroup_mapping
            mapping_tail = m_groups % workgroup_mapping
            mapping_tail = workgroup_mapping if (mapping_tail == 0) else mapping_tail
            mapping_span = mapping_tail if (mapping_block >= mapping_groups) else workgroup_mapping
            return mapping_linear // mapping_span

    @avelang.jit
    def _load_global_a(
        src_rsrc: al.Tensor((4,), al.u32),
        k: al.u32,
        group_row: al.u32,
        k_idx: al.u32,
        tid: al.u32,
        reg: al.Tensor((REG_WORDS_A,), al.u32),
    ):
        row = tid // GLOBAL_WORDS_PER_ROW
        col_word = tid - row * GLOBAL_WORDS_PER_ROW
        tile_offset = (group_row * GROUP_M * k + k_idx * GROUP_K) * BF16_BYTES
        thread_offset = row * k * BF16_BYTES + col_word * 4
        if LOAD_MODE == LOAD_MODE_BASE_OFFSET:
            thread_offset = thread_offset + tile_offset
            tile_offset = al.convert(0, al.u32)
        thread_offset_stride = GLOBAL_ROWS_PER_ROUND * k * BF16_BYTES
        for i in al.range(REG_WORDS_A):
            reg[i] = al.amdgpu.raw_buffer_load_x1(
                src_rsrc,
                thread_offset,
                tile_offset + i * thread_offset_stride,
                0,
            )

    @avelang.jit
    def _load_global_b(
        src_rsrc: al.Tensor((4,), al.u32),
        k: al.u32,
        group_row: al.u32,
        k_idx: al.u32,
        tid: al.u32,
        reg: al.Tensor((REG_WORDS_B,), al.u32),
    ):
        row = tid // GLOBAL_WORDS_PER_ROW
        col_word = tid - row * GLOBAL_WORDS_PER_ROW
        tile_offset = (group_row * GROUP_N * k + k_idx * GROUP_K) * BF16_BYTES
        thread_offset = row * k * BF16_BYTES + col_word * 4
        if LOAD_MODE == LOAD_MODE_BASE_OFFSET:
            thread_offset = thread_offset + tile_offset
            tile_offset = al.convert(0, al.u32)
        thread_offset_stride = GLOBAL_ROWS_PER_ROUND * k * BF16_BYTES
        for i in al.range(REG_WORDS_B):
            reg[i] = al.amdgpu.raw_buffer_load_x1(
                src_rsrc,
                thread_offset,
                tile_offset + i * thread_offset_stride,
                0,
            )

    @avelang.jit
    def _load_global_ab(
        a_rsrc: al.Tensor((4,), al.u32),
        b_rsrc: al.Tensor((4,), al.u32),
        k: al.u32,
        group_m: al.u32,
        group_n: al.u32,
        k_idx: al.u32,
        tid: al.u32,
        reg_a: al.Tensor((REG_WORDS_A,), al.u32),
        reg_b: al.Tensor((REG_WORDS_B,), al.u32),
    ):
        _load_global_b(b_rsrc, k, group_n, k_idx, tid, reg_b)
        _load_global_a(a_rsrc, k, group_m, k_idx, tid, reg_a)

    @avelang.jit
    def _load_global_next(
        a_rsrc: al.Tensor((4,), al.u32),
        b_rsrc: al.Tensor((4,), al.u32),
        k: al.u32,
        group_m: al.u32,
        group_n: al.u32,
        k_idx: al.u32,
        tid: al.u32,
        reg_a: al.Tensor((REG_WORDS_A,), al.u32),
        reg_b: al.Tensor((REG_WORDS_B,), al.u32),
    ):
        _load_global_b(b_rsrc, k, group_n, k_idx, tid, reg_b)
        _load_global_a(a_rsrc, k, group_m, k_idx, tid, reg_a)

    @avelang.jit
    def _store_shm_a(
        shm: al.Tensor((SHM_TOTAL_WORDS_A,), al.u32),
        reg: al.Tensor((REG_WORDS_A,), al.u32),
        tid: al.u32,
    ):
        row = tid // GLOBAL_WORDS_PER_ROW
        col_word = tid - row * GLOBAL_WORDS_PER_ROW
        shm_word = row * SHM_GROUP_WORDS_A + col_word
        shm_word_stride = GLOBAL_ROWS_PER_ROUND * SHM_GROUP_WORDS_A
        for i in al.range(REG_WORDS_A):
            shm[shm_word + i * shm_word_stride] = reg[i]

    @avelang.jit
    def _store_shm_b(
        shm: al.Tensor((SHM_TOTAL_WORDS_B,), al.u32),
        reg: al.Tensor((REG_WORDS_B,), al.u32),
        tid: al.u32,
    ):
        row = tid // GLOBAL_WORDS_PER_ROW
        col_word = tid - row * GLOBAL_WORDS_PER_ROW
        row_group = row // READ_ROWS_B
        row_in_group = row - row_group * READ_ROWS_B
        shm_word = (
            row_group * SHM_GROUP_WORDS_B
            + row_in_group * SHM_ROW_WORDS
            + col_word
        )
        shm_word_stride = (GLOBAL_ROWS_PER_ROUND // READ_ROWS_B) * SHM_GROUP_WORDS_B
        for i in al.range(REG_WORDS_B):
            shm[shm_word + i * shm_word_stride] = reg[i]

    @avelang.jit
    def _store_shm_ab(
        shm_a: al.Tensor((SHM_TOTAL_WORDS_A,), al.u32),
        shm_b: al.Tensor((SHM_TOTAL_WORDS_B,), al.u32),
        reg_a: al.Tensor((REG_WORDS_A,), al.u32),
        reg_b: al.Tensor((REG_WORDS_B,), al.u32),
        tid: al.u32,
    ):
        _batch4_store_shm_b(shm_b, reg_b, tid)
        _batch4_store_shm_a(shm_a, reg_a, tid)

    @avelang.jit
    def _load_shm_to_regs_a(
        shm: al.Tensor((SHM_TOTAL_WORDS_A,), al.u32),
        warp_row: al.u32,
        batch_id: al.u32,
        wtid: al.u32,
        data: al.Tensor((M_TILES_PER_WARP, 2), al.u32),
    ):
        lane = wtid % 16
        quad = wtid // 16
        start_row = warp_row * 16 + lane
        col_uint2 = quad + batch_id * 4
        for tile in al.range(M_TILES_PER_WARP):
            uint2_index = (
                start_row * (SHM_GROUP_WORDS_A // 2)
                + col_uint2
                + tile * 32 * (SHM_GROUP_WORDS_A // 2)
            )
            word_index = uint2_index * 2
            data[tile, 0] = shm[word_index]
            data[tile, 1] = shm[word_index + 1]

    @avelang.jit
    def _load_shm_to_regs_b(
        shm: al.Tensor((SHM_TOTAL_WORDS_B,), al.u32),
        warp_col: al.u32,
        batch_id: al.u32,
        wtid: al.u32,
        data: al.Tensor((N_TILES_PER_WARP, 2), al.u32),
    ):
        lane = wtid % 16
        quad = wtid // 16
        start_row = warp_col * WARP_MAT_N + lane * READ_ROWS_B
        col_uint2 = quad + batch_id * 4
        start_uint2 = (start_row // READ_ROWS_B) * (SHM_GROUP_WORDS_B // 2)
        for tile in al.range(N_TILES_PER_WARP):
            uint2_index = start_uint2 + col_uint2 + tile * 16
            word_index = uint2_index * 2
            data[tile, 0] = shm[word_index]
            data[tile, 1] = shm[word_index + 1]

    @avelang.jit
    def _read_shm_ab(
        shm_a: al.Tensor((SHM_TOTAL_WORDS_A,), al.u32),
        shm_b: al.Tensor((SHM_TOTAL_WORDS_B,), al.u32),
        wid: al.u32,
        batch_id: al.u32,
        wtid: al.u32,
        data_a: al.Tensor((M_TILES_PER_WARP, 2), al.u32),
        data_b: al.Tensor((N_TILES_PER_WARP, 2), al.u32),
    ):
        warp_row = wid // WARP_PER_COL
        warp_col = wid % WARP_PER_COL
        _load_shm_to_regs_b(shm_b, warp_col, batch_id, wtid, data_b)
        _load_shm_to_regs_a(shm_a, warp_row, batch_id, wtid, data_a)

    @avelang.jit
    def _matmul(
        data_a: al.Tensor((M_TILES_PER_WARP, 2), al.u32),
        data_b: al.Tensor((N_TILES_PER_WARP, 2), al.u32),
        acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32),
    ):
        for tile_m in al.range(M_TILES_PER_WARP):
            for tile_n in al.range(N_TILES_PER_WARP):
                acc[tile_m, tile_n] = al.amdgpu.mfma_16x16x16_bf16_f32(
                    data_a[tile_m],
                    data_b[tile_n],
                    acc[tile_m, tile_n],
                )

    @avelang.jit
    def _hot_loop_scheduler():
        if LOOP_SCHEDULER == 0:
            for _ in al.range(8):
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)
            for _ in al.range(8):
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 3, 0)
            for _ in al.range(8):
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)
        else:
            for _ in al.range(45):
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 8, 0)
            al.amdgpu.sched_group_barrier(0x0800, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
            for _ in al.range(30):
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 2, 0)
            al.amdgpu.sched_group_barrier(0x0800, 1, 0)
            for _ in al.range(15):
                al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 1, 0)
                al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, 1, 0)
            al.amdgpu.sched_group_barrier(SCHED_MASK_MFMA, 5, 0)

    @avelang.jit
    def _write_results(
        dst_rsrc: al.Tensor((4,), al.u32),
        n: al.u32,
        group_m: al.u32,
        group_n: al.u32,
        wtid: al.u32,
        wid: al.u32,
        acc: al.Tensor((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32),
    ):
        lane = wtid % 16
        quad = wtid // 16
        warp_row = wid // WARP_PER_COL
        warp_col = wid % WARP_PER_COL
        warp_offset = (group_m * GROUP_M * n + group_n * GROUP_N) * BF16_BYTES
        for tile_m in al.range(M_TILES_PER_WARP):
            for acc_idx in al.range(4):
                row = warp_row * 16 + quad * 4 + tile_m * 32 + acc_idx
                col_base = warp_col * WARP_MAT_N + lane * READ_ROWS_B
                if STORE_VEC == 4:
                    thread_offset = (row * n + col_base) * BF16_BYTES
                    packed = al.full((4,), 0, al.u32)
                    lo0 = al.bitcast(acc[tile_m, 0, acc_idx], al.u32)
                    hi0 = al.bitcast(acc[tile_m, 1, acc_idx], al.u32)
                    lo1 = al.bitcast(acc[tile_m, 2, acc_idx], al.u32)
                    hi1 = al.bitcast(acc[tile_m, 3, acc_idx], al.u32)
                    lo2 = al.bitcast(acc[tile_m, 4, acc_idx], al.u32)
                    hi2 = al.bitcast(acc[tile_m, 5, acc_idx], al.u32)
                    lo3 = al.bitcast(acc[tile_m, 6, acc_idx], al.u32)
                    hi3 = al.bitcast(acc[tile_m, 7, acc_idx], al.u32)
                    packed[0] = al.amdgpu.perm(hi0, lo0, 0x07060302)
                    packed[1] = al.amdgpu.perm(hi1, lo1, 0x07060302)
                    packed[2] = al.amdgpu.perm(hi2, lo2, 0x07060302)
                    packed[3] = al.amdgpu.perm(hi3, lo3, 0x07060302)
                    al.amdgpu.raw_buffer_store_x4(
                        packed, dst_rsrc, thread_offset, warp_offset, 0
                    )
                else:
                    for t in al.range(0, N_TILES_PER_WARP, 4):
                        col = col_base + t
                        thread_offset = (row * n + col) * BF16_BYTES
                        lo0 = al.bitcast(acc[tile_m, t, acc_idx], al.u32)
                        hi0 = al.bitcast(acc[tile_m, t + 1, acc_idx], al.u32)
                        lo1 = al.bitcast(acc[tile_m, t + 2, acc_idx], al.u32)
                        hi1 = al.bitcast(acc[tile_m, t + 3, acc_idx], al.u32)
                        packed = al.full((2,), 0, al.u32)
                        packed[0] = al.amdgpu.perm(hi0, lo0, 0x07060302)
                        packed[1] = al.amdgpu.perm(hi1, lo1, 0x07060302)
                        al.amdgpu.raw_buffer_store_x2(
                            packed, dst_rsrc, thread_offset, warp_offset, 0
                        )

    @avelang.jit
    def kernel(
        A: al.Pointer(al.bf16),
        B: al.Pointer(al.bf16),
        C: al.Pointer(al.bf16),
        m: al.u32,
        n: al.u32,
        k: al.u32,
    ):
        tid = al.thread_id(0)
        wid = tid // WARP_SIZE
        wtid = tid % WARP_SIZE
        group_m = _group_m(m, n)
        group_n = _group_n(m, n)

        a_tensor = al.make_tensor(A, al.bf16, al.make_layout((m, k), (k, 1)))
        b_tensor = al.make_tensor(B, al.bf16, al.make_layout((n, k), (k, 1)))
        c_tensor = al.make_tensor(C, al.bf16, al.make_layout((m * n,), (1,)))
        a_rsrc = al.amdgpu.make_rsrc(a_tensor, m * k * BF16_BYTES)
        b_rsrc = al.amdgpu.make_rsrc(b_tensor, n * k * BF16_BYTES)
        c_rsrc = al.amdgpu.make_rsrc(c_tensor, m * n * BF16_BYTES)

        shm_a = al.make_shared((SHM_TOTAL_WORDS_A,), al.u32)
        shm_b = al.make_shared((SHM_TOTAL_WORDS_B,), al.u32)
        reg_a = al.make_local((REG_WORDS_A,), al.u32)
        reg_b = al.make_local((REG_WORDS_B,), al.u32)
        data_a0 = al.make_local((M_TILES_PER_WARP, 2), al.u32)
        data_a1 = al.make_local((M_TILES_PER_WARP, 2), al.u32)
        data_a2 = al.make_local((M_TILES_PER_WARP, 2), al.u32)
        data_a3 = al.make_local((M_TILES_PER_WARP, 2), al.u32)
        data_b0 = al.make_local((N_TILES_PER_WARP, 2), al.u32)
        data_b1 = al.make_local((N_TILES_PER_WARP, 2), al.u32)
        data_b2 = al.make_local((N_TILES_PER_WARP, 2), al.u32)
        data_b3 = al.make_local((N_TILES_PER_WARP, 2), al.u32)
        acc = al.make_local((M_TILES_PER_WARP, N_TILES_PER_WARP, 4), al.f32)
        k_total = k // GROUP_K

        if PREFETCH_BEFORE_ZERO:
            _load_global_ab(
                a_rsrc, b_rsrc, k, group_m, group_n, al.convert(0, al.u32),
                tid, reg_a, reg_b
            )

        for tile_m in al.range(M_TILES_PER_WARP):
            for tile_n in al.range(N_TILES_PER_WARP):
                for acc_idx in al.range(4):
                    acc[tile_m, tile_n, acc_idx] = al.convert(0.0, al.f32)

        if not PREFETCH_BEFORE_ZERO:
            _load_global_ab(
                a_rsrc, b_rsrc, k, group_m, group_n, al.convert(0, al.u32),
                tid, reg_a, reg_b
            )

        _batch4_store_shm_ba(shm_a, shm_b, reg_a, reg_b, tid)
        al.syncthreads()
        _batch4_read_shm_ba(shm_a, shm_b, wid, 0, wtid, data_a0, data_b0)
        _load_global_next(
            a_rsrc, b_rsrc, k, group_m, group_n, al.convert(1, al.u32),
            tid, reg_a, reg_b
        )

        for k_idx in al.range(0, k_total - 2):
            _batch4_read_shm_ba(shm_a, shm_b, wid, 1, wtid, data_a1, data_b1)
            _matmul(data_a0, data_b0, acc)
            _batch4_read_shm_ba(shm_a, shm_b, wid, 2, wtid, data_a2, data_b2)
            _matmul(data_a1, data_b1, acc)
            _batch4_read_shm_ba(shm_a, shm_b, wid, 3, wtid, data_a3, data_b3)
            _matmul(data_a2, data_b2, acc)
            al.syncthreads()
            if PIPELINE_INTERLEAVE:
                _batch4_store_shm_b(shm_b, reg_b, tid)
                _load_global_b(b_rsrc, k, group_n, k_idx + 2, tid, reg_b)
                _batch4_store_shm_a(shm_a, reg_a, tid)
                _load_global_a(a_rsrc, k, group_m, k_idx + 2, tid, reg_a)
            else:
                _batch4_store_shm_ba(shm_a, shm_b, reg_a, reg_b, tid)
                _load_global_ab(
                    a_rsrc, b_rsrc, k, group_m, group_n, k_idx + 2,
                    tid, reg_a, reg_b
                )
            al.syncthreads()
            _batch4_read_shm_ba(shm_a, shm_b, wid, 0, wtid, data_a0, data_b0)
            _matmul(data_a3, data_b3, acc)
            _hot_loop_scheduler()

        _batch4_read_shm_ba(shm_a, shm_b, wid, 1, wtid, data_a1, data_b1)
        _matmul(data_a0, data_b0, acc)
        _batch4_read_shm_ba(shm_a, shm_b, wid, 2, wtid, data_a2, data_b2)
        _matmul(data_a1, data_b1, acc)
        _batch4_read_shm_ba(shm_a, shm_b, wid, 3, wtid, data_a3, data_b3)
        _matmul(data_a2, data_b2, acc)
        al.syncthreads()
        _batch4_store_shm_ba(shm_a, shm_b, reg_a, reg_b, tid)
        al.syncthreads()
        _batch4_read_shm_ba(shm_a, shm_b, wid, 0, wtid, data_a0, data_b0)
        _matmul(data_a3, data_b3, acc)
        _batch4_read_shm_ba(shm_a, shm_b, wid, 1, wtid, data_a1, data_b1)
        _matmul(data_a0, data_b0, acc)
        _batch4_read_shm_ba(shm_a, shm_b, wid, 2, wtid, data_a2, data_b2)
        _matmul(data_a1, data_b1, acc)
        _batch4_read_shm_ba(shm_a, shm_b, wid, 3, wtid, data_a3, data_b3)
        _matmul(data_a2, data_b2, acc)
        _matmul(data_a3, data_b3, acc)
        _write_results(c_rsrc, n, group_m, group_n, wtid, wid, acc)

    return kernel


@lru_cache(maxsize=None)
def _kernel_for_key(key: tuple[int, ...]):
    try:
        config = CONFIG_BY_KEY[key]
    except KeyError as exc:
        raise ValueError(f"No registered GEMM config for key {key}.") from exc
    if config.num_batch_k == 2:
        return _make_batch2_kernel(config)
    if config.num_batch_k == 4:
        return _make_batch4_kernel(config)
    raise ValueError(f"Unsupported num_batch_k={config.num_batch_k} for {config.name}.")


def gemm_pipeline_transposed_b(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    config: GemmConfig | str | None = None,
) -> torch.Tensor:
    if A.dtype != torch.bfloat16:
        raise ValueError(f"A must have dtype torch.bfloat16, got {A.dtype}.")
    if B.dtype != torch.bfloat16:
        raise ValueError(f"B must have dtype torch.bfloat16, got {B.dtype}.")
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {tuple(A.shape)}.")
    if B.ndim != 2:
        raise ValueError(f"B must be 2D, got shape {tuple(B.shape)}.")

    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[0]
    if B.shape[1] != k:
        raise ValueError(f"B.shape[1] must equal K={k}, got {B.shape[1]}.")

    if config is None:
        resolved = default_config(m, n, k)
    elif isinstance(config, str):
        resolved = get_config(config)
    else:
        resolved = config
    if not resolved.supports(m, n, k):
        raise ValueError(f"AMDGPU GEMM config {resolved.name!r} does not support M={m}, N={n}, K={k}.")

    out_shape = (m, n)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.bfloat16, device=A.device)
    elif out.shape != out_shape:
        raise ValueError(f"out must have shape {out_shape}, got {tuple(out.shape)}.")
    elif out.dtype != torch.bfloat16:
        raise ValueError(f"out must have dtype torch.bfloat16, got {out.dtype}.")
    elif out.device != A.device:
        raise ValueError(f"out must be on device {A.device}, got {out.device}.")

    if resolved.num_batch_k == 4:
        m_groups = (m + resolved.group_m - 1) // resolved.group_m
        n_groups = (n + resolved.group_n - 1) // resolved.group_n
    else:
        m_groups = m // resolved.group_m
        n_groups = n // resolved.group_n
    grid_size = m_groups * n_groups
    block_size = WARP_SIZE * resolved.partition_m * resolved.partition_n * resolved.partition_k
    kernel = _kernel_for_key(resolved.key)
    kernel[lambda: ((grid_size, 1, 1), (block_size, 1, 1))](A, B, out, m, n, k)
    return out
