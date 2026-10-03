"""AMDGPU FP4 GEMM device kernel and pipeline."""

from __future__ import annotations

from functools import cache

import avelang
import avelang.language as al

from .config import (
    ACCUM_VALUES,
    LAYOUT_K,
    MAX_SHM_BYTES,
    PACK_FACTOR,
    PIPELINE_STAGES,
    TILE,
    U16_BYTES,
    U32_BYTES,
    WARP_SIZE,
)
from .memory_ops import make_memory_ops
from .reduce import make_reduction
from .warp_schedule import make_warp_schedule
from .writeback import make_writeback_ops

SCHED_MASK_MFMA = 0x8
SCHED_MASK_BUFFER_LOAD = 0x20
SCHED_MASK_DS_READ = 0x100
SCHED_MASK_DS_WRITE = 0x200


@cache
def make_kernel(config):
    memory, advance_global_ptr, load_global, store_shm, read_shm_a, read_shm_b = make_memory_ops(config)
    result_shm_u32, write_results = make_writeback_ops(config)
    prefetch, pipeline_compute = make_warp_schedule(config, memory, read_shm_a, read_shm_b)
    reduction_shm_u32, reduce_k = make_reduction(config)
    shm_words = max(memory.shm_u32, reduction_shm_u32, result_shm_u32)
    if shm_words * U32_BYTES > MAX_SHM_BYTES:
        raise ValueError(f"configuration requires {shm_words * U32_BYTES} bytes of LDS")
    SHM_U32 = al.constexpr(shm_words)
    SINGLE_BUFFER = al.constexpr(memory.shm_stages == 1)
    global_loads = memory.a_loads + memory.b_loads + memory.scale_loads
    mfmas_per_load = config.m_tiles * config.n_tiles * (config.group_k // TILE // config.warp_k) // global_loads
    MFMAS_PER_LOAD = al.constexpr(4 if mfmas_per_load > 12 else 2 if mfmas_per_load > 6 else 1)
    GLOBAL_LOADS = al.constexpr(global_loads)
    GROUP_M = al.constexpr(config.group_m)
    GROUP_N = al.constexpr(config.group_n)
    GROUP_K = al.constexpr(config.group_k)
    WARP_M = al.constexpr(config.warp_m)
    WARP_N = al.constexpr(config.warp_n)
    WARP_K = al.constexpr(config.warp_k)
    M_TILES = al.constexpr(config.m_tiles)
    N_TILES = al.constexpr(config.n_tiles)
    ELEMENT_A_BYTES = al.constexpr(config.element_a_bytes)
    SCALE_GROUP_SIZE = al.constexpr(config.scale_group_size)
    GLOBAL_SCALE_FACTOR = al.constexpr(config.global_scale_factor)
    READ_BATCH_A = al.constexpr(config.read_batch_a)
    A_LOADS = al.constexpr(memory.a_loads)
    B_LOADS = al.constexpr(memory.b_loads)
    SCALE_LOADS = al.constexpr(memory.scale_loads)
    PIPELINE_SHM_U32 = al.constexpr(memory.shm_u32)
    RESULT_SHM_U32 = al.constexpr(result_shm_u32)
    REDUCTION_SHM_U32 = al.constexpr(reduction_shm_u32)

    @avelang.jit
    def hot_loop_scheduler():
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, A_LOADS + 2, 0)
        for _ in al.range(GLOBAL_LOADS):
            al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
            al.amdgpu.sched_group_barrier(
                SCHED_MASK_MFMA, MFMAS_PER_LOAD, 0
            )
        for _ in al.range(GLOBAL_LOADS):
            al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
            al.amdgpu.sched_group_barrier(
                SCHED_MASK_MFMA, MFMAS_PER_LOAD * 2, 0
            )
        al.amdgpu.sched_barrier(0)

    @avelang.jit
    def fp4_gemm_kernel(
        A: al.Pointer(al.u16),
        B: al.Pointer(al.u32),
        scales: al.Pointer(al.u8),
        global_scale: al.Tensor((1,), al.f32),
        C: al.Pointer(al.u16),
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
        alpha = global_scale[0] * al.convert(GLOBAL_SCALE_FACTOR, al.f32)

        block_row = group_m * GROUP_M
        valid_m = m - block_row
        if valid_m > GROUP_M:
            valid_m = al.convert(GROUP_M, al.u32)

        # Like Petit, descriptor bases advance with K while ranges cover one tile.
        a_tensor = al.make_tensor(A, al.u16, al.make_layout((m * k,), (1,)))
        b_tensor = al.make_tensor(
            B, al.u32, al.make_layout((n * k // PACK_FACTOR,), (1,))
        )
        s_tensor = al.make_tensor(
            scales, al.u8, al.make_layout((n * k // SCALE_GROUP_SIZE,), (1,))
        )
        c_tensor = al.make_tensor(C, al.u16, al.make_layout((m * n,), (1,)))

        # Subview offsets/extents are in elements; resource ranges are in bytes.
        a_block_elements = (valid_m - 1) * k + GROUP_K
        a_block = al.subview(
            a_tensor, (block_row * k,), (a_block_elements,), (1,)
        )
        b_block_elements = (
            ((GROUP_K // LAYOUT_K - 1) * n + GROUP_N) * LAYOUT_K // PACK_FACTOR
        )
        b_block = al.subview(
            b_tensor, (group_n * GROUP_N * LAYOUT_K // PACK_FACTOR,),
            (b_block_elements,), (1,),
        )
        scale_block_elements = (GROUP_K // SCALE_GROUP_SIZE - 1) * n + GROUP_N
        scale_block = al.subview(
            s_tensor, (group_n * GROUP_N * LAYOUT_K // SCALE_GROUP_SIZE,),
            (scale_block_elements,), (1,),
        )
        c_block_elements = (valid_m - 1) * n + GROUP_N
        c_block = al.subview(
            c_tensor, (block_row * n + group_n * GROUP_N,),
            (c_block_elements,), (1,),
        )
        a_rsrc = al.make_local((1, 4), al.u32)
        b_rsrc = al.make_local((1, 4), al.u32)
        scale_rsrc = al.make_local((1, 4), al.u32)
        a_rsrc[0] = al.amdgpu.make_rsrc(a_block, a_block_elements * ELEMENT_A_BYTES)
        b_rsrc[0] = al.amdgpu.make_rsrc(b_block, b_block_elements * U32_BYTES)
        scale_rsrc[0] = al.amdgpu.make_rsrc(scale_block, scale_block_elements)
        c_rsrc = al.amdgpu.make_rsrc(c_block, c_block_elements * U16_BYTES)

        shm = al.make_shared((SHM_U32,), al.u32)
        pipeline_shm = al.subview(shm, (0,), (PIPELINE_SHM_U32,), (1,))
        result_shm = al.subview(shm, (0,), (RESULT_SHM_U32,), (1,))
        reg_a = al.make_local((PIPELINE_STAGES, A_LOADS, 4), al.u32)
        reg_b = al.make_local((PIPELINE_STAGES, B_LOADS, 4), al.u32)
        reg_scale = al.make_local((PIPELINE_STAGES, SCALE_LOADS, 4), al.u32)
        data_a = al.make_local((M_TILES, READ_BATCH_A, 4), al.u32)
        qword = al.make_local((1, 4), al.u32)
        packed_scale = al.make_local((1, 1), al.u16)
        data_b = al.make_local((1, 4), al.u32)
        acc = al.make_local((M_TILES, N_TILES, 4), al.f32)

        for tile_m in al.range(M_TILES):
            for tile_n in al.range(N_TILES):
                for acc_idx in al.range(ACCUM_VALUES):
                    acc[tile_m, tile_n, acc_idx] = al.convert(0.0, al.f32)

        k_total = k // GROUP_K
        second_shm_stage = 0 if SINGLE_BUFFER else 1
        load_global(
            a_rsrc[0], b_rsrc[0], scale_rsrc[0], n, k, tid,
            reg_a[0], reg_b[0], reg_scale[0],
        )
        al.amdgpu.sched_barrier(0x7DF)
        al.amdgpu.s_waitcnt(0, 7, 15)
        store_shm(pipeline_shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid)

        if k_total > 1:
            advance_global_ptr(a_rsrc[0], b_rsrc[0], scale_rsrc[0], n)
            al.syncthreads()
            load_global(
                a_rsrc[0], b_rsrc[0], scale_rsrc[0], n, k, tid,
                reg_a[1], reg_b[1], reg_scale[1],
            )

        k_idx = al.convert(0, al.u32)
        while k_idx + 3 < k_total:
            al.amdgpu.sched_barrier(0)
            advance_global_ptr(a_rsrc[0], b_rsrc[0], scale_rsrc[0], n)
            al.syncthreads()

            prefetch(
                pipeline_shm, 0, data_a,
                qword, packed_scale, warp_m, warp_n, warp_k, wtid,
            )
            load_global(
                a_rsrc[0], b_rsrc[0], scale_rsrc[0], n, k, tid,
                reg_a[0], reg_b[0], reg_scale[0],
            )
            if SINGLE_BUFFER:
                pipeline_compute(
                    pipeline_shm, 0, data_a,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )
                al.syncthreads()
                store_shm(
                    pipeline_shm, 0, reg_a[1], reg_b[1], reg_scale[1], tid
                )
            else:
                store_shm(
                    pipeline_shm, 1, reg_a[1], reg_b[1], reg_scale[1], tid
                )
                pipeline_compute(
                    pipeline_shm, 0, data_a,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )
            hot_loop_scheduler()

            advance_global_ptr(a_rsrc[0], b_rsrc[0], scale_rsrc[0], n)
            al.syncthreads()
            prefetch(
                pipeline_shm, second_shm_stage, data_a,
                qword, packed_scale, warp_m, warp_n, warp_k, wtid,
            )
            load_global(
                a_rsrc[0], b_rsrc[0], scale_rsrc[0], n, k, tid,
                reg_a[1], reg_b[1], reg_scale[1],
            )
            if SINGLE_BUFFER:
                pipeline_compute(
                    pipeline_shm, 0, data_a,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )
                al.syncthreads()
                store_shm(
                    pipeline_shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid
                )
            else:
                store_shm(
                    pipeline_shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid
                )
                pipeline_compute(
                    pipeline_shm, 1, data_a,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )
            hot_loop_scheduler()
            k_idx = k_idx + 2

        # Epilogue: statically unroll up to three remaining K tiles.
        advance_global_ptr(a_rsrc[0], b_rsrc[0], scale_rsrc[0], n)
        al.syncthreads()
        prefetch(
            pipeline_shm, 0, data_a,
            qword, packed_scale, warp_m, warp_n, warp_k, wtid,
        )
        if k_idx + 2 < k_total:
            load_global(
                a_rsrc[0], b_rsrc[0], scale_rsrc[0], n, k, tid,
                reg_a[0], reg_b[0], reg_scale[0],
            )
        if SINGLE_BUFFER:
            pipeline_compute(
                pipeline_shm, 0, data_a,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )
            if k_idx + 1 < k_total:
                al.syncthreads()
                store_shm(
                    pipeline_shm, 0, reg_a[1], reg_b[1], reg_scale[1], tid
                )
        else:
            if k_idx + 1 < k_total:
                store_shm(
                    pipeline_shm, 1, reg_a[1], reg_b[1], reg_scale[1], tid
                )
            pipeline_compute(
                pipeline_shm, 0, data_a,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )

        if k_idx + 1 < k_total:
            advance_global_ptr(a_rsrc[0], b_rsrc[0], scale_rsrc[0], n)
            al.syncthreads()
            prefetch(
                pipeline_shm, second_shm_stage, data_a,
                qword, packed_scale, warp_m, warp_n, warp_k, wtid,
            )
            if SINGLE_BUFFER:
                pipeline_compute(
                    pipeline_shm, 0, data_a,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )
                if k_idx + 2 < k_total:
                    al.syncthreads()
                    store_shm(
                        pipeline_shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid
                    )
            else:
                if k_idx + 2 < k_total:
                    store_shm(
                        pipeline_shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid
                    )
                pipeline_compute(
                    pipeline_shm, 1, data_a,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )

        if k_idx + 2 < k_total:
            advance_global_ptr(a_rsrc[0], b_rsrc[0], scale_rsrc[0], n)
            al.syncthreads()
            prefetch(
                pipeline_shm, 0, data_a,
                qword, packed_scale, warp_m, warp_n, warp_k, wtid,
            )
            pipeline_compute(
                pipeline_shm, 0, data_a,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )

        # Unlike Petit, keep a barrier before reusing the LDS union: result
        # stores can overlap another warp's final A reads in single-buffer tiles.
        al.syncthreads()
        if WARP_K > 1:
            reduction_shm = al.subview(
                shm, (0,),
                (REDUCTION_SHM_U32,), (1,),
            )
            reduce_k(reduction_shm, warp_m, warp_n, warp_k, wtid, acc)
        write_results(
            c_rsrc, result_shm, n, warp_m, warp_n, warp_k,
            wtid, tid, alpha, acc,
        )

    return fp4_gemm_kernel
