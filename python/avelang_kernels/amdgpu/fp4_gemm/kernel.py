"""AMDGPU FP4 GEMM device kernel and pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import avelang
import avelang.language as al

from .config import FP4GemmConfig, U32_BYTES
from .memory_ops import FP4MemoryConfig, advance_global_ptr, load_global, store_shm
from .reduce import fp4_reduction_shm_u32, reduce_k
from .solution import SolutionId
from .warp_schedule import prefetch, pipeline_compute
from .writeback import FP4WritebackConfig, write_results

SCHED_MASK_MFMA = 0x8
SCHED_MASK_BUFFER_LOAD = 0x20
SCHED_MASK_DS_READ = 0x100
SCHED_MASK_DS_WRITE = 0x200


@dataclass(frozen=True, slots=True)
class FP4KernelConfig:
    gemm: FP4GemmConfig
    memory: FP4MemoryConfig
    writeback: FP4WritebackConfig
    shm_u32: int
    pipeline_global_loads: int
    pipeline_mfmas_per_load: int

    @classmethod
    @cache
    def from_solution(cls, solution: SolutionId, arch: str = "gfx942") -> "FP4KernelConfig":
        gemm = FP4GemmConfig.from_solution(solution, arch)
        memory = FP4MemoryConfig.from_config(gemm)
        writeback = FP4WritebackConfig.from_config(gemm)
        shm_u32 = max(memory.pipeline_shm_u32, fp4_reduction_shm_u32(gemm), writeback.shm_result_u32)
        if shm_u32 * U32_BYTES > gemm.max_shm_bytes:
            raise ValueError(f"solution {int(solution):#x} requires {shm_u32 * U32_BYTES} bytes of LDS")
        global_loads = memory.a_loads + memory.b_loads + memory.scale_loads
        mfmas_per_load = gemm.m_tiles * gemm.n_tiles * (gemm.group_k // gemm.tile // gemm.warp_k) // global_loads
        mfma_batch = 4 if mfmas_per_load > 12 else 2 if mfmas_per_load > 6 else 1
        return cls(gemm, memory, writeback, shm_u32, global_loads, mfma_batch)


@avelang.jit
def hot_loop_scheduler(config: al.constexpr):
    al.amdgpu.sched_group_barrier(SCHED_MASK_DS_READ, config.memory.a_loads + 2, 0)
    for _ in al.range(config.pipeline_global_loads):
        al.amdgpu.sched_group_barrier(SCHED_MASK_DS_WRITE, 1, 0)
        al.amdgpu.sched_group_barrier(
            SCHED_MASK_MFMA, config.pipeline_mfmas_per_load, 0
        )
    for _ in al.range(config.pipeline_global_loads):
        al.amdgpu.sched_group_barrier(SCHED_MASK_BUFFER_LOAD, 1, 0)
        al.amdgpu.sched_group_barrier(
            SCHED_MASK_MFMA, config.pipeline_mfmas_per_load * 2, 0
        )
    al.amdgpu.sched_barrier(0)


@avelang.jit
def fp4_gemm_kernel(
    config: al.constexpr,
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
    wid = tid // config.gemm.warp_size
    wtid = tid % config.gemm.warp_size
    warp_k = wid // config.gemm.warp_n // config.gemm.warp_m
    warp_n = wid % config.gemm.warp_n
    warp_m = (wid // config.gemm.warp_n) % config.gemm.warp_m

    group_m = al.block_id(0)
    group_n = al.block_id(1)
    alpha = global_scale[0] * al.convert(config.gemm.global_scale_factor, al.f32)

    block_row = group_m * config.gemm.group_m
    valid_m = m - block_row
    if valid_m > config.gemm.group_m:
        valid_m = al.convert(config.gemm.group_m, al.u32)

    # Like Petit, descriptor bases advance with K while ranges cover one tile.
    a_tensor = al.make_tensor(A, al.u16, al.make_layout((m * k,), (1,)))
    b_tensor = al.make_tensor(
        B, al.u32, al.make_layout((n * k // config.gemm.pack_factor,), (1,))
    )
    s_tensor = al.make_tensor(
        scales, al.u8, al.make_layout((n * k // config.gemm.scale_group_size,), (1,))
    )
    c_tensor = al.make_tensor(C, al.u16, al.make_layout((m * n,), (1,)))

    # Subview offsets/extents are in elements; resource ranges are in bytes.
    a_block_elements = (valid_m - 1) * k + config.gemm.group_k
    a_block = al.subview(
        a_tensor, (block_row * k,), (a_block_elements,), (1,)
    )
    b_block_elements = (
        ((config.gemm.group_k // config.gemm.layout_k - 1) * n + config.gemm.group_n) * config.gemm.layout_k // config.gemm.pack_factor
    )
    b_block = al.subview(
        b_tensor, (group_n * config.gemm.group_n * config.gemm.layout_k // config.gemm.pack_factor,),
        (b_block_elements,), (1,),
    )
    scale_block_elements = (config.gemm.group_k // config.gemm.scale_group_size - 1) * n + config.gemm.group_n
    scale_block = al.subview(
        s_tensor, (group_n * config.gemm.group_n * config.gemm.layout_k // config.gemm.scale_group_size,),
        (scale_block_elements,), (1,),
    )
    c_block_elements = (valid_m - 1) * n + config.gemm.group_n
    c_block = al.subview(
        c_tensor, (block_row * n + group_n * config.gemm.group_n,),
        (c_block_elements,), (1,),
    )
    a_rsrc = al.make_local((1, 4), al.u32)
    b_rsrc = al.make_local((1, 4), al.u32)
    scale_rsrc = al.make_local((1, 4), al.u32)
    a_rsrc[0] = al.amdgpu.make_rsrc(a_block, a_block_elements * config.gemm.element_a_bytes)
    b_rsrc[0] = al.amdgpu.make_rsrc(b_block, b_block_elements * U32_BYTES)
    scale_rsrc[0] = al.amdgpu.make_rsrc(scale_block, scale_block_elements)
    c_rsrc = al.amdgpu.make_rsrc(c_block, c_block_elements * config.gemm.output_bytes)

    shm = al.make_shared((config.shm_u32,), al.u32)
    reg_a = al.make_local((config.gemm.pipeline_stages, config.memory.a_loads, 4), al.u32)
    reg_b = al.make_local((config.gemm.pipeline_stages, config.memory.b_loads, 4), al.u32)
    reg_scale = al.make_local((config.gemm.pipeline_stages, config.memory.scale_loads, 4), al.u32)
    data_a = al.make_local((config.gemm.m_tiles, config.gemm.read_batch_a, 4), al.u32)
    qword = al.make_local((1, 4), al.u32)
    packed_scale = al.make_local((1, 1), al.u16)
    data_b = al.make_local((1, 4), al.u32)
    acc = al.make_local((config.gemm.m_tiles, config.gemm.n_tiles, 4), al.f32)

    for tile_m in al.range(config.gemm.m_tiles):
        for tile_n in al.range(config.gemm.n_tiles):
            for acc_idx in al.range(config.gemm.accum_values):
                acc[tile_m, tile_n, acc_idx] = al.convert(0.0, al.f32)

    k_total = k // config.gemm.group_k
    load_global(
        config, a_rsrc[0], b_rsrc[0], scale_rsrc[0], n, k, tid,
        reg_a[0], reg_b[0], reg_scale[0],
    )
    al.amdgpu.sched_barrier(0x7DF)
    al.amdgpu.s_waitcnt(0, 7, 15)
    store_shm(config, shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid)

    if k_total > 1:
        advance_global_ptr(config, a_rsrc[0], b_rsrc[0], scale_rsrc[0], n)
        al.syncthreads()
        load_global(
            config, a_rsrc[0], b_rsrc[0], scale_rsrc[0], n, k, tid,
            reg_a[1], reg_b[1], reg_scale[1],
        )

    k_idx = al.convert(0, al.u32)
    while k_idx + 3 < k_total:
        al.amdgpu.sched_barrier(0)
        advance_global_ptr(config, a_rsrc[0], b_rsrc[0], scale_rsrc[0], n)
        al.syncthreads()

        prefetch(
            config, shm, 0, data_a,
            qword, packed_scale, warp_m, warp_n, warp_k, wtid,
        )
        load_global(
            config, a_rsrc[0], b_rsrc[0], scale_rsrc[0], n, k, tid,
            reg_a[0], reg_b[0], reg_scale[0],
        )
        if config.memory.single_buffer:
            pipeline_compute(
                config, shm, 0, data_a,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )
            al.syncthreads()
            store_shm(
                config, shm, 0, reg_a[1], reg_b[1], reg_scale[1], tid
            )
        else:
            store_shm(
                config, shm, 1, reg_a[1], reg_b[1], reg_scale[1], tid
            )
            pipeline_compute(
                config, shm, 0, data_a,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )
        hot_loop_scheduler(config)

        advance_global_ptr(config, a_rsrc[0], b_rsrc[0], scale_rsrc[0], n)
        al.syncthreads()
        prefetch(
            config, shm, config.memory.second_shm_stage, data_a,
            qword, packed_scale, warp_m, warp_n, warp_k, wtid,
        )
        load_global(
            config, a_rsrc[0], b_rsrc[0], scale_rsrc[0], n, k, tid,
            reg_a[1], reg_b[1], reg_scale[1],
        )
        if config.memory.single_buffer:
            pipeline_compute(
                config, shm, 0, data_a,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )
            al.syncthreads()
            store_shm(
                config, shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid
            )
        else:
            store_shm(
                config, shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid
            )
            pipeline_compute(
                config, shm, 1, data_a,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )
        hot_loop_scheduler(config)
        k_idx = k_idx + 2

    # Epilogue: statically unroll up to three remaining K tiles.
    advance_global_ptr(config, a_rsrc[0], b_rsrc[0], scale_rsrc[0], n)
    al.syncthreads()
    prefetch(
        config, shm, 0, data_a,
        qword, packed_scale, warp_m, warp_n, warp_k, wtid,
    )
    if k_idx + 2 < k_total:
        load_global(
            config, a_rsrc[0], b_rsrc[0], scale_rsrc[0], n, k, tid,
            reg_a[0], reg_b[0], reg_scale[0],
        )
    if config.memory.single_buffer:
        pipeline_compute(
            config, shm, 0, data_a,
            qword, packed_scale, data_b,
            warp_m, warp_n, warp_k, wtid, acc,
        )
        if k_idx + 1 < k_total:
            al.syncthreads()
            store_shm(
                config, shm, 0, reg_a[1], reg_b[1], reg_scale[1], tid
            )
    else:
        if k_idx + 1 < k_total:
            store_shm(
                config, shm, 1, reg_a[1], reg_b[1], reg_scale[1], tid
            )
        pipeline_compute(
            config, shm, 0, data_a,
            qword, packed_scale, data_b,
            warp_m, warp_n, warp_k, wtid, acc,
        )

    if k_idx + 1 < k_total:
        advance_global_ptr(config, a_rsrc[0], b_rsrc[0], scale_rsrc[0], n)
        al.syncthreads()
        prefetch(
            config, shm, config.memory.second_shm_stage, data_a,
            qword, packed_scale, warp_m, warp_n, warp_k, wtid,
        )
        if config.memory.single_buffer:
            pipeline_compute(
                config, shm, 0, data_a,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )
            if k_idx + 2 < k_total:
                al.syncthreads()
                store_shm(
                    config, shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid
                )
        else:
            if k_idx + 2 < k_total:
                store_shm(
                    config, shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid
                )
            pipeline_compute(
                config, shm, 1, data_a,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )

    if k_idx + 2 < k_total:
        advance_global_ptr(config, a_rsrc[0], b_rsrc[0], scale_rsrc[0], n)
        al.syncthreads()
        prefetch(
            config, shm, 0, data_a,
            qword, packed_scale, warp_m, warp_n, warp_k, wtid,
        )
        pipeline_compute(
            config, shm, 0, data_a,
            qword, packed_scale, data_b,
            warp_m, warp_n, warp_k, wtid, acc,
        )

    # Unlike Petit, keep a barrier before reusing the LDS union: result
    # stores can overlap another warp's final A reads in single-buffer tiles.
    al.syncthreads()
    if config.gemm.warp_k > 1:
        reduce_k(config, shm, warp_m, warp_n, warp_k, wtid, acc)
    write_results(
        config, c_rsrc, shm, n, warp_m, warp_n, warp_k,
        wtid, tid, alpha, acc,
    )
