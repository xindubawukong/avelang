"""AMDGPU FP4 GEMM device kernel and pipeline."""

import avelang
import avelang.language as al

from .config import FP4GemmConfig, U32_BYTES
from .memory_ops import FP4MemoryConfig, make_fp4_memory_ops
from .solution import SolutionId
from .reduce import fp4_reduction_shm_u32, make_fp4_reduce
from .warp_schedule import make_fp4_warp_schedule
from .writeback import FP4WritebackConfig, make_fp4_writeback

SCHED_MASK_MFMA = 0x8
SCHED_MASK_BUFFER_LOAD = 0x20
SCHED_MASK_DS_READ = 0x100
SCHED_MASK_DS_WRITE = 0x200


def _make_fp4_gemm_kernel(solution: SolutionId, arch: str = "gfx942"):
    config = FP4GemmConfig.from_solution(solution, arch)
    GLOBAL_SCALE_FACTOR = config.global_scale_factor
    GROUP_M = config.group_m
    GROUP_N = config.group_n
    GROUP_K = config.group_k
    WARP_M = config.warp_m
    WARP_N = config.warp_n
    WARP_K = config.warp_k
    WARP_SIZE = config.warp_size
    M_TILES = config.m_tiles
    N_TILES = config.n_tiles
    READ_BATCH_A = config.read_batch_a
    TILE = config.tile
    LAYOUT_K = config.layout_k
    PACK_FACTOR = config.pack_factor
    GROUP_SIZE = config.group_size
    PIPELINE_STAGES = config.pipeline_stages
    ACCUM_VALUES = config.accumulator_shape[2]
    ELEMENT_A_BYTES = config.element_a_bytes
    OUTPUT_BYTES = config.output_bytes

    memory = FP4MemoryConfig.from_config(config)
    A_LOADS = memory.a_loads
    B_LOADS = memory.b_loads
    SCALE_LOADS = memory.scale_loads
    SINGLE_BUFFER = memory.single_buffer
    SECOND_SHM_STAGE = memory.second_shm_stage
    SHM_REDUCTION_U32 = fp4_reduction_shm_u32(config)
    writeback = FP4WritebackConfig.from_config(config)
    SHM_U32 = max(
        memory.pipeline_shm_u32,
        SHM_REDUCTION_U32,
        writeback.shm_result_u32,
    )
    if SHM_U32 * U32_BYTES > config.max_shm_bytes:
        raise ValueError(
            f"solution {int(solution):#x} requires "
            f"{SHM_U32 * U32_BYTES} bytes of LDS"
        )
    (
        _advance_global_offsets, _load_global, _store_shm,
        _read_shm_a, _read_shm_b,
    ) = make_fp4_memory_ops(memory, SHM_U32)
    _prefetch, _pipeline_compute = make_fp4_warp_schedule(
        config, SHM_U32, _read_shm_a, _read_shm_b
    )
    _reduce_k = make_fp4_reduce(config, SHM_U32)
    _write_results = make_fp4_writeback(writeback, SHM_U32)

    PIPELINE_GLOBAL_LOADS = A_LOADS + B_LOADS + SCALE_LOADS
    PIPELINE_MFMAS = M_TILES * N_TILES * (GROUP_K // TILE // WARP_K)
    PIPELINE_MFMAS_PER_LOAD = (
        4
        if PIPELINE_MFMAS // PIPELINE_GLOBAL_LOADS > 12
        else 2
        if PIPELINE_MFMAS // PIPELINE_GLOBAL_LOADS > 6
        else 1
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
    def _kernel(
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

        # Descriptor bases stay fixed; ranges span this block across all K tiles.
        a_tensor = al.make_tensor(A, al.u16, al.make_layout((m * k,), (1,)))
        b_tensor = al.make_tensor(
            B, al.u32, al.make_layout((n * k // PACK_FACTOR,), (1,))
        )
        s_tensor = al.make_tensor(
            scales, al.u8, al.make_layout((n * k // GROUP_SIZE,), (1,))
        )
        c_tensor = al.make_tensor(C, al.u16, al.make_layout((m * n,), (1,)))

        # Subview offsets/extents are in elements; resource ranges are in bytes.
        a_block_elements = valid_m * k
        a_block = al.subview(
            a_tensor, (block_row * k,), (a_block_elements,), (1,)
        )
        b_block_elements = (
            ((k // LAYOUT_K - 1) * n + GROUP_N) * LAYOUT_K // PACK_FACTOR
        )
        b_block = al.subview(
            b_tensor, (group_n * GROUP_N * LAYOUT_K // PACK_FACTOR,),
            (b_block_elements,), (1,),
        )
        scale_block_elements = (
            ((k // LAYOUT_K - 1) * n + GROUP_N) * LAYOUT_K // GROUP_SIZE
        )
        scale_block = al.subview(
            s_tensor, (group_n * GROUP_N * LAYOUT_K // GROUP_SIZE,),
            (scale_block_elements,), (1,),
        )
        c_block_elements = (valid_m - 1) * n + GROUP_N
        c_block = al.subview(
            c_tensor, (block_row * n + group_n * GROUP_N,),
            (c_block_elements,), (1,),
        )
        a_rsrc = al.amdgpu.make_rsrc(a_block, a_block_elements * ELEMENT_A_BYTES)
        b_rsrc = al.amdgpu.make_rsrc(b_block, b_block_elements * U32_BYTES)
        scale_rsrc = al.amdgpu.make_rsrc(scale_block, scale_block_elements)
        c_rsrc = al.amdgpu.make_rsrc(c_block, c_block_elements * OUTPUT_BYTES)

        shm = al.make_shared((SHM_U32,), al.u32)
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

        a_offset = al.convert(0, al.u32)
        b_offset = al.convert(0, al.u32)
        scale_offset = al.convert(0, al.u32)
        k_total = k // GROUP_K
        _load_global(
            a_rsrc, b_rsrc, scale_rsrc,
            n, k, a_offset, b_offset, scale_offset, tid,
            reg_a[0], reg_b[0], reg_scale[0],
        )
        al.amdgpu.sched_barrier(0x7DF)
        _store_shm(shm, 0, reg_a[0], reg_b[0], reg_scale[0], tid)

        if k_total > 1:
            a_offset, b_offset, scale_offset = _advance_global_offsets(
                a_offset, b_offset, scale_offset, n
            )
            al.syncthreads()
            _load_global(
                a_rsrc, b_rsrc, scale_rsrc,
                n, k, a_offset, b_offset, scale_offset, tid,
                reg_a[1], reg_b[1], reg_scale[1],
            )

        k_idx = al.convert(0, al.u32)
        while k_idx + 3 < k_total:
            al.amdgpu.sched_barrier(0)
            al.syncthreads()

            _prefetch(
                shm, 0, data_a,
                qword, packed_scale, warp_m, warp_n, warp_k, wtid,
            )
            a_offset, b_offset, scale_offset = _advance_global_offsets(
                a_offset, b_offset, scale_offset, n
            )
            _load_global(
                a_rsrc, b_rsrc, scale_rsrc,
                n, k, a_offset, b_offset, scale_offset, tid,
                reg_a[0], reg_b[0], reg_scale[0],
            )
            if SINGLE_BUFFER:
                _pipeline_compute(
                    shm, 0, data_a,
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
                    shm, 0, data_a,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )
            _hot_loop_scheduler()

            al.syncthreads()
            _prefetch(
                shm, SECOND_SHM_STAGE, data_a,
                qword, packed_scale, warp_m, warp_n, warp_k, wtid,
            )
            a_offset, b_offset, scale_offset = _advance_global_offsets(
                a_offset, b_offset, scale_offset, n
            )
            _load_global(
                a_rsrc, b_rsrc, scale_rsrc,
                n, k, a_offset, b_offset, scale_offset, tid,
                reg_a[1], reg_b[1], reg_scale[1],
            )
            if SINGLE_BUFFER:
                _pipeline_compute(
                    shm, 0, data_a,
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
                    shm, 1, data_a,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )
            _hot_loop_scheduler()
            k_idx = k_idx + 2

        # Epilogue: statically unroll up to three remaining K tiles.
        al.syncthreads()
        _prefetch(
            shm, 0, data_a,
            qword, packed_scale, warp_m, warp_n, warp_k, wtid,
        )
        if k_idx + 2 < k_total:
            a_offset, b_offset, scale_offset = _advance_global_offsets(
                a_offset, b_offset, scale_offset, n
            )
            _load_global(
                a_rsrc, b_rsrc, scale_rsrc,
                n, k, a_offset, b_offset, scale_offset, tid,
                reg_a[0], reg_b[0], reg_scale[0],
            )
        if SINGLE_BUFFER:
            _pipeline_compute(
                shm, 0, data_a,
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
                shm, 0, data_a,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )

        if k_idx + 1 < k_total:
            al.syncthreads()
            _prefetch(
                shm, SECOND_SHM_STAGE, data_a,
                qword, packed_scale, warp_m, warp_n, warp_k, wtid,
            )
            if SINGLE_BUFFER:
                _pipeline_compute(
                    shm, 0, data_a,
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
                    shm, 1, data_a,
                    qword, packed_scale, data_b,
                    warp_m, warp_n, warp_k, wtid, acc,
                )

        if k_idx + 2 < k_total:
            al.syncthreads()
            _prefetch(
                shm, 0, data_a,
                qword, packed_scale, warp_m, warp_n, warp_k, wtid,
            )
            _pipeline_compute(
                shm, 0, data_a,
                qword, packed_scale, data_b,
                warp_m, warp_n, warp_k, wtid, acc,
            )

        # All warps must finish reading A/B/scales before the LDS union is
        # overwritten by reduction or result writeback.
        al.syncthreads()
        if WARP_K > 1:
            _reduce_k(shm, warp_m, warp_n, warp_k, wtid, acc)
        _write_results(
            c_rsrc, shm, n, warp_m, warp_n, warp_k,
            wtid, tid, alpha, acc,
        )

    _kernel.__name__ = f"fp4_gemm_{int(solution):012x}"
    return _kernel
