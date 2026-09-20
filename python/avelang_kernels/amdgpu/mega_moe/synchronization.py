"""Scoped publication and epoch polling used by the MegaMoE device protocol."""

from functools import cache

import avelang
import avelang.language as al


@avelang.jit
def complete_stores():
    al.amdgpu.compiler_barrier()
    al.amdgpu.s_waitcnt(0, 0, 0)
    al.amdgpu.fence(1, 2)
    al.amdgpu.compiler_barrier()


@avelang.jit
def wait_equal(resource: al.Tensor((4,), al.u32), offset: al.u32, expected: al.u32):
    observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)
    while observed != expected:
        al.amdgpu.compiler_barrier()
        observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)

    al.amdgpu.fence(0, 2)


@avelang.jit
def wait_epoch(resource: al.Tensor((4,), al.u32), offset: al.u32, expected: al.u32):
    observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)
    while al.bitcast(observed - expected, al.i32) < 0:
        al.amdgpu.compiler_barrier()
        observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)

    al.amdgpu.fence(0, 2)


@avelang.jit
def wait_mask(resource: al.Tensor((4,), al.u32), offset: al.u32, mask: al.u32):
    observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)
    while (observed & mask) != mask:
        al.amdgpu.s_sleep(1)
        observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)

    al.amdgpu.fence(0, 2)


@cache
def make_global_barrier(config):
    R = config.solution.world_size
    from .workspace import WorkspaceLayout

    layout = WorkspaceLayout(config)
    SIZE, RECORD = layout.workspace_bytes, layout.barrier_record_bytes

    @avelang.jit
    def global_barrier(heap: al.Pointer(al.u8), rank: al.u32):
        tid = al.convert(al.thread_id(0), al.u32)
        memory = al.make_tensor(heap, al.u8, al.make_layout((SIZE,), (1,)))
        resource = al.amdgpu.make_rsrc(memory, SIZE)
        epoch_shared = al.make_shared((2,), al.u32)
        if tid == 0:
            next_epoch = al.amdgpu.raw_buffer_load_x1(resource, rank * RECORD + 128, 0, 17) + 1
            al.amdgpu.raw_buffer_store_x1(next_epoch, resource, rank * RECORD + 128, 0, 17)
            epoch_shared[0] = next_epoch
        al.syncthreads()
        epoch = epoch_shared[0]
        if tid < R:
            al.amdgpu.fence(1, 2)
            al.amdgpu.raw_buffer_store_x1(epoch, resource, tid * RECORD + rank * 4, 0, 17)
            al.amdgpu.s_waitcnt(0, 0, 0)
            observed = al.amdgpu.raw_buffer_load_x1(resource, rank * RECORD + tid * 4, 0, 17)
            while al.bitcast(observed - epoch, al.i32) < 0:
                al.amdgpu.s_sleep(1)
                observed = al.amdgpu.raw_buffer_load_x1(resource, rank * RECORD + tid * 4, 0, 17)
            al.amdgpu.fence(0, 2)
        al.syncthreads()

    return global_barrier
