"""Scoped publication and epoch polling used by the MegaMoE device protocol."""

from functools import cache

import avelang
import avelang.language as al


@avelang.jit
def complete_stores():
    al.amdgpu.compiler_barrier()
    al.amdgpu.s_waitcnt(0, 0, 0)
    al.amdgpu.compiler_barrier()


@avelang.jit
def wait_equal(resource: al.Tensor((4,), al.u32), offset: al.u32, expected: al.u32):
    observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)
    while observed != expected:
        al.amdgpu.compiler_barrier()
        observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)


@avelang.jit
def wait_epoch(resource: al.Tensor((4,), al.u32), offset: al.u32, expected: al.u32):
    observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)
    while al.bitcast(observed - expected, al.i32) < 0:
        al.amdgpu.compiler_barrier()
        observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)


@avelang.jit
def wait_mask(resource: al.Tensor((4,), al.u32), offset: al.u32, mask: al.u32):
    observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)
    while (observed & mask) != mask:
        al.amdgpu.s_sleep(1)
        observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)


@cache
def make_grid_barrier(blocks, system):
    SCOPE, AUX = (2, 16) if system else (1, 0)

    @avelang.jit
    def grid_barrier(resource: al.Tensor((4,), al.u32), offset: al.u32, block: al.u32, tid: al.u32):
        al.syncthreads()
        if tid == 0:
            delta = 1 + ((block - 1) >> 31) * (0x80000000 - blocks)
            al.amdgpu.fence(1, SCOPE)
            old = al.amdgpu.raw_buffer_atomic_add_u32(delta, resource, offset, 0, AUX)
            observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)
            while ((observed ^ old) & 0x80000000) == 0:
                al.amdgpu.s_sleep(1)
                observed = al.amdgpu.raw_buffer_load_x1(resource, offset, 0, 17)
            al.amdgpu.compiler_barrier()
        al.syncthreads()

    return grid_barrier
