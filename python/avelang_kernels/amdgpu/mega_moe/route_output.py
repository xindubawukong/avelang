"""System publication and source-owned BF16 route reduction in FP32."""

from functools import cache

import avelang
import avelang.language as al

from .synchronization import make_grid_barrier, wait_epoch
from .workspace import WorkspaceLayout


@cache
def make_route_reduce(config, blocks, waves):
    layout = WorkspaceLayout(config)
    R, TOPK, D = config.solution.world_size, config.solution.topk, config.solution.hidden
    B, SLOT, OUT = layout.rank_sym_buffer_base, layout.rank_slot_bytes, layout.route_output
    VECS, BLOCKS, WAVES, TOTAL_WAVES = D // 8, blocks, waves, blocks * waves

    @avelang.jit
    def reduce_routes(
        resource: al.Tensor((4,), al.u32),
        out: al.Pointer(al.u32),
        tokens: al.u32,
        stride: al.u32,
        rank: al.u32,
        block: al.u32,
        tid: al.u32,
    ):
        wave, lane = tid // 64, tid % 64
        if tokens != 0:
            global_wave = block * WAVES + wave
            if R == 1 and tokens == 8:
                global_wave = wave * BLOCKS + block
            waves_per_token = (TOTAL_WAVES + tokens - 1) // tokens if R > 1 else (VECS + 63) // 64
            vecs_per_wave = (VECS + waves_per_token - 1) // waves_per_token if R > 1 else al.convert(64, al.u32)
            output = al.make_tensor(out, al.u32, al.make_layout((tokens, stride // 8, 4), (stride // 2, 4, 1)))
            for task in al.range(global_wave, tokens * waves_per_token, TOTAL_WAVES):
                token, wave_in_token = task // waves_per_token, task % waves_per_token
                for vec_in_wave in al.range(lane, vecs_per_wave, 64):
                    col = wave_in_token * vecs_per_wave + vec_in_wave
                    if col < VECS:
                        values = al.make_local((TOPK, 4), al.u32)
                        for slot in al.static_range(TOPK):
                            offset = al.amdgpu.readfirstlane(
                                al.convert(B + rank * SLOT + OUT + (token * TOPK + slot) * D * 2, al.u32)
                            )
                            values[slot] = al.amdgpu.raw_buffer_load_x4(
                                resource, al.convert(col * 16, al.u32), offset, 18
                            )
                        bf = al.view(values, al.bf16, al.make_layout((TOPK, 4, 2), (8, 2, 1)))
                        accum = al.full((4, 2), 0, al.f32)
                        for slot in al.static_range(TOPK):
                            for pair in al.static_range(4):
                                accum[pair] = accum[pair] + al.convert(bf[slot, pair], al.f32)
                        result = al.make_local((4, 2), al.bf16)
                        for pair in al.static_range(4):
                            result[pair] = al.convert(accum[pair], al.bf16)
                        packed = al.view(result, al.u32, al.make_layout((4,), (1,)))
                        output[token, col] = packed

    return reduce_routes


@cache
def make_combine_kernel(config):
    layout = WorkspaceLayout(config)
    R = config.solution.world_size
    SIZE, C = layout.workspace_bytes, layout.barrier_record_bytes
    GATE, GRID = layout.epoch_gate, layout.grid_sync + 4 * 4
    grid_barrier = make_grid_barrier(128, True)
    reduce_routes = make_route_reduce(config, 128, 8)

    @avelang.jit
    def combine(heap: al.Pointer(al.u8), out: al.Pointer(al.u32), tokens: al.u32, stride: al.u32, rank: al.u32):
        tid, block = al.convert(al.thread_id(0), al.u32), al.convert(al.block_id(0), al.u32)
        memory = al.make_tensor(heap, al.u8, al.make_layout((SIZE,), (1,)))
        resource = al.amdgpu.make_rsrc(memory, SIZE)
        epoch = al.amdgpu.raw_buffer_load_x1(resource, rank * C + GATE, 0, 17)
        grid_barrier(resource, al.convert(GRID, al.u32), block, tid)
        if tid < 64:
            if block == 0:
                al.amdgpu.fence(0, 2)
                if tid < R:
                    al.amdgpu.fence(1, 2)
                    al.amdgpu.raw_buffer_store_x1(epoch, resource, tid * C + 256 + rank * 4, 0, 17)
                al.amdgpu.s_waitcnt(0, 0, 0)
            if tid < R:
                wait_epoch(resource, rank * C + 256 + tid * 4, epoch)
        al.syncthreads()
        reduce_routes(resource, out, tokens, stride, rank, block, tid)

    return combine
