"""Combine source-owned route rows after a full cross-GPU handoff."""

from functools import cache

import avelang
import avelang.language as al

from .workspace import WorkspaceLayout


@cache
def make_combine_kernel(config):
    layout = WorkspaceLayout(config)
    D, K = config.solution.hidden, config.solution.topk
    SIZE, B, SLOT, OUT = (
        layout.workspace_bytes,
        layout.rank_sym_buffer_base,
        layout.rank_slot_bytes,
        layout.route_output,
    )

    @avelang.jit
    def combine(heap: al.Pointer(al.u8), out: al.Pointer(al.bf16), tokens: al.u32, stride: al.u32, rank: al.u32):
        tid, token = al.convert(al.thread_id(0), al.u32), al.convert(al.block_id(0), al.u32)
        memory = al.make_tensor(heap, al.u8, al.make_layout((SIZE,), (1,)))
        resource = al.amdgpu.make_rsrc(memory, SIZE)
        output = al.make_tensor(out, al.bf16, al.make_layout((tokens, D // 8, 8), (stride, 8, 1)))
        if token < tokens:
            for vector in al.range(tid, D // 8, 256):
                accum = al.make_local((8,), al.f32)
                for element in al.static_range(8):
                    accum[element] = al.convert(0.0, al.f32)
                packed = al.make_local((1, 4), al.u32)
                values = al.view(packed, al.bf16, al.make_layout((1, 8), (8, 1)))
                for route in al.static_range(K):
                    packed[0] = al.amdgpu.raw_buffer_load_x4(
                        resource, B + rank * SLOT + OUT + (token * K + route) * D * 2 + vector * 16, 0, 17
                    )
                    for element in al.static_range(8):
                        accum[element] = accum[element] + al.convert(values[0, element], al.f32)
                for element in al.static_range(8):
                    output[token, vector, element] = al.convert(accum[element], al.bf16)

    return combine
