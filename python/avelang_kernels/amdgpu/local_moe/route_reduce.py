"""Local route output reduction with FP32 accumulation and one BF16 rounding."""

from functools import cache

import avelang
import avelang.language as al


@cache
def make_route_reduce(hidden: int, topk: int):
    D, TOPK = (hidden, topk)

    @avelang.jit
    def reduce_routes(src: al.Pointer(al.u32), dst: al.Pointer(al.bf16), tokens: al.u32):
        token = al.block_id(0)
        col = al.thread_id(0) + al.block_id(1) * 512
        if col >= D // 8:
            return
        source = al.make_tensor(src, al.u32, al.make_layout((al.convert(tokens, al.u64) * TOPK * D // 2,), (1,)))
        row = al.subview(source, (token * (TOPK * D // 2),), (TOPK * D // 2,), (1,))
        resource = al.amdgpu.make_rsrc(row, TOPK * D * 2)
        packed = al.make_local((TOPK, 4), al.u32)
        for slot in al.static_range(TOPK):
            packed[slot] = al.amdgpu.raw_buffer_load_x4(
                resource, al.convert(col * 16, al.u32), al.convert(slot * D * 2, al.u32), 0
            )
        values = al.view(packed, al.bf16, al.make_layout((TOPK, 8), (8, 1)))
        sums = al.make_local((8,), al.f32)
        for component in al.static_range(8):
            sums[component] = al.convert(0.0, al.f32)
        for slot in al.static_range(TOPK):
            for component in al.static_range(8):
                sums[component] = sums[component] + al.convert(values[slot, component], al.f32)
        result = al.make_local((1, 8), al.bf16)
        for component in al.static_range(8):
            result[0, component] = al.convert(sums[component], al.bf16)
        output = al.make_tensor(dst, al.bf16, al.make_layout((tokens, D // 8, 8), (D, 8, 1)))
        output[token, col] = result[0]

    return reduce_routes
