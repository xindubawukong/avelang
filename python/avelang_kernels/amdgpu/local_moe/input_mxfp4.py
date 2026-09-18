"""Direct global input reads for one 16x16 MFMA cell."""

from functools import cache

import avelang
import avelang.language as al


@cache
def make_mxfp4_input(config):
    BM, TOPK = config.stage1_tile_m, config.topk

    @avelang.jit
    def load_input_fragment(
        act: al.Tensor((4,), al.u32),
        scales: al.Tensor((4,), al.u32),
        routes: al.Tensor((4,), al.u32),
        hidden: al.u32,
        tokens: al.u32,
        block: al.u32,
        m16: al.u32,
        k128: al.u32,
        lane: al.u32,
    ) -> (al.Tensor((4,), al.u32), al.u32):
        row = m16 * 16 + lane % 16
        route = al.amdgpu.raw_buffer_load_x1(routes, row * 4, 0, 0)
        token, slot = route & 0xFFFFFF, route >> 24
        values = al.full((4,), 0, al.u32)
        scale = al.convert(0, al.u32)
        if token < tokens and slot < TOPK:
            offset = token * (hidden // 2) + k128 * 64 + (lane // 16) * 16
            values = al.amdgpu.raw_buffer_load_x4(act, offset, 0, 0)
            scale_offset = ((block * (BM // 32) + m16 // 2) * (hidden // 256) + k128 // 2) * 256 + lane * 4
            word = al.amdgpu.raw_buffer_load_x1(scales, scale_offset, 0, 0)
            scale = (word >> ((2 * (k128 % 2) + m16 % 2) * 8)) & 255
        return values, scale

    return load_input_fragment
