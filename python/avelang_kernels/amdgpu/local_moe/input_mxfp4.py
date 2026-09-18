"""Load routed activation fragments and one native scale word per K256 tile."""

from functools import cache

import avelang
import avelang.language as al


@cache
def make_mxfp4_input(config):
    D, BM, TOPK = config.hidden, config.stage1_tile_m, config.topk
    MR = BM // 16

    @avelang.jit
    def load_input(
        act: al.Tensor((4,), al.u32),
        scales: al.Tensor((4,), al.u32),
        routes: al.Tensor((4,), al.u32),
        fragments: al.Tensor((MR, 2, 4), al.u32),
        tokens: al.u32,
        block: al.u32,
        k: al.u32,
        lane: al.u32,
    ) -> al.u32:
        for m in al.static_range(MR):
            route = al.amdgpu.raw_buffer_load_x1(routes, (m * 16 + lane % 16) * 4, 0, 0)
            token, slot = route & 0xFFFFFF, route >> 24
            for half_k in al.static_range(2):
                values = al.full((4,), 0, al.u32)
                if token < tokens and slot < TOPK:
                    offset = token * (D // 2) + k * 128 + half_k * 64 + lane // 16 * 16
                    values = al.amdgpu.raw_buffer_load_x4(act, offset, 0, 0)
                fragments[m, half_k] = values
        offset_s = (block * (D // 256) + k) * 256 + lane * 4
        return al.amdgpu.raw_buffer_load_x1(scales, offset_s, 0, 0)

    return load_input
