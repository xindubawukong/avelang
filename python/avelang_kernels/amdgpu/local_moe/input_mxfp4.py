"""Direct buffer-to-LDS copies into one linear Stage1 input tile."""

from functools import cache

import avelang
import avelang.language as al


@cache
def make_mxfp4_input(config):
    D, BM, TOPK = config.hidden, config.stage1_tile_m, config.topk
    MR, WORDS = BM // 16, config.stage1_lds_words
    ACT_WORDS = BM * 32

    @avelang.jit
    def prefetch_input(
        act: al.Tensor((4,), al.u32),
        scales: al.Tensor((4,), al.u32),
        routes: al.Tensor((4,), al.u32),
        storage: al.Tensor((WORDS,), al.u32),
        tokens: al.u32,
        block: al.u32,
        k: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        row, vector = wave * 8 + lane // 8, lane % 8
        route = al.amdgpu.raw_buffer_load_x1(routes, row * 4, 0, 0)
        token, slot = route & 0xFFFFFF, route >> 24
        offset = al.select(
            token < tokens and slot < TOPK,
            token * (D // 2) + k * 128 + vector * 16,
            al.convert(0xFFFFFFF0, al.u32),
        )
        al.amdgpu.raw_buffer_load_x4_lds(act, storage, 16, offset, 0, wave * 8 * 32 * 4, 0)
        if wave == 0:
            offset_s = (block * (D // 256) + k) * 256 + lane * 4
            al.amdgpu.raw_buffer_load_x1_lds(scales, storage, 4, offset_s, 0, ACT_WORDS * 4, 0)

    @avelang.jit
    def read_input(
        storage: al.Tensor((WORDS,), al.u32),
        fragments: al.Tensor((MR, 2, 4), al.u32),
        lane: al.u32,
    ) -> al.u32:
        lds = al.view(storage, al.u32, al.make_layout((BM, 8, 4), (32, 4, 1)))
        for m in al.static_range(MR):
            for half_k in al.static_range(2):
                fragments[m, half_k] = lds[m * 16 + lane % 16, lane // 16 + half_k * 4]
        return storage[ACT_WORDS + lane]

    return prefetch_input, read_input
