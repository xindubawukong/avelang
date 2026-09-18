"""Direct buffer-to-LDS copies into alternating linear Stage1 input tiles."""

from functools import cache

import avelang
import avelang.language as al


@cache
def make_mxfp4_input(config):
    D, BM, TOPK = config.hidden, config.stage1_tile_m, config.topk
    WM, WN = config.stage1_wave_m, config.stage1_warps_n
    MR, WORDS = WM // 16, config.stage1_lds_words
    LOADS, TB = BM // 32, BM // 4
    ACT_WORDS = BM * 32
    STRIDE = config.stage1_input_stage_words

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
        for load in al.static_range(LOADS):
            row, vector = wave * TB + load * 8 + lane // 8, lane % 8
            route = al.amdgpu.raw_buffer_load_x1(routes, row * 4, 0, 0)
            token, slot = route & 0xFFFFFF, route >> 24
            offset = al.select(
                token < tokens and slot < TOPK,
                token * (D // 2) + k * 128 + vector * 16,
                al.convert(0xFFFFFFF0, al.u32),
            )
            destination = (k % 2 * STRIDE + wave * TB * 32 + load * 256) * 4
            al.amdgpu.raw_buffer_load_x4_lds(act, storage, 16, offset, 0, destination, 0)
        if wave < BM // 32:
            offset_s = ((block * (BM // 32) + wave) * (D // 256) + k) * 256 + lane * 4
            destination_s = (k % 2 * STRIDE + ACT_WORDS + wave * 64) * 4
            al.amdgpu.raw_buffer_load_x1_lds(scales, storage, 4, offset_s, 0, destination_s, 0)

    @avelang.jit
    def read_input(
        storage: al.Tensor((WORDS,), al.u32),
        fragments: al.Tensor((MR, 2, 4), al.u32),
        k: al.u32,
        wave: al.u32,
        lane: al.u32,
    ) -> al.u32:
        lds = al.view(storage, al.u32, al.make_layout((2, BM, 8, 4), (STRIDE, 32, 4, 1)))
        for m in al.static_range(MR):
            for half_k in al.static_range(2):
                fragments[m, half_k] = lds[k % 2, (wave // WN) * WM + m * 16 + lane % 16, lane // 16 + half_k * 4]
        return storage[k % 2 * STRIDE + ACT_WORDS + (wave // WN) * 64 + lane]

    return prefetch_input, read_input
