"""Stage1 access to routed MXFP4 activations and native E8M0 scales."""

from functools import cache

import avelang
import avelang.language as al


@cache
def make_mxfp4_input(config):
    """Local routed input: issue DMA without waiting; read after caller's barrier.

    LDS act is [stage, k_group, row, vector8, word4], with XOR row swizzle.
    Scale words follow each stage's act. Each u32 packs four E8M0 scales.
    """
    D = config.compute_hidden
    BM, WN, WM = (config.stage1_tile_m, config.stage1_warps_n, config.stage1_wave_m)
    KG, MR = (config.stage1_k_groups, WM // 16)
    SX, TB = (WM // 32, BM // 4)
    LOADS, WORDS = (BM // 32, config.stage1_lds_words)
    STRIDE, ACT_WORDS = (config.stage1_input_stage_words, KG * BM * 32)

    @avelang.jit
    def prefetch_input(
        act_resource: al.Tensor((4,), al.u32),
        act_scale_resource: al.Tensor((4,), al.u32),
        storage: al.Tensor((WORDS,), al.u32),
        input_offsets: al.Tensor((2,), al.u32),
        block: al.u32,
        k: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        stage = k % 2
        for group in al.static_range(KG):
            for load in al.static_range(LOADS):
                offset = input_offsets[load] + group * (D // KG // 2) + k * 128
                destination = (stage * STRIDE + group * BM * 32 + wave * TB * 32 + load * 256) * 4
                al.amdgpu.raw_buffer_load_x4_lds(act_resource, storage, 16, offset, 0, destination, 0)
        if wave < BM // 32:
            for group in al.static_range(KG):
                offset_s = ((block * (BM // 32) + wave) * (D // 256) + k + group * (D // KG // 256)) * 256 + lane * 4
                destination_s = (stage * STRIDE + ACT_WORDS + (group * (BM // 32) + wave) * 64) * 4
                al.amdgpu.raw_buffer_load_x1_lds(act_scale_resource, storage, 4, offset_s, 0, destination_s, 0)

    @avelang.jit
    def read_input(
        storage: al.Tensor((WORDS,), al.u32),
        fragments: al.Tensor((MR, 2, 4), al.u32),
        scales: al.Tensor((2,), al.u32),
        k: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        stage = k % 2
        lds = al.view(storage, al.u32, al.make_layout((2, KG, BM, 8, 4), (STRIDE, BM * 32, 32, 4, 1)))
        group = wave // WN if KG == 2 else al.convert(0, al.u32)
        wave_m = al.convert(0, al.u32) if KG == 2 else wave // WN
        for m in al.static_range(MR):
            for half_k in al.static_range(2):
                row = wave_m * WM + m * 16 + lane % 16
                fragments[m, half_k] = lds[stage, group, row, lane // 16 + half_k * 4 ^ row & 7]
        for m32 in al.static_range(SX):
            scales[m32] = storage[stage * STRIDE + ACT_WORDS + (group * (BM // 32) + wave_m * SX + m32) * 64 + lane]

    return (prefetch_input, read_input)
