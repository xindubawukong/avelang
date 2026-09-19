"""Packed-row input loads with per-K scale-byte assembly."""

from functools import cache

import avelang
import avelang.language as al


@cache
def make_mxfp4_packed_input(config, logical_hidden, row_bytes, cache_policy):
    BM, WN, WM = config.stage1_tile_m, config.stage1_warps_n, config.stage1_wave_m
    WORDS, STRIDE, ACT_WORDS = config.stage1_lds_words, config.stage1_input_stage_words, BM * 32
    TB, MR = BM // config.stage1_num_warps, WM // 16
    LOADS = TB // 8
    ROW_BYTES, LOGICAL, CACHE = row_bytes, logical_hidden, cache_policy

    @avelang.jit
    def prefetch_input(
        act_resource: al.Tensor((4,), al.u32),
        act_scale_resource: al.Tensor((4,), al.u32),
        storage: al.Tensor((WORDS,), al.u32),
        offsets: al.Tensor((2,), al.u32),
        block: al.u32,
        k: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        lds = al.view(storage, al.u32, al.make_layout((2, BM, 8, 4), (STRIDE, 32, 4, 1)))
        for load in al.static_range(LOADS):
            row = wave * TB + load * 8 + lane // 8
            vector = (lane % 8) ^ (row & 7)
            offset = al.select(
                (offsets[load] != 0xFFFFFFFF) and (k * 256 + vector * 32 < LOGICAL),
                offsets[load] + k * 128,
                al.convert(0xFFFFFFFF, al.u32),
            )
            lds[k % 2, row, lane % 8] = al.amdgpu.raw_buffer_load_x4(act_resource, offset, 0, CACHE)
        if wave == 0:
            row_base = al.amdgpu.readfirstlane(offsets[0])
            packed = al.convert(0, al.u32)
            for byte in al.static_range(4):
                row = lane % 16 + (byte % 2) * 16
                col = k * 8 + lane // 16 + (byte // 2) * 4
                offset = row_base + row * ROW_BYTES + LOGICAL // 2 + (col // 4) * 4
                word = al.amdgpu.raw_buffer_load_x1(act_scale_resource, offset, 0, CACHE)
                value = al.select(col < LOGICAL // 32, (word >> ((col % 4) * 8)) & 255, al.convert(0, al.u32))
                packed = packed | (value << (byte * 8))
            storage[(k % 2) * STRIDE + ACT_WORDS + lane] = packed

    @avelang.jit
    def read_input(
        storage: al.Tensor((WORDS,), al.u32),
        fragments: al.Tensor((MR, 2, 4), al.u32),
        scales: al.Tensor((2,), al.u32),
        k: al.u32,
        wave: al.u32,
        lane: al.u32,
    ):
        lds = al.view(storage, al.u32, al.make_layout((2, BM, 8, 4), (STRIDE, 32, 4, 1)))
        for m in al.static_range(MR):
            row = (wave // WN) * WM + m * 16 + lane % 16
            for half in al.static_range(2):
                fragments[m, half] = lds[k % 2, row, (lane // 16 + half * 4) ^ (row & 7)]
        scales[0] = storage[(k % 2) * STRIDE + ACT_WORDS + lane]

    return prefetch_input, read_input
