"""Packed transport input adapter for the local MoE W13 pipeline.

Scale preparation completes its own LDS repacking barriers. Act prefetch
is asynchronous; the W13 pipeline owns its wait and read barriers.
"""

from functools import cache

import avelang
import avelang.language as al


@cache
def make_mxfp4_packed_input(config, logical_hidden, row_bytes, cache_policy):
    D, LOGICAL = config.compute_hidden, logical_hidden
    BM, WM, WN = config.stage1_tile_m, config.stage1_wave_m, config.stage1_warps_n
    WAVES, WORDS, STRIDE = config.stage1_num_warps, config.stage1_lds_words, config.stage1_input_stage_words
    ROW_BYTES = row_bytes
    ACT_WORDS, SCALE_WORDS = BM * 32, STRIDE - BM * 32
    SX = WM // 32
    MR, TB, LOADS = WM // 16, BM // WAVES, BM // WAVES // 8
    RAW_WORDS, TILES = (ROW_BYTES - LOGICAL // 2) // 4, D // 256
    SCALE_TASKS = (TILES * (BM // 32) * 64) // 4
    TASKS_PER_THREAD = (SCALE_TASKS + WAVES * 16 - 1) // (WAVES * 16)
    SCALE_VECS = SCALE_WORDS // 4
    SCALE_LOADS = (SCALE_VECS + WAVES // 2 * 64 - 1) // (WAVES // 2 * 64)
    CACHE = cache_policy

    @avelang.jit
    def prepare_scales(
        resource: al.Tensor((4,), al.u32), storage: al.Tensor((WORDS,), al.u32), row_base: al.u32, tid: al.u32
    ):
        wave, lane = al.amdgpu.readfirstlane(tid // 64), tid % 64
        stage, local_wave = wave // (WAVES // 2), wave % (WAVES // 2)
        for load in al.static_range(SCALE_LOADS):
            vec = local_wave * 64 + lane + load * (WAVES // 2) * 64
            linear = stage * SCALE_VECS + vec
            row, row_vector = linear // (RAW_WORDS // 4), linear % (RAW_WORDS // 4)
            offset = al.select(
                vec < SCALE_VECS,
                row_base + row * ROW_BYTES + LOGICAL // 2 + row_vector * 16,
                al.convert(0xFFFFFFFF, al.u32),
            )
            # DMA adds lane*16; guard complete waves for a partial final batch.
            destination = (stage * STRIDE + ACT_WORDS + (local_wave * 64 + load * (WAVES // 2) * 64) * 4) * 4
            if local_wave * 64 + load * (WAVES // 2) * 64 < SCALE_VECS:
                al.amdgpu.raw_buffer_load_x4_lds(resource, storage, 16, offset, 0, destination, CACHE)
        al.amdgpu.s_waitcnt(0, 0, 0)
        al.syncthreads()
        packed = al.make_local((TASKS_PER_THREAD,), al.u32)
        quad, quad_lane = tid // 4, tid % 4
        for i in al.static_range(TASKS_PER_THREAD):
            task = (quad + i * WAVES * 16) % SCALE_TASKS
            block, row16 = task // 16, task % 16
            tile, wave_m = block // (BM // 32), block % (BM // 32)
            row = wave_m * 32 + row16 + 16 * (quad_lane & 1)
            raw_word = row * RAW_WORDS + tile * 2 + (quad_lane >> 1)
            raw = storage[(raw_word // SCALE_WORDS) * STRIDE + ACT_WORDS + raw_word % SCALE_WORDS]
            v0 = al.amdgpu.get_dpp(raw, raw, 0x00, 15, 15, 0)
            v1 = al.amdgpu.get_dpp(raw, raw, 0x55, 15, 15, 0)
            v2 = al.amdgpu.get_dpp(raw, raw, 0xAA, 15, 15, 0)
            v3 = al.amdgpu.get_dpp(raw, raw, 0xFF, 15, 15, 0)
            select = 0x0C0C0400 + quad_lane * 0x101
            p01 = al.amdgpu.perm(v1, v0, select)
            p23 = al.amdgpu.perm(v3, v2, select)
            packed[i] = al.amdgpu.perm(p23, p01, 0x05040100)
        al.syncthreads()
        for i in al.static_range(TASKS_PER_THREAD):
            task = (quad + i * WAVES * 16) % SCALE_TASKS
            block, row16 = task // 16, task % 16
            tile, wave_m = block // (BM // 32), block % (BM // 32)
            storage[
                (tile % 2) * STRIDE + ACT_WORDS + (tile // 2 * (BM // 32) + wave_m) * 64 + quad_lane * 16 + row16
            ] = packed[i]
        al.syncthreads()

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
        for load in al.static_range(LOADS):
            source_vec = (lane % 8) ^ ((lane // 8 + load * 8) & 7)
            offset = al.select(
                (offsets[load] != 0xFFFFFFFF) and (k * 256 + source_vec * 32 < LOGICAL),
                offsets[load] + k * 128,
                al.convert(0xFFFFFFFF, al.u32),
            )
            destination = ((k % 2) * STRIDE + wave * TB * 32 + load * 256) * 4
            al.amdgpu.raw_buffer_load_x4_lds(act_resource, storage, 16, offset, 0, destination, CACHE)

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
        lds = al.view(storage, al.u32, al.make_layout((2, BM, 8, 4), (STRIDE, 32, 4, 1)))
        wave_m = wave // WN
        for m in al.static_range(MR):
            for half in al.static_range(2):
                row = wave_m * WM + m * 16 + lane % 16
                fragments[m, half] = lds[stage, row, (lane // 16 + half * 4) ^ (row & 7)]
        for m32 in al.static_range(SX):
            scales[m32] = storage[
                stage * STRIDE + ACT_WORDS + (k // 2 * (BM // 32) + wave_m * (WM // 32) + m32) * 64 + lane
            ]

    return prepare_scales, prefetch_input, read_input
