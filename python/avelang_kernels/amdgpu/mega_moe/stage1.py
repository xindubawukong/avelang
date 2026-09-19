"""Reuse Stage1 workgroups across destination-pool tiles after push completes."""

from functools import cache

import avelang
import avelang.language as al

from ..local_moe.intermediate_mxfp4 import make_intermediate_store
from ..local_moe.stage1 import make_stage1_compute
from ..local_moe.weight_mxfp4 import make_w13_resources
from .input_mxfp4_packed import make_mxfp4_packed_input
from .scheduler import make_scheduler
from .workspace import WorkspaceLayout


@cache
def make_stage1(config):
    layout = WorkspaceLayout(config)
    BM, BN, WORDS = config.stage1_tile_m, config.stage1_projection_n, config.stage1_lds_words
    I, D, E = config.intermediate, config.compute_hidden, config.local_experts
    SIZE, B, SLOT, L1 = layout.workspace_bytes, layout.rank_sym_buffer_base, layout.rank_slot_bytes, layout.l1_tokens
    ROW_BYTES, L2, SCALES, SC = config.input_token_bytes, layout.l2_tokens, layout.l2_scales, layout.scale_cols
    TB = BM // config.stage1_num_warps
    LOADS, SLICES, SEGMENTS = TB // 8, BM // 8, (BN + 127) // 128
    get_work = make_scheduler(layout, BM, I // BN)
    prepare_scales, prefetch_input, read_input = make_mxfp4_packed_input(config, config.solution.hidden, ROW_BYTES, 17)
    stage1_compute = make_stage1_compute(config, prefetch_input=prefetch_input, read_input=read_input)
    initialize_w13_resources = make_w13_resources(D, I, E, BN, I)
    store_intermediate = make_intermediate_store(I, SC, act_aux=17, scale_aux=17)

    @avelang.jit
    def stage1(
        heap: al.Pointer(al.u8),
        weight: al.Pointer(al.u32),
        scales: al.Pointer(al.u32),
        bias: al.Pointer(al.bf16),
        rank: al.u32,
        bias_enabled: al.u32,
    ):
        tid, logical = al.convert(al.thread_id(0), al.u32), al.convert(al.block_id(0), al.u32)
        wave, lane = al.amdgpu.readfirstlane(tid // 64), tid % 64
        memory = al.make_tensor(heap, al.u8, al.make_layout((SIZE,), (1,)))
        resource = al.amdgpu.make_rsrc(memory, SIZE)
        storage = al.make_shared((WORDS,), al.u32)
        active = al.convert(1, al.u32)
        while active != 0:
            expert, pool, rows, tile, found = get_work(resource, rank, logical)
            active = al.amdgpu.readfirstlane(found)
            if active != 0:
                row_base = B + rank * SLOT + L1 + pool * ROW_BYTES
                prepare_scales(resource, storage, row_base, tid)
                input_offsets = al.make_local((2,), al.u32)
                for load in al.static_range(LOADS):
                    row = wave * TB + load * 8 + lane // 8
                    vector = (lane % 8) ^ (row & 7)
                    input_offsets[load] = al.select(
                        row < rows, row_base + row * ROW_BYTES + vector * 16, al.convert(0xFFFFFFFF, al.u32)
                    )
                wr, sr, br = initialize_w13_resources(weight, scales, bias, expert, tile, bias_enabled)
                stage1_compute(resource, resource, wr, sr, br, storage, input_offsets, pool // BM, wave, lane)
                hidden = al.view(storage, al.f32, al.make_layout((BM, BN // 4, 4), (BN, 4, 1)))
                al.syncthreads()
                for batch in al.static_range(SLICES):
                    row = batch * 8 + tid // 32
                    if row < rows:
                        for segment in al.static_range(SEGMENTS):
                            col_lane = segment * 32 + tid % 32
                            values = hidden[row, col_lane]
                            store_intermediate(
                                values,
                                resource,
                                pool + row,
                                pool + row,
                                tile * BN,
                                col_lane,
                                al.convert(L2, al.u32),
                                al.convert(SCALES, al.u32),
                            )
                al.amdgpu.fence(1, 2)
                al.syncthreads()
                logical = logical + 256

    return stage1
